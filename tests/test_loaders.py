"""TEST-D04/D05/C08: offline keyed transforms, batches, spawn and injection boundary."""

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import data
from src.utils import load_config, stable_seed


def identity_policy(x, generator):
    torch.rand(100, generator=generator)  # Consuming policy draws cannot move geometry.
    return x.clone(), {'event': 'synthetic identity'}


def half_policy(x, generator):
    assert x.dtype == torch.float32 and x.shape == (3, 32, 32)
    assert 0 <= x.min() <= x.max() <= 1
    return x * .5, {'event': 'synthetic boundary oracle'}


@pytest.fixture
def base_stats():
    raw = np.random.default_rng(18).integers(0, 256, (17, 32, 32, 3), dtype=np.uint8)
    base = SimpleNamespace(data=raw, targets=np.arange(17) % 10, train=True,
                           transform=None, target_transform=None)
    return base, data.compute_normalization(raw, np.arange(11))


def dataset(base_stats, mode='train', seed=42, epoch=0, policy=None):
    base, stats = base_stats
    base = deepcopy(base)
    base.train = mode != 'test'
    return data.IndexedCIFAR10(base, np.arange(17), mode, stats, seed, epoch, policy)


def collect(loader):
    batches = list(loader)
    return tuple(torch.cat([batch[i] for batch in batches]) for i in range(3))


def test_transform_order_dtype_shape_labels_and_geometry_oracle(base_stats):
    base, stats = base_stats
    clean = dataset(base_stats)
    robust = dataset(base_stats, policy=half_policy)
    raw = base.data.copy()
    original = 3
    gen = torch.Generator().manual_seed(stable_seed(42, 0, original, 'geometry'))
    top, left = torch.randint(0, 9, (2,), generator=gen).tolist()
    padded = np.pad(raw[original], ((4, 4), (4, 4), (0, 0)), constant_values=0)
    cropped = padded[top:top + 32, left:left + 32]
    if torch.rand((), generator=gen).item() < .5: cropped = cropped[:, ::-1]
    pixels = torch.from_numpy(cropped.copy()).permute(2, 0, 1).float() / 255
    mean = torch.tensor(stats['mean'], dtype=torch.float32)[:, None, None]
    std = torch.tensor(stats['std'], dtype=torch.float32)[:, None, None]
    image, label, index = robust[original]
    assert image.dtype == torch.float32 and image.shape == (3, 32, 32) and image.is_contiguous()
    assert type(label) is type(index) is int
    assert label == int(base.targets[index]) and index == original
    assert torch.equal(image, (pixels * .5 - mean) / std)
    assert not torch.equal(image, clean[original][0] * .5)
    np.testing.assert_array_equal(base.data, raw)


@pytest.mark.parametrize('mode', ['val', 'test'])
def test_deterministic_eval_no_geometry_shuffle_or_policy(base_stats, mode):
    ds = dataset(base_stats, mode)
    other = dataset(base_stats, mode, seed=2026, epoch=8)
    base, stats = base_stats
    for i in range(17):
        expected = torch.from_numpy(base.data[i].copy()).permute(2, 0, 1).float() / 255
        mean = torch.tensor(stats['mean'], dtype=torch.float32)[:, None, None]
        std = torch.tensor(stats['std'], dtype=torch.float32)[:, None, None]
        assert torch.equal(ds[i][0], (expected - mean) / std)
        assert torch.equal(ds[i][0], other[i][0])
    x, y, ids = collect(data._make_loader(ds, 6))
    assert ids.tolist() == list(range(17))
    assert ds.namespace.endswith('original-test' if mode == 'test' else 'original-train')
    with pytest.raises(ValueError, match='train-only'):
        dataset(base_stats, mode, policy=identity_policy)


def test_epoch_paired_order_geometry_nonfull_and_global_rng_isolation(base_stats):
    clean, robust = dataset(base_stats), dataset(base_stats, policy=identity_policy)
    before = random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone()
    loaders = [data._make_loader(ds, 6) for ds in (clean, robust)]
    a, b = (collect(loader) for loader in loaders)
    for left, right in zip(a, b): assert torch.equal(left, right)
    assert a[0].dtype == torch.float32 and a[0].shape == (17, 3, 32, 32)
    assert a[1].dtype == a[2].dtype == torch.int64
    assert sorted(a[2].tolist()) == list(range(17))
    np.testing.assert_array_equal(a[1].numpy(), base_stats[0].targets[a[2]])
    assert [len(batch[0]) for batch in loaders[0]] == [6, 6, 5]
    for left, right in zip(a, collect(loaders[0])): assert torch.equal(left, right)
    clean.set_epoch(1)
    robust.set_epoch(1)
    c, d = (collect(loader) for loader in loaders)
    for left, right in zip(c, d): assert torch.equal(left, right)
    assert not torch.equal(a[2], c[2])
    assert not torch.equal(a[0][torch.argsort(a[2])], c[0][torch.argsort(c[2])])
    assert random.getstate() == before[0]
    now = np.random.get_state()
    assert now[0] == before[1][0] and now[2:] == before[1][2:]
    np.testing.assert_array_equal(now[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_spawn_workers_two_equal_reference_across_epochs(base_stats):
    # Actual spawn processes even on Windows; no conditional worker SKIP.
    reference, parallel = dataset(base_stats), dataset(base_stats)
    single = data._make_loader(reference, 6, 0)
    spawned = data._make_loader(parallel, 6, 2)
    assert spawned.persistent_workers is False and spawned.pin_memory is False
    for epoch in (0, 1):
        reference.set_epoch(epoch)
        parallel.set_epoch(epoch)
        for a, b in zip(collect(single), collect(spawned)): assert torch.equal(a, b)


@pytest.mark.parametrize('kind', ['dtype', 'shape', 'range', 'nan', 'mutation', 'tuple'])
def test_bad_policy_contract(base_stats, kind):
    def bad(x, generator):
        if kind == 'dtype': return x.double(), {}
        if kind == 'shape': return x[0], {}
        if kind == 'range': return x + 2, {}
        if kind == 'nan': return x * float('nan'), {}
        if kind == 'mutation': x.zero_(); return x, {}
        return x
    with pytest.raises(ValueError, match='policy'): dataset(base_stats, policy=bad)[0]


@pytest.mark.parametrize('field,value', [('run_seed', True), ('run_seed', -1),
    ('epoch', 1.0), ('epoch', -1), ('mode', 'bad'), ('policy', 42)])
def test_invalid_dataset_arguments(base_stats, field, value):
    base, stats = base_stats
    kwargs = dict(base=base, indices=np.arange(17), mode='train', normalization=stats, run_seed=42)
    kwargs[field] = value
    with pytest.raises(ValueError): data.IndexedCIFAR10(**kwargs)


def test_partition_labels_raw_and_stats_validation(base_stats):
    base, stats = base_stats
    base.train = False
    with pytest.raises(ValueError, match='partition'):
        data.IndexedCIFAR10(base, [0], 'train', stats, 42)
    base.train = True
    base.transform = lambda x: x
    with pytest.raises(ValueError, match='raw'): data.IndexedCIFAR10(base, [0], 'train', stats, 42)
    base.transform = None
    base.targets = [10] * 17
    with pytest.raises(ValueError, match='labels'): data.IndexedCIFAR10(base, [0], 'train', stats, 42)


def test_make_loaders_verification_boundary_no_test_decoding(base_stats, monkeypatch, tmp_path):
    base, stats = base_stats
    config = load_config(Path('configs/clean_cnn.yaml'))
    for path_key, hash_key in [('split_path', 'split_sha256'), ('normalization_path', 'normalization_sha256')]:
        path = tmp_path / path_key
        path.write_bytes(b'synthetic identity')
        config['dataset'][path_key] = str(path)
        config['dataset'][hash_key] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Machine path validation is tested elsewhere; use portable repository relative paths.
    monkeypatch.setattr(data, 'validate_config', lambda cfg: None)
    calls = []
    def load(root, train, download):
        assert download is False
        calls.append(train)
        result = deepcopy(base)
        result.train = train
        return result
    monkeypatch.setattr(data, 'load_cifar10', load)
    monkeypatch.setattr(data, 'load_split', lambda *a: (np.arange(11, dtype='<i8'), np.arange(11, 17, dtype='<i8')))
    monkeypatch.setattr(data, 'load_normalization', lambda *a: {'statistics': stats})
    config['runtime']['device'] = 'cpu'
    loaders = data.make_loaders(config, 42, 0, partitions=('train', 'val'))
    assert calls == [True] and set(loaders) == {'train', 'val'}
    assert loaders['train'].drop_last is False and loaders['val'].pin_memory is False
    assert data._make_loader(loaders['train'].dataset, 4, device='cuda').pin_memory is True
    assert data._make_loader(loaders['train'].dataset, 4, device='cuda', pin_memory=False).pin_memory is False
    config['runtime']['persistent_workers'] = True
    with pytest.raises(ValueError, match='persistent_workers'): data.make_loaders(config, 42, 0)
    config['runtime']['persistent_workers'] = False
    calls.clear()
    assert set(data.make_loaders(config, 123, 1)) == {'train', 'val', 'test'}
    assert calls == [True, False]
    config['dataset']['normalization_sha256'] = None
    calls.clear()
    with pytest.raises(data.DataIntegrityError, match='normalization_sha256'): data.make_loaders(config, 42, 0)
    assert calls == []
    config['augmentation']['enabled'] = True
    with pytest.raises(ValueError, match='policy'): data.make_loaders(config, 42, 0)


def test_stable_seed_golden_encoding_types_and_fresh_process():
    parts = (42, 0, 3, 'geometry')
    payload = b'csci494-stable-seed-v1\n[42,0,3,"geometry"]'
    expected = int(hashlib.sha256(payload).hexdigest(), 16) % (2 ** 63)
    assert stable_seed(*parts) == expected
    assert stable_seed(42, '0') != stable_seed('42', 0)
    assert stable_seed('train_corruption') != stable_seed('cifar10-test')
    assert stable_seed('ab', 'c') != stable_seed('a', 'bc')
    for bad in (True, 1.0, None, {}, np.int64(4)):
        with pytest.raises(TypeError): stable_seed(bad)
    env = dict(os.environ, PYTHONHASHSEED='random')
    command = [sys.executable, '-c', 'from src.utils import stable_seed; print(stable_seed(42,0,3,"geometry"))']
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == expected
