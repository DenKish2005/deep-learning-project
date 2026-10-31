"""Actual DATA-003 opt-in: no training, tuning, downloads or corruptions.

RUN_CIFAR_NORMALIZATION=1 requires existing verified raw/split/stats. Missing or
corrupt artifacts FAIL. Test images are read only for the fixed loader contract.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
import pytest
import torch

from scripts import prepare_normalization as cli
from src import data
from src.utils import load_config, validate_controlled_configs
from tests.test_loaders import collect, identity_policy
from tests.test_split_local import snapshot


pytestmark = pytest.mark.skipif(os.environ.get('RUN_CIFAR_NORMALIZATION') != '1',
                                reason='Actual stats/loaders require RUN_CIFAR_NORMALIZATION=1')
ROOT = Path(os.environ.get('CIFAR10_ROOT', 'data/raw'))
SPLITS = Path(os.environ.get('CIFAR10_SPLIT_DIR', 'data/splits'))
STATS = SPLITS / data.NORMALIZATION_NAME


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*a, **kw): pytest.fail('Actual DATA-003 checks must remain offline')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)


def announce(kind, value):
    print('DATA003_EVIDENCE ' + json.dumps({'kind': kind, 'value': value}, sort_keys=True))


def configs():
    clean, robust = [load_config(Path('configs') / name) for name in ('clean_cnn.yaml', 'robust_cnn.yaml')]
    validate_controlled_configs(clean, robust)
    for cfg in (clean, robust):
        cfg['runtime']['device'] = 'cpu'  # Explicit mechanics reference, no model.
        cfg['dataset']['root'] = ROOT.as_posix()
        cfg['dataset']['split_path'] = (SPLITS / data.SPLIT_NAME).as_posix()
        cfg['dataset']['normalization_path'] = STATS.as_posix()
    return clean, robust


def test_actual_stats_independent_integer_oracle_and_reuse(monkeypatch):
    original = data._read_partition
    def train_only(root, train):
        assert train is True, 'Stats must not decode original test'
        return original(root, train)
    monkeypatch.setattr(data, '_read_partition', train_only)
    before = snapshot(ROOT, SPLITS)
    base = data.load_cifar10(ROOT, train=True)
    train, val = data.load_split(SPLITS / data.SPLIT_NAME, base.targets)
    record = data.load_normalization(base.data, base.targets, SPLITS / data.SPLIT_NAME, STATS)
    stats = record['statistics']
    assert stats['samples'] == 45000 and stats['pixels_per_channel'] == 46080000
    assert stats['train_indices_sha256'] == data._array_sha256(train)
    # Independent exact integer accumulation before converting scale/moments.
    sums, squares = np.zeros(3, dtype=np.uint64), np.zeros(3, dtype=np.uint64)
    for start in range(0, len(train), 137):
        values = base.data[train[start:start + 137]].astype(np.uint64)
        sums += values.sum(axis=(0, 1, 2), dtype=np.uint64)
        squares += (values * values).sum(axis=(0, 1, 2), dtype=np.uint64)
    mean = sums.astype(np.float64) / (45000 * 1024 * 255)
    std = np.sqrt(squares.astype(np.float64) / (45000 * 1024 * 255**2) - mean**2)
    np.testing.assert_allclose(stats['mean'], mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(stats['std'], std, rtol=0, atol=1e-12)
    assert data.prepare_normalization(base.data, base.targets, SPLITS / data.SPLIT_NAME, STATS) == record
    args = ['--root', str(ROOT), '--split-path', str(SPLITS / data.SPLIT_NAME), '--output', str(STATS)]
    assert cli.main(args) == cli.main(args + ['--verify-only']) == 0
    assert snapshot(ROOT, SPLITS) == before
    for config in configs():
        assert config['dataset']['normalization_sha256'] == data._hash(STATS, 'sha256')
    announce('normalization', {'statistics': stats, 'file_sha256': data._hash(STATS, 'sha256'),
        'semantic_sha256': record['statistics_sha256'], 'integer_oracle_atol': 1e-12,
        'max_mean_error': float(np.max(np.abs(np.array(stats['mean']) - mean))),
        'max_std_error': float(np.max(np.abs(np.array(stats['std']) - std))),
        'raw_split_stats_hashes_mtimes_preserved': True, 'test_decoded': False})


def test_actual_all_typed_batches_coverage_labels_and_eval_contract():
    before = snapshot(ROOT, SPLITS)
    cfg, _ = configs()
    loaders = data.make_loaders(cfg, 42, 0)
    results = {}
    for mode, loader in loaders.items():
        ids, labels, sizes, digest = [], [], [], hashlib.sha256()
        base = loader.dataset.base
        mean, std = loader.dataset.mean, loader.dataset.std
        for x, y, index in loader:
            assert x.dtype == torch.float32 and x.shape == (len(index), 3, 32, 32)
            assert y.dtype == index.dtype == torch.int64 and torch.isfinite(x).all()
            np.testing.assert_array_equal(y.numpy(), np.asarray(base.targets)[index.numpy()])
            if mode != 'train':
                expected = torch.from_numpy(base.data[index.numpy()].copy()).permute(0, 3, 1, 2).float() / 255
                assert torch.equal(x, (expected - mean) / std)
            ids.extend(index.tolist()); labels.extend(y.tolist()); sizes.append(len(index))
            digest.update(x.numpy().astype('<f4', copy=False).tobytes())
        expected_ids = loader.dataset.indices
        np.testing.assert_array_equal(np.sort(ids), np.sort(expected_ids))
        assert len(set(ids)) == len(expected_ids)
        if mode != 'train': assert ids == expected_ids.tolist()
        counts = np.bincount(labels, minlength=10).tolist()
        assert counts == [{'train': 4500, 'val': 500, 'test': 1000}[mode]] * 10
        results[mode] = {'samples': len(ids), 'batches': len(sizes), 'last_batch': sizes[-1],
            'images_dtype': 'float32', 'image_shape': ['N', 3, 32, 32], 'labels_ids_dtype': 'int64',
            'class_counts': counts, 'label_alignment': True, 'unique_complete_ids': True,
            'order_semantic_sha256': data._array_sha256(np.asarray(ids, dtype='<i8')),
            'normalized_images_stream_sha256': digest.hexdigest(), 'namespace': loader.dataset.namespace}
    assert results['train']['batches'] == 352 and results['train']['last_batch'] == 72
    assert snapshot(ROOT, SPLITS) == before
    announce('full_loaders', results)


def test_actual_paired_boundary_epoch_workers_and_model_seed_independence(monkeypatch):
    original = data._read_partition
    def train_only(root, train):
        assert train is True
        return original(root, train)
    monkeypatch.setattr(data, '_read_partition', train_only)
    before = snapshot(ROOT, SPLITS)
    clean_cfg, robust_cfg = configs()
    clean = data.make_loaders(clean_cfg, 42, 0, partitions=('train', 'val'))
    robust = data.make_loaders(robust_cfg, 42, 0, policy=identity_policy, partitions=('train', 'val'))
    train = clean['train'].dataset.indices
    for epoch in (0, 1):
        clean['train'].dataset.set_epoch(epoch)
        robust['train'].dataset.set_epoch(epoch)
        a, b = next(iter(clean['train'])), next(iter(robust['train']))
        for left, right in zip(a, b): assert torch.equal(left, right)
        if epoch == 0: first = a[2].clone()
        else: assert not torch.equal(first, a[2])
    # Actual raw train subset, same persisted IDs/stats. Spawn2 vs workers0,
    # changing batch size and epoch; no scientific run and no test decoding.
    base = clean['train'].dataset.base
    stats = data.load_normalization(base.data, base.targets, SPLITS / data.SPLIT_NAME, STATS)['statistics']
    for seed in (42, 123, 2026):
        subset = data.IndexedCIFAR10(base, train[:129], 'train', stats, seed)
        assert subset.indices.tolist() == train[:129].tolist()
        if seed == 42:
            parallel = data.IndexedCIFAR10(base, train[:129], 'train', stats, seed, policy=identity_policy)
            for epoch in (0, 1):
                subset.set_epoch(epoch); parallel.set_epoch(epoch)
                for left, right in zip(collect(data._make_loader(subset, 64)),
                                        collect(data._make_loader(parallel, 37, 2))):
                    assert torch.equal(left, right)
    assert snapshot(ROOT, SPLITS) == before
    announce('paired_workers', {'paired_order_geometry': True, 'policy': 'synthetic identity only',
        'epochs': [0, 1], 'workers': [0, 2], 'spawn_actual_train_ids': 129, 'batch_sizes': [64, 37],
        'bitwise_equal': True, 'model_seeds_preserve_split_stats': [42, 123, 2026],
        'test_decoded': False, 'raw_split_stats_hashes_mtimes_preserved': True})
