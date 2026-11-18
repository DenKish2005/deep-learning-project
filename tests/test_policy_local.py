"""Actual ROBUST-001 train-only integration; opted-in missing/corrupt data FAIL.

Fixed before outputs: persisted train IDs0..11,13..17; seeds42/123/2026;
epochs0/1. Spawn0/2, batches7/5 on seed42. No training/test decoding/download.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import socket

import numpy as np
import pytest
import torch

from src import data
from src.utils import load_config, stable_seed, validate_controlled_configs
from tests.test_loaders import collect
from tests.test_policy import EVENTS, pixel_oracle, policy
from tests.test_policy_integration import geometry_oracle
from tests.test_split_local import snapshot


pytestmark = pytest.mark.skipif(os.environ.get('RUN_ROBUST_POLICY') != '1',
                                reason='Actual policy requires RUN_ROBUST_POLICY=1')
TRAIN_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
SEEDS, EPOCHS = (42, 123, 2026), (0, 1)


def tensor_hash(x):
    return hashlib.sha256(x.numpy().astype('<f4').tobytes()).hexdigest()


def announce(kind, value):
    print('ROBUST001_EVIDENCE ' + json.dumps({'kind': kind, 'value': value}, sort_keys=True))


@pytest.fixture
def actual(monkeypatch):
    root = Path(os.environ.get('CIFAR10_ROOT', 'data/raw'))
    before = snapshot(root, Path('data/splits'), Path('configs'), Path('report'), Path('outputs/figures'))
    decoded = []
    read = data._read_partition
    def train_only(root, train):
        assert train is True, 'ROBUST-001 must never decode official test'
        decoded.append('train')
        return read(root, train)
    def forbidden(*args, **kwargs): pytest.fail('ROBUST-001 must stay offline/read-only')
    monkeypatch.setattr(data, '_read_partition', train_only)
    monkeypatch.setattr(data, '_download_archive', forbidden)
    monkeypatch.setattr(data, 'ensure_cifar10', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    clean, robust = [load_config(Path('configs') / name) for name in ('clean_cnn.yaml', 'robust_cnn.yaml')]
    for cfg in (clean, robust):
        cfg['dataset']['root'] = root.as_posix()
        cfg['runtime']['device'] = 'cpu'
    validate_controlled_configs(clean, robust)
    clean_loaders = data.make_loaders(clean, 42, 0, partitions=('train', 'val'))
    robust_loaders = data.make_loaders(robust, 42, 0, policy=policy(), partitions=('train', 'val'))
    assert set(clean_loaders) == set(robust_loaders) == {'train', 'val'}
    for mode in ('train', 'val'):
        assert np.array_equal(clean_loaders[mode].dataset.indices, robust_loaders[mode].dataset.indices)
    assert set(TRAIN_IDS) <= set(robust_loaders['train'].dataset.indices.tolist())
    assert robust_loaders['val'].dataset.policy is None
    assert data._split_code_sha256() == '4ec051baa81ab0c70bfd2ca41f7a6750ca3b13f61930cd832b2a84dbd9244f96'
    assert data._normalization_code_sha256() == '1cbf01e93d4b1a838daf222c7e1672316117ad65ec1559c4e3075151dae8327c'
    yield clean_loaders, robust_loaders
    assert decoded and set(decoded) == {'train'}
    assert snapshot(root, Path('data/splits'), Path('configs'), Path('report'), Path('outputs/figures')) == before


def test_actual_policy_numeric_oracles_and_make_loaders(actual):
    clean, robust = actual
    ds = robust['train'].dataset
    base = ds.base
    stats = json.loads(Path('data/splits/normalization.json').read_text())['statistics']
    raw_hash = hashlib.sha256(base.data.tobytes()).hexdigest()
    counts = {event: 0 for event in EVENTS}
    streams, max_error, max_pixel_error = [], 0., 0.
    for seed in SEEDS:
        subset = data.IndexedCIFAR10(base, TRAIN_IDS, 'train', stats, seed, policy=policy())
        for epoch in EPOCHS:
            subset.set_epoch(epoch)
            values, metadata = [], []
            for position, original in enumerate(TRAIN_IDS):
                pixels = geometry_oracle(base, original, seed, epoch)
                key = stable_seed(seed, epoch, original, 'train_corruption')
                expected, audit = pixel_oracle(pixels, torch.Generator().manual_seed(key))
                produced, production_audit = policy()(pixels, torch.Generator().manual_seed(key))
                assert production_audit == audit
                max_pixel_error = max(max_pixel_error, (produced - expected).abs().max().item())
                torch.testing.assert_close(produced, expected, atol=1e-6, rtol=0)
                y, label, index = subset[position]
                assert (label, index) == (base.targets[original], original)
                assert torch.equal(y, (produced - subset.mean) / subset.std)
                normalized = (expected - subset.mean) / subset.std
                max_error = max(max_error, (y - normalized).abs().max().item())
                torch.testing.assert_close(y, normalized, atol=5e-6, rtol=0)
                counts[audit['event']] += 1
                values.append(y)
                metadata.append([original, audit['event'], audit['severity']])
            streams.append({'seed': seed, 'epoch': epoch, 'normalized_sha256': tensor_hash(torch.stack(values)),
                            'events_severities': metadata})
    assert all(counts.values()) and sum(counts.values()) == 102
    # Exercise returned make_loaders objects themselves on the predeclared IDs.
    for loaders in (clean, robust):
        loaders['train'].dataset.indices = np.asarray(TRAIN_IDS, dtype='<i8')
    clean_batch, robust_batch = next(iter(clean['train'])), next(iter(robust['train']))
    assert torch.equal(clean_batch[1], robust_batch[1]) and torch.equal(clean_batch[2], robust_batch[2])
    for x, original in zip(robust_batch[0], robust_batch[2].tolist()):
        assert torch.equal(x, ds[TRAIN_IDS.index(original)][0])
    # Bounded clean validation from the same verified train namespace, no policy.
    for position in range(17):
        a, b = clean['val'].dataset[position], robust['val'].dataset[position]
        assert torch.equal(a[0], b[0]) and a[1:] == b[1:]
        original = a[2]
        raw = torch.from_numpy(base.data[original].copy()).permute(2, 0, 1).float() / 255
        assert torch.equal(a[0], (raw - ds.mean) / ds.std)
    assert hashlib.sha256(base.data.tobytes()).hexdigest() == raw_hash
    announce('actual_numeric', {'ids': TRAIN_IDS, 'seeds': SEEDS, 'epochs': EPOCHS,
        'image_conditions': 102, 'event_counts': counts, 'streams': streams,
        'max_pixel_numpy2d_oracle_error': max_pixel_error, 'max_normalized_oracle_error': max_error,
        'make_loaders_explicit_production_policy': True, 'validation_samples': 17, 'test_decoded': False})


def test_actual_policy_spawn_batch_epoch_order_and_rng(actual):
    clean, robust = actual
    for loaders in (clean, robust):
        loaders['train'].dataset.indices = np.asarray(TRAIN_IDS, dtype='<i8')
    a, b = clean['train'].dataset, robust['train'].dataset
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    streams = []
    for epoch in EPOCHS:
        a.set_epoch(epoch)
        b.set_epoch(epoch)
        clean_values = collect(data._make_loader(a, 7, 0))
        serial = collect(data._make_loader(b, 7, 0))
        spawned = collect(data._make_loader(b, 5, 2))
        assert all(torch.equal(x, y) for x, y in zip(serial, spawned))
        assert torch.equal(clean_values[1], serial[1]) and torch.equal(clean_values[2], serial[2])
        assert sorted(serial[2].tolist()) == TRAIN_IDS
        assert serial[0].dtype == torch.float32 and serial[1].dtype == serial[2].dtype == torch.int64
        streams.append({'epoch': epoch, 'normalized_sha256': tensor_hash(serial[0]),
                        'ordered_ids': serial[2].tolist()})
    assert streams[0]['ordered_ids'] != streams[1]['ordered_ids']
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and after[2:] == before[1][2:]
    np.testing.assert_array_equal(after[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    announce('actual_spawn', {'seed': 42, 'workers': [0, 2], 'batch_sizes': [7, 5],
        'streams': streams, 'global_rng_unchanged': True, 'paired_order_labels_ids': True, 'test_decoded': False})
