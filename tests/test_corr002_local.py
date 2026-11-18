"""Actual CORR-002 mechanics on pre-fixed persisted train IDs, no model/tuning.

RUN_CIFAR_CORRUPTIONS=1 requires real verified inputs and read-only references.
FixedCorruptionProbe is a test injection callable, not a training policy.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src import data
from src.corruptions import apply_corruption, severity_params
from src.utils import stable_seed
from tests.test_blur import blur_oracle
from tests.test_corruption_dispatch import CONDITIONS
from tests.test_corruptions_local import (actual_train, offline_and_preserved,
    train_pixel_oracle, tensor_hash, TRAIN_IDS, REFERENCE as OLD_REFERENCE)
from tests.test_loaders import collect
from tests.test_split_local import snapshot


pytestmark = pytest.mark.skipif(os.environ.get('RUN_CIFAR_CORRUPTIONS') != '1',
                                reason='Actual dispatcher requires RUN_CIFAR_CORRUPTIONS=1')
REFERENCE = Path('report/corr002_reference.npz')
REFERENCE_CONDITIONS = [('clean', 0), ('gaussian_noise', 3), ('gaussian_blur', 3), ('brightness', 3)]


class FixedCorruptionProbe:
    def __init__(self, kind, severity):
        self.kind, self.severity = kind, severity

    def __call__(self, x, generator):
        return apply_corruption(x, self.kind, self.severity, generator), {'event': 'fixed-dispatch-probe'}


@pytest.fixture(autouse=True)
def references_preserved():
    before = snapshot(Path('report'))
    yield
    assert snapshot(Path('report')) == before


def announce(kind, value):
    print('CORR002_EVIDENCE ' + json.dumps({'kind': kind, 'value': value}, sort_keys=True))


def test_actual_all_conditions_before_normalize_and_saved_references(actual_train, monkeypatch):
    base, stats = actual_train
    monkeypatch.setattr(data, '_read_partition', lambda *a, **kw: pytest.fail('No test decoding in CORR-002 train probe'))
    raw_before = hashlib.sha256(base.data.tobytes()).hexdigest()
    mean = torch.tensor(stats['mean'], dtype=torch.float32)[:, None, None]
    std = torch.tensor(stats['std'], dtype=torch.float32)[:, None, None]
    max_blur_error = 0.
    observations = []
    with np.load(REFERENCE, allow_pickle=False) as saved, np.load(OLD_REFERENCE, allow_pickle=False) as old:
        assert set(saved.files) == {'train_ids', 'kinds', 'severities', 'train_pixels', 'train_normalized'}
        np.testing.assert_array_equal(saved['train_ids'], TRAIN_IDS[:3])
        assert saved['train_ids'].dtype == saved['severities'].dtype == np.dtype('<i8')
        assert list(zip(saved['kinds'].tolist(), saved['severities'].tolist())) == REFERENCE_CONDITIONS
        assert saved['train_pixels'].dtype == saved['train_normalized'].dtype == np.dtype('<f4')
        assert saved['train_pixels'].shape == saved['train_normalized'].shape == (4, 3, 3, 32, 32)
        for kind, severity in CONDITIONS:
            ds = data.IndexedCIFAR10(base, TRAIN_IDS, 'train', stats, 42,
                                    policy=FixedCorruptionProbe(kind, severity))
            for epoch in (0, 1):
                ds.set_epoch(epoch)
                outputs = []
                for pos, index in enumerate(TRAIN_IDS):
                    x = train_pixel_oracle(base, index, epoch)
                    seed = stable_seed(42, epoch, index, 'train_corruption')
                    gen = torch.Generator().manual_seed(seed)
                    before = gen.get_state().clone()
                    pixels = apply_corruption(x, kind, severity, gen)
                    if kind == 'clean': expected = x.clone()
                    elif kind == 'gaussian_noise':
                        epsilon = torch.randn(x.shape, generator=torch.Generator().manual_seed(seed))
                        expected = (x + severity_params(kind, severity)['sigma'] * epsilon).clamp(0, 1)
                    elif kind == 'gaussian_blur':
                        expected = blur_oracle(x, **severity_params(kind, severity))
                        max_blur_error = max(max_blur_error, (expected - pixels).abs().max().item())
                    else:
                        expected = x * severity_params(kind, severity)['factor']
                    torch.testing.assert_close(pixels, expected, atol=1e-6, rtol=0)
                    if kind != 'gaussian_noise': assert torch.equal(before, gen.get_state())
                    y, label, original = ds[pos]
                    assert label == base.targets[index] and original == index
                    assert torch.equal(y, (pixels - mean) / std)
                    torch.testing.assert_close(y, (expected - mean) / std, atol=5e-6, rtol=0)
                    assert torch.isfinite(y).all()
                    outputs.append(y)
                    if epoch == 0 and pos < 3:
                        if (kind, severity) in REFERENCE_CONDITIONS:
                            c = REFERENCE_CONDITIONS.index((kind, severity))
                            np.testing.assert_array_equal(saved['train_pixels'][c, pos], pixels.numpy())
                            np.testing.assert_array_equal(saved['train_normalized'][c, pos], y.numpy())
                        if (kind, severity) == ('gaussian_noise', 3):
                            np.testing.assert_array_equal(old['train_noisy'][pos], pixels.numpy())
                            np.testing.assert_array_equal(old['train_normalized'][pos], y.numpy())
                observations.append({'kind': kind, 'severity': severity, 'epoch': epoch,
                                     'normalized_sha256': tensor_hash(torch.stack(outputs))})
    assert hashlib.sha256(base.data.tobytes()).hexdigest() == raw_before
    announce('all_conditions_train', {'ids': TRAIN_IDS, 'epochs': [0, 1], 'run_seed': 42,
        'conditions': 16, 'image_conditions': 544, 'max_blur_oracle_error': max_blur_error,
        'normalize_after_dispatch': True, 'labels_ids_unchanged': True, 'test_decoded': False,
        'reference_sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest(), 'streams': observations})


def test_actual_dispatch_spawn_and_batch_invariance(actual_train):
    base, stats = actual_train
    observations = []
    for kind, severity in REFERENCE_CONDITIONS:
        ds = data.IndexedCIFAR10(base, TRAIN_IDS, 'train', stats, 42,
                                policy=FixedCorruptionProbe(kind, severity))
        for epoch in (0, 1):
            ds.set_epoch(epoch)
            serial = collect(data._make_loader(ds, 7, 0))
            parallel = collect(data._make_loader(ds, 5, 2))
            assert all(torch.equal(a, b) for a, b in zip(serial, parallel))
            assert sorted(serial[2].tolist()) == TRAIN_IDS
            assert serial[0].dtype == torch.float32 and serial[1].dtype == serial[2].dtype == torch.int64
            observations.append({'kind': kind, 'severity': severity, 'epoch': epoch,
                                 'normalized_sha256': tensor_hash(serial[0])})
    announce('spawn', {'workers': [0, 2], 'batch_sizes': [7, 5], 'streams': observations})
