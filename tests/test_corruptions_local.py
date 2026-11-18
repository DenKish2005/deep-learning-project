"""CORR-001 actual opt-in; fixed mechanics only, no model/tuning/download.

RUN_CIFAR_CORRUPTIONS=1 requires verified local inputs and saved reference.
Train IDs were fixed before observing outputs; test IDs only verify keyed inputs.
"""

import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
import pytest
import torch

from src import data
from src.corruptions import (gaussian_noise, keyed_test_epsilon,
                             keyed_test_generator, severity_params)
from src.utils import load_config, stable_seed, validate_controlled_configs
from tests.test_loaders import collect
from tests.test_split_local import snapshot


pytestmark = pytest.mark.skipif(os.environ.get('RUN_CIFAR_CORRUPTIONS') != '1',
                                reason='Actual noise requires RUN_CIFAR_CORRUPTIONS=1')
ROOT = Path(os.environ.get('CIFAR10_ROOT', 'data/raw'))
SPLITS = Path(os.environ.get('CIFAR10_SPLIT_DIR', 'data/splits'))
REFERENCE = Path('report/corr001_reference.npz')
TRAIN_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
TEST_IDS = [0, 71, 9999]
NAMESPACE = 'cifar10-python/original-test'


def tensor_hash(x):
    return hashlib.sha256(x.numpy().astype('<f4').tobytes()).hexdigest()


def announce(kind, value):
    print('CORR001_EVIDENCE ' + json.dumps({'kind': kind, 'value': value}, sort_keys=True))


@pytest.fixture(autouse=True)
def offline_and_preserved(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('Actual CORR-001 checks must stay offline')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)
    before = snapshot(ROOT, SPLITS, Path('configs'))
    yield
    assert snapshot(ROOT, SPLITS, Path('configs')) == before


@pytest.fixture(scope='module')
def actual_train():
    base = data.load_cifar10(ROOT, train=True, download=False)
    train, _ = data.load_split(SPLITS / data.SPLIT_NAME, base.targets)
    stats = data.load_normalization(base.data, base.targets, SPLITS / data.SPLIT_NAME,
                                    SPLITS / data.NORMALIZATION_NAME)['statistics']
    clean, robust = [load_config(Path('configs') / name) for name in
                     ('clean_cnn.yaml', 'robust_cnn.yaml')]
    validate_controlled_configs(clean, robust)
    for cfg in (clean, robust):
        for field, filename in [('split_sha256', data.SPLIT_NAME),
                                ('normalization_sha256', data.NORMALIZATION_NAME)]:
            assert cfg['dataset'][field] == hashlib.sha256((SPLITS / filename).read_bytes()).hexdigest()
    assert set(TRAIN_IDS) <= set(train.tolist())
    assert data._split_code_sha256() == '4ec051baa81ab0c70bfd2ca41f7a6750ca3b13f61930cd832b2a84dbd9244f96'
    assert data._normalization_code_sha256() == '1cbf01e93d4b1a838daf222c7e1672316117ad65ec1559c4e3075151dae8327c'
    return base, stats


def fixed_noise_probe(x, generator):
    """Single fixed event for boundary testing; not a robust training policy."""
    return gaussian_noise(x, .08, generator), {'event': 'fixed-noise-probe'}


def train_pixel_oracle(base, index, epoch):
    # Integer geometry oracle, separate from IndexedCIFAR10 implementation.
    gen = torch.Generator().manual_seed(stable_seed(42, epoch, index, 'geometry'))
    top, left = torch.randint(0, 9, (2,), generator=gen).tolist()
    padded = np.pad(base.data[index], ((4, 4), (4, 4), (0, 0)), constant_values=0)
    cropped = padded[top:top + 32, left:left + 32]
    if torch.rand((), generator=gen).item() < .5:
        cropped = cropped[:, ::-1]
    return torch.from_numpy(cropped.copy()).permute(2, 0, 1).float() / 255


def test_actual_train_noise_before_normalize_labels_ids_and_spawn(actual_train, monkeypatch):
    base, stats = actual_train
    # Training mechanics must not decode the official test partition.
    monkeypatch.setattr(data, '_read_partition', lambda *a, **kw: pytest.fail('Unexpected partition decoding'))
    raw_before = hashlib.sha256(base.data.tobytes()).hexdigest()
    mean = torch.tensor(stats['mean'], dtype=torch.float32)[:, None, None]
    std = torch.tensor(stats['std'], dtype=torch.float32)[:, None, None]
    ds = data.IndexedCIFAR10(base, TRAIN_IDS, 'train', stats, 42, policy=fixed_noise_probe)
    observations = []
    with np.load(REFERENCE, allow_pickle=False) as saved:
        assert set(saved.files) == {'train_ids', 'train_noisy', 'train_normalized', 'test_ids', 'test_epsilon'}
        assert saved['train_ids'].dtype == np.dtype('<i8')
        np.testing.assert_array_equal(saved['train_ids'], TRAIN_IDS[:3])
        for epoch in (0, 1):
            ds.set_epoch(epoch)
            for pos, index in enumerate(TRAIN_IDS):
                x = train_pixel_oracle(base, index, epoch)
                gen = torch.Generator().manual_seed(stable_seed(42, epoch, index, 'train_corruption'))
                epsilon = torch.randn(x.shape, generator=gen)
                noisy = (x + .08 * epsilon).clamp(0, 1)
                expected = (noisy - mean) / std
                y, label, original = ds[pos]
                assert original == index and label == base.targets[index]
                assert torch.equal(y, expected) and torch.isfinite(y).all()
                assert not torch.allclose(y, ((x - mean) / std + .08 * epsilon))
                if epoch == 0 and pos < 3:
                    assert saved['train_noisy'].dtype == saved['train_normalized'].dtype == np.dtype('<f4')
                    np.testing.assert_array_equal(saved['train_noisy'][pos], noisy.numpy())
                    np.testing.assert_array_equal(saved['train_normalized'][pos], y.numpy())
            reference = collect(data._make_loader(ds, 7, 0))
            parallel = collect(data._make_loader(ds, 5, 2))
            assert all(torch.equal(a, b) for a, b in zip(reference, parallel))
            assert sorted(reference[2].tolist()) == TRAIN_IDS
            observations.append({'epoch': epoch, 'normalized_stream_sha256': tensor_hash(reference[0])})
    assert hashlib.sha256(base.data.tobytes()).hexdigest() == raw_before
    announce('train_boundary', {'ids': TRAIN_IDS, 'run_seed': 42, 'sigma': .08,
        'epochs': observations, 'batch_sizes': [7, 5], 'workers': [0, 2],
        'normalize_after_noise': True, 'labels_ids_unchanged': True,
        'raw_split_stats_configs_preserved': True, 'test_decoded': False,
        'reference_sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest()})


def test_actual_original_test_fixed_keyed_input_contract(actual_train):
    base = data.load_cifar10(ROOT, train=False, download=False)
    before = hashlib.sha256(base.data.tobytes()).hexdigest(), tuple(base.targets)
    observations = []
    with np.load(REFERENCE, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['test_ids'], TEST_IDS)
        assert saved['test_ids'].dtype == np.dtype('<i8') and saved['test_epsilon'].dtype == np.dtype('<f4')
        for pos, index in enumerate(TEST_IDS):
            x = torch.from_numpy(base.data[index].copy()).permute(2, 0, 1).float() / 255
            epsilon = keyed_test_epsilon(index, namespace=NAMESPACE)
            np.testing.assert_array_equal(saved['test_epsilon'][pos], epsilon.numpy())
            hashes = []
            for severity in range(1, 6):
                sigma = severity_params('gaussian_noise', severity)['sigma']
                y = gaussian_noise(x, sigma, keyed_test_generator(index, namespace=NAMESPACE))
                assert torch.equal(y, (x + sigma * epsilon).clamp(0, 1))
                hashes.append(tensor_hash(y))
            observations.append({'original_index': index, 'epsilon_sha256': tensor_hash(epsilon),
                                 'severity_1_to_5_pixels_sha256': hashes})
    assert before == (hashlib.sha256(base.data.tobytes()).hexdigest(), tuple(base.targets))
    announce('test_keyed_contract', {'namespace': NAMESPACE, 'eval_seed': 100494,
        'fixed_inputs_only': True, 'no_model_or_tuning': True, 'samples': observations})
