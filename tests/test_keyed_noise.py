"""TEST-C03/U02: fixed original-test keys, common epsilon, order/spawn/processes.

All images here are synthetic. This fixture is not a production dataset wrapper.
"""

from copy import deepcopy
import hashlib
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.corruptions import (gaussian_noise, keyed_test_epsilon,
                             keyed_test_generator, severity_params)
from src.utils import seed_worker, stable_seed


NAMESPACE = 'cifar10-python/original-test'
IDS = (9999, 0, 71, 7, 500, 42, 888, 1, 123, 2026, 494)


@pytest.mark.parametrize('index', [0, 7, 9999])
def test_exact_seed_payload_and_common_preclamp_epsilon(index, monkeypatch):
    payload = f'csci494-stable-seed-v1\n[100494,"cifar10-test",{index},"gaussian_noise","v1"]'
    expected_seed = int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest(), 'big') & ((1 << 63) - 1)
    generator = keyed_test_generator(index, namespace=NAMESPACE)
    assert generator.device.type == 'cpu' and generator.initial_seed() == expected_seed
    assert generator.initial_seed() == stable_seed(100494, 'cifar10-test', index, 'gaussian_noise', 'v1')
    epsilon = keyed_test_epsilon(index, namespace=NAMESPACE)
    oracle = torch.randn((3, 32, 32), generator=torch.Generator().manual_seed(expected_seed))
    assert torch.equal(epsilon, oracle)
    assert epsilon.shape == (3, 32, 32) and epsilon.dtype == torch.float32
    assert epsilon.device.type == 'cpu' and torch.isfinite(epsilon).all()
    assert not torch.equal(epsilon, keyed_test_epsilon((index + 1) % 10000, namespace=NAMESPACE))
    draw = torch.randn
    observed = []
    def capture(*args, **kwargs):
        result = draw(*args, **kwargs)
        observed.append(result.clone())
        return result
    monkeypatch.setattr(torch, 'randn', capture)
    x = torch.full_like(epsilon, .5)
    # Six model identities, all five severities. No model is instantiated.
    with torch.random.fork_rng(devices=[]):
        for model in ('clean_cnn', 'robust_cnn'):
            for seed in (42, 123, 2026):
                torch.random.default_generator.manual_seed(seed)
                for severity in range(1, 6):
                    sigma = severity_params('gaussian_noise', severity)['sigma']
                    y = gaussian_noise(x, sigma, keyed_test_generator(index, namespace=NAMESPACE))
                    assert torch.equal(y, (x + sigma * epsilon).clamp(0, 1)), model
    assert len(observed) == 30 and all(torch.equal(epsilon, e) for e in observed)


@pytest.mark.parametrize('namespace', ['cifar10-test', 'cifar10-python/original-train',
                                      'train', 'val', '', None, True, [], {}])
def test_reject_foreign_namespace(namespace):
    with pytest.raises(ValueError, match='namespace'):
        keyed_test_epsilon(0, namespace=namespace)


@pytest.mark.parametrize('index', [-1, 10000, True, False, 1.0, '1', None, np.int64(1)])
def test_reject_non_original_test_ids(index):
    with pytest.raises(ValueError, match='original_index'):
        keyed_test_epsilon(index, namespace=NAMESPACE)


def test_keyed_helpers_preserve_global_rng_and_fresh_storage():
    state = random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone()
    for index in IDS:
        a = keyed_test_epsilon(index, namespace=NAMESPACE)
        b = keyed_test_epsilon(index, namespace=NAMESPACE)
        assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
        b.zero_()
        assert torch.equal(a, keyed_test_epsilon(index, namespace=NAMESPACE))
        gaussian_noise(torch.zeros_like(a), .18, keyed_test_generator(index, namespace=NAMESPACE))
    assert random.getstate() == state[0]
    now = np.random.get_state()
    assert now[0] == state[1][0] and now[2:] == state[1][2:]
    np.testing.assert_array_equal(now[1], state[1][1])
    assert torch.equal(torch.get_rng_state(), state[2])


class KeyedSyntheticFixture(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, position):
        index = self.ids[position]
        x = torch.full((3, 32, 32), .5)
        return index, keyed_test_epsilon(index, namespace=NAMESPACE), gaussian_noise(
            x, .12, keyed_test_generator(index, namespace=NAMESPACE))


def collect(ids, batch_size, workers):
    loader = DataLoader(KeyedSyntheticFixture(ids), batch_size=batch_size,
        num_workers=workers, drop_last=False, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(101), persistent_workers=False,
        multiprocessing_context='spawn' if workers else None)
    result, sizes = {}, []
    for indices, epsilon, noisy in loader:
        sizes.append(len(indices))
        for i, e, y in zip(indices.tolist(), epsilon, noisy):
            assert i not in result
            result[i] = (e, y)
    return result, sizes


def test_batch_order_and_real_spawn_workers_equal_cpu_reference():
    before = torch.get_rng_state().clone()
    reference, sizes = collect(IDS, 4, 0)
    assert sizes == [4, 4, 3]
    for ids, batch, workers in [(IDS[::-1], 7, 0), (IDS[3:] + IDS[:3], 3, 2)]:
        other, _ = collect(ids, batch, workers)
        assert set(reference) == set(other) == set(IDS)
        for index in IDS:
            assert all(torch.equal(a, b) for a, b in zip(reference[index], other[index]))
    assert torch.equal(before, torch.get_rng_state())


def test_fresh_process_and_pythonhashseed_independence():
    code = '''import hashlib, torch
from src.corruptions import keyed_test_epsilon, gaussian_noise, keyed_test_generator
namespace = 'cifar10-python/original-test'
e = keyed_test_epsilon(71, namespace=namespace)
y = gaussian_noise(torch.full_like(e,.5),.12,keyed_test_generator(71,namespace=namespace))
print(hashlib.sha256(e.numpy().astype('<f4').tobytes()+y.numpy().astype('<f4').tobytes()).hexdigest())
'''
    e = keyed_test_epsilon(71, namespace=NAMESPACE)
    y = gaussian_noise(torch.full_like(e, .5), .12, keyed_test_generator(71, namespace=NAMESPACE))
    expected = hashlib.sha256(e.numpy().astype('<f4').tobytes() + y.numpy().astype('<f4').tobytes()).hexdigest()
    for hashseed in ('1', '991'):
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
            env=dict(os.environ, PYTHONHASHSEED=hashseed), timeout=60)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected
