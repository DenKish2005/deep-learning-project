"""Optional actual CUDA primitive contract; canonical keyed test noise stays CPU."""

import json
import os

import pytest
import torch

from src.corruptions import gaussian_noise, keyed_test_epsilon


@pytest.mark.skipif(os.environ.get('RUN_CORR_CUDA') != '1',
                    reason='CUDA primitive check requires RUN_CORR_CUDA=1')
def test_actual_cuda_generator_device_clipping_and_rng_isolation():
    assert torch.cuda.is_available(), 'Opted-in CUDA must be available; never synthetic PASS/skip'
    device = torch.device('cuda', torch.cuda.current_device())
    x = torch.full((3, 32, 32), .5, dtype=torch.float32, device=device)
    cpu_state = torch.get_rng_state().clone()
    cuda_states = [s.clone() for s in torch.cuda.get_rng_state_all()]
    for sigma in (0, .02, .05, .08, .12, .18):
        gen = torch.Generator(device=device).manual_seed(100494)
        oracle = torch.Generator(device=device).manual_seed(100494)
        y = gaussian_noise(x, sigma, gen)
        expected = x.clone() if sigma == 0 else (x + sigma * torch.randn(
            x.shape, device=device, dtype=x.dtype, generator=oracle)).clamp(0, 1)
        assert torch.equal(y, expected) and torch.equal(gen.get_state(), oracle.get_state())
        assert y.device == device and y.dtype == x.dtype and y.shape == x.shape
        assert y.data_ptr() != x.data_ptr() and torch.isfinite(y).all()
    for image, gen in [(x, torch.Generator()),
                       (x.cpu(), torch.Generator(device=device))]:
        before = gen.get_state().clone()
        with pytest.raises(ValueError, match='generator device'):
            gaussian_noise(image, .08, gen)
        assert torch.equal(before, gen.get_state())
    assert torch.equal(x, torch.full_like(x, .5))
    assert keyed_test_epsilon(0, namespace='cifar10-python/original-test').device.type == 'cpu'
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_states, torch.cuda.get_rng_state_all()))
    print('CORR001_EVIDENCE ' + json.dumps({'kind': 'cuda_primitive', 'value': {
        'device': str(device), 'gpu': torch.cuda.get_device_name(device),
        'same_device_determinism': True, 'cross_device_generator_rejected': True,
        'global_rng_unchanged': True, 'cpu_cuda_equality_claimed': False}}))
