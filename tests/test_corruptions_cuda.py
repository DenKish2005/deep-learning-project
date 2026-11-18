"""Optional actual CUDA primitive contract; canonical keyed test noise stays CPU."""

import json
import os

import pytest
import torch

from src.corruptions import apply_corruption, gaussian_noise, keyed_test_epsilon
from tests.test_blur import blur_oracle
from tests.test_corruption_dispatch import CONDITIONS


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


@pytest.mark.skipif(os.environ.get('RUN_CORR_CUDA') != '1',
                    reason='CUDA dispatcher check requires RUN_CORR_CUDA=1')
def test_actual_cuda_all_families_and_unused_generators():
    assert torch.cuda.is_available(), 'Opted-in CUDA must be available'
    device = torch.device('cuda', torch.cuda.current_device())
    cpu = torch.linspace(0, 1, 3072).reshape(3, 32, 32)
    x = cpu.to(device)
    before = x.clone()
    cpu_state = torch.get_rng_state().clone()
    cuda_states = [s.clone() for s in torch.cuda.get_rng_state_all()]
    max_blur_error = 0.
    for kind, severity in CONDITIONS:
        gen = torch.Generator(device=device).manual_seed(494)
        oracle_gen = torch.Generator(device=device).manual_seed(494)
        y = apply_corruption(x, kind, severity, gen)
        if kind == 'gaussian_noise':
            sigma = (.02, .05, .08, .12, .18)[severity - 1]
            expected = (x + sigma * torch.randn(x.shape, device=device, generator=oracle_gen)).clamp(0, 1)
            assert torch.equal(y, expected) and torch.equal(gen.get_state(), oracle_gen.get_state())
            for image, wrong in [(x, torch.Generator()), (cpu, torch.Generator(device=device))]:
                state = wrong.get_state().clone()
                with pytest.raises(ValueError, match='generator device'):
                    apply_corruption(image, kind, severity, wrong)
                assert torch.equal(state, wrong.get_state())
        else:
            assert torch.equal(gen.get_state(), oracle_gen.get_state())
            assert torch.equal(y, apply_corruption(x, kind, severity))
            # Deterministic families intentionally ignore even an unused cross-device stream.
            unused = torch.Generator().manual_seed(42)
            state = unused.get_state().clone()
            assert torch.equal(y, apply_corruption(x, kind, severity, unused))
            assert torch.equal(state, unused.get_state())
            reverse = torch.Generator(device=device).manual_seed(42)
            state = reverse.get_state().clone()
            apply_corruption(cpu, kind, severity, reverse)
            assert torch.equal(state, reverse.get_state())
            if kind == 'clean': assert torch.equal(y, x)
            elif kind == 'brightness':
                torch.testing.assert_close(y.cpu(), cpu * (.9, .8, .7, .55, .4)[severity-1], atol=1e-7, rtol=0)
            else:
                sigma, kernel = list(zip((.3, .7, 1.1, 1.5, 2.), (3, 7, 9, 11, 13)))[severity-1]
                expected = blur_oracle(cpu, sigma, kernel)
                max_blur_error = max(max_blur_error, (y.cpu() - expected).abs().max().item())
                torch.testing.assert_close(y.cpu(), expected, atol=2e-6, rtol=0)
                const = apply_corruption(torch.full_like(x, .9), kind, severity)
                torch.testing.assert_close(const, torch.full_like(x, .9), atol=1e-6, rtol=0)
        assert y.shape == x.shape and y.device == device and y.dtype == x.dtype
        assert y.data_ptr() != x.data_ptr() and torch.equal(x, before)
        assert torch.isfinite(y).all() and ((y >= 0) & (y <= 1)).all()
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_states, torch.cuda.get_rng_state_all()))
    print('CORR002_EVIDENCE ' + json.dumps({'kind': 'cuda_dispatch', 'value': {
        'device': str(device), 'gpu': torch.cuda.get_device_name(device), 'conditions': 16,
        'max_blur_oracle_error': max_blur_error, 'global_rng_unchanged': True,
        'noise_wrong_device_rejected_both_ways': True, 'unused_generators_unchanged': True,
        'cpu_cuda_bitwise_equality_claimed': False}}))
