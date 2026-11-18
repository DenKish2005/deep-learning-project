"""Opt-in actual CUDA policy contract; no model or CPU/CUDA equality claim."""

import json
import os

import pytest
import torch

from tests.test_policy import EVENTS, pixel_oracle, policy


pytestmark = pytest.mark.skipif(os.environ.get('RUN_CORR_CUDA') != '1',
                                reason='Actual CUDA policy requires RUN_CORR_CUDA=1')


def test_actual_cuda_policy_numeric_device_and_rng():
    assert torch.cuda.is_available(), 'Opted-in CUDA must be available'
    device = torch.device('cuda:0')
    x = torch.linspace(0, 1, 3 * 8 * 8, device=device).reshape(3, 8, 8)
    original = x.clone()
    before = torch.get_rng_state().clone(), [s.clone() for s in torch.cuda.get_rng_state_all()]
    events, maximum = set(), 0.
    for seed in range(32):
        generator, oracle_generator = [torch.Generator(device=device).manual_seed(seed) for _ in range(2)]
        produced, audit = policy()(x, generator)
        expected, expected_audit = pixel_oracle(x, oracle_generator)
        assert audit == expected_audit
        maximum = max(maximum, (produced - expected).abs().max().item())
        torch.testing.assert_close(produced, expected, atol=1e-6, rtol=0)
        replay, replay_audit = policy()(x, torch.Generator(device=device).manual_seed(seed))
        assert torch.equal(produced, replay) and audit == replay_audit
        assert torch.equal(generator.get_state(), oracle_generator.get_state())
        assert produced.device == device and produced.dtype == x.dtype and produced.shape == x.shape
        assert produced.data_ptr() != x.data_ptr() and torch.equal(x, original)
        assert torch.isfinite(produced).all() and ((produced >= 0) & (produced <= 1)).all()
        events.add(audit['event'])
    assert events == set(EVENTS)
    for pixels, generator in [(x.cpu(), torch.Generator(device=device)), (x, torch.Generator())]:
        state = generator.get_state().clone()
        with pytest.raises(ValueError, match='device must match'): policy()(pixels, generator)
        assert torch.equal(state, generator.get_state())
    generator = torch.Generator(device=device).manual_seed(42)
    state = generator.get_state().clone()
    with pytest.raises(ValueError): policy()(x.double(), generator)
    assert torch.equal(state, generator.get_state())
    assert torch.equal(torch.get_rng_state(), before[0])
    assert all(torch.equal(a, b) for a, b in zip(before[1], torch.cuda.get_rng_state_all()))
    print('ROBUST001_EVIDENCE ' + json.dumps({'kind': 'actual_cuda', 'value': {
        'device': str(device), 'gpu': torch.cuda.get_device_name(device), 'seeds': list(range(32)),
        'max_numpy2d_oracle_error': maximum, 'same_device_replay': True,
        'wrong_generator_devices_rejected_before_draws': True, 'global_cpu_cuda_rng_unchanged': True,
        'cpu_cuda_bitwise_equality_claimed': False}}, sort_keys=True))
