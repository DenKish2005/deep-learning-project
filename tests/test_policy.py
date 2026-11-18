"""ROBUST-001: exact event boundaries, independent pixels and keyed sampling."""

from dataclasses import FrozenInstanceError
import math
import random

import numpy as np
import pytest
import torch

from src import corruptions as corr
from src.utils import stable_seed
from tests.test_blur import blur_oracle


PROBABILITIES = [.50, .17, .17, .16]
EVENTS = ('identity', 'gaussian_noise', 'gaussian_blur', 'brightness')


def policy():
    return corr.CorruptionPolicy(PROBABILITIES)


def pixel_oracle(x, generator):
    """Literal protocol thresholds/tables, no dispatcher or severity mapping."""
    u = torch.rand((), dtype=torch.float64, device=x.device, generator=generator).item()
    event = EVENTS[int(u >= .5) + int(u >= .67) + int(u >= .84)]
    severity = 0 if event == 'identity' else torch.randint(1, 4, (), device=x.device, generator=generator).item()
    if event == 'identity':
        y = x.clone()
    elif event == 'gaussian_noise':
        epsilon = torch.randn(x.shape, device=x.device, dtype=torch.float32, generator=generator)
        y = (x + (.02, .05, .08)[severity - 1] * epsilon).clamp(0, 1)
    elif event == 'gaussian_blur':
        y = blur_oracle(x.cpu(), (.3, .7, 1.1)[severity - 1], (3, 7, 9)[severity - 1]).to(x.device)
    else:
        y = x * (.9, .8, .7)[severity - 1]
    return y, {'event': event, 'severity': severity}


@pytest.mark.parametrize('u,event', [
    (0., 'identity'), (math.nextafter(.5, 0.), 'identity'), (.5, 'gaussian_noise'),
    (math.nextafter(.67, 0.), 'gaussian_noise'), (.67, 'gaussian_blur'),
    (math.nextafter(.84, 0.), 'gaussian_blur'), (.84, 'brightness'),
    (math.nextafter(1., 0.), 'brightness')])
@pytest.mark.parametrize('severity', [1, 2, 3])
def test_literal_interval_endpoints_numeric_and_draw_order(monkeypatch, u, event, severity):
    x = torch.linspace(0, 1, 3 * 8 * 8).reshape(3, 8, 8)
    generator = torch.Generator().manual_seed(81)
    calls = []
    def uniform(shape, **kwargs):
        assert shape == () and kwargs == {'dtype': torch.float64, 'device': x.device, 'generator': generator}
        calls.append('event')
        return torch.tensor(u, dtype=torch.float64)
    def integer(low, high, shape, **kwargs):
        assert event != 'identity' and calls == ['event']
        assert (low, high, shape) == (1, 4, ())
        assert kwargs == {'device': x.device, 'generator': generator}
        calls.append('severity')
        return torch.tensor(severity)
    monkeypatch.setattr(torch, 'rand', uniform)
    monkeypatch.setattr(torch, 'randint', integer)
    expected_state = torch.Generator().manual_seed(81)
    if event == 'identity': expected = x.clone()
    elif event == 'gaussian_noise':
        expected = (x + (.02, .05, .08)[severity - 1] * torch.randn(x.shape, generator=expected_state)).clamp(0, 1)
    elif event == 'gaussian_blur':
        expected = blur_oracle(x, (.3, .7, 1.1)[severity - 1], (3, 7, 9)[severity - 1])
    else: expected = x * (.9, .8, .7)[severity - 1]
    before = x.clone()
    y, metadata = policy()(x, generator)
    assert metadata == {'event': event, 'severity': 0 if event == 'identity' else severity}
    assert calls == (['event'] if event == 'identity' else ['event', 'severity'])
    assert torch.equal(generator.get_state(), expected_state.get_state())
    torch.testing.assert_close(y, expected, atol=1e-6, rtol=0)
    assert y.data_ptr() != x.data_ptr() and torch.equal(x, before)


@pytest.mark.parametrize('seed', range(24))
def test_seeded_pixels_metadata_state_and_replay(seed):
    x = torch.linspace(0, 1, 3 * 9 * 8).reshape(3, 9, 8).transpose(1, 2)
    a, b = (torch.Generator().manual_seed(seed) for _ in range(2))
    y, metadata = policy()(x, a)
    expected, expected_metadata = pixel_oracle(x, b)
    torch.testing.assert_close(y, expected, atol=1e-6, rtol=0)
    assert metadata == expected_metadata and torch.equal(a.get_state(), b.get_state())
    assert y.shape == x.shape and y.device == x.device and y.dtype == x.dtype
    assert torch.isfinite(y).all() and ((y >= 0) & (y <= 1)).all()
    again, audit = policy()(x, torch.Generator().manual_seed(seed))
    assert torch.equal(y, again) and audit == metadata and y.data_ptr() != again.data_ptr()


@pytest.mark.parametrize('probabilities', [None, [], [.5, .5], [1., 0., 0., 0.],
    [.49, .18, .17, .16], [.5, .17, .17, -.16], [True, .17, .17, .16],
    [float('nan'), .17, .17, .16], [float('inf'), .17, .17, .16], ['.5', .17, .17, .16],
    torch.tensor([.5, .17, .17, .16])])
def test_reject_nonprotocol_probabilities(probabilities):
    with pytest.raises(ValueError, match='probabilities'): corr.CorruptionPolicy(probabilities)


@pytest.mark.parametrize('maximum', [True, 0, 1, 2, 4, 5, 3., None, '3'])
def test_reject_nonprotocol_severity(maximum):
    with pytest.raises(ValueError, match='max_train_severity'):
        corr.CorruptionPolicy(PROBABILITIES, maximum)


def test_constructor_owns_immutable_settings():
    supplied = list(PROBABILITIES)
    p = corr.CorruptionPolicy(supplied)
    supplied[0] = 0.
    assert p.probabilities == (.5, .17, .17, .16)
    with pytest.raises(FrozenInstanceError): p.max_train_severity = 5


@pytest.mark.parametrize('invalid', ['double', 'rgb', 'batch', 'empty', 'tiny', 'nan', 'inf', 'negative', 'large', 'list'])
def test_invalid_pixels_fail_before_draws(monkeypatch, invalid):
    x = torch.full((3, 8, 8), .5)
    if invalid == 'double': x = x.double()
    if invalid == 'rgb': x = x[:1]
    if invalid == 'batch': x = x[None]
    if invalid == 'empty': x = x[:, :0]
    if invalid == 'tiny': x = x[:, :4]
    if invalid in ('nan', 'inf', 'negative', 'large'):
        x[0, 0, 0] = {'nan': float('nan'), 'inf': float('inf'), 'negative': -.1, 'large': 1.1}[invalid]
    if invalid == 'list': x = x.tolist()
    generator = torch.Generator().manual_seed(42)
    state = generator.get_state().clone()
    def forbidden(*args, **kwargs): pytest.fail('invalid input drew RNG')
    monkeypatch.setattr(torch, 'rand', forbidden)
    with pytest.raises(ValueError): policy()(x, generator)
    assert torch.equal(state, generator.get_state())


@pytest.mark.parametrize('generator', [None, 42, 'cpu'])
def test_explicit_generator_required(generator):
    with pytest.raises(ValueError, match='explicit torch.Generator'):
        policy()(torch.full((3, 8, 8), .5), generator)


def test_real_dispatch_different_keys_and_global_rng_isolation():
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    x, p = torch.full((3, 8, 8), .5), policy()
    outputs, events = [], set()
    for original in range(32):
        key = stable_seed(123, 1, original, 'train_corruption')
        y, metadata = p(x, torch.Generator().manual_seed(key))
        replay, replay_metadata = p(x, torch.Generator().manual_seed(key))
        assert torch.equal(y, replay) and metadata == replay_metadata
        outputs.append(y)
        events.add(metadata['event'])
    assert events == set(EVENTS)
    assert any(not torch.equal(outputs[0], y) for y in outputs[1:])
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and after[2:] == before[1][2:]
    np.testing.assert_array_equal(after[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_100000_keyed_selections_frequencies_and_global_isolation(monkeypatch):
    # Sampling audit calls the production selector/validation/draws on each image;
    # dispatcher pixels are independently covered above, so avoid 17k blur ops.
    monkeypatch.setattr(corr, 'apply_corruption', lambda x, *args: x.clone())
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    counts = np.zeros(4, dtype=np.int64)
    levels = np.zeros((3, 3), dtype=np.int64)
    p, x = policy(), torch.full((3, 5, 5), .5)
    for original in range(100000):
        generator = torch.Generator().manual_seed(stable_seed(42, 0, original, 'train_corruption'))
        _, metadata = p(x, generator)
        event = EVENTS.index(metadata['event'])
        counts[event] += 1
        if event:
            assert 1 <= metadata['severity'] <= 3
            levels[event - 1, metadata['severity'] - 1] += 1
        else: assert metadata['severity'] == 0
    assert np.max(np.abs(counts / 100000 - PROBABILITIES)) < .01
    assert np.max(np.abs(levels / counts[1:, None] - 1/3)) < .02
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and after[2:] == before[1][2:]
    np.testing.assert_array_equal(after[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    print('ROBUST001_SAMPLING', {'counts': counts.tolist(), 'conditional_severity_counts': levels.tolist()})
