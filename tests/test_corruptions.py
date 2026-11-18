"""TEST-C01/C02/C04: mapping, pixel contracts and independent numeric oracles."""

from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from src.corruptions import gaussian_noise, severity_params
from src.utils import load_config


@pytest.mark.parametrize('kind,expected', [
    ('gaussian_noise', [{'sigma': v} for v in (.02, .05, .08, .12, .18)]),
    ('gaussian_blur', [{'sigma': s, 'kernel_size': k} for s, k in
                       zip((.3, .7, 1.1, 1.5, 2.), (3, 7, 9, 11, 13))]),
    ('brightness', [{'factor': v} for v in (.9, .8, .7, .55, .4)]),
])
def test_parameter_mapping_exact_and_fresh(kind, expected):
    assert [severity_params(kind, s) for s in range(1, 6)] == expected
    params = severity_params(kind, 1)
    params.clear()
    assert severity_params(kind, 1) == expected[0]


@pytest.mark.parametrize('kind', ['clean', 'noise', 'darken', '', None, True, 1, [], {}])
def test_unknown_kind_and_wrong_types(kind):
    with pytest.raises(ValueError, match='kind'):
        severity_params(kind, 1)


@pytest.mark.parametrize('severity', [0, -1, 6, True, False, 1.0, '1', None,
                                     np.int64(1), [], float('nan'), float('inf')])
def test_invalid_severity_types_and_levels(severity):
    with pytest.raises(ValueError, match='severity'):
        severity_params('gaussian_noise', severity)


@pytest.mark.parametrize('filename', ['clean_cnn.yaml', 'robust_cnn.yaml'])
def test_mapping_matches_both_frozen_configs(filename):
    cfg = load_config(Path('configs') / filename)['corruptions']
    for i in range(5):
        assert severity_params('gaussian_noise', i + 1)['sigma'] == cfg['noise_sigma'][i]
        assert severity_params('gaussian_blur', i + 1)['sigma'] == cfg['blur_sigma'][i]
        assert severity_params('brightness', i + 1)['factor'] == cfg['brightness_factor'][i]


@pytest.mark.parametrize('severity', range(1, 6))
@pytest.mark.parametrize('pattern', ['zero', 'one', 'ramp'])
def test_noise_contract_shape_range_no_mutation(severity, pattern):
    x = {'zero': torch.zeros(3, 32, 32), 'one': torch.ones(3, 32, 32),
         'ramp': torch.linspace(0, 1, 3072).reshape(3, 32, 32)}[pattern]
    before = x.clone()
    y = gaussian_noise(x, **severity_params('gaussian_noise', severity),
                       generator=torch.Generator().manual_seed(494))
    assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
    assert torch.isfinite(y).all() and ((y >= 0) & (y <= 1)).all()
    assert torch.equal(x, before) and y.data_ptr() != x.data_ptr()
    assert not torch.equal(y, x)


def test_sigma_zero_clones_noncontiguous_input_without_rng_draw():
    x = torch.full((3, 7, 9), .5).transpose(1, 2)
    gen = torch.Generator().manual_seed(42)
    before = gen.get_state().clone()
    y = gaussian_noise(x, 0, gen)
    assert torch.equal(y, x) and y.data_ptr() != x.data_ptr()
    assert torch.equal(before, gen.get_state())
    noisy = gaussian_noise(x, .05, gen)
    assert noisy.shape == x.shape and torch.equal(x, torch.full_like(x, .5))


@pytest.mark.parametrize('sigma', [-.01, float('nan'), float('inf'), -float('inf'),
                                  True, '0.02', None, [], np.float32(.02), 10**400])
def test_invalid_sigma_does_not_consume_generator(sigma):
    gen = torch.Generator().manual_seed(3)
    before = gen.get_state().clone()
    with pytest.raises(ValueError, match='sigma'):
        gaussian_noise(torch.zeros(3, 4, 4), sigma, gen)
    assert torch.equal(before, gen.get_state())


@pytest.mark.parametrize('x', [None, np.zeros((3, 4, 4)), torch.zeros(3, 4, 4).double(),
    torch.zeros(3, 4, 4, dtype=torch.uint8), torch.zeros(1, 3, 4, 4), torch.zeros(4, 4, 3),
    torch.zeros(1, 4, 4), torch.zeros(3, 0, 4), torch.zeros(3, 4, 0),
    torch.full((3, 4, 4), float('nan')), torch.full((3, 4, 4), float('inf')),
    torch.full((3, 4, 4), -.001), torch.full((3, 4, 4), 1.001),
    torch.zeros(3, 4, 4, device='meta'), torch.zeros(3, 4, 4).to_sparse()])
def test_invalid_pixels_fail_before_rng_draw_even_at_sigma_zero(x):
    gen = torch.Generator().manual_seed(3)
    before = gen.get_state().clone()
    with pytest.raises(ValueError, match='x must'):
        gaussian_noise(x, 0, gen)
    assert torch.equal(before, gen.get_state())


@pytest.mark.parametrize('generator', [None, 42, True, np.random.default_rng(1)])
def test_generator_is_required(generator):
    with pytest.raises(ValueError, match='explicit torch.Generator'):
        gaussian_noise(torch.zeros(3, 4, 4), .02, generator)


def test_exact_clipping_with_known_preclamp_epsilon(monkeypatch):
    epsilon = torch.tensor([-2., -1., 0., 1., 2.]).repeat(3, 1, 1)
    def known(shape, *, dtype, device, generator):
        assert shape == epsilon.shape and dtype == torch.float32 and device.type == 'cpu'
        assert isinstance(generator, torch.Generator)
        return epsilon.clone()
    monkeypatch.setattr(torch, 'randn', known)
    x = torch.full_like(epsilon, .5)
    y = gaussian_noise(x, .5, torch.Generator())
    assert torch.equal(y, torch.tensor([0., 0., .5, 1., 1.]).repeat(3, 1, 1))
    huge = gaussian_noise(x, 1e300, torch.Generator())
    assert torch.equal(huge, torch.tensor([0., 0., .5, 1., 1.]).repeat(3, 1, 1))


def test_large_tensor_statistics_preclamp_and_global_rng_isolation():
    state = random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone()
    gen = torch.Generator().manual_seed(100494)
    oracle = torch.Generator().manual_seed(100494)
    x = torch.full((3, 512, 512), .5)
    epsilon = torch.randn(x.shape, generator=oracle)
    assert abs(epsilon.double().mean().item()) < .01
    assert abs(epsilon.double().std(unbiased=False).item() - 1) < .02
    y = gaussian_noise(x, .02, gen)
    assert torch.equal(y, (x + .02 * epsilon).clamp(0, 1))
    assert ((y > 0) & (y < 1)).all()  # No clipping for this fixed low-sigma fixture.
    recovered = (y - x) / .02
    assert abs(recovered.double().mean().item()) < .01
    assert abs(recovered.double().std(unbiased=False).item() - 1) < .02
    assert random.getstate() == state[0]
    now = np.random.get_state()
    assert now[0] == state[1][0] and now[2:] == state[1][2:]
    np.testing.assert_array_equal(now[1], state[1][1])
    assert torch.equal(torch.get_rng_state(), state[2])
    assert torch.equal(gen.get_state(), oracle.get_state())
