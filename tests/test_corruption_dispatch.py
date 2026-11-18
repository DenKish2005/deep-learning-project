"""TEST-C01/C02/C08: all-family dispatch, local RNG and Normalize regression."""

from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from src.corruptions import (apply_corruption, gaussian_blur, darken, gaussian_noise,
                             keyed_test_epsilon, keyed_test_generator, severity_params)
from tests.test_blur import INVALID_PIXELS, blur_oracle


CONDITIONS = [('clean', 0)] + [(kind, s) for kind in
    ('gaussian_noise', 'gaussian_blur', 'brightness') for s in range(1, 6)]


@pytest.mark.parametrize('kind,severity', CONDITIONS)
@pytest.mark.parametrize('pattern', ['zero', 'one', 'ramp'])
def test_all_conditions_contract_and_dispatch(kind, severity, pattern):
    x = {'zero': torch.zeros(3, 32, 32), 'one': torch.ones(3, 32, 32),
         'ramp': torch.linspace(0, 1, 3072).reshape(3, 32, 32)}[pattern]
    before = x.clone()
    gen = torch.Generator().manual_seed(494)
    oracle_gen = torch.Generator().manual_seed(494)
    y = apply_corruption(x, kind, severity, gen)
    if kind == 'clean': expected = x.clone()
    else:
        params = severity_params(kind, severity)
        primitive = {'gaussian_noise': gaussian_noise, 'gaussian_blur': gaussian_blur,
                     'brightness': darken}[kind]
        expected = primitive(x, **params, **({'generator': oracle_gen} if kind == 'gaussian_noise' else {}))
    assert torch.equal(y, expected) and torch.equal(gen.get_state(), oracle_gen.get_state())
    assert y.shape == x.shape and y.device == x.device and y.dtype == x.dtype
    assert y.data_ptr() != x.data_ptr() and torch.equal(x, before)
    assert torch.isfinite(y).all() and ((y >= 0) & (y <= 1)).all()
    if kind != 'gaussian_noise':
        assert torch.equal(y, apply_corruption(x, kind, severity))


@pytest.mark.parametrize('kind,severity', [('clean', 0), ('gaussian_noise', 1),
                                         ('gaussian_blur', 1), ('brightness', 1)])
@pytest.mark.parametrize('x', INVALID_PIXELS)
def test_dispatch_rejects_pixels_before_draw(kind, severity, x):
    gen = torch.Generator().manual_seed(494)
    before = gen.get_state().clone()
    with pytest.raises(ValueError, match='x must'):
        apply_corruption(x, kind, severity, gen)
    assert torch.equal(gen.get_state(), before)


@pytest.mark.parametrize('kind', ['clean', 'gaussian_noise', 'gaussian_blur', 'brightness'])
@pytest.mark.parametrize('severity', [-1, 6, True, False, None, 1., '1', np.int64(1), []])
def test_dispatch_invalid_severity_before_draw(kind, severity):
    gen = torch.Generator().manual_seed(494)
    before = gen.get_state().clone()
    with pytest.raises(ValueError, match='severity'):
        apply_corruption(torch.zeros(3, 32, 32), kind, severity, gen)
    assert torch.equal(gen.get_state(), before)


@pytest.mark.parametrize('kind,severity', [('clean', 1), ('gaussian_noise', 0),
    ('gaussian_blur', 0), ('brightness', 0), ('noise', 1), ('darken', 1), ('', 0),
    (True, 0), (None, 0), ([], 1), ({}, 1)])
def test_dispatch_clean_is_separate_and_unknown_kinds_rejected(kind, severity):
    with pytest.raises(ValueError):
        apply_corruption(torch.zeros(3, 32, 32), kind, severity)


@pytest.mark.parametrize('kind,severity', CONDITIONS)
def test_dispatch_wrong_generator_type(kind, severity):
    with pytest.raises(ValueError, match='generator'):
        apply_corruption(torch.zeros(3, 32, 32), kind, severity, generator=42)


def test_noise_requires_generator_and_preserves_common_keyed_epsilon():
    x = torch.linspace(0, 1, 3072).reshape(3, 32, 32)
    with pytest.raises(ValueError, match='explicit torch.Generator'):
        apply_corruption(x, 'gaussian_noise', 1)
    namespace = 'cifar10-python/original-test'
    epsilon = keyed_test_epsilon(71, namespace=namespace)
    for severity, sigma in enumerate((.02, .05, .08, .12, .18), 1):
        y = apply_corruption(x, 'gaussian_noise', severity,
                             keyed_test_generator(71, namespace=namespace))
        assert torch.equal(y, (x + sigma * epsilon).clamp(0, 1))


def test_dispatch_global_rng_isolation_success_and_failure():
    state = random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone()
    x = torch.full((3, 32, 32), .5)
    for kind, severity in CONDITIONS:
        gen = torch.Generator().manual_seed(494)
        apply_corruption(x, kind, severity, gen)
        with pytest.raises(ValueError): apply_corruption(x, kind, -1, gen)
    assert random.getstate() == state[0]
    now = np.random.get_state()
    assert now[0] == state[1][0] and now[2:] == state[1][2:]
    np.testing.assert_array_equal(now[1], state[1][1])
    assert torch.equal(torch.get_rng_state(), state[2])


@pytest.mark.parametrize('kind', ['gaussian_noise', 'gaussian_blur', 'brightness'])
def test_dispatch_then_normalize_independent_pixel_oracle(kind):
    x = torch.linspace(0, 1, 3072).reshape(3, 32, 32)
    mean = torch.tensor([.49, .48, .45])[:, None, None]
    std = torch.tensor([.25, .24, .26])[:, None, None]
    params = severity_params(kind, 3)
    gen = torch.Generator().manual_seed(494)
    y = apply_corruption(x, kind, 3, gen)
    if kind == 'gaussian_noise':
        expected = (x + .08 * torch.randn(x.shape, generator=torch.Generator().manual_seed(494))).clamp(0, 1)
    elif kind == 'gaussian_blur': expected = blur_oracle(x, **params)
    else: expected = x * .7
    torch.testing.assert_close((y - mean) / std, (expected - mean) / std, atol=3e-6, rtol=0)
    # Already-normalized data cannot accidentally cross the pixel boundary.
    with pytest.raises(ValueError, match='before normalization'):
        apply_corruption((x - mean) / std, kind, 3, gen)
