"""TEST-C01/C02/C06: exact multiplicative progressive darkening."""

import numpy as np
import pytest
import torch

from src.corruptions import darken, severity_params
from tests.test_blur import INVALID_PIXELS


@pytest.mark.parametrize('factor', [0, 1, .9, .8, .7, .55, .4])
@pytest.mark.parametrize('pattern', ['zero', 'one', 'ramp', 'noncontiguous'])
def test_darken_exact_pixels_contract_and_boundary_helpers(factor, pattern):
    x = {'zero': torch.zeros(3, 32, 32), 'one': torch.ones(3, 32, 32),
         'ramp': torch.linspace(0, 1, 3072).reshape(3, 32, 32),
         'noncontiguous': torch.linspace(0, 1, 189).reshape(3, 7, 9).transpose(1, 2)}[pattern]
    before = x.clone()
    y = darken(x, factor)
    expected = torch.from_numpy((x.double().numpy() * factor).astype(np.float32))
    torch.testing.assert_close(y, expected, atol=1e-7, rtol=0)
    assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
    assert torch.equal(x, before) and x.data_ptr() != y.data_ptr()
    assert torch.isfinite(y).all() and ((y >= 0) & (y <= 1)).all()
    assert (y <= x).all() and torch.equal(y, darken(x, factor))
    if factor == 1: assert torch.equal(y, x)
    if factor == 0: assert torch.count_nonzero(y) == 0


def test_exact_severity_order_and_no_additive_shift():
    x = torch.tensor([0., .25, .5, 1.]).repeat(3, 1, 1)
    previous = x
    for level, factor in enumerate((.9, .8, .7, .55, .4), 1):
        params = severity_params('brightness', level)
        assert params == {'factor': factor}
        y = darken(x, **params)
        torch.testing.assert_close(y, factor * x, atol=1e-7, rtol=0)
        assert (y <= previous).all() and torch.equal(y[:, :, 0], x[:, :, 0])
        assert not torch.equal(y, (x - (1 - factor)).clamp(0, 1))
        previous = y


@pytest.mark.parametrize('factor', [-.01, 1.01, True, False, None, '0.9', [],
                                   np.float32(.9), float('nan'), float('inf'), -float('inf'), 10**400])
def test_darken_rejects_invalid_factors(factor):
    with pytest.raises(ValueError, match='factor'):
        darken(torch.zeros(3, 32, 32), factor)


@pytest.mark.parametrize('x', INVALID_PIXELS)
@pytest.mark.parametrize('factor', [0, 1])
def test_darken_validates_pixels_even_at_boundary_helpers(x, factor):
    with pytest.raises(ValueError, match='x must'):
        darken(x, factor)
