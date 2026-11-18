"""TEST-C01/C02/C05/C08: reflect blur with an independent float64 2D oracle."""

import numpy as np
import pytest
import torch

from src.corruptions import gaussian_blur, severity_params


def blur_oracle(x, sigma, kernel_size):
    """NumPy 2D window sum; independent of torch's two 1D convolutions."""
    radius = kernel_size // 2
    q = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-q**2 / (2 * sigma**2))
    weights /= weights.sum()
    kernel = np.outer(weights, weights)
    padded = np.pad(x.double().numpy(), ((0, 0), (radius, radius), (radius, radius)), mode='reflect')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size), axis=(1, 2))
    return torch.from_numpy(np.einsum('chwij,ij->chw', windows, kernel)).float()


@pytest.mark.parametrize('severity', range(1, 6))
@pytest.mark.parametrize('pattern', ['zero', 'one', 'constant', 'impulse', 'edge', 'ramp', 'checkerboard'])
def test_blur_contract_and_independent_oracle(severity, pattern):
    x = torch.zeros(3, 32, 32)
    if pattern == 'one': x.fill_(1)
    if pattern == 'constant': x[:] = torch.tensor([.1, .5, .9])[:, None, None]
    if pattern == 'impulse': x[0, 16, 16] = 1
    if pattern == 'edge': x[1, 0, 1] = 1
    if pattern == 'ramp': x = torch.linspace(0, 1, 3072).reshape(3, 32, 32)
    if pattern == 'checkerboard': x[:] = (torch.arange(32)[:, None] + torch.arange(32)) % 2
    before = x.clone()
    params = severity_params('gaussian_blur', severity)
    y = gaussian_blur(x, **params)
    assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
    assert y.data_ptr() != x.data_ptr() and torch.equal(x, before)
    assert torch.isfinite(y).all() and ((y >= 0) & (y <= 1)).all()
    torch.testing.assert_close(y, blur_oracle(x, **params), atol=1e-6, rtol=0)
    assert torch.equal(y, gaussian_blur(x, **params))
    if pattern in ('zero', 'one', 'constant'):
        torch.testing.assert_close(y, x, atol=1e-6, rtol=0)
    if pattern == 'impulse':
        radius = params['kernel_size'] // 2
        kernel = y[0, 16-radius:17+radius, 16-radius:17+radius]
        assert abs(kernel.sum().item() - 1) < 1e-6
        assert torch.equal(kernel, kernel.flip(0)) and torch.equal(kernel, kernel.flip(1))
        assert torch.count_nonzero(y[1:]) == 0
    if pattern == 'checkerboard':
        assert (y[:, :, 1:] - y[:, :, :-1]).abs().mean() < 1


@pytest.mark.parametrize('sigma,kernel', [(0, 3), (2., 1), (1e-300, 3), (1e300, 3)])
def test_blur_helpers_noncontiguous_and_extreme_sigma(sigma, kernel):
    x = torch.linspace(0, 1, 189).reshape(3, 7, 9).transpose(1, 2)
    y = gaussian_blur(x, sigma, kernel)
    assert y.data_ptr() != x.data_ptr() and y.shape == x.shape
    assert torch.isfinite(y).all()
    if sigma == 0 or kernel == 1 or sigma < 1e-200:
        assert torch.equal(x, y)
    else:
        torch.testing.assert_close(y, blur_oracle(x, 1e100, kernel), atol=1e-6, rtol=0)


@pytest.mark.parametrize('sigma', [-1, True, None, '1', np.float32(.3), float('nan'), float('inf'), 10**400])
def test_blur_invalid_sigma(sigma):
    with pytest.raises(ValueError, match='sigma'):
        gaussian_blur(torch.zeros(3, 32, 32), sigma, 3)


@pytest.mark.parametrize('kernel', [0, -1, 2, 4, True, 3., None, '3', np.int64(3)])
def test_blur_invalid_kernel(kernel):
    with pytest.raises(ValueError, match='kernel_size'):
        gaussian_blur(torch.zeros(3, 32, 32), .3, kernel)


@pytest.mark.parametrize('shape,kernel', [((3, 6, 32), 13), ((3, 32, 6), 13), ((3, 1, 2), 3)])
@pytest.mark.parametrize('sigma', [0, 2.])
def test_blur_invalid_reflect_size_even_for_identity(shape, kernel, sigma):
    with pytest.raises(ValueError, match='reflect radius'):
        gaussian_blur(torch.zeros(shape), sigma, kernel)


INVALID_PIXELS = [None, np.zeros((3, 4, 4)), torch.zeros(3, 4, 4).double(),
    torch.zeros(3, 4, 4, dtype=torch.uint8), torch.zeros(1, 3, 4, 4), torch.zeros(1, 4, 4),
    torch.zeros(4, 4, 3), torch.zeros(3, 0, 4), torch.zeros(3, 4, 0),
    torch.full((3, 4, 4), float('nan')), torch.full((3, 4, 4), float('inf')),
    torch.full((3, 4, 4), -.001), torch.full((3, 4, 4), 1.001),
    torch.zeros(3, 4, 4, device='meta'), torch.zeros(3, 4, 4).to_sparse()]


@pytest.mark.parametrize('x', INVALID_PIXELS)
def test_blur_rejects_invalid_pixels_even_for_identity(x):
    with pytest.raises(ValueError, match='x must'):
        gaussian_blur(x, 0, 3)
