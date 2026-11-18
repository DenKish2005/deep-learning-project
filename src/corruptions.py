"""Pixel-space corruption primitives. Blur/darkening/policy await later tasks."""

import math

import torch


def severity_params(kind: str, severity: int) -> dict:
    """Exact protocol-v1 parameters; clean is separate from levels 1..5.

    Return a fresh dict so callers cannot change the frozen mapping. Mapping
    blur/brightness parameters does not implement those corruption families.
    """
    if type(kind) is not str or kind not in ('gaussian_noise', 'gaussian_blur', 'brightness'):
        raise ValueError('kind must be gaussian_noise, gaussian_blur or brightness')
    if type(severity) is not int or not 1 <= severity <= 5:
        raise ValueError('severity must be a Python integer in 1..5 (clean is separate)')
    index = severity - 1
    if kind == 'gaussian_noise':
        return {'sigma': (0.02, 0.05, 0.08, 0.12, 0.18)[index]}
    if kind == 'gaussian_blur':
        sigma = (0.3, 0.7, 1.1, 1.5, 2.0)[index]
        return {'sigma': sigma, 'kernel_size': 2 * math.ceil(3 * sigma) + 1}
    return {'factor': (0.90, 0.80, 0.70, 0.55, 0.40)[index]}


def gaussian_noise(x: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    """clip(x + sigma * N(0,1), 0, 1) using only the supplied local generator.

    Dense float32 RGB CHW pixels, positive spatial dimensions, CPU or CUDA.
    Return new storage, preserving shape/device/dtype. Sigma zero clones without
    drawing; invalid arguments fail before consuming the supplied generator.
    This primitive does not assign image identity: callers own the seed key.
    """
    if (not isinstance(x, torch.Tensor) or x.layout != torch.strided
            or x.dtype != torch.float32 or x.ndim != 3 or x.shape[0] != 3
            or min(x.shape[1:]) <= 0 or x.device.type not in ('cpu', 'cuda')):
        raise ValueError('x must be dense CPU/CUDA float32 RGB CHW with positive H,W')
    if not isinstance(generator, torch.Generator):
        raise ValueError('generator must be an explicit torch.Generator')
    if generator.device != x.device:
        raise ValueError('generator device must match x device (including CUDA index)')
    try:
        valid_sigma = type(sigma) in (int, float) and math.isfinite(sigma) and sigma >= 0
    except OverflowError:
        valid_sigma = False
    if not valid_sigma:
        raise ValueError('sigma must be a finite nonnegative Python number (not bool)')
    if not torch.isfinite(x).all() or not ((x >= 0) & (x <= 1)).all():
        raise ValueError('x must contain finite pixels in [0,1] before normalization')
    if sigma == 0:
        return x.clone()
    epsilon = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    # Finite Python sigma may exceed float32. Avoid scalar conversion overflow;
    # clipping still defines finite float32 pixels for those boundary helpers.
    if sigma > torch.finfo(torch.float32).max:
        return (x.double() + sigma * epsilon.double()).clamp(0, 1).to(x.dtype)
    return (x + sigma * epsilon).clamp(0, 1)
