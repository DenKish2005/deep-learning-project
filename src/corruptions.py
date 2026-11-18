"""Pixel-space corruption primitives. Blur/darkening/policy await later tasks."""

import math

import torch
import torch.nn.functional as F

from src.utils import stable_seed


def _validate_pixels(x: torch.Tensor) -> None:
    """The common pixel boundary for deterministic primitives and clean."""
    if (not isinstance(x, torch.Tensor) or x.layout != torch.strided
            or x.dtype != torch.float32 or x.ndim != 3 or x.shape[0] != 3
            or min(x.shape[1:]) <= 0 or x.device.type not in ('cpu', 'cuda')):
        raise ValueError('x must be dense CPU/CUDA float32 RGB CHW with positive H,W')
    if not torch.isfinite(x).all() or not ((x >= 0) & (x <= 1)).all():
        raise ValueError('x must contain finite pixels in [0,1] before normalization')


def gaussian_blur(x: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
    """Normalized discrete separable Gaussian, reflect padding, no channel mixing.

    Finite nonnegative Python sigma and positive odd Python kernel are required.
    Radius must be smaller than both H and W, including identity calls. Sigma0
    (or kernel1) returns a clone. Positive sigma weights are computed in float64,
    normalized then cast to pixel dtype; convolution and output are float32.
    The final clamp contains roundoff at the [0,1] boundary. No RNG is used.
    """
    _validate_pixels(x)
    try:
        valid = type(sigma) in (int, float) and math.isfinite(sigma) and sigma >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError('sigma must be a finite nonnegative Python number (not bool)')
    if type(kernel_size) is not int or kernel_size <= 0 or kernel_size % 2 != 1:
        raise ValueError('kernel_size must be a positive odd Python integer')
    radius = kernel_size // 2
    if radius >= min(x.shape[1:]):
        raise ValueError('reflect radius must be smaller than both image dimensions')
    if sigma == 0 or kernel_size == 1:
        return x.clone()
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float64, device=x.device)
    # Divide before squaring: tiny sigma tends to a delta, huge sigma to uniform.
    weights = torch.exp(-0.5 * (coordinates / sigma).square())
    weights = (weights / weights.sum()).to(x.dtype)
    horizontal = weights.reshape(1, 1, 1, -1).repeat(3, 1, 1, 1)
    vertical = weights.reshape(1, 1, -1, 1).repeat(3, 1, 1, 1)
    y = F.conv2d(F.pad(x.unsqueeze(0), (radius, radius, 0, 0), mode='reflect'),
                 horizontal, groups=3)
    y = F.conv2d(F.pad(y, (0, 0, radius, radius), mode='reflect'), vertical, groups=3)
    return y.squeeze(0).clamp(0, 1)


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


def keyed_test_generator(original_index: int, *, namespace: str) -> torch.Generator:
    """Fresh CPU stream for one official original-test ID; eval seed is 100494.

    Require the dataset's full namespace to reject accidental train/val IDs.
    Its protocol seed alias is 'cifar10-test'. No severity, model or run seed
    enters the key. Recreate this generator for each image/condition; reusing
    a consumed generator would change epsilon. No dataset/evaluator is built.
    """
    if type(namespace) is not str or namespace != 'cifar10-python/original-test':
        raise ValueError('namespace must be cifar10-python/original-test')
    if type(original_index) is not int or not 0 <= original_index < 10000:
        raise ValueError('original_index must be a Python integer in original-test 0..9999')
    return torch.Generator(device='cpu').manual_seed(stable_seed(
        100494, 'cifar10-test', original_index, 'gaussian_noise', 'v1'))


def keyed_test_epsilon(original_index: int, *, namespace: str) -> torch.Tensor:
    """Reusable pre-clamp CPU float32 [3,32,32] N(0,1) for that test identity.

    For pixels use gaussian_noise(x, sigma, keyed_test_generator(...)); both
    boundaries draw exactly the same epsilon in the locked CPU environment.
    This tensor is unbounded noise, not an image to normalize or clamp.
    """
    generator = keyed_test_generator(original_index, namespace=namespace)
    return torch.randn((3, 32, 32), dtype=torch.float32, device='cpu', generator=generator)
