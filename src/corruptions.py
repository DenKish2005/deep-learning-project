"""Pixel-space primitives, dispatch and explicit-generator training policy."""

from dataclasses import dataclass
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


def darken(x: torch.Tensor, factor: float) -> torch.Tensor:
    """clip(factor*x,0,1) for a finite Python factor in [0,1], without mutation.

    Factor1 returns equal pixels in new storage; factor0 returns black. Larger
    factors are rejected: this helper only darkens. No generator/RNG is used.
    """
    _validate_pixels(x)
    try:
        valid = type(factor) in (int, float) and math.isfinite(factor) and 0 <= factor <= 1
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError('factor must be a finite Python number in [0,1] (not bool)')
    return (x * factor).clamp(0, 1)


def severity_params(kind: str, severity: int) -> dict:
    """Exact protocol-v1 parameters; clean is separate from levels 1..5.

    Return a fresh dict so callers cannot change the frozen mapping.
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


def apply_corruption(x: torch.Tensor, kind: str, severity: int,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Dispatch clean0 or one protocol family at severity1..5 before Normalize.

    Noise requires an explicit same-device generator; there is no global fallback
    or seed assignment. Clean/blur/brightness accept None or a torch.Generator
    on any device and leave it untouched (unused, hence no device constraint).
    Wrong generator types are rejected for every family. Clean validates pixels
    and Python integer severity0, then clones. Invalid arguments never draw RNG.
    """
    if type(kind) is str and kind == 'clean':
        if type(severity) is not int or severity != 0:
            raise ValueError('clean severity must be the Python integer 0')
        params = None
    else:
        params = severity_params(kind, severity)
    if generator is not None and not isinstance(generator, torch.Generator):
        raise ValueError('generator must be None or an explicit torch.Generator')
    if kind == 'clean':
        _validate_pixels(x)
        return x.clone()
    if kind == 'gaussian_noise':
        return gaussian_noise(x, **params, generator=generator)
    if kind == 'gaussian_blur':
        return gaussian_blur(x, **params)
    return darken(x, **params)


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


@dataclass(frozen=True)
class CorruptionPolicy:
    """One protocol-v1 event per image; caller owns the explicit local stream.

    Constructor accepts the frozen robust probabilities only, in event order
    identity/noise/blur/brightness, and max_train_severity=3. Copy the sequence
    into an immutable tuple; this API does not provide per-run policy tuning.

    Draw order on x.device: scalar float64 rand for intervals [0,.50),
    [.50,.67), [.67,.84), [.84,1); only corrupted events draw scalar int64
    randint(1,4); only noise then draws float32 randn(x.shape). Identity draws
    no severity and reports 0. Metadata contains only event and severity.
    All validation precedes draws, including radius4 compatibility for any
    possible blur. Thus policy pixels require H,W>4; CIFAR uses32x32.
    Normalize afterward. No seed assignment, global RNG, labels or batch state.
    """

    probabilities: tuple
    max_train_severity: int = 3

    def __post_init__(self):
        probabilities = self.probabilities
        if (not isinstance(probabilities, (tuple, list)) or len(probabilities) != 4
                or any(type(p) not in (int, float) for p in probabilities)
                or tuple(probabilities) != (.50, .17, .17, .16)):
            raise ValueError('probabilities must be protocol-v1 [.50,.17,.17,.16]')
        if type(self.max_train_severity) is not int or self.max_train_severity != 3:
            raise ValueError('max_train_severity must be the protocol-v1 Python integer 3')
        object.__setattr__(self, 'probabilities', tuple(probabilities))

    def __call__(self, x: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, dict]:
        _validate_pixels(x)
        if not isinstance(generator, torch.Generator):
            raise ValueError('generator must be an explicit torch.Generator')
        if generator.device != x.device:
            raise ValueError('generator device must match x device (including CUDA index)')
        if min(x.shape[1:]) <= 4:
            raise ValueError('policy requires H,W>4 for every possible reflect blur')
        u = torch.rand((), dtype=torch.float64, device=x.device, generator=generator).item()
        if u < .50:
            event, kind, severity = 'identity', 'clean', 0
        else:
            kind = ('gaussian_noise' if u < .67 else
                    'gaussian_blur' if u < .84 else 'brightness')
            event = kind
            severity = int(torch.randint(1, 4, (), device=x.device, generator=generator).item())
        return apply_corruption(x, kind, severity, generator), {'event': event, 'severity': severity}
