"""Pixel-space corruption primitives. Blur/darkening/policy await later tasks."""

import math


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
