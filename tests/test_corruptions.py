"""TEST-C02: exact protocol mapping, independent numeric oracles and errors."""

from pathlib import Path

import numpy as np
import pytest

from src.corruptions import severity_params
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
