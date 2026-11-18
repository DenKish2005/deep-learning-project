"""Actual CORR-003 replay only; explicit opt-in requires real verified inputs.

No downloads, regeneration, test decoding, scientific evaluation or human approval.
"""

import hashlib
import json
import os
from pathlib import Path
import pickle
import random

import numpy as np
from PIL import Image
import pytest
import torch

from scripts import make_plots as plots
from src import data
from src.utils import stable_seed
from tests.test_blur import blur_oracle


pytestmark = pytest.mark.skipif(os.environ.get('RUN_CORR_GRID') != '1',
                                reason='Actual grid requires RUN_CORR_GRID=1')
IDS = [29, 4, 6, 9, 3, 27, 0, 7, 8, 1]
OUTPUT = Path('outputs/figures')


def snapshot(paths):
    return {p.as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in paths if p.is_file()}


@pytest.fixture
def actual(monkeypatch):
    root = Path(os.environ.get('CIFAR10_ROOT', 'data/raw'))
    split = Path(os.environ.get('CIFAR10_SPLITS', 'data/splits')) / 'cifar10_split_v1.npz'
    paths = list(root.rglob('*')) + list(split.parent.rglob('*'))
    paths += list(Path('report').glob('corr00*_reference.npz'))
    paths += [OUTPUT / name for name in plots.OUTPUT_NAMES]
    before = snapshot(paths)
    read = data._read_partition
    decoded = []
    def train_only(root, train):
        assert train is True, 'CORR-003 must never decode official test'
        decoded.append('train')
        return read(root, train)
    def forbidden(*a, **kw): pytest.fail('CORR-003 is offline and read-only')
    monkeypatch.setattr(data, '_read_partition', train_only)
    monkeypatch.setattr(data, '_download_archive', forbidden)
    monkeypatch.setattr(data, 'ensure_cifar10', forbidden)
    pixels, source = plots.load_audit_grid(root, split)
    yield root, split, pixels, source
    assert decoded and set(decoded) == {'train'}
    assert snapshot(paths) == before


def announce(kind, value):
    print('CORR003_EVIDENCE ' + json.dumps({'kind': kind, 'value': value}, sort_keys=True))


def test_actual_source_and_independent_numeric_replay(actual):
    root, _, pixels, source = actual
    assert [r['original_id'] for r in source['rows']] == IDS
    assert source['test_decoded'] is False
    # Independent raw decode after checksum verification, exclusively train batch1.
    with (root / data.BASE_FOLDER / 'data_batch_1').open('rb') as stream:
        batch = pickle.load(stream, encoding='latin1')
    raw = batch['data'].reshape(-1, 3, 32, 32)
    max_blur_error = 0.
    clipped = [0] * 5
    for row, original in enumerate(IDS):
        assert batch['labels'][original] == row
        x = torch.from_numpy(raw[original].copy()).float() / 255
        assert torch.equal(pixels[row, 0], x)
        epsilon = torch.randn(x.shape, generator=torch.Generator().manual_seed(stable_seed(
            494, 'cifar10-python/original-train/corr003-audit', original, 'gaussian_noise', 'v1')))
        for col, sigma in enumerate((.02, .05, .08, .12, .18), 1):
            preclamp = x + sigma * epsilon
            clipped[col - 1] += int(((preclamp < 0) | (preclamp > 1)).sum())
            assert torch.equal(pixels[row, col], preclamp.clamp(0, 1))
        for col, (sigma, kernel) in enumerate(zip((.3, .7, 1.1, 1.5, 2.), (3, 7, 9, 11, 13)), 6):
            oracle = blur_oracle(x, sigma, kernel)
            error = (pixels[row, col] - oracle).abs().max().item()
            max_blur_error = max(max_blur_error, error)
            torch.testing.assert_close(pixels[row, col], oracle, atol=1e-6, rtol=0)
        for col, factor in enumerate((.9, .8, .7, .55, .4), 11):
            assert torch.equal(pixels[row, col], x * factor)
    announce('numeric_replay', {'ids_class_order': IDS, 'image_conditions': 160,
        'max_blur_numpy2d_error': max_blur_error, 'noise_clipped_values_per_level': clipped,
        'values_per_noise_level': 30720, 'noise_and_darken_bitwise_oracles': True,
        'pixel_sha256_c_bytes': hashlib.sha256(pixels.numpy().astype('<f4').tobytes()).hexdigest(),
        'test_decoded': False})


def test_actual_readonly_exports_rng_and_png_rgb(actual):
    root, split, pixels, source = actual
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    for verify in (True, False):
        assert plots.export_audit(root, split, OUTPUT, verify_only=verify)['status'] == 'verified'
    assert random.getstate() == python_rng
    after = np.random.get_state()
    assert after[0] == numpy_rng[0] and np.array_equal(after[1], numpy_rng[1]) and after[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)
    record = json.loads((OUTPUT / plots.OUTPUT_NAMES[2]).read_text())
    assert record['source'] == plots._source_record(pixels, source)
    figure = plots.plot_corruption_grid(pixels, source['rows'])
    figure.canvas.draw()
    max_rgb_error = 0.
    with Image.open(OUTPUT / plots.OUTPUT_NAMES[0]) as image:
        rgb = np.asarray(image)[:, :, :3]
        # Center of each source pixel block; avoid interpolation/rounding boundaries.
        for r in range(10):
            for c in range(16):
                ax = figure.axes[r * 16 + c]
                for px, py in ((3, 3), (16, 16), (28, 28)):
                    x, y = ax.transData.transform((px, py))
                    actual_rgb = rgb[image.height - 1 - int(y), int(x)].astype(float)
                    expected = pixels[r, c, :, py, px].numpy() * 255
                    error = float(np.max(np.abs(actual_rgb - expected)))
                    max_rgb_error = max(max_rgb_error, error)
                    np.testing.assert_allclose(actual_rgb, expected, atol=1, rtol=0)
    figure.clear()
    announce('exports', {'png_rgb_samples': 480, 'max_rgb_quantization_error_255': max_rgb_error,
        'reuse_and_verify_preserve_bytes_mtimes': True, 'global_rng_unchanged': True,
        'outputs': record['outputs'], 'source_manifest': plots._file_record(OUTPUT / plots.OUTPUT_NAMES[2])})
