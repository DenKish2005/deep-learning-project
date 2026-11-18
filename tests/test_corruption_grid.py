"""Offline selection, train-only source boundary and fixed audit noise oracles."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import make_plots as plots
from src import data
from src.utils import stable_seed


@pytest.fixture
def raw():
    return np.random.default_rng(7).integers(0, 256, (30, 32, 32, 3), dtype=np.uint8)


def test_minimum_train_id_per_class_independent_of_input_order():
    labels = np.tile(np.arange(10), 3)
    indices = np.arange(10, 30)[::-1]  # IDs0..9 excluded: validation, never selected.
    assert plots.select_train_ids(labels, indices) == list(range(10, 20))
    assert plots.select_train_ids(labels, indices[::-1]) == list(range(10, 20))


@pytest.mark.parametrize('labels,indices', [
    ([0] * 10, list(range(10))), (list(range(10)), [True]),
    (list(range(10)), [0, 0]), (list(range(10)), [-1]),
    (list(range(10)), [10]), (list(range(10)), []),
    ([False] * 10, list(range(10))), ([1.] * 10, list(range(10))),
    (list(range(11)), list(range(11))), ([[0]], [0]),
])
def test_selection_errors(labels, indices):
    with pytest.raises((ValueError, data.DataIntegrityError)):
        plots.select_train_ids(labels, indices)


def test_exact_conditions_pixels_shared_epsilon_and_rng_isolation(raw):
    before = raw.copy()
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    pixels, rows = plots.prepare_grid_pixels(raw, np.tile(np.arange(10), 3), np.arange(10, 30))
    assert plots.CONDITIONS == (('clean', 0),) + tuple(
        (k, s) for k in ('gaussian_noise', 'gaussian_blur', 'brightness') for s in range(1, 6))
    assert pixels.shape == (10, 16, 3, 32, 32) and pixels.dtype == torch.float32
    assert pixels.device.type == 'cpu' and torch.isfinite(pixels).all()
    assert 0 <= pixels.min() <= pixels.max() <= 1
    for row, info in enumerate(rows):
        original = row + 10
        assert info['original_id'] == original and info['label'] == row
        assert info['class_name'] == data.CLASSES[row]
        seed = stable_seed(494, 'cifar10-python/original-train/corr003-audit', original, 'gaussian_noise', 'v1')
        assert info['noise_seed'] == seed
        x = torch.from_numpy(raw[original].copy()).permute(2, 0, 1).float() / 255
        assert torch.equal(pixels[row, 0], x)
        epsilon = torch.randn(x.shape, generator=torch.Generator().manual_seed(seed))
        for column, sigma in enumerate((.02, .05, .08, .12, .18), 1):
            assert torch.equal(pixels[row, column], (x + sigma * epsilon).clamp(0, 1))
        for column, factor in enumerate((.9, .8, .7, .55, .4), 11):
            assert torch.equal(pixels[row, column], x * factor)
    np.testing.assert_array_equal(raw, before)
    assert random.getstate() == python_rng
    after = np.random.get_state()
    assert after[0] == numpy_rng[0] and np.array_equal(after[1], numpy_rng[1]) and after[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)
    # Reordering candidates replays bitwise and the key differs between IDs.
    replay, _ = plots.prepare_grid_pixels(raw, np.tile(np.arange(10), 3), np.arange(29, 9, -1))
    assert torch.equal(pixels, replay)
    assert len({r['noise_seed'] for r in rows}) == 10


@pytest.mark.parametrize('case', ['float', 'shape', 'length'])
def test_raw_boundary_errors(raw, case):
    labels = np.tile(np.arange(10), 3)
    if case == 'float': raw = raw.astype(np.float32) / 255
    if case == 'shape': raw = raw[:, :, :, :1]
    if case == 'length': labels = labels[:-1]
    with pytest.raises((ValueError, data.DataIntegrityError)):
        plots.prepare_grid_pixels(raw, labels, np.arange(10, 30))


def test_offline_train_only_verification_precedes_selection(raw, tmp_path, monkeypatch):
    events = []
    def load(root, train, download):
        assert train is True and download is False
        events.append('verified train')
        return SimpleNamespace(data=raw, targets=np.tile(np.arange(10), 3))
    def split(path, targets):
        events.append('verified split')
        return np.arange(10, 30), np.arange(10)
    prepare = plots.prepare_grid_pixels
    def selected(*args):
        assert events == ['verified train', 'verified split']
        events.append('selected')
        return prepare(*args)
    monkeypatch.setattr(data, 'load_cifar10', load)
    monkeypatch.setattr(data, 'load_split', split)
    monkeypatch.setattr(data, 'verify_cifar10_files', lambda root: {'files': 'fixture'})
    monkeypatch.setattr(plots, 'prepare_grid_pixels', selected)
    path = tmp_path / 'split.npz'
    path.write_bytes(b'synthetic split')
    path.with_name('split_manifest.json').write_text('{}')
    _, source = plots.load_audit_grid(tmp_path, path)
    assert source['test_decoded'] is False and len(source['rows']) == 10
    assert source['noise_key']['namespace'] == plots.AUDIT_NAMESPACE
    def fail(*args): raise data.DataIntegrityError('corrupt split')
    monkeypatch.setattr(data, 'load_split', fail)
    with pytest.raises(data.DataIntegrityError, match='corrupt split'):
        plots.load_audit_grid(tmp_path, path)
    assert events == ['verified train', 'verified split', 'selected', 'verified train']
