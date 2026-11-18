"""Bounded PNG/PDF export, immutable publication, provenance and CLI failures."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from scripts import make_plots as plots
from src import data


@pytest.fixture
def grid():
    # Known RGB constants per class/column detect auto-contrast or channel swaps.
    pixels = torch.empty(10, 16, 3, 32, 32)
    for r in range(10):
        for c in range(16):
            pixels[r, c] = torch.tensor([r / 10, c / 16, .25])[:, None, None]
    rows = [{'original_id': i + 10, 'label': i, 'class_name': data.CLASSES[i]} for i in range(10)]
    return pixels, rows


@pytest.fixture
def exported(grid, tmp_path, monkeypatch):
    pixels, rows = grid
    monkeypatch.setattr(plots, 'load_audit_grid', lambda *args: (pixels, {'rows': rows, 'fixture': 'synthetic'}))
    out = tmp_path / 'figures'
    result = plots.export_audit(tmp_path, tmp_path / 'split.npz', out)
    assert result['status'] == 'created'
    return out, result


def snapshot(out):
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}


def test_figure_axes_range_labels_and_no_mutation(grid):
    pixels, rows = grid
    before = pixels.clone()
    figure = plots.plot_corruption_grid(pixels, rows)
    assert len(figure.axes) == 160
    for i, ax in enumerate(figure.axes):
        image = ax.images[0]
        assert image.get_clim() == (0, 1) and image.get_interpolation() == 'nearest'
        np.testing.assert_array_equal(image.get_array(), pixels[i // 16, i % 16].permute(1, 2, 0).numpy())
    assert 'ID 10' in figure.axes[0].get_ylabel()
    assert [ax.get_title() for ax in figure.axes[:3]] == ['clean\ns0', 's1\n0.02', 's2\n0.05']
    assert torch.equal(pixels, before)
    figure.clear()


def test_exports_open_rgb_sampling_manifest_and_readonly_reuse(exported, grid):
    out, result = exported
    record = json.loads((out / plots.OUTPUT_NAMES[2]).read_text())
    assert record['outputs'] == result['outputs']
    assert record['source']['pixels']['shape'] == [10, 16, 3, 32, 32]
    assert set(record['source']['implementation_sha256_lf']) == {
        'scripts/make_plots.py', 'src/data.py', 'src/corruptions.py', 'src/utils.py'}
    assert (out / plots.OUTPUT_NAMES[1]).read_bytes().startswith(b'%PDF-')
    pixels, rows = grid
    figure = plots.plot_corruption_grid(pixels, rows)
    figure.canvas.draw()
    with Image.open(out / plots.OUTPUT_NAMES[0]) as png:
        assert png.size == (3520, 2320)
        for r, c in [(0, 0), (4, 9), (9, 15)]:
            x, y = figure.axes[r * 16 + c].transData.transform((16, 16))
            observed = np.asarray(png)[png.height - 1 - int(y), int(x), :3]
            expected = pixels[r, c, :, 16, 16].numpy() * 255
            np.testing.assert_allclose(observed, expected, atol=1, rtol=0)
    figure.clear()
    before = snapshot(out)
    assert plots.export_audit(Path('unused'), Path('unused'), out)['mode'] == 'reuse'
    assert plots.export_audit(Path('unused'), Path('unused'), out, verify_only=True)['mode'] == 'verify-only'
    assert snapshot(out) == before


@pytest.mark.parametrize('case', ['shape', 'dtype', 'range', 'nan', 'rows', 'labels', 'ids'])
def test_invalid_numeric_or_source_boundary(grid, case):
    pixels, rows = grid
    if case == 'shape': pixels = pixels[:9]
    if case == 'dtype': pixels = pixels.double()
    if case == 'range': pixels[0, 0, 0, 0, 0] = -1
    if case == 'nan': pixels[0, 0, 0, 0, 0] = float('nan')
    if case == 'rows': rows = rows[:9]
    if case == 'labels': rows[0]['label'] = True
    if case == 'ids': rows[0]['original_id'] = rows[1]['original_id']
    with pytest.raises(ValueError): plots.plot_corruption_grid(pixels, rows)


@pytest.mark.parametrize('target', [0, 1, 2])
def test_tampered_export_or_manifest_preserved(exported, target):
    out, _ = exported
    (out / plots.OUTPUT_NAMES[target]).write_bytes(b'corrupt or conflicting')
    before = snapshot(out)
    with pytest.raises(ValueError): plots.export_audit(Path('unused'), Path('unused'), out)
    assert snapshot(out) == before


def test_changed_source_rejected(exported, monkeypatch, grid):
    out, _ = exported
    pixels, rows = grid
    monkeypatch.setattr(plots, 'load_audit_grid', lambda *args: (pixels, {'rows': rows, 'fixture': 'changed'}))
    before = snapshot(out)
    with pytest.raises(ValueError, match='mismatch'):
        plots.export_audit(Path('unused'), Path('unused'), out)
    assert snapshot(out) == before


def test_missing_partial_and_invalid_outputs_preserved(tmp_path):
    out = tmp_path / 'absent'
    with pytest.raises(ValueError, match='requires existing'):
        plots.export_audit(tmp_path, tmp_path, out, verify_only=True)
    assert not out.exists()
    out.mkdir()
    (out / plots.OUTPUT_NAMES[0]).write_bytes(b'preserve')
    before = snapshot(out)
    with pytest.raises(ValueError, match='incomplete'):
        plots.export_audit(tmp_path, tmp_path, out)
    assert snapshot(out) == before
    (out / plots.OUTPUT_NAMES[1]).mkdir()
    with pytest.raises(ValueError, match='regular files'):
        plots.export_audit(tmp_path, tmp_path, out)


def test_render_failure_never_publishes(grid, tmp_path, monkeypatch):
    pixels, rows = grid
    monkeypatch.setattr(plots, 'load_audit_grid', lambda *args: (pixels, {'rows': rows}))
    def fail(*args, **kwargs): raise OSError('injected render failure')
    monkeypatch.setattr(plots.Figure, 'savefig', fail)
    out = tmp_path / 'figures'
    with pytest.raises(OSError, match='injected'):
        plots.export_audit(tmp_path, tmp_path, out)
    assert not list(out.iterdir())


def test_exclusive_publication_preserves_racing_file(grid, tmp_path, monkeypatch):
    pixels, rows = grid
    monkeypatch.setattr(plots, 'load_audit_grid', lambda *args: (pixels, {'rows': rows}))
    link = plots.os.link
    def race(source, destination):
        if destination.suffix == '.pdf': destination.write_bytes(b'preserve racing file')
        link(source, destination)
    monkeypatch.setattr(plots.os, 'link', race)
    out = tmp_path / 'figures'
    with pytest.raises(FileExistsError): plots.export_audit(tmp_path, tmp_path, out)
    assert (out / plots.OUTPUT_NAMES[1]).read_bytes() == b'preserve racing file'
    assert not (out / plots.OUTPUT_NAMES[2]).exists()


@pytest.mark.parametrize('args', [[], ['--metrics', 'x'], ['--corruption-grid', '--severity', '3'],
                                  ['--corruption-grid', '--root']])
def test_cli_invalid_arguments(args):
    with pytest.raises(SystemExit) as exc: plots.main(args)
    assert exc.value.code == 2


def test_cli_success_defaults_and_structured_error(exported, monkeypatch, capsys):
    _, result = exported
    seen = []
    def run(root, split, out, verify_only):
        seen.append((root, split, out, verify_only))
        return result
    monkeypatch.setattr(plots, 'export_audit', run)
    assert plots.main(['--corruption-grid', '--verify-only']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'created'
    assert seen == [(Path('data/raw'), Path('data/splits/cifar10_split_v1.npz'), Path('outputs/figures'), True)]
    def fail(*a, **kw): raise data.DataIntegrityError('corrupt raw input')
    monkeypatch.setattr(plots, 'export_audit', fail)
    assert plots.main(['--corruption-grid']) == 1
    assert json.loads(capsys.readouterr().err)['status'] == 'failed'


def test_subprocess_help_and_missing_actual_input(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'scripts.make_plots', '--help'],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0 and '--corruption-grid' in result.stdout
    out = tmp_path / 'figures'
    result = subprocess.run([sys.executable, '-m', 'scripts.make_plots', '--corruption-grid',
                             '--root', str(tmp_path / 'missing'), '--output', str(out)],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 1 and json.loads(result.stderr)['status'] == 'failed'
    assert not out.exists()
