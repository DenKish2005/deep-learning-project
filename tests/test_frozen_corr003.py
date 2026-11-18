"""Authenticate bounded recorded-source replay; preserve strict source drift errors."""

import hashlib

import pytest

from scripts import verify_frozen_corr003 as frozen
from scripts import make_plots as plots
from tests.test_corruption_grid_cli import grid, exported  # noqa: F401


def test_snapshot_has_exact_recorded_implementation_hashes(tmp_path):
    hashes = frozen.snapshot_source(tmp_path)
    assert set(hashes) == {'src/data.py', 'src/utils.py', 'src/corruptions.py',
                           'scripts/make_plots.py'}
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()} == {
        *hashes, 'src/__init__.py'}
    for name, expected in hashes.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected


def test_snapshot_rejects_changed_manifest_without_touching_it(tmp_path, monkeypatch):
    manifest = tmp_path / 'changed.json'
    manifest.write_text('{}')
    before = manifest.read_bytes(), manifest.stat().st_mtime_ns
    monkeypatch.setattr(frozen, 'MANIFEST', manifest)
    with pytest.raises(ValueError, match='manifest mismatch'):
        frozen.snapshot_source(tmp_path / 'snapshot')
    assert (manifest.read_bytes(), manifest.stat().st_mtime_ns) == before
    assert not (tmp_path / 'snapshot').exists()


def test_changed_implementation_is_strict_export_error(exported, monkeypatch):
    # Synthetic export fixture lives in the existing CLI tests; authenticate the
    # full source record, including implementation, rather than just pixel hashes.
    out, _ = exported
    original = plots._source_record
    def changed(*args):
        record = original(*args)
        record['implementation_sha256_lf']['src/corruptions.py'] = '0' * 64
        return record
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}
    monkeypatch.setattr(plots, '_source_record', changed)
    with pytest.raises(ValueError, match='mismatch'):
        plots.export_audit(out, out, out, verify_only=True)
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()} == before
