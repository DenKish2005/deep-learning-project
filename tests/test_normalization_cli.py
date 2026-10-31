"""Offline stats CLI checks; frozen split and raw acquisition CLIs stay separate."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import prepare_normalization as cli
from src import data
from tests.test_normalization import artifact, pixels, snapshot


def test_prepare_verify_reuse_and_train_only(artifact, monkeypatch, capsys):
    raw, labels, split, output = artifact
    calls = []
    def load(root, train, download):
        assert train is True and download is False
        calls.append(root)
        return SimpleNamespace(data=raw, targets=labels)
    monkeypatch.setattr(cli, 'load_cifar10', load)
    args = ['--root', 'synthetic-root', '--split-path', str(split), '--output', str(output)]
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'stats_verified' and result['samples'] == 2
    assert result['normalization_file_sha256'] == data._hash(output, 'sha256')
    before = {p.name: snapshot(p) for p in output.parent.iterdir() if p.is_file()}
    assert cli.main(args + ['--verify-only']) == 0
    assert json.loads(capsys.readouterr().out)['mode'] == 'verify-only'
    assert cli.main(args) == 0
    assert {p.name: snapshot(p) for p in output.parent.iterdir() if p.is_file()} == before
    assert calls == [Path('synthetic-root')] * 3


def test_defaults_missing_and_tamper_errors(artifact, monkeypatch, capsys):
    raw, labels, split, output = artifact
    monkeypatch.setattr(cli, 'load_cifar10', lambda *a, **kw: SimpleNamespace(data=raw, targets=labels))
    seen = []
    def defaults(raw, labels, split_path, path):
        seen.append((split_path, path))
        raise data.DataIntegrityError('missing artifact')
    monkeypatch.setattr(cli, 'load_normalization', defaults)
    assert cli.main(['--verify-only']) == 1
    assert seen == [(Path('data/splits/cifar10_split_v1.npz'), Path('data/splits/normalization.json'))]
    assert json.loads(capsys.readouterr().err)['status'] == 'failed'
    monkeypatch.setattr(cli, 'load_normalization', data.load_normalization)
    args = ['--split-path', str(split), '--output', str(output)]
    assert cli.main(args + ['--verify-only']) == 1
    assert not output.exists()
    assert cli.main(args) == 0
    output.write_bytes(b'partial')
    before = snapshot(output)
    assert cli.main(args) == 1
    assert snapshot(output) == before


@pytest.mark.parametrize('args', [['--split-seed', '42'], ['--prepare-split'], ['--unknown'], ['--root']])
def test_invalid_arguments_exit_two(args):
    with pytest.raises(SystemExit) as error: cli.main(args)
    assert error.value.code == 2


def test_help_and_module_missing_inputs(tmp_path):
    help_result = subprocess.run([sys.executable, '-m', 'scripts.prepare_normalization', '--help'],
                                 capture_output=True, text=True, timeout=45)
    assert help_result.returncode == 0
    assert '--verify-only' in help_result.stdout and 'data/splits/normalization.json' in help_result.stdout
    result = subprocess.run([sys.executable, '-m', 'scripts.prepare_normalization',
                             '--root', str(tmp_path / 'missing'), '--verify-only'],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 1
    assert json.loads(result.stderr)['status'] == 'failed'
    assert not (tmp_path / 'missing').exists()
