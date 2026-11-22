"""Local run identity and artifact verification without official CIFAR-10 data."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import CompactCNN
from src.repro import backup_run_record, verify_run_record, write_run_record
from src.train import _run_epochs, _training_runtime
from src.utils import (ConfigError, capture_code_identity, capture_environment, config_hash,
                       load_config, sha256_file, write_json_atomic)


ROOT = Path(__file__).resolve().parents[1]


def test_config_hash_is_portable_and_binds_controls():
    path = ROOT / 'configs/clean_cnn.yaml'
    clean = load_config(path)
    robust = load_config(ROOT / 'configs/robust_cnn.yaml')
    relocated = dict(reversed(list(clean.items())))
    relocated['dataset'] = dict(reversed(list(clean['dataset'].items())))
    relocated['dataset']['root'] = 'other/data'
    relocated['dataset']['split_path'] = 'other/split.npz'
    relocated['output'] = {**clean['output'], 'root': 'other/runs'}
    digest = config_hash(clean)
    assert digest == config_hash(relocated)
    assert digest != config_hash(robust)
    relocated['dataset']['split_sha256'] = 'a' * 64
    assert digest != config_hash(relocated)

    output = subprocess.check_output([sys.executable, '-c',
        'import sys; from src.utils import config_hash, load_config; print(config_hash(load_config(sys.argv[1])))',
        str(path)], cwd=ROOT, env={**os.environ, 'PYTHONHASHSEED': 'random'}, text=True)
    assert output.strip() == digest
    relocated['dataset']['normalization_sha256'] = None
    with pytest.raises(ConfigError, match='resolved split and normalization hashes'):
        config_hash(relocated)


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / 'checkout'
    root.mkdir()

    def git(*args):
        subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)

    git('init', '-q')
    (root / '.gitignore').write_text('outputs/\n', encoding='utf-8')
    (root / 'model.py').write_text('SIZE = 32\n', encoding='utf-8')
    git('add', '.')
    git('-c', 'user.name=REPRO fixture', '-c', 'user.email=repro@example.invalid',
        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'synthetic source')
    return root


def test_working_source_fingerprint_tracks_edits_and_ignores_outputs(checkout):
    clean = capture_code_identity(checkout)
    assert clean['git_dirty'] is False
    (checkout / 'outputs').mkdir()
    (checkout / 'outputs/training.log').write_text('ignored file', encoding='utf-8')
    assert capture_code_identity(checkout) == clean

    (checkout / 'model.py').write_text('SIZE = 16\n', encoding='utf-8')
    dirty = capture_code_identity(checkout)
    assert dirty['git_sha'] == clean['git_sha']
    assert dirty['git_dirty'] is True
    assert dirty['code_snapshot_sha256'] != clean['code_snapshot_sha256']
    (checkout / 'extra.py').write_text('VERSION = 1\n', encoding='utf-8')
    assert capture_code_identity(checkout)['code_snapshot_sha256'] != dirty['code_snapshot_sha256']
    assert capture_code_identity(checkout) == capture_code_identity(checkout)


def test_environment_identity_is_stable_and_leaves_runtime_flags():
    flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic)
    initial = capture_environment('cpu')
    assert initial == capture_environment('cpu')
    assert initial['device'] == 'cpu' and initial['gpu'] is None
    assert initial['packages']['torch']
    assert initial['environment_id'] != capture_environment('cpu', num_workers=2)['environment_id']
    assert flags == (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic)


@pytest.fixture
def record_inputs(tmp_path, checkout):
    config = load_config(ROOT / 'configs/clean_cnn.yaml')
    artifact = tmp_path / 'trainer.csv'
    artifact.write_text('epoch,train_loss\n1,1.0\n', encoding='utf-8')
    started = datetime(2026, 11, 22, 8, tzinfo=timezone.utc)
    finished = datetime(2026, 11, 22, 9, tzinfo=timezone.utc)
    return {
        'config': config, 'code_identity': capture_code_identity(checkout),
        'seed': 42, 'dataset_sha256': '1' * 64, 'init_sha256': '2' * 64,
        'started_at': started, 'finished_at': finished, 'environment': capture_environment('cpu'),
        'training_result': {'status': 'complete', 'optimizer_steps': 7, 'best_epoch': 1},
        'artifacts': {'history.csv': artifact},
    }


def test_write_verify_and_back_up_actual_synthetic_training(record_inputs, tmp_path):
    class EpochDataset(TensorDataset):
        def set_epoch(self, epoch):
            self.epoch = epoch

    generator = torch.Generator().manual_seed(5)
    record_inputs['code_identity'] = capture_code_identity(ROOT)
    images = torch.rand(6, 3, 32, 32, generator=generator)
    labels = torch.arange(6)
    training_data = DataLoader(EpochDataset(images[:3], labels[:3]), batch_size=2, generator=generator)
    validation_data = DataLoader(TensorDataset(images[3:5], labels[3:5]), batch_size=2, generator=generator)
    config = record_inputs['config']
    config['training'].update(epochs=2, batch_size=2)
    config['evaluation']['batch_size'] = 2
    config['runtime']['device'] = 'cpu'
    config['dataset'].update(
        train_size=3, val_size=2, test_size=1,
        split_sha256=hashlib.sha256(b'train:0,1,2;val:3,4;test:5').hexdigest(),
        normalization_sha256=hashlib.sha256(b'synthetic identity normalization').hexdigest(),
    )
    device = torch.device('cpu')
    with _training_runtime(device, 42):
        record_inputs['started_at'] = datetime.now(timezone.utc)
        model = CompactCNN()
        initial_state = tmp_path / 'initial.pt'
        torch.save(model.state_dict(), initial_state)
        record_inputs['init_sha256'] = sha256_file(initial_state)
        record_inputs['environment'] = capture_environment('cpu')
        record_inputs['training_result'] = _run_epochs(
            model, training_data, validation_data, config['training'], device)
        record_inputs['finished_at'] = datetime.now(timezone.utc)
    record_inputs['dataset_sha256'] = hashlib.sha256(images.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
    history = record_inputs['artifacts']['history.csv']
    rows = record_inputs['training_result']['history']
    with history.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    run = tmp_path / 'runs' / 'clean-s42-fixture'
    manifest = write_run_record(run, **record_inputs)
    assert manifest['profile'] == 'synthetic' and manifest['status'] == 'complete'
    assert manifest['run_id'] == 'clean-s42-fixture'
    assert manifest['total_steps'] == 4
    assert manifest['best_epoch'] == record_inputs['training_result']['best_epoch']
    assert manifest['environment_id'] == record_inputs['environment']['environment_id']
    assert manifest['code_snapshot_sha256'] == record_inputs['code_identity']['code_snapshot_sha256']
    assert load_config(run / 'config_resolved.yaml', require_core_protocol=False) == config
    with pytest.raises(ConfigError, match='protocol v1'):
        load_config(run / 'config_resolved.yaml')
    assert manifest['artifacts']['history.csv'] == sha256_file(history)
    assert verify_run_record(run) == manifest
    assert (run / 'history.csv').read_bytes() == history.read_bytes()

    destination = tmp_path / 'backup' / 'portable-copy'
    assert backup_run_record(run, destination) == manifest
    assert verify_run_record(destination) == manifest
    with pytest.raises(FileExistsError):
        write_run_record(run, **record_inputs)
    with pytest.raises(FileExistsError):
        backup_run_record(run, destination)
    assert verify_run_record(run) == verify_run_record(destination) == manifest


@pytest.mark.parametrize('action,error', [('modify', 'checksum mismatch'), ('delete', 'missing run artifact')])
def test_verification_detects_corrupted_backups(record_inputs, tmp_path, action, error):
    run, backup = tmp_path / 'run', tmp_path / 'backup'
    original = write_run_record(run, **record_inputs)
    backup_run_record(run, backup)
    history = backup / 'history.csv'
    if action == 'modify':
        history.write_text('tampered', encoding='utf-8')
    else:
        history.unlink()
    with pytest.raises(ValueError, match=error):
        verify_run_record(backup)
    assert verify_run_record(run) == original


def test_missing_metadata_and_interrupted_write_do_not_publish(record_inputs, tmp_path, monkeypatch):
    run = tmp_path / 'run'
    incomplete = {**record_inputs, 'environment': {**record_inputs['environment']}}
    del incomplete['environment']['python']
    with pytest.raises(ValueError, match='missing required metadata'):
        write_run_record(run, **incomplete)
    assert not run.exists()

    def stop_manifest(source, destination):
        if Path(destination).name == 'run_manifest.json':
            raise OSError('simulated write failure')
        real_link(source, destination)

    real_link = os.link
    with monkeypatch.context() as patch:
        patch.setattr('src.utils.os.link', stop_manifest)
        with pytest.raises(OSError, match='simulated write failure'):
            write_run_record(run, **record_inputs)
    assert not run.exists()
    original = write_run_record(run, **record_inputs)
    assert verify_run_record(run) == original
    (run / 'run_manifest.json').unlink()
    with pytest.raises(ValueError, match='missing a complete manifest'):
        verify_run_record(run)


def test_verification_checks_manifest_links_and_artifact_paths(record_inputs, tmp_path):
    with pytest.raises(ValueError, match='invalid relative artifact path'):
        write_run_record(tmp_path / 'invalid', **{**record_inputs,
                         'artifacts': {'../escape.csv': record_inputs['artifacts']['history.csv']}})
    run = tmp_path / 'run'
    manifest = write_run_record(run, **record_inputs)
    manifest['environment_id'] = '3' * 64
    (run / 'run_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='environment identity'):
        verify_run_record(run)


def test_write_json_is_exclusive_and_rejects_nan(tmp_path):
    path = tmp_path / 'record.json'
    write_json_atomic(path, {'value': 1})
    with pytest.raises(FileExistsError):
        write_json_atomic(path, {'value': 2})
    assert json.loads(path.read_text(encoding='utf-8')) == {'value': 1}
    invalid = tmp_path / 'invalid.json'
    with pytest.raises(ValueError):
        write_json_atomic(invalid, {'value': float('nan')})
    assert not invalid.exists()
