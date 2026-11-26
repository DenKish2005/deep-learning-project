from pathlib import Path

import pytest

from scripts import run_experiments as launcher
from src.checkpoints import CheckpointRun
from src.models import CompactCNN
from src.repro import verify_run_record


CONFIGS = (Path('configs/clean_cnn.yaml'), Path('configs/robust_cnn.yaml'))
SEEDS = (42, 123, 2026)


def test_dry_run_lists_six_distinct_core_runs(tmp_path, capsys):
    import json

    root = tmp_path / 'uncreated'
    assert launcher.main(['--dry-run', '--runs-root', str(root)]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [(row['model_name'], row['seed']) for row in rows] == [
        (model, seed) for model in ('clean_cnn', 'robust_cnn') for seed in SEEDS
    ]
    assert len({row['run_id'] for row in rows}) == 6
    assert all(row['epochs'] == 40 for row in rows)
    assert not root.exists()
    with pytest.raises(ValueError, match='exactly once'):
        launcher.build_plan(CONFIGS, (42, 42, 123))


def test_sequential_launcher_retries_then_verifies_and_skips(tmp_path, monkeypatch):
    plan = launcher.build_plan(CONFIGS, SEEDS, device='cpu', runs_root=tmp_path)
    code = launcher.capture_code_identity(launcher._ROOT)
    device = launcher._resolve_training_device(plan[0].config)
    with launcher._training_runtime(device, 42):
        environment = launcher.capture_environment(device)
    calls = []

    def fake_train(config, seed, *, run_dir):
        run_dir.mkdir()
        calls.append(run_dir.name)
        if len(calls) == 1:
            raise OSError('simulated transient error before first epoch')
        spec = next(item for item in plan if item.directory == run_dir)
        identity = {**code, 'run_id': run_dir.name, 'profile': 'core', 'model_name': config['model_name'],
                    'seed': seed, 'config_hash': spec.config_hash, 'dataset_sha256': '1' * 64,
                    'split_hash': config['dataset']['split_sha256'],
                    'normalization_hash': config['dataset']['normalization_sha256'],
                    'init_sha256': '2' * 64, 'environment_id': environment['environment_id'], 'device': 'cpu'}
        weights = CompactCNN().state_dict()
        history = [{'epoch': epoch, 'best_epoch': 1, 'train_loss': 1.0, 'val_loss': 1.0,
                    'train_accuracy': 0.5, 'val_accuracy': 0.5, 'num_train': 45000,
                    'num_val': 5000, 'optimizer_steps': 352} for epoch in range(1, 41)]
        CheckpointRun(run_dir, identity, config).save_epoch({
            'epoch': 40, 'best_epoch': 1, 'model': weights, 'best_model': weights, 'history': history,
            'best_val_accuracy': 0.5, 'best_val_loss': 1.0, 'optimizer': {}, 'scheduler': {}, 'rng': {},
        }, improved=True)

    monkeypatch.setattr(launcher, 'train_run', fake_train)
    monkeypatch.setattr(launcher, 'sleep', lambda seconds: None)
    assert launcher.run_plan(plan, resume_incomplete=True) == 0
    assert calls == [plan[0].directory.name, *(item.directory.name for item in plan)]
    for spec in plan:
        manifest = verify_run_record(spec.directory)
        assert manifest['profile'] == 'core' and manifest['total_steps'] == 14080
    assert launcher.run_plan(plan, resume_incomplete=True) == 0
    assert len(calls) == 7

    (plan[0].directory / 'history.csv').write_text('damaged')
    assert launcher.run_plan(plan, resume_incomplete=True) == 1
    assert len(calls) == 7


def test_existing_checkpoint_dispatches_resume(tmp_path, monkeypatch):
    spec = launcher.build_plan(CONFIGS, SEEDS, device='cpu', runs_root=tmp_path)[0]
    spec.directory.mkdir()
    path = spec.directory / 'last.pt'
    path.touch()
    calls = []
    monkeypatch.setattr(launcher, 'train_run', lambda config, seed, **kwargs: calls.append((seed, kwargs)))
    with pytest.raises(FileExistsError):
        launcher._train_or_resume(spec, resume_incomplete=False, owned=False)
    launcher._train_or_resume(spec, resume_incomplete=True, owned=False)
    assert calls == [(42, {'resume': path})]
