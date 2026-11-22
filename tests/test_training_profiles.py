from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data import IndexedCIFAR10, _make_loader, compute_normalization
from src.training_profiles import run_training_check
from src import training_profiles
from src.utils import ConfigError, load_config, validate_config


_CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


@pytest.fixture
def tiny_partitions(monkeypatch):
    images = np.random.default_rng(5).integers(0, 256, (120, 32, 32, 3), dtype=np.uint8)
    targets = np.arange(len(images)) % 10
    dataset = SimpleNamespace(data=images, targets=targets, train=True)
    train_ids, val_ids = np.arange(80), np.arange(80, 120)
    statistics = compute_normalization(images, train_ids)
    requests = []

    def make_loaders(config, seed, epoch, *, policy, partitions):
        requests.append(partitions)
        assert set(partitions) <= {'train', 'val'}
        inputs = {
            'train': IndexedCIFAR10(dataset, train_ids, 'train', statistics, seed, epoch, policy),
            'val': IndexedCIFAR10(dataset, val_ids, 'val', statistics, seed, epoch),
        }
        return {name: _make_loader(inputs[name], 16) for name in partitions}

    monkeypatch.setattr(training_profiles, 'make_loaders', make_loaders)
    monkeypatch.setattr(training_profiles, '_PILOT_TRAIN_SAMPLES', 20)
    monkeypatch.setattr(training_profiles, '_PILOT_VAL_SAMPLES', 10)
    return dataset, statistics, requests


def _config(name='clean_cnn'):
    config = load_config(_CONFIGS / f'{name}.yaml')
    config['runtime']['device'] = 'cpu'
    return config


def test_debug_uses_training_ids_without_geometry_or_dropout(tiny_partitions, tmp_path, monkeypatch):
    base, stats, requests = tiny_partitions
    checked = []
    original = training_profiles._debug_run

    def inspect_model_and_inputs(model, loader, *args):
        assert model.dropout.p == 0 and loader.dataset.policy is None
        images, _, ids = next(iter(loader))
        raw = torch.from_numpy(base.data[ids.numpy()].copy()).permute(0, 3, 1, 2).float() / 255
        mean = torch.tensor(stats['mean'])[None, :, None, None]
        std = torch.tensor(stats['std'])[None, :, None, None]
        torch.testing.assert_close(images, (raw - mean) / std, rtol=0, atol=0)
        checked.extend(ids.tolist())
        return original(model, loader, *args)

    monkeypatch.setattr(training_profiles, '_debug_run', inspect_model_and_inputs)
    run_dir = tmp_path / 'debug-short'
    result = run_training_check(_config(), 42, profile='debug', run_dir=run_dir, max_steps=1)
    assert requests == [('train',)] and result['val_indices'] == []
    assert result['train_indices'] == checked
    assert len(set(checked)) == 64 and set(checked) <= set(range(80))
    assert [row['optimizer_steps'] for row in result['history']] == [0, 1]
    assert result['optimizer_steps'] == 1 and (run_dir / 'summary.json').is_file()
    with pytest.raises(FileExistsError):
        run_training_check(_config(), 42, profile='debug', run_dir=run_dir, max_steps=1)


def test_pilot_pair_uses_same_samples_order_and_initial_weights(tiny_partitions, tmp_path):
    _, _, requests = tiny_partitions
    clean, robust = [run_training_check(_config(f'{name}_cnn'), 42, profile='pilot',
                                       run_dir=tmp_path / f'pilot-{name}')
                     for name in ('clean', 'robust')]
    assert requests == [('train', 'val')] * 2
    for key in ('train_indices', 'val_indices', 'init_sha256', 'train_order_sha256', 'optimizer_steps'):
        assert clean[key] == robust[key]
    assert clean['config_hash'] != robust['config_hash']
    for outcome in (clean, robust):
        assert outcome['profile'] == 'pilot' and outcome['status'] == 'complete'
        assert outcome['optimizer_steps'] == 1
        row, = outcome['history']
        assert row['num_train'] == 20 and row['num_val'] == 10
        assert row['train_seconds'] > 0 and row['validation_seconds'] > 0
        assert outcome['estimated_core_run_seconds'] > outcome['elapsed_seconds']


def test_diagnostic_profiles_reject_core_paths_and_core_budget_changes(tmp_path):
    config = _config()
    with pytest.raises(ValueError, match='matching the profile'):
        run_training_check(config, 42, profile='pilot', run_dir=tmp_path / 'core-run')
    with pytest.raises(ValueError, match='between 1 and 300'):
        run_training_check(config, 42, profile='debug', run_dir=tmp_path / 'debug-run', max_steps=301)
    config['training']['epochs'] = 1
    with pytest.raises(ConfigError, match=r'training.epochs.*40'):
        validate_config(config)
    assert list(tmp_path.iterdir()) == []
