"""Epoch checkpoint and resume checks using only synthetic images."""

import csv
import hashlib
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import train
from src.checkpoints import checkpoint_directory, load_checkpoint, save_checkpoint_atomic
from src.models import CompactCNN
from src.utils import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_training(monkeypatch):
    config = load_config(ROOT / 'configs/clean_cnn.yaml')
    config['runtime']['device'] = 'cpu'
    config['training'].update(epochs=2, batch_size=2)
    config['dataset'].update(
        train_size=5, val_size=3, test_size=1,
        split_sha256=hashlib.sha256(b'fixture:train:0-4;val:5-7').hexdigest(),
        normalization_sha256=hashlib.sha256(b'fixture:unit scale').hexdigest(),
    )
    config['evaluation']['batch_size'] = 2
    images = torch.rand(8, 3, 32, 32, generator=torch.Generator().manual_seed(100))
    targets = torch.arange(len(images))

    class EpochDataset(TensorDataset):
        def set_epoch(self, epoch):
            self.epoch = epoch

        def __getitem__(self, index):
            image, target = super().__getitem__(index)
            return image + 0.01 * (random.random() + np.random.random()), target

    def make_loaders(config, seed, epoch, *, policy, partitions):
        assert partitions == ('train', 'val') and policy is None
        return {
            'train': DataLoader(EpochDataset(images[:5], targets[:5]), batch_size=2, shuffle=True,
                                generator=torch.Generator().manual_seed(seed + 1)),
            'val': DataLoader(TensorDataset(images[5:], targets[5:]), batch_size=2,
                              generator=torch.Generator().manual_seed(seed + 2)),
        }

    def verified_files(root):
        data = images.numpy().tobytes() + targets.numpy().tobytes()
        return {'files': {'synthetic': {'sha256': hashlib.sha256(data).hexdigest()}}}

    # Allow the small fixture budget at this entry point. The core config tests
    # independently exercise production validation and all 40 configured epochs.
    monkeypatch.setattr(train, 'validate_config', lambda config: None)
    monkeypatch.setattr(train, 'make_loaders', make_loaders)
    monkeypatch.setattr(train, 'verify_cifar10_files', verified_files)
    return config, images


def _assert_identical(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_identical(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        for before, after in zip(left, right, strict=True):
            _assert_identical(before, after)
    else:
        assert left == right


def test_resume_matches_uninterrupted_cpu(synthetic_training, tmp_path, monkeypatch):
    config, images = synthetic_training
    uninterrupted, interrupted = tmp_path / 'full', tmp_path / 'restarted'

    # Force a worse second score so the test also exercises selection of epoch 1.
    original_validate = train.validate
    validation_scores = iter([1.0, 0.0, 1.0, 0.0])

    def validate(*args):
        return {**original_validate(*args), 'accuracy': next(validation_scores)}

    monkeypatch.setattr(train, 'validate', validate)
    train.train_run(config, 42, run_dir=uninterrupted)
    expected = load_checkpoint(uninterrupted / 'last.pt', kind='last')
    train_epoch = train.train_one_epoch

    def interrupt_second_epoch(model, loader, *args):
        if loader.dataset.epoch == 2:
            raise KeyboardInterrupt
        return train_epoch(model, loader, *args)

    with monkeypatch.context() as patch:
        patch.setattr(train, 'train_one_epoch', interrupt_second_epoch)
        with pytest.raises(KeyboardInterrupt):
            train.train_run(config, 42, run_dir=interrupted)

    resume_path = interrupted / 'last.pt'
    assert load_checkpoint(resume_path, kind='last')['epoch'] == 1
    # Model and history files can lag the completed epoch at process exit.
    (interrupted / 'best.pt').unlink()
    (interrupted / 'history.csv').write_text('partial history', encoding='utf-8')
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    result = train.train_run(config, 42, resume=resume_path)
    actual = load_checkpoint(resume_path, kind='last')

    assert actual['epoch'] == 2 and actual['best_epoch'] == 1 and result['optimizer_steps'] == 6
    assert actual['identity']['init_sha256'] == expected['identity']['init_sha256']
    for key in ('model', 'optimizer', 'scheduler', 'rng', 'best_model'):
        _assert_identical(expected[key], actual[key])
    for expected_row, actual_row in zip(expected['history'], actual['history'], strict=True):
        for key in expected_row.keys() - {'epoch_seconds', 'cumulative_seconds'}:
            assert expected_row[key] == actual_row[key]
    assert result['history'][1]['cumulative_seconds'] >= result['history'][0]['cumulative_seconds']
    with (interrupted / 'history.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
        assert [int(row['epoch']) for row in rows] == [1, 2]
        assert all(row['experiment_id'] == 'restarted' for row in rows)

    best = load_checkpoint(interrupted / 'best.pt', identity=actual['identity'], kind='best')
    assert best['epoch'] == 1 and best['val_accuracy'] == 1.0
    _assert_identical(best['model'], actual['best_model'])
    assert random.getstate() == python_rng
    assert np.random.get_state()[0] == numpy_rng[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    assert np.random.get_state()[2:] == numpy_rng[2:]
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    model = CompactCNN().eval()
    model.load_state_dict(best['model'], strict=True)
    with torch.inference_mode():
        logits = model(images[:2])
    assert logits.shape == (2, 10) and torch.isfinite(logits).all()


def test_checkpoint_rejects_identity_mismatch_and_corrupt_file(synthetic_training, tmp_path):
    config, _ = synthetic_training
    run_dir = tmp_path / 'run'
    train.train_run(config, 42, run_dir=run_dir)
    last = run_dir / 'last.pt'
    original = last.read_bytes()
    identity = load_checkpoint(last)['identity']
    for key, value in [('seed', 123), ('split_hash', '3' * 64), ('config_hash', '4' * 64)]:
        with pytest.raises(ValueError, match=key):
            load_checkpoint(last, identity={**identity, key: value})
    with pytest.raises(ValueError, match='seed'):
        train.train_run(config, 123, resume=last)
    with pytest.raises(ValueError, match='resume must point to last.pt'):
        train.train_run(config, 42, resume=run_dir / 'best.pt')
    assert last.read_bytes() == original

    bad = tmp_path / 'corrupt.pt'
    bad.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError, match='invalid checkpoint'):
        load_checkpoint(bad)


def test_atomic_checkpoint_keeps_previous_bytes_on_failed_write(tmp_path, monkeypatch):
    path = tmp_path / 'last.pt'
    save_checkpoint_atomic(path, {'value': torch.tensor([1])})
    original = path.read_bytes()

    def fail_write(state, stream):
        stream.write(b'partial data')
        raise OSError('simulated disk write failure')

    with monkeypatch.context() as patch:
        patch.setattr(torch, 'save', fail_write)
        with pytest.raises(OSError, match='simulated disk write failure'):
            save_checkpoint_atomic(path, {'value': torch.tensor([2])})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_checkpoint_directory_excludes_a_second_writer(tmp_path):
    directory = tmp_path / 'run'
    with checkpoint_directory(directory, resume=False):
        with pytest.raises(FileExistsError):
            with checkpoint_directory(directory, resume=False):
                pytest.fail('new run unexpectedly reused an existing directory')
        with pytest.raises(RuntimeError, match='already in use'):
            with checkpoint_directory(directory, resume=True):
                pytest.fail('second writer unexpectedly obtained the lock')
    with checkpoint_directory(directory, resume=True):
        assert directory.is_dir()
