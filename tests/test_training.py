"""Synthetic acceptance checks for the shared trainer and its data boundary."""

from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src import data, train
from src.corruptions import CorruptionPolicy
from src.models import CompactCNN
from src.utils import load_config


_CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


@pytest.mark.parametrize('model_name', ['clean_cnn', 'robust_cnn'])
def test_epoch_logging_and_validation(model_name, monkeypatch):
    config = load_config(_CONFIGS / f'{model_name}.yaml')
    raw = np.random.default_rng(5).integers(0, 256, (8, 32, 32, 3), dtype=np.uint8)
    base = SimpleNamespace(data=raw, targets=np.arange(8), train=True)
    train_ids, val_ids = np.arange(5), np.arange(5, 8)
    stats = data.compute_normalization(raw, train_ids)
    policy = CorruptionPolicy(config['augmentation']['probabilities']) if config['augmentation']['enabled'] else None
    train_data = data.IndexedCIFAR10(base, train_ids, 'train', stats, 42, policy=policy)
    val_data = data.IndexedCIFAR10(base, val_ids, 'val', stats, 42)
    train_loader, val_loader = data._make_loader(train_data, 2), data._make_loader(val_data, 2)
    validation_epochs = []
    original_validate = train.validate

    def verify_validation(model, loader, criterion, device):
        assert loader is val_loader
        before = deepcopy(model.state_dict())
        gradients = [parameter.grad.clone() for parameter in model.parameters()]
        metrics = original_validate(model, loader, criterion, device)

        for name, value in model.state_dict().items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            else:
                assert value == before[name]
        for parameter, gradient in zip(model.parameters(), gradients, strict=True):
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
        assert metrics['optimizer_steps'] == 0
        validation_epochs.append(train_data.epoch)
        return metrics

    monkeypatch.setattr(train, 'validate', verify_validation)
    device = torch.device('cpu')
    forward_modes = []
    with train._training_runtime(device, 42):
        model = CompactCNN()
        initial = model.classifier.weight.detach().clone()

        def observe_forward(module, inputs):
            forward_modes.append((module.training, torch.is_inference_mode_enabled(), train_data.epoch))

        with model.register_forward_pre_hook(observe_forward):
            result = train._run_epochs(model, train_loader, val_loader,
                                       {**config['training'], 'epochs': 2}, device)

    assert not torch.equal(initial, model.classifier.weight)
    assert validation_epochs == [1, 2]
    assert forward_modes == [
        mode for epoch in (1, 2) for mode in [(True, False, epoch)] * 3 + [(False, True, epoch)] * 2
    ]
    assert result['status'] == 'complete'
    assert result['optimizer_steps'] == 6
    history = result['history']
    assert [row['epoch'] for row in history] == [1, 2]
    for row in history:
        assert (row['num_train'], row['num_val'], row['optimizer_steps']) == (5, 3, 3)
        assert all(math.isfinite(row[key]) for key in ('train_loss', 'val_loss'))
        assert all(0 <= row[key] <= 1 for key in ('train_accuracy', 'val_accuracy'))
    best = max(history, key=lambda row: (row['val_accuracy'], -row['val_loss']))
    assert result['best_epoch'] == best['epoch']
    assert result['best_val_accuracy'] == best['val_accuracy']
    assert result['best_val_loss'] == best['val_loss']


def test_epoch_metrics_weight_the_smaller_final_batch():
    inputs = torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0]])
    targets = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    criterion = nn.CrossEntropyLoss()
    training = train.train_one_epoch(model, loader, optimizer, criterion, 'cpu')
    validation = train.validate(model, loader, criterion, 'cpu')
    expected_loss = F.cross_entropy(inputs, targets).item()
    for metrics in (training, validation):
        assert metrics['n_samples'] == 3
        assert metrics['loss'] == pytest.approx(expected_loss)
        assert metrics['accuracy'] == 2 / 3
    assert training['optimizer_steps'] == 2


def test_no_test_loader_in_training_and_core_budget(monkeypatch):
    epoch_updates = []
    train_loader = SimpleNamespace(dataset=SimpleNamespace(set_epoch=epoch_updates.append))
    val_loader = object()
    loader_requests = []
    initial_weights = []
    val_scores = [(0.4, 1.0), (0.5, 2.0), (0.5, 1.5), (0.5, 1.5)]

    # Exercise the real run entry point with stand-ins for full-dataset computation.
    def make_loaders(config, run_seed, epoch, *, policy, partitions):
        loader_requests.append((config['model_name'], run_seed, epoch, policy, partitions))
        return {'train': train_loader, 'val': val_loader}

    def train_epoch(model, loader, optimizer, criterion, device):
        assert loader is train_loader and device.type == 'cpu'
        assert isinstance(criterion, nn.CrossEntropyLoss)
        assert torch.are_deterministic_algorithms_enabled()
        assert isinstance(optimizer, torch.optim.AdamW)
        settings = optimizer.param_groups[0]
        assert (settings['betas'], settings['eps'], settings['weight_decay']) == ((0.9, 0.999), 1e-8, 0.0001)
        if epoch_updates[-1] == 1:
            initial_weights.append(model.classifier.weight.detach().clone())
        optimizer.step()
        return {'accuracy': 0.4, 'loss': 2.0, 'n_samples': 45000, 'optimizer_steps': 352}

    def validate(model, loader, criterion, device):
        assert loader is val_loader
        accuracy, loss = val_scores[min(epoch_updates[-1] - 1, len(val_scores) - 1)]
        return {'accuracy': accuracy, 'loss': loss, 'n_samples': 5000}

    monkeypatch.setattr(train, 'make_loaders', make_loaders)
    monkeypatch.setattr(train, 'train_one_epoch', train_epoch)
    monkeypatch.setattr(train, 'validate', validate)
    rng_before = torch.get_rng_state().clone()
    deterministic_before = torch.are_deterministic_algorithms_enabled()

    for model_name in ('clean_cnn', 'robust_cnn'):
        config = load_config(_CONFIGS / f'{model_name}.yaml')
        config['runtime']['device'] = 'cpu'
        result = train.train_run(config, 42)
        assert result['model_name'] == model_name
        assert result['optimizer_steps'] == 14080
        assert len(result['history']) == 40
        assert result['best_epoch'] == 3
        assert [row['best_epoch'] for row in result['history'][:4]] == [1, 2, 3, 3]
        for index, row in enumerate(result['history']):
            expected_lr = 0.001 * (1 + math.cos(math.pi * index / 40)) / 2
            assert row['learning_rate'] == pytest.approx(expected_lr)

    assert loader_requests == [
        ('clean_cnn', 42, 1, None, ('train', 'val')),
        ('robust_cnn', 42, 1, CorruptionPolicy([0.50, 0.17, 0.17, 0.16]), ('train', 'val')),
    ]
    assert epoch_updates == list(range(1, 41)) * 2
    torch.testing.assert_close(initial_weights[0], initial_weights[1], rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.are_deterministic_algorithms_enabled() == deterministic_before


def test_nonfinite_gradients_fail_before_parameter_update():
    model = nn.Linear(2, 2)
    loader = [(torch.ones(1, 2), torch.tensor([1]))]
    optimizer = torch.optim.AdamW(model.parameters())
    initial = model.weight.detach().clone()
    model.weight.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
    with pytest.raises(ValueError, match='finite parameter gradients'):
        train.train_one_epoch(model, loader, optimizer, nn.CrossEntropyLoss(), 'cpu')
    torch.testing.assert_close(model.weight, initial, rtol=0, atol=0)
    assert not optimizer.state
