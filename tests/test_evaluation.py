from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src import evaluate
from src.checkpoints import save_checkpoint_atomic
from src.corruptions import keyed_test_epsilon
from src.data import CLASSES
from src.evaluate import EvaluationProvenance, build_conditions, evaluate_condition, save_evaluation
from src.models import CompactCNN
from src.utils import config_hash, json_sha256, load_config, sha256_file


class KnownLogitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('score', torch.tensor(3.0))
        self.inputs = []

    def forward(self, inputs):
        assert not self.training and torch.is_inference_mode_enabled()
        self.inputs.append(inputs.clone())
        winners = (inputs[:, 0, 0, 0] > 0).long()
        return F.one_hot(winners, num_classes=10).float() * self.score


@pytest.fixture
def evaluation_inputs(tmp_path):
    values = torch.tensor([5, 200, 180, 0, 255], dtype=torch.uint8)
    images = values[:, None, None, None].expand(-1, 3, 32, 32).contiguous()
    targets = torch.tensor([0, 0, 1, 0, 1], dtype=torch.int64)
    indices = torch.tensor([41, 3, 6, 9, 24], dtype=torch.int64)
    normalization = {
        'mean': [0.25, 0.5, 0.75], 'std': [0.25, 0.125, 0.5],
        'channels': ['R', 'G', 'B'], 'scale': 1 / 255, 'ddof': 0, 'samples': 1,
        'pixels_per_channel': 1024, 'train_indices_sha256': '1' * 64,
        'algorithm': 'float64-two-pass-chunks256-v1',
    }
    checkpoint = tmp_path / 'known-logits.pt'
    torch.save(KnownLogitClassifier().state_dict(), checkpoint)
    model = KnownLogitClassifier()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    provenance = EvaluationProvenance(
        run_id='fixture-s42', model_name='known-logits', seed=42, dataset='synthetic-cifar10',
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        config_hash='2' * 64, split_hash='3' * 64,
        normalization_hash=hashlib.sha256(json.dumps(normalization, sort_keys=True).encode()).hexdigest(),
        git_sha='fixture', environment_id='pytest-cpu',
    )
    options = {'normalization': normalization, 'expected_indices': indices.tolist(),
               'provenance': provenance}
    return model, TensorDataset(images, targets, indices), options


def test_protocol_condition_order_and_parameters():
    conditions = build_conditions()
    assert [(condition['corruption'], condition['severity']) for condition in conditions] == [
        ('clean', 0),
        *((family, level) for family in ('gaussian_noise', 'gaussian_blur', 'brightness')
          for level in range(1, 6)),
    ]
    assert conditions[0]['parameter_value'] is None
    assert conditions[1]['parameter_value'] == {'sigma': 0.02}
    assert conditions[10]['parameter_value'] == {'sigma': 2.0, 'kernel_size': 13}
    assert conditions[-1]['parameter_value'] == {'factor': 0.40}
    conditions[1]['parameter_value']['sigma'] = 1.0
    assert build_conditions()[1]['parameter_value'] == {'sigma': 0.02}


def test_checkpoint_cli_uses_full_normalization_and_reuses_verified_outputs(evaluation_inputs, tmp_path, monkeypatch):
    _, dataset, options = evaluation_inputs
    images, labels, _ = dataset.tensors
    config = load_config(Path('configs/clean_cnn.yaml').resolve())
    monkeypatch.chdir(tmp_path)
    config['dataset']['normalization_path'] = 'normalization.json'
    norm_path = Path('normalization.json')
    norm_path.write_text(json.dumps({'statistics': options['normalization']}))
    config['dataset']['normalization_sha256'] = sha256_file(norm_path)
    files = {'test_batch': {'sha256': '1' * 64}}
    test_data = SimpleNamespace(base=SimpleNamespace(data=images.permute(0, 2, 3, 1).numpy()),
                                indices=np.arange(len(images)), labels=labels.numpy())

    def make_loaders(*args, partitions, **kwargs):
        assert partitions == ('test',)
        return {'test': SimpleNamespace(dataset=test_data)}

    monkeypatch.setattr(evaluate, 'load_config', lambda _: config)
    monkeypatch.setattr(evaluate, 'verify_cifar10_files', lambda _: {'files': files})
    monkeypatch.setattr(evaluate, 'make_loaders', make_loaders)
    monkeypatch.setattr(evaluate, '_TEST_SIZE', len(images))
    monkeypatch.setattr(evaluate, 'capture_environment', lambda *a, **kw: {'environment_id': 'fixture'})
    checkpoint = tmp_path / 'best.pt'
    save_checkpoint_atomic(checkpoint, {
        'schema_version': 1, 'kind': 'best', 'epoch': 1, 'config': config,
        'model': CompactCNN().state_dict(), 'identity': {
            'profile': 'core', 'model_name': 'clean_cnn', 'seed': 42, 'run_id': 'fixture',
            'config_hash': config_hash(config), 'architecture': config['architecture'],
            'classes': list(CLASSES), 'split_hash': config['dataset']['split_sha256'],
            'normalization_hash': config['dataset']['normalization_sha256'],
            'git_sha': 'fixture', 'dataset_sha256': json_sha256({'test_batch': '1' * 64}),
        },
    })
    dest = tmp_path / 'eval'
    summaries = evaluate.evaluate_checkpoint(checkpoint, Path('unused'), dest, device='cpu')
    assert len(summaries) == 16 and all(row['n_samples'] == len(images) for row in summaries)

    def unexpected_inference(*args, **kwargs):
        pytest.fail('existing verified conditions should be reused')

    monkeypatch.setattr(evaluate, 'evaluate_condition', unexpected_inference)
    assert evaluate.evaluate_checkpoint(checkpoint, Path('unused'), dest, device='cpu', resume=True) == summaries
    Path(summaries[0]['predictions_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        evaluate.evaluate_checkpoint(checkpoint, Path('unused'), dest, device='cpu', resume=True)


@pytest.mark.parametrize('condition_index', [0, 1])
def test_checkpoint_to_evaluation(evaluation_inputs, tmp_path, condition_index):
    model, dataset, options = evaluation_inputs
    condition = build_conditions()[condition_index]
    result = evaluate_condition(model, DataLoader(dataset, batch_size=2), condition, **options)
    summary_path, predictions_path = tmp_path / 'metrics' / 'summary.json', tmp_path / 'predictions.npz'
    summary = save_evaluation(summary_path, predictions_path, result)

    oracle = []
    mean = torch.tensor([0.25, 0.5, 0.75])[:, None, None]
    std = torch.tensor([0.25, 0.125, 0.5])[:, None, None]
    for raw, _, original_id in dataset:
        pixels = raw.float() / 255
        if condition_index == 1:
            epsilon = keyed_test_epsilon(int(original_id), namespace='cifar10-python/original-test')
            pixels = (pixels + 0.02 * epsilon).clamp(0, 1)
        oracle.append((pixels - mean) / std)
    torch.testing.assert_close(torch.cat(model.inputs), torch.stack(oracle), rtol=0, atol=0)

    assert json.loads(summary_path.read_text()) == summary
    assert summary['status'] == 'complete'
    assert summary['profile'] == 'synthetic'
    assert summary['checkpoint_sha256'] == options['provenance'].checkpoint_sha256
    assert summary['predictions_path'] == str(predictions_path)
    assert summary['predictions_sha256'] == hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    assert summary['n_samples'] == 5
    assert summary['n_correct'] == 4
    assert summary['accuracy'] == 0.8
    assert summary['macro_f1'] == pytest.approx(0.16)
    expected_loss = math.log(math.exp(3.0) + 9) - 3.0 * 4 / 5
    assert summary['loss'] == pytest.approx(expected_loss, abs=1e-12)

    with np.load(predictions_path, allow_pickle=False) as predictions:
        assert set(predictions.files) == {'indices', 'targets', 'logits'}
        np.testing.assert_array_equal(predictions['indices'], np.array([3, 6, 9, 24, 41]), strict=True)
        np.testing.assert_array_equal(predictions['targets'], np.array([0, 1, 0, 1, 0]), strict=True)
        logits = predictions['logits']
        assert logits.shape == (5, 10) and logits.dtype == np.float32
        targets = predictions['targets']
        winners = logits.argmax(axis=1)
        reconstructed_loss = F.cross_entropy(torch.from_numpy(logits).double(), torch.from_numpy(targets))
        assert reconstructed_loss.item() == summary['loss']
    counts = np.zeros((10, 10), dtype=np.int64)
    np.add.at(counts, (targets, winners), 1)
    assert summary['confusion_matrix'] == counts.tolist()
    assert summary['per_class_accuracy'] == pytest.approx([2 / 3, 1, *([0] * 8)])


def test_eval_idempotent_and_order_independent(evaluation_inputs):
    model, dataset, options = evaluation_inputs
    condition = build_conditions()[4]
    first = evaluate_condition(model, DataLoader(dataset, batch_size=2), condition, **options)
    reordered = torch.utils.data.Subset(dataset, [4, 2, 0, 3, 1])
    other_run = replace(options['provenance'], run_id='fixture-s123', seed=123,
                        checkpoint_sha256='4' * 64, config_hash='5' * 64)
    for loader, provenance in ((DataLoader(dataset, batch_size=2), options['provenance']),
                               (DataLoader(reordered, batch_size=4), other_run)):
        again = evaluate_condition(model, loader, condition, **{**options, 'provenance': provenance})
        np.testing.assert_array_equal(again.indices, first.indices)
        np.testing.assert_array_equal(again.targets, first.targets)
        np.testing.assert_array_equal(again.logits, first.logits)
        for key in ('condition_hash', 'accuracy', 'loss', 'macro_f1', 'confusion_matrix'):
            assert again.summary[key] == first.summary[key]

    different_condition = evaluate_condition(model, DataLoader(dataset, batch_size=2),
                                             build_conditions()[5], **options)
    assert different_condition.summary['condition_hash'] != first.summary['condition_hash']
    images, targets, indices = dataset.tensors
    altered = images.clone()
    altered[0, 1, 1, 1] = 123
    different_pixels = evaluate_condition(
        model, DataLoader(TensorDataset(altered, targets, indices), batch_size=2), condition, **options)
    assert different_pixels.summary['condition_hash'] != first.summary['condition_hash']


@pytest.mark.parametrize(('order', 'message'), [
    ([0, 1, 2, 3], 'missing 1'), ([0, 1, 2, 3, 0], 'duplicate original test ID'),
    ([0, 1, 2, 3, 4], 'unexpected original test ID'),
])
def test_invalid_index_coverage(evaluation_inputs, order, message):
    model, dataset, options = evaluation_inputs
    if message.startswith('unexpected'):
        options['expected_indices'][-1] = 9999
    with pytest.raises(ValueError, match=message):
        evaluate_condition(model, DataLoader(torch.utils.data.Subset(dataset, order), batch_size=2),
                           build_conditions()[0], **options)


def test_rejects_invalid_condition_and_logits(evaluation_inputs):
    model, dataset, options = evaluation_inputs
    condition = build_conditions()[1]
    with pytest.raises(ValueError, match='condition parameters'):
        evaluate_condition(model, DataLoader(dataset), {**condition, 'parameter_value': {}}, **options)
    model.score.fill_(float('inf'))
    with pytest.raises(ValueError, match='logits must be finite'):
        evaluate_condition(model, DataLoader(dataset), condition, **options)


def test_uses_compactcnn_on_synthetic_images(evaluation_inputs, tmp_path):
    _, dataset, options = evaluation_inputs
    checkpoint = tmp_path / 'compactcnn.pt'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        torch.save(CompactCNN().state_dict(), checkpoint)
        model = CompactCNN()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    options['provenance'] = replace(options['provenance'], model_name='clean_cnn',
                                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    initial = model.features[1].running_mean.clone()
    result = evaluate_condition(model, DataLoader(dataset, batch_size=3), build_conditions()[6], **options)
    assert result.logits.shape == (5, 10)
    assert np.isfinite(result.logits).all()
    torch.testing.assert_close(model.features[1].running_mean, initial, rtol=0, atol=0)


def test_save_evaluation_preserves_existing_artifacts(evaluation_inputs, tmp_path):
    model, dataset, options = evaluation_inputs
    result = evaluate_condition(model, DataLoader(dataset), build_conditions()[0], **options)
    summary_path, predictions_path = tmp_path / 'summary.json', tmp_path / 'predictions.npz'
    predictions_path.write_bytes(b'existing predictions')
    with pytest.raises(FileExistsError):
        save_evaluation(summary_path, predictions_path, result)
    assert predictions_path.read_bytes() == b'existing predictions'
    assert not summary_path.exists()

    summary_path.write_text('existing summary')
    with pytest.raises(FileExistsError):
        save_evaluation(summary_path, tmp_path / 'another.npz', result)
    assert summary_path.read_text() == 'existing summary'
    assert not (tmp_path / 'another.npz').exists()


def test_save_evaluation_can_retry_after_summary_write_failure(evaluation_inputs, tmp_path, monkeypatch):
    model, dataset, options = evaluation_inputs
    result = evaluate_condition(model, DataLoader(dataset), build_conditions()[0], **options)
    summary_path, predictions_path = tmp_path / 'summary.json', tmp_path / 'predictions.npz'
    link = os.link

    def fail_summary_link(source, destination):
        if destination == summary_path:
            raise OSError('simulated summary write failure')
        link(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr('src.evaluate.os.link', fail_summary_link)
        with pytest.raises(OSError, match='simulated summary write failure'):
            save_evaluation(summary_path, predictions_path, result)
    assert not summary_path.exists()
    assert not predictions_path.exists()

    assert save_evaluation(summary_path, predictions_path, result)['status'] == 'complete'
