"""Single-condition inference and prediction artifacts for CIFAR-10."""

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.corruptions import CorruptionPolicy, apply_corruption, keyed_test_generator, severity_params
from src.checkpoints import load_checkpoint
from src.data import (CLASSES, PARTITION_SIZE, make_loaders, validate_normalization,
                     verify_cifar10_files)
from src.metrics import classification_metrics
from src.models import CompactCNN
from src.utils import capture_environment, config_hash, json_sha256, load_config, sha256_file


_TEST_SIZE = PARTITION_SIZE[False]
_TEST_NAMESPACE = 'cifar10-python/original-test'
_EVAL_SEED = 100494


@dataclass(frozen=True)
class EvaluationProvenance:
    """Identity supplied by the caller's fixture or future checkpoint adapter."""

    run_id: str
    model_name: str
    seed: int
    dataset: str
    checkpoint_sha256: str
    config_hash: str
    split_hash: str
    normalization_hash: str
    git_sha: str
    environment_id: str

    def __post_init__(self):
        for name in ('run_id', 'model_name', 'dataset', 'git_sha', 'environment_id'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a nonempty string')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        for name in ('checkpoint_sha256', 'config_hash', 'split_hash', 'normalization_hash'):
            value = getattr(self, name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in '0123456789abcdef' for char in value)):
                raise ValueError(f'{name} must be a lowercase SHA-256 digest')


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict
    indices: np.ndarray
    targets: np.ndarray
    logits: np.ndarray


def _condition(corruption: str, severity: int) -> dict:
    if type(corruption) is str and corruption == 'clean':
        if type(severity) is not int or severity != 0:
            raise ValueError('clean severity must be the integer 0')
        parameters = None
    else:
        parameters = severity_params(corruption, severity)
    return {'corruption': corruption, 'severity': severity, 'parameter_value': parameters}


def build_conditions() -> list[dict]:
    """Return the sixteen conditions in protocol order, with fresh parameters."""
    return [_condition('clean', 0)] + [
        _condition(family, severity)
        for family in ('gaussian_noise', 'gaussian_blur', 'brightness')
        for severity in range(1, 6)
    ]


def _ordered_ids(expected_indices: Sequence[int] | np.ndarray) -> np.ndarray:
    indices = np.asarray(expected_indices)
    if (indices.ndim != 1 or not indices.size or indices.dtype.kind not in 'iu'
            or (not isinstance(expected_indices, np.ndarray)
                and any(isinstance(index, (bool, np.bool_)) for index in expected_indices))
            or np.any(indices < 0) or np.any(indices >= _TEST_SIZE)
            or np.unique(indices).size != indices.size):
        raise ValueError('expected_indices must contain unique integer test IDs in [0, 9999]')
    return np.sort(indices.astype(np.int64))


def _check_batch(images: torch.Tensor, targets: torch.Tensor, indices: torch.Tensor) -> None:
    if (not isinstance(images, torch.Tensor) or images.layout != torch.strided
            or images.device.type != 'cpu' or images.dtype != torch.uint8 or images.ndim != 4
            or images.shape[1:] != (3, 32, 32) or not len(images)):
        raise ValueError('images must be a nonempty CPU uint8 tensor shaped [N, 3, 32, 32]')
    for name, values in (('targets', targets), ('indices', indices)):
        if (not isinstance(values, torch.Tensor) or values.layout != torch.strided
                or values.device.type != 'cpu' or values.dtype != torch.int64
                or values.shape != (len(images),)):
            raise ValueError(f'{name} must be a CPU int64 vector aligned with images')
    if not ((targets >= 0) & (targets < len(CLASSES))).all():
        raise ValueError('targets contain an invalid CIFAR-10 class ID')


def _condition_hash(condition: dict, provenance: EvaluationProvenance, indices: np.ndarray,
                    targets: np.ndarray, image_digests: list[bytes]) -> str:
    metadata = {
        'version': 'cifar10-evaluation-input-v1',
        'dataset': provenance.dataset,
        'namespace': _TEST_NAMESPACE,
        'condition': condition,
        'eval_seed': _EVAL_SEED,
        'normalization_hash': provenance.normalization_hash,
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode())
    digest.update(indices.astype('<i8', copy=False).tobytes())
    digest.update(targets.astype('<i8', copy=False).tobytes())
    for image_digest in image_digests:
        digest.update(image_digest)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_condition(model: nn.Module, batches: Iterable[Sequence[torch.Tensor]],
                       condition: dict, *, normalization: dict,
                       expected_indices: Sequence[int] | np.ndarray, provenance: EvaluationProvenance,
                       device: str | torch.device = 'cpu',
                       profile: str = 'synthetic') -> EvaluationResult:
    """Evaluate raw (NCHW uint8 images, int64 targets, int64 original IDs) batches.

    The caller supplies the model and the expected test IDs. This entry point
    marks results synthetic while production checkpoint verification is pending.
    """
    started = perf_counter()
    if profile not in ('synthetic', 'core'):
        raise ValueError("profile must be 'synthetic' or 'core'")
    if not isinstance(condition, dict):
        raise ValueError('condition must be a mapping from build_conditions()')
    canonical = _condition(condition.get('corruption'), condition.get('severity'))
    if canonical != condition:
        raise ValueError('condition parameters must match build_conditions()')
    validate_normalization(normalization)
    ordered_ids = _ordered_ids(expected_indices)
    positions = {original_id: position for position, original_id in enumerate(ordered_ids.tolist())}
    seen = np.zeros(len(ordered_ids), dtype=bool)
    targets_by_id = np.empty(len(ordered_ids), dtype=np.int64)
    logits_by_id = np.empty((len(ordered_ids), len(CLASSES)), dtype=np.float32)
    image_digests = [b''] * len(ordered_ids)
    mean = torch.tensor(normalization['mean'], dtype=torch.float32, device='cpu').view(3, 1, 1)
    std = torch.tensor(normalization['std'], dtype=torch.float32, device='cpu').view(3, 1, 1)
    corruption, severity = canonical['corruption'], canonical['severity']
    model.to(device).eval()

    for images, targets, indices in batches:
        _check_batch(images, targets, indices)
        normalized = []
        batch_positions = []
        pixels = images.to(torch.float32).div_(255.0)
        for pixel, original_id in zip(pixels, indices.tolist(), strict=True):
            if original_id not in positions:
                raise ValueError(f'unexpected original test ID {original_id}')
            position = positions[original_id]
            if seen[position]:
                raise ValueError(f'duplicate original test ID {original_id}')
            seen[position] = True
            generator = (keyed_test_generator(original_id, namespace=_TEST_NAMESPACE)
                         if corruption == 'gaussian_noise' else None)
            image = (apply_corruption(pixel, corruption, severity, generator) - mean) / std
            image_bytes = image.numpy().astype('<f4', copy=False).tobytes()
            image_digests[position] = hashlib.sha256(image_bytes).digest()
            normalized.append(image)
            batch_positions.append(position)

        inputs = torch.stack(normalized)
        if not torch.isfinite(inputs).all():
            raise ValueError('normalized inputs must be finite')
        logits = model(inputs.to(device))
        if (not isinstance(logits, torch.Tensor) or logits.layout != torch.strided
                or not logits.is_floating_point() or logits.shape != (len(images), len(CLASSES))):
            raise ValueError('model must return floating-point logits shaped [N, 10]')
        logits = logits.to(device='cpu', dtype=torch.float32)
        if not torch.isfinite(logits).all():
            raise ValueError('model logits must be finite in float32')
        targets_by_id[batch_positions] = targets.numpy()
        logits_by_id[batch_positions] = logits.numpy()

    if not seen.all():
        raise ValueError(f'missing {int((~seen).sum())} expected original test IDs')
    metrics = classification_metrics(targets_by_id, logits_by_id.argmax(axis=1))
    # Compute over the persisted logits with equal weight per image, in original ID order.
    loss = F.cross_entropy(torch.from_numpy(logits_by_id).double(), torch.from_numpy(targets_by_id))
    summary = {
        'schema_version': 1,
        **asdict(provenance),
        'profile': profile,
        **canonical,
        **metrics,
        'loss': loss.item(),
        'per_class_accuracy': metrics['per_class_accuracy'].tolist(),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'condition_hash': _condition_hash(canonical, provenance, ordered_ids, targets_by_id, image_digests),
        'eval_seed': _EVAL_SEED,
        'duration_seconds': perf_counter() - started,
    }
    return EvaluationResult(summary, ordered_ids, targets_by_id, logits_by_id)


def save_predictions(path: Path, indices: np.ndarray, targets: np.ndarray, logits: np.ndarray) -> None:
    """Write one NPZ with aligned, sorted original IDs; reject existing files."""
    if (indices.dtype != np.int64 or not np.array_equal(indices, _ordered_ids(indices))
            or targets.dtype != np.int64 or targets.shape != indices.shape
            or not np.all((targets >= 0) & (targets < len(CLASSES)))
            or logits.dtype != np.float32 or logits.shape != (len(indices), len(CLASSES))
            or not np.isfinite(logits).all()):
        raise ValueError('predictions must have sorted int64 IDs, aligned int64 targets, and finite float32 logits')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.evaluation-', dir=path.parent) as temporary:
        staged = Path(temporary) / 'predictions.npz'
        np.savez_compressed(staged, indices=indices, targets=targets, logits=logits)
        os.link(staged, path)


def save_evaluation(summary_path: Path, predictions_path: Path, result: EvaluationResult) -> dict:
    """Write a prediction pair without replacement; the complete summary is written last."""
    summary_path, predictions_path = Path(summary_path), Path(predictions_path)
    if summary_path.resolve() == predictions_path.resolve():
        raise ValueError('summary and predictions paths must be different')
    if summary_path.exists() or summary_path.is_symlink():
        raise FileExistsError(f'evaluation summary already exists: {summary_path}')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.evaluation-', dir=summary_path.parent) as temporary:
        staged = Path(temporary) / 'summary.json'
        save_predictions(predictions_path, result.indices, result.targets, result.logits)
        try:
            with predictions_path.open('rb') as stream:
                predictions_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
            summary = {
                **result.summary,
                'predictions_path': str(predictions_path),
                'predictions_sha256': predictions_sha256,
                'status': 'complete',
            }
            staged.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
                              encoding='utf-8')
            os.link(staged, summary_path)
        except (OSError, TypeError, ValueError):
            predictions_path.unlink()
            raise
    return summary


def _raw_test_batches(dataset, batch_size: int):
    """Yield verified test images as raw CPU uint8 NCHW batches with original IDs."""
    base = dataset.base
    labels = dataset.labels
    for start in range(0, len(dataset.indices), batch_size):
        stop = min(start + batch_size, len(dataset.indices))
        indices = dataset.indices[start:stop]
        images = torch.from_numpy(base.data[indices].copy()).permute(0, 3, 1, 2).contiguous()
        targets = torch.as_tensor(labels[indices].copy(), dtype=torch.int64)
        original_ids = torch.as_tensor(indices.copy(), dtype=torch.int64)
        yield images, targets, original_ids


def _read_core_evaluation(summary_path: Path, prediction_path: Path, *, condition: dict,
                          provenance: EvaluationProvenance, targets: np.ndarray) -> dict:
    if (summary_path.is_symlink() or prediction_path.is_symlink()
            or not summary_path.is_file() or not prediction_path.is_file()):
        raise ValueError(f'missing or non-regular evaluation pair: {summary_path}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    expected = {'schema_version': 1, 'profile': 'core', 'status': 'complete', 'eval_seed': _EVAL_SEED,
                **asdict(provenance), **condition}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError(f'evaluation metadata does not match its checkpoint or condition: {summary_path}')
    if (Path(summary['predictions_path']).resolve() != prediction_path.resolve()
            or sha256_file(prediction_path) != summary['predictions_sha256']):
        raise ValueError(f'evaluation prediction checksum or path mismatch: {prediction_path}')
    with np.load(prediction_path, allow_pickle=False) as saved:
        if set(saved.files) != {'indices', 'targets', 'logits'}:
            raise ValueError(f'invalid prediction file fields: {prediction_path}')
        indices, labels, logits = (saved[key] for key in ('indices', 'targets', 'logits'))
    if (indices.dtype != np.int64 or not np.array_equal(indices, np.arange(len(targets)))
            or labels.dtype != np.int64 or not np.array_equal(labels, targets)
            or logits.dtype != np.float32 or logits.shape != (len(targets), len(CLASSES))
            or not np.isfinite(logits).all()):
        raise ValueError(f'prediction artifacts disagree with the verified test examples: {prediction_path}')
    scores = classification_metrics(labels, logits.argmax(axis=1))
    loss = F.cross_entropy(torch.from_numpy(logits).double(), torch.from_numpy(labels)).item()
    if (any(not np.array_equal(summary[key], value) for key, value in scores.items())
            or not np.isclose(summary['loss'], loss, rtol=1e-10)):
        raise ValueError(f'evaluation metrics disagree with saved predictions: {prediction_path}')
    return summary


def evaluate_checkpoint(checkpoint_path: Path, config_path: Path, output_dir: Path, *,
                        device: str = 'auto', resume: bool = False) -> list[dict]:
    """Run all 16 protocol conditions for one validation-selected core checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.name != 'best.pt' or checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError('checkpoint must be a regular best.pt file')
    config = load_config(config_path)
    resolved = 'cuda' if device == 'auto' and torch.cuda.is_available() else (
        'cpu' if device == 'auto' else device)
    if resolved not in ('cpu', 'cuda'):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if resolved == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    torch_device = torch.device(resolved)

    checkpoint = load_checkpoint(checkpoint_path, device=torch_device, kind='best')
    identity = checkpoint['identity']
    expected_config_hash = config_hash(config)
    checks = {
        'profile': 'core',
        'model_name': config['model_name'],
        'config_hash': expected_config_hash,
        'split_hash': config['dataset']['split_sha256'],
        'normalization_hash': config['dataset']['normalization_sha256'],
        'architecture': config['architecture'],
        'classes': list(CLASSES),
    }
    mismatches = [key for key, value in checks.items() if identity.get(key) != value]
    if mismatches:
        raise ValueError(f'checkpoint does not match evaluation config: {", ".join(mismatches)}')
    if type(identity.get('seed')) is not int or identity['seed'] not in config['seeds']:
        raise ValueError('checkpoint seed is not one of the configured core seeds')
    run_id = identity.get('run_id')
    if (not isinstance(run_id, str) or not run_id.strip()
            or Path(run_id).name != run_id or run_id in ('.', '..')):
        raise ValueError('checkpoint has an invalid run_id')

    files = verify_cifar10_files(Path(config['dataset']['root']))['files']
    dataset_sha256 = json_sha256({name: info['sha256'] for name, info in files.items()})
    if identity.get('dataset_sha256') != dataset_sha256:
        raise ValueError('checkpoint CIFAR-10 file identity does not match the verified dataset')

    # This performs the frozen split and train-only normalization checks without
    # changing either artifact. Only its raw test partition is used below.
    policy = (CorruptionPolicy(config['augmentation']['probabilities'])
              if config['augmentation']['enabled'] else None)
    loaders = make_loaders(
        config, identity['seed'], 0, policy=policy, partitions=('test',)
    )
    test_dataset = loaders['test'].dataset
    # make_loaders verified the file against the training split. Read the full
    # statistics record and confirm these are the same file bytes.
    normalization_bytes = Path(config['dataset']['normalization_path']).read_bytes()
    if hashlib.sha256(normalization_bytes).hexdigest() != identity['normalization_hash']:
        raise ValueError('normalization file changed after verification')
    normalization = json.loads(normalization_bytes)['statistics']
    validate_normalization(normalization)
    expected_indices = np.arange(_TEST_SIZE, dtype=np.int64)
    targets = test_dataset.labels[expected_indices]

    model = CompactCNN(num_classes=config['num_classes'], dropout=config['dropout'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(torch_device).eval()

    environment_id = capture_environment(torch_device, num_workers=0)['environment_id']
    provenance = EvaluationProvenance(
        run_id=run_id,
        model_name=identity['model_name'],
        seed=identity['seed'],
        dataset=f"cifar10:{dataset_sha256}",
        checkpoint_sha256=sha256_file(checkpoint_path),
        config_hash=identity['config_hash'],
        split_hash=identity['split_hash'],
        normalization_hash=identity['normalization_hash'],
        git_sha=identity['git_sha'],
        environment_id=environment_id,
    )
    root = Path(output_dir)
    metrics_dir = root / 'metrics' / run_id
    predictions_dir = root / 'predictions' / run_id
    batch_size = config['evaluation']['batch_size']
    summaries = []
    for condition in build_conditions():
        condition_id = f"{condition['corruption']}-s{condition['severity']}"
        summary_path = metrics_dir / f'{condition_id}.json'
        prediction_path = predictions_dir / f'{condition_id}.npz'
        if summary_path.exists() or summary_path.is_symlink():
            if not resume:
                raise FileExistsError(f'evaluation summary already exists: {summary_path}')
        else:
            if resume and prediction_path.is_file() and not prediction_path.is_symlink():
                # A prediction written before an interruption has no summary or provenance.
                abandoned = prediction_path.parent / '.incomplete'
                abandoned.mkdir(exist_ok=True)
                folder = Path(mkdtemp(prefix='orphan-', dir=abandoned))
                prediction_path.rename(folder / prediction_path.name)
            result = evaluate_condition(
                model, _raw_test_batches(test_dataset, batch_size), condition,
                normalization=normalization, expected_indices=expected_indices,
                provenance=provenance, device=torch_device, profile='core',
            )
            save_evaluation(summary_path, prediction_path, result)
        summaries.append(_read_core_evaluation(summary_path, prediction_path, condition=condition,
                                               provenance=provenance, targets=targets))
    return summaries


def main(argv=None) -> int:
    """Evaluate one core checkpoint on clean and all fifteen corruption conditions."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path,
                        help='validation-selected best.pt from one completed core run')
    parser.add_argument('--config', required=True, type=Path,
                        help='matching clean_cnn.yaml or robust_cnn.yaml')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'),
                        help='root for separate metrics/ and predictions/ run folders')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--resume', action='store_true', help='Verify and reuse completed condition files')
    args = parser.parse_args(argv)
    summaries = evaluate_checkpoint(
        args.checkpoint, args.config, args.output_dir, device=args.device, resume=args.resume
    )
    print(f"Saved {len(summaries)} core conditions for {summaries[0]['run_id']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
