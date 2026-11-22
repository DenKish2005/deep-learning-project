"""Shared training and clean validation for both CompactCNN regimes."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch import nn

from src.checkpoints import CheckpointRun, capture_rng, checkpoint_directory, restore_rng
from src.corruptions import CorruptionPolicy
from src.data import CLASSES, make_loaders, verify_cifar10_files
from src.models import CompactCNN
from src.utils import capture_code_identity, capture_environment, config_hash, json_sha256, validate_config


def _run_epoch(model, loader, criterion, device, optimizer=None) -> dict:
    started = perf_counter()
    training = optimizer is not None
    model.train(training)
    device = torch.device(device)
    non_blocking = device.type == 'cuda' and getattr(loader, 'pin_memory', False)
    loss_sum = 0.0
    correct = samples = steps = 0

    with torch.enable_grad() if training else torch.inference_mode():
        for images, targets, *_ in loader:
            if targets.ndim != 1 or targets.numel() == 0 or images.shape[0] != len(targets):
                raise ValueError('each batch needs a nonempty target vector aligned with the images')
            images = images.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError('criterion must return a finite scalar batch loss')

            batch_size = len(targets)
            loss_sum += loss.detach().item() * batch_size
            correct += (logits.detach().argmax(dim=1) == targets).sum().item()
            samples += batch_size

            if training:
                loss.backward()
                finite_gradients = [torch.isfinite(parameter.grad).all()
                                    for parameter in model.parameters() if parameter.grad is not None]
                if not finite_gradients or not torch.stack(finite_gradients).all():
                    raise ValueError('training requires finite parameter gradients')
                optimizer.step()
                steps += 1

    if samples == 0:
        raise ValueError('epoch loader must contain at least one example')
    return {
        'loss': loss_sum / samples,
        'accuracy': correct / samples,
        'n_samples': samples,
        'optimizer_steps': steps,
        'elapsed_seconds': perf_counter() - started,
    }


def train_one_epoch(model, loader, optimizer, criterion, device) -> dict:
    """Train on one partition using a criterion that returns mean batch loss."""
    return _run_epoch(model, loader, criterion, device, optimizer)


def validate(model, loader, criterion, device) -> dict:
    """Score the clean validation loader in inference mode."""
    return _run_epoch(model, loader, criterion, device)


def _run_epochs(model, train_loader, val_loader, training: dict, device: torch.device, *,
                expected_counts: tuple[int, int, int] | None = None,
                checkpoints: CheckpointRun | None = None, resume: bool = False) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['lr'],
                                  betas=tuple(training['betas']), eps=training['eps'],
                                  weight_decay=training['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training['t_max'], eta_min=training['eta_min'])
    criterion = nn.CrossEntropyLoss()
    history = []
    best_score = None
    best_epoch = None
    best_model = None
    next_epoch = 1
    elapsed = 0.0
    loaders = (train_loader, val_loader)
    if resume:
        if checkpoints is None:
            raise ValueError('resuming requires a checkpoint run')
        saved = checkpoints.load_last()
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history = saved['history']
        best_model = saved['best_model']
        best_epoch = saved['best_epoch']
        best_score = (saved['best_val_accuracy'], -saved['best_val_loss'])
        elapsed = history[-1]['cumulative_seconds']
        next_epoch = saved['epoch'] + 1
        restore_rng(saved['rng'], device, loaders)
        checkpoints.sync_outputs(saved)
    started = perf_counter()

    for epoch in range(next_epoch, training['epochs'] + 1):
        epoch_started = perf_counter()
        learning_rate = optimizer.param_groups[0]['lr']
        train_loader.dataset.set_epoch(epoch)
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        counts = (train_metrics['n_samples'], val_metrics['n_samples'], train_metrics['optimizer_steps'])
        if expected_counts is not None and counts != expected_counts:
            raise RuntimeError(f'epoch {epoch} counts {counts} do not match configured budget {expected_counts}')

        score = (val_metrics['accuracy'], -val_metrics['loss'])
        improved = best_score is None or score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            if checkpoints is not None:
                best_model = {name: value.detach().cpu().clone() if isinstance(value, torch.Tensor)
                              else deepcopy(value) for name, value in model.state_dict().items()}
        finished = perf_counter()
        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'learning_rate': learning_rate,
            'epoch_seconds': finished - epoch_started,
            'cumulative_seconds': elapsed + (finished - started),
            'best_epoch': best_epoch,
            'num_train': counts[0],
            'num_val': counts[1],
            'optimizer_steps': counts[2],
        })
        scheduler.step()
        if checkpoints is not None:
            checkpoints.save_epoch({
                'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'rng': capture_rng(device, loaders),
                'history': history, 'best_epoch': best_epoch, 'best_model': best_model,
                'best_val_accuracy': best_score[0], 'best_val_loss': -best_score[1],
            }, improved=improved)

    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_val_accuracy': best_score[0],
        'best_val_loss': -best_score[1],
        'optimizer_steps': sum(row['optimizer_steps'] for row in history),
        'status': 'complete',
    }


@contextmanager
def _training_runtime(device: torch.device, seed: int):
    """Use the protocol's seeds and Torch settings, then restore the caller's."""
    cuda = device.type == 'cuda'
    if cuda and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise ValueError('set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA training')
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    if cuda:
        cudnn = torch.backends.cudnn
        matmul = torch.backends.cuda.matmul
        backend_settings = (cudnn.benchmark, cudnn.deterministic,
                            cudnn.conv.fp32_precision, matmul.fp32_precision)

    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if cuda else []):
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.default_generator.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            if cuda:
                torch.cuda.manual_seed(seed)
                cudnn.benchmark = False
                cudnn.deterministic = True
                cudnn.conv.fp32_precision = 'ieee'
                matmul.fp32_precision = 'ieee'
            yield
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            if cuda:
                (cudnn.benchmark, cudnn.deterministic,
                 cudnn.conv.fp32_precision, matmul.fp32_precision) = backend_settings


def _checkpoint_identity(config: dict, seed: int, model: CompactCNN, device: torch.device,
                         run_dir: Path) -> dict:
    files = verify_cifar10_files(Path(config['dataset']['root']))['files']
    model_state = {}
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            value = {'dtype': array.dtype.str, 'shape': list(array.shape),
                     'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
        model_state[name] = value
    environment = capture_environment(device, num_workers=config['runtime']['num_workers'])
    return {
        'run_id': run_dir.name, 'profile': 'core', 'model_name': config['model_name'], 'seed': seed,
        'config_hash': config_hash(config),
        'dataset_sha256': json_sha256({name: info['sha256'] for name, info in files.items()}),
        'split_hash': config['dataset']['split_sha256'],
        'normalization_hash': config['dataset']['normalization_sha256'],
        'architecture': config['architecture'], 'classes': list(CLASSES),
        'init_sha256': json_sha256(model_state),
        **capture_code_identity(Path(__file__).resolve().parents[1]),
        'environment_id': environment['environment_id'], 'device': environment['device'],
    }


def train_run(config: dict, seed: int, *, run_dir: Path | None = None,
              resume: Path | None = None) -> dict:
    """Train a configured CompactCNN and return its history and best validation scores.

    Supply a new run_dir to save checkpoints or point resume at an existing last.pt.
    Each call returns epoch rows with train/validation metrics and step counts.
    """
    validate_config(config)
    if type(seed) is not int or seed not in config['seeds']:
        raise ValueError(f"seed must be one of {config['seeds']}")
    if resume is not None:
        resume = Path(resume)
        if resume.name != 'last.pt' or (run_dir is not None and Path(run_dir).resolve() != resume.parent.resolve()):
            raise ValueError('resume must point to last.pt in the run directory')
        run_dir = resume.parent
    device_name = config['runtime']['device']
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    device = torch.device(device_name)
    augmentation = config['augmentation']
    policy = CorruptionPolicy(augmentation['probabilities']) if augmentation['enabled'] else None
    training = config['training']
    train_size, val_size = config['dataset']['train_size'], config['dataset']['val_size']
    batch_size = training['batch_size']
    steps_per_epoch = (train_size + batch_size - 1) // batch_size

    counts = (train_size, val_size, steps_per_epoch)
    with _training_runtime(device, seed):
        loaders = make_loaders(config, seed, 1, policy=policy, partitions=('train', 'val'))
        model = CompactCNN(num_classes=config['num_classes'], dropout=config['dropout']).to(device)
        if run_dir is None:
            result = _run_epochs(model, loaders['train'], loaders['val'], training, device,
                                 expected_counts=counts)
        else:
            directory = Path(run_dir).resolve()
            identity = _checkpoint_identity(config, seed, model, device, directory)
            checkpoints = CheckpointRun(directory, identity, config)
            with checkpoint_directory(directory, resume=resume is not None):
                result = _run_epochs(model, loaders['train'], loaders['val'], training, device,
                                     expected_counts=counts, checkpoints=checkpoints, resume=resume is not None)
    return {'model_name': config['model_name'], 'seed': seed, 'device': str(device), **result}
