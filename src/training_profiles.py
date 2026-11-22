"""Bounded CIFAR-10 training checks using the verified train and validation splits."""

import argparse
from copy import copy
import json
from math import ceil
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn

from src.corruptions import CorruptionPolicy
from src.data import CLASSES, _make_loader, make_loaders
from src.models import CompactCNN
from src.train import (_initial_state_sha256, _make_optimizer, _resolve_training_device,
                       _run_epochs, _training_runtime, train_one_epoch, validate)
from src.utils import (capture_code_identity, capture_environment, config_hash, json_sha256,
                       load_config, stable_seed, validate_config, write_json_atomic)


_DEBUG_SAMPLES = 64
_DEBUG_MAX_STEPS = 300
_PILOT_TRAIN_SAMPLES = 4096
_PILOT_VAL_SAMPLES = 512


def _subset(dataset, size: int, seed: int):
    """Sample original IDs by class from a persisted partition."""
    if type(size) is not int or not len(CLASSES) <= size <= len(dataset.indices):
        raise ValueError(f'invalid {dataset.mode} subset size: {size}')
    base_count, extra = divmod(size, len(CLASSES))
    generator = np.random.default_rng(stable_seed(seed, dataset.mode, 'train-003-subset'))
    labels = dataset.labels[dataset.indices]
    selected = [generator.choice(dataset.indices[labels == label], base_count + (label < extra), replace=False)
                for label in range(len(CLASSES))]
    result = copy(dataset)
    result.indices = np.sort(np.concatenate(selected))
    return result


def _debug_run(model, loader, training: dict, device: torch.device, max_steps: int) -> dict:
    optimizer = _make_optimizer(model, training)
    criterion = nn.CrossEntropyLoss()

    def measure(step):
        report = validate(model, loader, criterion, device)
        return {'optimizer_steps': step, 'loss': report['loss'], 'accuracy': report['accuracy']}

    history = [measure(0)]
    for steps in range(1, max_steps + 1):
        # The whole subset is one batch, so each call is exactly one optimizer update.
        if train_one_epoch(model, loader, optimizer, criterion, device)['optimizer_steps'] != 1:
            raise RuntimeError('debug training must make exactly one update per pass')
        if steps % 5 == 0 or steps == max_steps:
            history.append(measure(steps))
            if history[-1]['accuracy'] >= 0.95 and history[-1]['loss'] < history[0]['loss']:
                break
    final = history[-1]
    passed = final['accuracy'] >= 0.95 and final['loss'] < history[0]['loss']
    return {'history': history, 'optimizer_steps': steps, 'status': 'complete' if passed else 'failed',
            'target_reached': passed, 'train_accuracy': final['accuracy'], 'train_loss': final['loss']}


def _pilot_run(model, train_loader, val_loader, config: dict, device: torch.device) -> dict:
    training = {**config['training'], 'epochs': 1}
    counts = (len(train_loader.dataset), len(val_loader.dataset), len(train_loader))
    result = _run_epochs(model, train_loader, val_loader, training, device,
                         expected_counts=counts, record_phase_times=True)
    row = result['history'][0]
    full_train_batches = ceil(config['dataset']['train_size'] / config['training']['batch_size'])
    full_val_batches = ceil(config['dataset']['val_size'] / config['evaluation']['batch_size'])
    epoch_seconds = (row['train_seconds'] * full_train_batches / len(train_loader)
                     + row['validation_seconds'] * full_val_batches / len(val_loader))
    order = [int(train_loader.dataset.indices[position]) for position in train_loader.sampler]
    return {**result, 'train_order_sha256': json_sha256(order),
            'estimated_core_epoch_seconds': epoch_seconds,
            'estimated_core_run_seconds': epoch_seconds * config['training']['epochs']}


def _peak_memory(device: torch.device) -> dict:
    try:
        import resource
    except ImportError:
        host_bytes = None
    else:
        unit = 1 if sys.platform == 'darwin' else 1024
        host_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
    memory = {'peak_process_rss_bytes': host_bytes}
    if device.type == 'cuda':
        memory.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return memory


def run_training_check(config: dict, seed: int, *, profile: str, run_dir: Path,
                       max_steps: int | None = None) -> dict:
    """Run a separate debug or pilot job and write a local diagnostic summary."""
    validate_config(config)
    if type(seed) is not int or seed not in config['seeds']:
        raise ValueError(f"seed must be one of {config['seeds']}")
    run_dir = Path(run_dir)
    if profile not in ('debug', 'pilot') or not run_dir.name.startswith(f'{profile}-'):
        raise ValueError('training checks require a debug-* or pilot-* directory matching the profile')
    debug = profile == 'debug'
    if debug:
        if config['augmentation']['enabled']:
            raise ValueError('use the Clean config for the 64-image debug check')
        max_steps = _DEBUG_MAX_STEPS if max_steps is None else max_steps
        if type(max_steps) is not int or not 1 <= max_steps <= _DEBUG_MAX_STEPS:
            raise ValueError(f'debug max_steps must be between 1 and {_DEBUG_MAX_STEPS}')
    elif max_steps is not None:
        raise ValueError('max_steps applies to debug only')
    device = _resolve_training_device(config)
    runtime = config['runtime']
    policy = CorruptionPolicy(config['augmentation']['probabilities']) if config['augmentation']['enabled'] else None
    code_identity = capture_code_identity(Path(__file__).resolve().parents[1])
    run_dir.mkdir(parents=True)

    with _training_runtime(device, seed):
        partitions = ('train',) if debug else ('train', 'val')
        loaders = make_loaders(config, seed, 1, policy=policy, partitions=partitions)
        train_data = _subset(loaders['train'].dataset, _DEBUG_SAMPLES if debug else _PILOT_TRAIN_SAMPLES, seed)
        dropout = 0.0 if debug else config['dropout']
        if debug:
            # Clean preprocessing on training IDs disables geometry and corruption.
            train_data.mode, train_data.policy = 'val', None
            train_loader = _make_loader(train_data, _DEBUG_SAMPLES, runtime['num_workers'],
                                        device, runtime['pin_memory'])
            val_ids = []
            overrides = {'train_batch_size': _DEBUG_SAMPLES, 'max_steps': max_steps,
                         'geometry': False, 'corruption': False, 'dropout': dropout, 'scheduler': None}
        else:
            val_data = _subset(loaders['val'].dataset, _PILOT_VAL_SAMPLES, seed)
            train_loader = _make_loader(train_data, config['training']['batch_size'], runtime['num_workers'],
                                        device, runtime['pin_memory'])
            val_loader = _make_loader(val_data, config['evaluation']['batch_size'], runtime['num_workers'],
                                      device, runtime['pin_memory'])
            val_ids = val_data.indices.tolist()
            overrides = {'epochs': 1}

        model = CompactCNN(num_classes=config['num_classes'], dropout=dropout).to(device)
        initialization = _initial_state_sha256(model)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        if debug:
            result = _debug_run(model, train_loader, config['training'], device, max_steps)
        else:
            result = _pilot_run(model, train_loader, val_loader, config, device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
        memory = _peak_memory(device)
        environment = capture_environment(device, num_workers=runtime['num_workers'])

    summary = {
        'schema_version': 1, 'profile': profile, 'run_id': run_dir.name, 'seed': seed,
        'model_name': config['model_name'], 'config': config, 'config_hash': config_hash(config),
        'overrides': overrides, 'train_indices': train_data.indices.tolist(), 'val_indices': val_ids,
        'partitions': list(partitions), 'init_sha256': initialization, 'environment': environment,
        **code_identity,
        'elapsed_seconds': elapsed, 'memory': memory, **result,
    }
    write_json_atomic(run_dir / 'summary.json', summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Run a local CIFAR-10 diagnostic or training pilot.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--profile', choices=('debug', 'pilot'), required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--max-steps', type=int, help='Debug update limit (default 300).')
    parser.add_argument('--overfit-samples', type=int, choices=(_DEBUG_SAMPLES,), help='Debug subset size.')
    args = parser.parse_args(argv)
    if args.overfit_samples is not None and args.profile != 'debug':
        parser.error('--overfit-samples applies to debug only')
    try:
        config = load_config(args.config)
        if args.device:
            config['runtime']['device'] = args.device
        run_dir = args.run_dir or (Path(config['output']['root']) / f'{args.profile}-{config["model_name"]}-s{args.seed}')
        result = run_training_check(config, args.seed, profile=args.profile, run_dir=run_dir,
                                    max_steps=args.max_steps)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps({'profile': args.profile, 'status': result['status'],
                      'summary': str(run_dir / 'summary.json'), 'optimizer_steps': result['optimizer_steps']}))
    return 0 if result['status'] == 'complete' else 1
