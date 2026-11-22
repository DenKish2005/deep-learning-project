"""Versioned, local CompactCNN checkpoints and epoch-history recovery."""

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import random
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.utils import config_hash


@contextmanager
def _atomic_output(path: Path, mode: str):
    with TemporaryDirectory(prefix='.checkpoint-', dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        options = {'encoding': 'utf-8', 'newline': ''} if mode == 'w' else {}
        with staged.open(mode, **options) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
        if os.name == 'posix':
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)


def save_checkpoint_atomic(path: Path, state: dict) -> None:
    """Replace a checkpoint only after its new contents have been serialized and flushed."""
    with _atomic_output(Path(path), 'wb') as stream:
        torch.save(state, stream)


def load_checkpoint(path: Path, device: str | torch.device = 'cpu', *,
                    identity: dict | None = None, kind: str | None = None) -> dict:
    """Load a trusted experiment checkpoint with explicit device and identity checks."""
    path = Path(path)
    try:
        with path.open('rb') as stream:
            state = torch.load(stream, map_location=device, weights_only=True)
    except (EOFError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise ValueError(f'invalid checkpoint: {path}') from exc
    if (not isinstance(state, dict) or type(state.get('schema_version')) is not int
            or state['schema_version'] != 1 or state.get('kind') not in ('best', 'last')
            or (kind is not None and kind != state['kind'])):
        raise ValueError(f'unsupported checkpoint version or kind: {path}')
    saved_identity = state.get('identity')
    if not isinstance(saved_identity, dict) or (identity is not None and saved_identity != identity):
        differing = [] if not isinstance(saved_identity, dict) or identity is None else sorted(
            key for key in saved_identity.keys() | identity.keys() if saved_identity.get(key) != identity.get(key))
        raise ValueError(f'checkpoint run identity mismatch: {", ".join(differing)}')
    if (not isinstance(state.get('config'), dict) or config_hash(state['config']) != saved_identity.get('config_hash')
            or not isinstance(state.get('model'), dict) or type(state.get('epoch')) is not int
            or state['epoch'] < 1 or state['epoch'] > state['config']['training']['epochs']):
        raise ValueError('invalid checkpoint config, model, or epoch')
    if state['kind'] == 'last':
        history = state.get('history')
        best_epoch = state.get('best_epoch')
        if (not isinstance(history, list) or len(history) != state['epoch']
                or any(not isinstance(row, dict) or row.get('epoch') != epoch
                       for epoch, row in enumerate(history, 1))
                or type(best_epoch) is not int or not 1 <= best_epoch <= state['epoch']
                or history[-1].get('best_epoch') != best_epoch
                or any(not isinstance(state.get(key), dict)
                       for key in ('best_model', 'optimizer', 'scheduler', 'rng'))):
            raise ValueError('invalid resume state or checkpoint history')
    return state


def capture_rng(device: torch.device, loaders: tuple) -> dict:
    numpy = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (numpy[0], numpy[1].tolist(), *numpy[2:]),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
        'loaders': [loader.generator.get_state() if getattr(loader, 'generator', None) is not None else None
                    for loader in loaders],
    }


def restore_rng(state: dict, device: torch.device, loaders: tuple) -> None:
    numpy = state['numpy']
    random.setstate(state['python'])
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch_cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['torch_cuda'], device)
    for loader, saved in zip(loaders, state['loaders'], strict=True):
        generator = getattr(loader, 'generator', None)
        if (generator is None) != (saved is None):
            raise ValueError('checkpoint data-loader generators do not match the run')
        if generator is not None:
            generator.set_state(saved)


@contextmanager
def checkpoint_directory(run_dir: Path, *, resume: bool):
    """Reserve new runs and allow one active checkpoint writer per directory."""
    from sys import platform

    if not resume:
        run_dir.mkdir(parents=True)
    elif not run_dir.is_dir():
        raise FileNotFoundError(f'checkpoint run directory does not exist: {run_dir}')
    # Keep the lock inode in place so concurrent openers always lock the same file.
    with (run_dir.parent / f'.{run_dir.name}.lock').open('a+b') as stream:
        stream.seek(0)
        try:
            if platform == 'win32':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'checkpoint directory is already in use: {run_dir}') from exc
        yield


@dataclass(frozen=True)
class CheckpointRun:
    directory: Path
    identity: dict
    config: dict

    def load_last(self) -> dict:
        return load_checkpoint(self.directory / 'last.pt', identity=self.identity, kind='last')

    def save_epoch(self, state: dict, *, improved: bool) -> None:
        last = {'schema_version': 1, 'kind': 'last', 'identity': self.identity,
                'config': self.config, **state}
        save_checkpoint_atomic(self.directory / 'last.pt', last)
        self.sync_outputs(last, rewrite_best=improved)

    def sync_outputs(self, last: dict, *, rewrite_best: bool = True) -> None:
        """Materialize outputs from the authoritative completed epoch in last.pt."""
        if rewrite_best:
            best = {'schema_version': 1, 'kind': 'best', 'identity': self.identity,
                    'config': self.config, 'epoch': last['best_epoch'], 'model': last['best_model'],
                    'val_accuracy': last['best_val_accuracy'], 'val_loss': last['best_val_loss']}
            save_checkpoint_atomic(self.directory / 'best.pt', best)
        metadata = {
            'schema_version': 1, 'experiment_id': self.identity['run_id'],
            'model_name': self.identity['model_name'], 'seed': self.identity['seed'],
            'checkpoint_path': 'last.pt', 'config_hash': self.identity['config_hash'],
            'git_sha': self.identity['git_sha'], 'device': self.identity['device'],
            'environment_id': self.identity['environment_id'],
        }
        rows = [{**metadata, **row} for row in last['history']]
        with _atomic_output(self.directory / 'history.csv', 'w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
