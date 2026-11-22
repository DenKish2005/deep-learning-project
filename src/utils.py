"""Configuration, reproducible seeds, and local experiment identity helpers."""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import subprocess
from tempfile import TemporaryDirectory

import yaml


def stable_seed(*parts) -> int:
    """v1: SHA256(prefix + compact UTF8 JSON list), low 63 bits, big endian.

    Only Python int (not bool) and str parts: no repr/hash/platform ambiguity.
    This local seed primitive does not change any global RNG state.
    """
    if any(type(part) not in (int, str) for part in parts):
        raise TypeError('stable_seed parts must be Python int or str (not bool)')
    payload = 'csci494-stable-seed-v1\n' + json.dumps(
        list(parts), ensure_ascii=False, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest(), 'big') & ((1 << 63) - 1)


def seed_worker(worker_id: int) -> None:
    """Top-level Windows-spawn worker hook; torch has already seeded its worker."""
    import random
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class ConfigError(ValueError):
    """Invalid YAML or a field violating the configuration contract."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject silent last-key-wins behavior, including nested mappings."""


def _construct_mapping(loader, node):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ConfigError(f'YAML line {key_node.start_mark.line + 1}: keys must be strings')
        if key in mapping:
            raise ConfigError(f'YAML line {key_node.start_mark.line + 1}: duplicate key {key!r}')
        mapping[key] = loader.construct_object(value_node)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _fail(path, message):
    raise ConfigError(f'{path}: {message}')


def _integer(value, path):
    if type(value) is not int:
        _fail(path, 'expected integer (booleans are not integers)')


def _number(value, path):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        _fail(path, 'expected finite number (not boolean/string/NaN/Inf)')


def _boolean(value, path):
    if type(value) is not bool:
        _fail(path, 'expected boolean')


def _range(base, lower, upper=None, *, open_lower=False, open_upper=False):
    def check(value, path):
        base(value, path)
        if (value < lower or (open_lower and value == lower)
                or (upper is not None and (value > upper or (open_upper and value == upper)))):
            left, right = ('(' if open_lower else '['), (')' if open_upper else ']')
            _fail(path, f'expected value in {left}{lower}, {upper if upper is not None else "infinity"}{right}')
    return check


_positive_int = _range(_integer, 1)
_seed = _range(_integer, 0, 2**63 - 1)
_positive = _range(_number, 0, open_lower=True)
_nonnegative = _range(_number, 0)
_probability = _range(_number, 0, 1)


def _choice(*values):
    def check(value, path):
        if not any(type(value) is type(v) and value == v for v in values):
            _fail(path, f'expected one of {values!r}')
    return check


def _sequence(item_check, length):
    def check(value, path):
        if type(value) is not list or len(value) != length:
            _fail(path, f'expected list of length {length}')
        for index, item in enumerate(value):
            item_check(item, f'{path}[{index}]')
    return check


def _path(value, path):
    if (type(value) is not str or not value.strip() or value != value.strip()
            or '\\' in value or '\x00' in value or ':' in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or '..' in PurePosixPath(value).parts or value == '.'):
        _fail(path, 'expected nonempty repository-relative path using / without ..')


def _hash(value, path):
    if value is not None and (type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None):
        _fail(path, 'expected null (unresolved) or 64 lowercase hex SHA-256 characters')


_SCHEMA = {
    'schema_version': _choice(1),
    'model_name': _choice('clean_cnn', 'robust_cnn'),
    'architecture': _choice('compact_cnn_v1'),
    'num_classes': _positive_int,
    'dropout': _range(_number, 0, 1, open_upper=True),
    'dataset': {
        'name': _choice('cifar10'), 'root': _path, 'split_path': _path,
        'normalization_path': _path, 'split_sha256': _hash,
        'normalization_sha256': _hash, 'split_seed': _seed,
        'train_size': _positive_int, 'val_size': _positive_int, 'test_size': _positive_int,
    },
    'seeds': _sequence(_seed, 3),
    'training': {
        'epochs': _positive_int, 'batch_size': _positive_int,
        'optimizer': _choice('AdamW'), 'lr': _positive, 'weight_decay': _nonnegative,
        'betas': _sequence(_range(_number, 0, 1, open_upper=True), 2),
        'eps': _positive, 'loss': _choice('cross_entropy'), 'scheduler': _choice('cosine'),
        't_max': _positive_int, 'eta_min': _nonnegative,
        'early_stopping': _boolean, 'amp': _boolean, 'gradient_clip': _choice(None),
        'selection': _choice('clean_val_accuracy_then_loss_then_earliest'),
        'crop_size': _positive_int, 'crop_padding': _range(_integer, 0),
        'crop_padding_mode': _choice('constant'), 'horizontal_flip_p': _probability,
    },
    'augmentation': {
        'enabled': _boolean, 'probabilities': _sequence(_probability, 4),
        'train_severities': _sequence(_range(_integer, 1, 5), 3),
    },
    'corruptions': {
        'noise_sigma': _sequence(_positive, 5), 'blur_sigma': _sequence(_positive, 5),
        'blur_kernel_rule': _choice('2*ceil(3*sigma)+1'), 'blur_padding': _choice('reflect'),
        'brightness_factor': _sequence(_range(_number, 0, 1, open_lower=True), 5),
    },
    'evaluation': {
        'severities': _sequence(_range(_integer, 1, 5), 5),
        'batch_size': _positive_int, 'seed': _seed,
    },
    'runtime': {
        'device': _choice('auto', 'cpu', 'cuda'), 'num_workers': _range(_integer, 0),
        'pin_memory': _choice('auto', True, False), 'deterministic': _boolean,
        'persistent_workers': _boolean,
    },
    'output': {name: _path for name in (
        'root', 'checkpoint_root', 'metrics_root', 'predictions_root', 'figures_root')},
}

# Frozen scientific settings of the proposed core protocol v1; changes require
# explicit protocol/schema/test updates, not silent hyperparameter overrides.
_PROTOCOL = {
    'num_classes': 10, 'dropout': 0.2,
    'dataset': {'split_seed': 494, 'train_size': 45000, 'val_size': 5000, 'test_size': 10000},
    'seeds': [42, 123, 2026],
    'training': {
        'epochs': 40, 'batch_size': 128, 'lr': 0.001, 'weight_decay': 0.0001,
        'betas': [0.9, 0.999], 'eps': 1e-8, 't_max': 40, 'eta_min': 0.0,
        'early_stopping': False, 'amp': False, 'crop_size': 32, 'crop_padding': 4,
        'horizontal_flip_p': 0.5,
    },
    'augmentation': {'train_severities': [1, 2, 3]},
    'corruptions': {
        'noise_sigma': [0.02, 0.05, 0.08, 0.12, 0.18],
        'blur_sigma': [0.3, 0.7, 1.1, 1.5, 2.0],
        'brightness_factor': [0.90, 0.80, 0.70, 0.55, 0.40],
    },
    'evaluation': {'severities': [1, 2, 3, 4, 5], 'batch_size': 256, 'seed': 100494},
    'runtime': {'deterministic': True},
}


def _validate_tree(cfg, schema, path='config'):
    if type(cfg) is not dict or not cfg:
        _fail(path, 'expected nonempty mapping')
    if any(type(k) is not str for k in cfg):
        _fail(path, 'keys must be strings')
    missing, unknown = schema.keys() - cfg.keys(), cfg.keys() - schema.keys()
    if missing:
        _fail(path, f'missing required keys: {", ".join(sorted(missing))}')
    if unknown:
        _fail(path, f'unknown keys: {", ".join(sorted(unknown))}')
    for key, rule in schema.items():
        child_path = f'{path}.{key}'
        if isinstance(rule, dict):
            _validate_tree(cfg[key], rule, child_path)
        else:
            rule(cfg[key], child_path)


def _check_protocol(cfg, expected, path='config'):
    for key, value in expected.items():
        child_path = f'{path}.{key}'
        if isinstance(value, dict):
            _check_protocol(cfg[key], value, child_path)
        elif cfg[key] != value:
            _fail(child_path, f'protocol v1 requires {value!r}; got {cfg[key]!r}')


def validate_config(cfg: dict) -> None:
    """Validate complete preparation config without resolving artifacts or hardware."""
    _validate_tree(cfg, _SCHEMA)
    policy = cfg['augmentation']
    if not math.isclose(math.fsum(policy['probabilities']), 1.0, rel_tol=0, abs_tol=1e-12):
        _fail('config.augmentation.probabilities', 'probabilities must sum to 1')
    expected_enabled = cfg['model_name'] == 'robust_cnn'
    if policy['enabled'] != expected_enabled:
        _fail('config.augmentation.enabled', f'must be {expected_enabled} for {cfg["model_name"]}')
    expected_probs = [0.5, 0.17, 0.17, 0.16] if expected_enabled else [1.0, 0.0, 0.0, 0.0]
    if policy['probabilities'] != expected_probs:
        _fail('config.augmentation.probabilities', f'protocol v1 requires {expected_probs!r}')
    runtime = cfg['runtime']
    if runtime['persistent_workers'] and runtime['num_workers'] == 0:
        _fail('config.runtime.persistent_workers', 'requires num_workers > 0')
    _check_protocol(cfg, _PROTOCOL)


def load_config(path: str | Path, *, require_core_protocol: bool = True) -> dict:
    """Read UTF-8 YAML; synthetic run records can request schema-only validation."""
    path = Path(path)
    try:
        with path.open(encoding='utf-8') as stream:
            cfg = yaml.load(stream, Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError, ConfigError) as exc:
        raise ConfigError(f'{path}: cannot read config: {exc}') from exc
    try:
        if require_core_protocol:
            validate_config(cfg)
        else:
            _validate_tree(cfg, _SCHEMA)
    except ConfigError as exc:
        raise ConfigError(f'{path}: {exc}') from exc
    return cfg


def validate_controlled_configs(clean: dict, robust: dict) -> None:
    """Require the named pair and equal controls; allow only treatment/output paths."""
    validate_config(clean)
    validate_config(robust)
    if clean['model_name'] != 'clean_cnn' or robust['model_name'] != 'robust_cnn':
        _fail('controlled configs', 'expected clean_cnn then robust_cnn')

    def compare(left, right, path='config'):
        for key, value in left.items():
            child_path = f'{path}.{key}'
            if child_path in ('config.model_name', 'config.augmentation.enabled',
                              'config.augmentation.probabilities', 'config.output'):
                continue
            if isinstance(value, dict):
                compare(value, right[key], child_path)
            elif value != right[key]:
                _fail(child_path, f'controlled setting differs: {value!r} != {right[key]!r}')
    compare(clean, robust)


def sha256_file(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_sha256(value) -> str:
    """Hash canonical JSON with sorted keys, compact separators, and finite numbers."""
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def config_hash(cfg: dict) -> str:
    """Fingerprint resolved settings, including synthetic overrides and data identities."""
    _validate_tree(cfg, _SCHEMA)
    dataset = cfg['dataset']
    if dataset['split_sha256'] is None or dataset['normalization_sha256'] is None:
        raise ConfigError('config identity requires resolved split and normalization hashes')
    settings = {key: value for key, value in cfg.items() if key not in ('dataset', 'output')}
    settings['dataset'] = {key: value for key, value in dataset.items()
                           if key not in ('root', 'split_path', 'normalization_path')}
    return json_sha256({'kind': 'csci494-config-v1', 'settings': settings})


def capture_code_identity(repository: Path) -> dict:
    """Record HEAD and raw contents of tracked and non-ignored checkout files."""
    git_environment = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}

    def git(root, *args):
        return subprocess.check_output(['git', '-C', os.fspath(root), *args], env=git_environment)

    root = Path(os.fsdecode(git(repository, 'rev-parse', '--show-toplevel').rstrip(b'\r\n')))
    git_sha = git(root, 'rev-parse', 'HEAD').decode().strip()
    status_args = ('status', '--porcelain=v1', '-z', '--untracked-files=all')
    status = git(root, *status_args)
    names = git(root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard')
    files = {}
    for name in sorted(set(os.fsdecode(names).split('\0')) - {''}):
        path = root / name
        if path.is_symlink():
            files[name] = {'symlink': os.readlink(path)}
        elif path.is_file():
            files[name] = sha256_file(path)
        elif not path.exists():
            files[name] = None
        else:
            raise ValueError(f'cannot fingerprint checkout path: {name}')
    if git(root, 'rev-parse', 'HEAD').decode().strip() != git_sha or git(root, *status_args) != status:
        raise RuntimeError('Git HEAD or working tree status changed during identity capture')
    return {'git_sha': git_sha, 'git_dirty': bool(status),
            'code_snapshot_sha256': json_sha256({'kind': 'csci494-checkout-v1', 'files': files})}


def capture_environment(device: str = 'cpu', *, num_workers: int = 0) -> dict:
    """Read source-independent runtime metadata without changing backend flags."""
    import torch

    if type(num_workers) is not int or num_workers < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    gpu = None
    driver_versions = None
    if device.type == 'cuda':
        index = torch.cuda.current_device() if device.index is None else device.index
        device = torch.device('cuda', index)
        properties = torch.cuda.get_device_properties(device)
        gpu = {'name': properties.name, 'vram_bytes': properties.total_memory,
               'capability': list(torch.cuda.get_device_capability(device))}
        try:
            driver_versions = sorted(set(subprocess.check_output(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                stderr=subprocess.DEVNULL, text=True, timeout=10).splitlines()))
        except (OSError, subprocess.SubprocessError):
            pass
    packages = {}
    for package in ('torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'matplotlib',
                    'tqdm', 'PyYAML', 'pytest', 'Pillow'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    snapshot = {
        'schema_version': 1,
        'python': platform.python_version(),
        'platform': platform.system(),
        'os_release': platform.release(),
        'machine': platform.machine(),
        'packages': packages,
        'device': str(device),
        'num_workers': num_workers,
        'torch_num_threads': torch.get_num_threads(),
        'torch_cuda_build': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu': gpu,
        'nvidia_driver': driver_versions,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cuda_matmul_fp32_precision': torch.backends.cuda.matmul.fp32_precision,
        'cudnn_conv_fp32_precision': torch.backends.cudnn.conv.fp32_precision,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG') if gpu else None,
    }
    return {**snapshot, 'environment_id': json_sha256(snapshot)}


def write_json_atomic(path: Path, value) -> None:
    """Publish a new JSON file atomically without replacing existing content."""
    path = Path(path)
    with TemporaryDirectory(prefix='.json-', dir=path.parent) as directory:
        staged = Path(directory) / 'record.json'
        with staged.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Validate preparation config(s); does not verify data artifacts.')
    parser.add_argument('--configs', nargs='+', required=True, type=Path)
    parser.add_argument('--controlled-pair', action='store_true', help='Require exactly clean then robust config')
    args = parser.parse_args(argv)
    if args.controlled_pair and len(args.configs) != 2:
        parser.error('--controlled-pair requires exactly two configs: clean then robust')
    try:
        configs = [load_config(path) for path in args.configs]
        if args.controlled_pair:
            validate_controlled_configs(*configs)
    except ConfigError as exc:
        parser.exit(2, f'Configuration error: {exc}\n')
    print(f'PASS: {len(configs)} config(s)' + ('; controlled settings equal.' if args.controlled_pair else '.'))
    print('Preparation schema only; dataset artifacts/hashes and training readiness are NOT verified.')


if __name__ == '__main__':
    main()
