"""Local, verified run records for the REPRO-001 synthetic milestone."""

from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shutil

import yaml

from src.utils import config_hash, json_sha256, load_config, sha256_file, write_json_atomic


_CONFIG = 'config_resolved.yaml'
_ENVIRONMENT = 'environment.json'
_MANIFEST = 'run_manifest.json'
_RECORD_FILES = {_CONFIG, _ENVIRONMENT, _MANIFEST}
_CODE_FIELDS = {'git_sha', 'git_dirty', 'code_snapshot_sha256'}
_HASH_FIELDS = ('config_hash', 'dataset_sha256', 'split_hash', 'normalization_hash',
                'init_sha256', 'environment_id', 'code_snapshot_sha256')


def _require_sha256(value, field: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{field} must be a lowercase SHA-256 digest')


def _utc_text(value: datetime) -> str:
    if value.utcoffset() is None:
        raise ValueError('run timestamps must include a timezone')
    return value.astimezone(timezone.utc).isoformat()


def _validate_manifest(manifest: dict) -> None:
    required = {
        'schema_version', 'protocol_version', 'run_id', 'model_name', 'seed',
        'profile', 'dataset', 'started_at', 'finished_at', 'status', 'total_steps',
        'best_epoch', 'artifacts', *_HASH_FIELDS, *_CODE_FIELDS,
    }
    if not isinstance(manifest, dict) or manifest.keys() != required:
        raise ValueError('run manifest fields do not match schema v1')
    for key in ('schema_version', 'protocol_version'):
        if type(manifest[key]) is not int or manifest[key] != 1:
            raise ValueError(f'{key} must be 1')
    if manifest['profile'] != 'synthetic' or manifest['status'] != 'complete':
        raise ValueError('run record requires a complete synthetic run')
    if not isinstance(manifest['run_id'], str) or not manifest['run_id'].strip():
        raise ValueError('run_id must be nonempty')
    for key in ('seed', 'total_steps', 'best_epoch'):
        if type(manifest[key]) is not int or manifest[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'invalid run manifest {key}')
    for key in _HASH_FIELDS:
        _require_sha256(manifest[key], key)
    if (not isinstance(manifest['git_sha'], str)
            or re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', manifest['git_sha']) is None
            or type(manifest['git_dirty']) is not bool):
        raise ValueError('invalid Git identity')
    started, finished = (datetime.fromisoformat(manifest[key]) for key in ('started_at', 'finished_at'))
    if (_utc_text(started) != manifest['started_at'] or _utc_text(finished) != manifest['finished_at']
            or finished < started):
        raise ValueError('run timestamps must be ordered UTC times')
    artifacts = manifest['artifacts']
    if (not isinstance(artifacts, dict) or not {_CONFIG, _ENVIRONMENT}.issubset(artifacts)
            or _MANIFEST in artifacts):
        raise ValueError('run artifact list must include resolved config and environment')
    for name, checksum in artifacts.items():
        _require_sha256(checksum, f'artifact {name}')


def _artifact_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (relative.is_absolute() or not relative.parts or '..' in relative.parts
            or relative.as_posix() != name or '\\' in name or ':' in name):
        raise ValueError(f'invalid relative artifact path: {name}')
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f'artifact path contains a symbolic link: {name}')
    return path


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f'duplicate JSON field: {key}')
        record[key] = value
    return record


def _read_json(path: Path) -> dict:
    with path.open(encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=_unique_keys)


def _verify_artifacts(run_dir: Path, artifacts: dict) -> None:
    for name, expected in artifacts.items():
        path = _artifact_path(run_dir, name)
        if not path.is_file():
            raise ValueError(f'missing run artifact: {name}')
        if sha256_file(path) != expected:
            raise ValueError(f'run artifact checksum mismatch: {name}')
    files = {path.relative_to(run_dir).as_posix() for path in run_dir.rglob('*')
             if path.is_file() or path.is_symlink()}
    unexpected = files - artifacts.keys() - {_MANIFEST}
    if unexpected:
        raise ValueError(f'unlisted run artifacts: {sorted(unexpected)}')


def _verify_links(run_dir: Path, manifest: dict) -> None:
    config = load_config(run_dir / _CONFIG, require_core_protocol=False)
    dataset = config['dataset']
    if (manifest['config_hash'] != config_hash(config) or manifest['model_name'] != config['model_name']
            or manifest['seed'] not in config['seeds'] or manifest['dataset'] != dataset['name']
            or manifest['best_epoch'] > config['training']['epochs']
            or manifest['split_hash'] != dataset['split_sha256']
            or manifest['normalization_hash'] != dataset['normalization_sha256']):
        raise ValueError('run manifest does not match the resolved config')

    environment = _read_json(run_dir / _ENVIRONMENT)
    required = {'schema_version', 'python', 'platform', 'machine', 'packages', 'device', 'gpu',
                'num_workers', 'deterministic_algorithms', 'environment_id'}
    if not isinstance(environment, dict) or not required.issubset(environment):
        raise ValueError('run environment is missing required metadata')
    environment_id = environment.pop('environment_id')
    configured_device = config['runtime']['device']
    if (environment_id != json_sha256(environment) or environment_id != manifest['environment_id']
            or environment['num_workers'] != config['runtime']['num_workers']
            or (configured_device != 'auto' and environment['device'].split(':')[0] != configured_device)):
        raise ValueError('run environment identity does not match the manifest or config')


@contextmanager
def _new_directory(path: Path):
    path.mkdir(parents=True)
    try:
        yield
    except Exception:
        shutil.rmtree(path)
        raise


def write_run_record(run_dir: Path, *, config: dict, seed: int, dataset_sha256: str,
                     init_sha256: str, code_identity: dict, environment: dict,
                     started_at: datetime, finished_at: datetime, training_result: dict,
                     artifacts: Mapping[str, Path]) -> dict:
    """Copy a completed synthetic run's artifacts and publish its manifest last."""
    run_dir = Path(run_dir)
    if code_identity.keys() != _CODE_FIELDS:
        raise ValueError('code identity fields do not match schema v1')
    identity = config_hash(config)
    dataset = config['dataset']
    with _new_directory(run_dir):
        (run_dir / _CONFIG).write_text(yaml.safe_dump(config, sort_keys=True), encoding='utf-8', newline='\n')
        write_json_atomic(run_dir / _ENVIRONMENT, environment)
        checksums = {name: sha256_file(run_dir / name) for name in (_CONFIG, _ENVIRONMENT)}
        for name, source in artifacts.items():
            if name in _RECORD_FILES:
                raise ValueError(f'reserved run artifact name: {name}')
            destination = _artifact_path(run_dir, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = Path(source)
            if not source.is_file():
                raise ValueError(f'artifact source is missing or not a file: {source}')
            shutil.copyfile(source, destination)
            checksums[name] = sha256_file(destination)
        manifest = {
            'schema_version': 1, 'protocol_version': 1, 'profile': 'synthetic',
            'run_id': run_dir.name, 'model_name': config['model_name'], 'seed': seed,
            'dataset': dataset['name'], 'dataset_sha256': dataset_sha256,
            'split_hash': dataset['split_sha256'], 'normalization_hash': dataset['normalization_sha256'],
            'config_hash': identity, 'init_sha256': init_sha256, **code_identity,
            'environment_id': environment.get('environment_id'), 'started_at': _utc_text(started_at),
            'finished_at': _utc_text(finished_at), 'status': training_result['status'],
            'total_steps': training_result['optimizer_steps'], 'best_epoch': training_result['best_epoch'],
            'artifacts': checksums,
        }
        _validate_manifest(manifest)
        _verify_artifacts(run_dir, checksums)
        _verify_links(run_dir, manifest)
        write_json_atomic(run_dir / _MANIFEST, manifest)
    return manifest


def verify_run_record(run_dir: Path) -> dict:
    """Read a completed run, checking its manifest links and exact artifact inventory."""
    run_dir = Path(run_dir)
    path = _artifact_path(run_dir, _MANIFEST)
    if not path.is_file():
        raise ValueError('run is missing a complete manifest')
    manifest = _read_json(path)
    _validate_manifest(manifest)
    _verify_artifacts(run_dir, manifest['artifacts'])
    _verify_links(run_dir, manifest)
    return manifest


def backup_run_record(run_dir: Path, destination: Path) -> dict:
    """Copy a verified local run to a new directory and check the copy before completion."""
    run_dir, destination = Path(run_dir), Path(destination)
    if destination.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError('backup destination must be outside the source run')
    manifest = verify_run_record(run_dir)
    with _new_directory(destination):
        for name in manifest['artifacts']:
            target = _artifact_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_artifact_path(run_dir, name), target)
        _verify_artifacts(destination, manifest['artifacts'])
        _verify_links(destination, manifest)
        write_json_atomic(destination / _MANIFEST, manifest)
    return manifest
