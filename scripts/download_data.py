"""CIFAR-10 acquisition, raw verification, or offline split prepare/verify."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import torchvision

from src.data import (DataIntegrityError, SPLIT_NAME, _allowed_url, _hash,
                      create_split, ensure_cifar10, inspect_cifar10,
                      load_cifar10, load_split)


MANIFEST = 'cifar10_integrity_v1.json'


def _validate_record_metadata(record: dict) -> None:
    provenance, verifier = record['provenance'], record['verifier']
    if (set(provenance) != {'method', 'download_url', 'note'}
            or provenance['method'] not in {'preexisting_files', 'preexisting_archive', 'official_https_download'}
            or not isinstance(provenance['note'], str) or not provenance['note']):
        raise ValueError('invalid acquisition provenance')
    url = provenance['download_url']
    if provenance['method'] == 'official_https_download':
        if not isinstance(url, str) or not _allowed_url(url):
            raise ValueError('invalid recorded official download URL')
    elif url is not None:
        raise ValueError('preexisting artifacts cannot claim a download URL')
    if (set(verifier) != {'python', 'torchvision', 'code_sha256_lf'}
            or any(not isinstance(verifier[k], str) or not verifier[k] for k in ('python', 'torchvision'))
            or not isinstance(verifier['code_sha256_lf'], dict)
            or set(verifier['code_sha256_lf']) != {'src/data.py', 'scripts/download_data.py'}):
        raise ValueError('invalid verifier provenance')
    for checksum in verifier['code_sha256_lf'].values():
        if (not isinstance(checksum, str) or len(checksum) != 64
                or any(c not in '0123456789abcdef' for c in checksum)):
            raise ValueError('invalid verifier code hash')
    if datetime.fromisoformat(record['verified_utc']).utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError('verification timestamp must have UTC timezone')


def _existing_manifest(path: Path, inspection: dict) -> dict | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise DataIntegrityError(f'Non-regular integrity manifest: {path}; preserve it and use a new --root')
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
        if (not isinstance(record, dict)
                or set(record) != {'schema_version', 'verified_utc', 'inspection', 'provenance', 'verifier'}
                or type(record['schema_version']) is not int or record['schema_version'] != 1
                or record['inspection'] != inspection
                or not isinstance(record['provenance'], dict)
                or not isinstance(record['verifier'], dict)
                or not isinstance(record['verified_utc'], str)):
            raise ValueError('schema or fresh inspection mismatch')
        _validate_record_metadata(record)
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f'Conflicting/invalid integrity manifest: {path}: {exc}; '
                                 'preserve it and use a new --root. It was not overwritten.') from exc
    return record


def _write_manifest(path: Path, inspection: dict, provenance: dict) -> None:
    """Create once, exclusively. Interrupted/invalid records fail on next read.

    No replacement of an existing record; this is not the general atomic-write
    or backup infrastructure planned in REPRO-001.
    """
    code = {}
    for relative in ('src/data.py', 'scripts/download_data.py'):
        source = Path(__file__).resolve().parents[1] / relative
        code[relative] = hashlib.sha256(source.read_text(encoding='utf-8').encode()).hexdigest()
    record = {'schema_version': 1, 'verified_utc': datetime.now(timezone.utc).isoformat(),
              'inspection': inspection, 'provenance': provenance,
              'verifier': {'python': platform.python_version(),
                           'torchvision': torchvision.__version__,
                           'code_sha256_lf': code}}
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/raw'),
                        help='Raw dataset directory (default: data/raw)')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--verify-only', action='store_true',
                       help='Read-only raw integrity inspection; no split required or created')
    modes.add_argument('--prepare-split', action='store_true',
                       help='Offline: verify original train, create or reuse split only (no stats/loaders)')
    modes.add_argument('--verify-split', action='store_true',
                       help='Offline, read-only: verify original train and existing NPZ/manifest')
    parser.add_argument('--split-seed', type=int, choices=[494],
                        help='Split modes only: shared protocol seed (default: 494)')
    parser.add_argument('--split-dir', type=Path,
                        help='Split modes only: artifact directory (default: data/splits)')
    args = parser.parse_args(argv)
    split_mode = args.prepare_split or args.verify_split
    if not split_mode and (args.split_seed is not None or args.split_dir is not None):
        parser.error('--split-seed/--split-dir require --prepare-split or --verify-split')
    try:
        if split_mode:
            # Offline existing raw data only; never decode test images/targets.
            dataset = load_cifar10(args.root, train=True, download=False)
            out_dir = args.split_dir if args.split_dir is not None else Path('data/splits')
            seed = args.split_seed if args.split_seed is not None else 494
            if args.prepare_split:
                create_split(dataset.targets, seed, out_dir)
            train, val = load_split(out_dir / SPLIT_NAME, dataset.targets)
            print(json.dumps({'status': 'verified', 'mode': 'prepare-split' if args.prepare_split else 'verify-split',
                              'seed': seed, 'namespace': 'cifar10-python/original-train',
                              'train': len(train), 'validation': len(val),
                              'split_sha256': _hash(out_dir / SPLIT_NAME, 'sha256')}, sort_keys=True))
            return 0
        provenance = None
        if not args.verify_only:
            provenance = ensure_cifar10(args.root)
        inspection = inspect_cifar10(args.root)
        manifest = args.root / MANIFEST
        record = _existing_manifest(manifest, inspection)
        if record is None and not args.verify_only:
            _write_manifest(manifest, inspection, provenance)
        print(json.dumps({'status': 'verified', 'mode': 'verify-only' if args.verify_only else 'acquire',
                          'train': inspection['partitions']['train']['count'],
                          'test': inspection['partitions']['test']['count'],
                          'archive_verified': inspection['archive'] is not None,
                          'manifest': 'unchanged' if record is not None else
                                      ('not_created' if args.verify_only else 'created')}, sort_keys=True))
        return 0
    except (DataIntegrityError, OSError, ValueError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
