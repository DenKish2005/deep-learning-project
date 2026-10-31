"""Offline train-only normalization preparation or read-only verification.

Requires existing verified raw data and frozen DATA-002 split. Never downloads,
creates a split, decodes test images or announces training readiness.
"""

import argparse
import json
from pathlib import Path
import sys

from src.data import (DataIntegrityError, _hash, load_cifar10,
                      load_normalization, prepare_normalization)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/raw'),
                        help='Existing raw CIFAR directory (default: data/raw)')
    parser.add_argument('--split-path', type=Path, default=Path('data/splits/cifar10_split_v1.npz'),
                        help='Frozen NPZ with sibling manifest (default: data/splits/cifar10_split_v1.npz)')
    parser.add_argument('--output', type=Path, default=Path('data/splits/normalization.json'),
                        help='Stats artifact to create/reuse/verify (default: data/splits/normalization.json)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Read-only: require existing stats; never create or repair artifacts')
    args = parser.parse_args(argv)
    try:
        base = load_cifar10(args.root, train=True, download=False)
        operation = load_normalization if args.verify_only else prepare_normalization
        record = operation(base.data, base.targets, args.split_path, args.output)
        stats = record['statistics']
        print(json.dumps({'status': 'stats_verified',
                          'mode': 'verify-only' if args.verify_only else 'prepare',
                          'samples': stats['samples'], 'mean': stats['mean'], 'std': stats['std'],
                          'normalization_file_sha256': _hash(args.output, 'sha256'),
                          'statistics_semantic_sha256': record['statistics_sha256']}, sort_keys=True))
        return 0
    except (DataIntegrityError, OSError, ValueError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
