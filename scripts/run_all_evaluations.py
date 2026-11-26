"""Evaluate the six verified core checkpoints and publish the 96-row result matrix."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from src.checkpoints import _atomic_output, checkpoint_directory
from src.evaluate import build_conditions, evaluate_checkpoint
from src.metrics import _csv_safe, aggregate_seeds, validate_matrix, write_aggregate_tables
from src.repro import verify_run_record
from src.utils import load_config, validate_controlled_configs


_MODELS = ('clean_cnn', 'robust_cnn')
_SEEDS = (42, 123, 2026)


@dataclass(frozen=True)
class EvaluationJob:
    directory: Path
    manifest: dict
    config: dict


def build_plan(checkpoints_root: Path) -> list[EvaluationJob]:
    """Check completed training manifests and paired experiment settings."""
    paths = sorted(path for path in checkpoints_root.iterdir() if path.name.startswith('core-v1-'))
    jobs = {}
    for directory in paths:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f'core run path is not a regular directory: {directory}')
        manifest = verify_run_record(directory)
        if manifest['profile'] != 'core' or manifest['run_id'] != directory.name:
            raise ValueError(f'not a completed core run: {directory}')
        key = (manifest['model_name'], manifest['seed'])
        if key in jobs:
            raise ValueError(f'multiple core checkpoints for model and seed {key}')
        config = load_config(directory / 'config_resolved.yaml')
        jobs[key] = EvaluationJob(directory, manifest, config)

    expected = [(model, seed) for model in _MODELS for seed in _SEEDS]
    if jobs.keys() != set(expected):
        raise ValueError(f'expected six completed core runs, missing {sorted(set(expected) - jobs.keys())}, '
                         f'unexpected {sorted(jobs.keys() - set(expected))}')
    first = jobs[expected[0]].manifest
    shared = ('dataset_sha256', 'split_hash', 'normalization_hash', 'git_sha',
              'code_snapshot_sha256', 'total_steps')
    for job in jobs.values():
        if any(job.manifest[key] != first[key] for key in shared):
            raise ValueError(f'core run does not match the shared experiment identities: {job.directory}')
    for seed in _SEEDS:
        clean, robust = (jobs[(model, seed)] for model in _MODELS)
        validate_controlled_configs(clean.config, robust.config)
        if clean.manifest['init_sha256'] != robust.manifest['init_sha256']:
            raise ValueError(f'Clean and Robust initial model weights differ for seed {seed}')
    if any(jobs[(model, seed)].manifest['config_hash'] != jobs[(model, _SEEDS[0])].manifest['config_hash']
           for model, seed in expected):
        raise ValueError('a model uses different settings across its three seeds')
    return [jobs[key] for key in expected]


def _publish_results(summaries: list[dict], metrics_dir: Path) -> None:
    frame = validate_matrix(pd.DataFrame(summaries))
    results_path = metrics_dir / 'results.csv'
    csv_text = _csv_safe(frame).to_csv(index=False)
    if results_path.exists():
        if results_path.is_symlink() or results_path.read_text(encoding='utf-8') != csv_text:
            raise ValueError(f'existing results table differs from the verified evaluations: {results_path}')
    else:
        with _atomic_output(results_path, 'w') as stream:
            stream.write(csv_text)

    directory = metrics_dir / 'aggregate' / 'core-v1'
    tables = aggregate_seeds(frame)
    if directory.exists():
        expected = {f'{name}.csv' for name in tables}
        if directory.is_symlink() or {path.name for path in directory.iterdir()} != expected:
            raise ValueError(f'existing aggregate directory has unexpected contents: {directory}')
        for name, table in tables.items():
            if name == 'results':
                table = table.drop(columns=['confusion_matrix', 'per_class_accuracy'], errors='ignore')
            destination = directory / f'{name}.csv'
            if (destination.is_symlink()
                    or destination.read_text(encoding='utf-8') != _csv_safe(table).to_csv(index=False)):
                raise ValueError(f'existing aggregate table differs from the verified evaluations: {destination}')
    else:
        write_aggregate_tables(tables, directory)
    print(f'Verified {len(frame)} conditions and wrote results under {metrics_dir}', flush=True)


def run_plan(jobs: list[EvaluationJob], output_dir: Path, *, device: str) -> None:
    metrics_dir = output_dir / 'metrics'
    lock = metrics_dir / 'aggregate'
    lock.mkdir(parents=True, exist_ok=True)
    with checkpoint_directory(lock, resume=True):
        summaries = []
        for job in jobs:
            run_id = job.manifest['run_id']
            print(f'Evaluating {run_id}', flush=True)
            rows = evaluate_checkpoint(job.directory / 'best.pt', job.directory / 'config_resolved.yaml',
                                       output_dir, device=device, resume=True)
            if any(row['checkpoint_sha256'] != job.manifest['artifacts']['best.pt'] for row in rows):
                raise ValueError(f'evaluation checkpoint checksum does not match its training manifest: {run_id}')
            summaries.extend(rows)
            print(f'Completed {run_id} ({len(summaries)}/96 verified conditions)', flush=True)
        _publish_results(summaries, metrics_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints-root', type=Path, default=Path('outputs/runs'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--require-complete', action='store_true',
                        help='Require the full verified core matrix (always enforced)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        jobs = build_plan(args.checkpoints_root.resolve())
        if args.dry_run:
            count = len(build_conditions())
            print(json.dumps({'evaluations': len(jobs) * count,
                              'checkpoints': [str(job.directory / 'best.pt') for job in jobs]}, indent=2))
        else:
            run_plan(jobs, args.output_dir.resolve(), device=args.device)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
