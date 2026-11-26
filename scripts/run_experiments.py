"""Plan and run the six core CIFAR-10 trainings sequentially."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import signal
from time import sleep
import traceback

import torch

from src.checkpoints import _atomic_output, checkpoint_directory
from src.repro import finalize_core_run, verify_run_record
from src.train import _resolve_training_device, _training_runtime, train_run
from src.utils import (capture_code_identity, capture_environment, config_hash, json_sha256,
                       load_config, validate_controlled_configs)


_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunSpec:
    directory: Path
    config: dict
    seed: int
    config_hash: str

    def as_record(self) -> dict:
        return {'run_id': self.directory.name, 'model_name': self.config['model_name'],
                'seed': self.seed, 'config_hash': self.config_hash,
                'run_dir': str(self.directory), 'epochs': self.config['training']['epochs']}


def build_plan(configs, seeds, *, device='auto', runs_root=None) -> list[RunSpec]:
    clean, robust = [load_config(path) for path in configs]
    if device != 'auto':
        for config in (clean, robust):
            config['runtime']['device'] = device
    validate_controlled_configs(clean, robust)
    if len(seeds) != len(clean['seeds']) or set(seeds) != set(clean['seeds']):
        raise ValueError(f"core plan requires each configured seed exactly once: {clean['seeds']}")
    root = Path(runs_root or clean['output']['root']).resolve()
    plan = []
    for config in (clean, robust):
        fingerprint = config_hash(config)
        for seed in seeds:
            run_id = f'core-v1-{config["model_name"]}-s{seed}-{fingerprint[:8]}'
            plan.append(RunSpec(root / run_id, config, seed, fingerprint))
    return plan


def _completed_run(spec, code_identity, environment_id):
    manifest = verify_run_record(spec.directory)
    expected = {'profile': 'core', 'run_id': spec.directory.name,
                'model_name': spec.config['model_name'], 'seed': spec.seed,
                'config_hash': spec.config_hash, 'environment_id': environment_id, **code_identity}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f'existing manifest does not match the core plan: {spec.directory}')
    return manifest


def _train_or_resume(spec, *, resume_incomplete, owned):
    directory = spec.directory
    if directory.is_symlink():
        raise ValueError(f'run directory is a symbolic link: {directory}')
    if directory.exists():
        if not resume_incomplete:
            raise FileExistsError(f'pass --resume-incomplete to resume: {directory}')
        last = directory / 'last.pt'
        if last.is_file() and not last.is_symlink():
            train_run(spec.config, spec.seed, resume=last)
            return
        if not owned or not directory.is_dir() or any(directory.iterdir()):
            raise ValueError(f'run directory has no usable last.pt: {directory}')
        directory.rmdir()  # The launcher's own empty directory from a pre-epoch interruption.
    train_run(spec.config, spec.seed, run_dir=directory)


def run_plan(plan: list[RunSpec], *, resume_incomplete: bool) -> int:
    device = _resolve_training_device(plan[0].config)
    if device.type == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    code_identity = capture_code_identity(_ROOT)
    with _training_runtime(device, plan[0].seed):
        environment = capture_environment(device, num_workers=plan[0].config['runtime']['num_workers'])

    root = plan[0].directory.parent
    run_ids = [item.directory.name for item in plan]
    state_dir = root / ('launcher-' + json_sha256(sorted(run_ids))[:12])
    state_dir.mkdir(parents=True, exist_ok=True)
    status_path = state_dir / 'status.json'
    print(f'Launcher status file: {status_path}', flush=True)

    with checkpoint_directory(state_dir, resume=True):
        previous = {}
        if status_path.is_file():
            previous = {row['run_id']: row for row in json.loads(status_path.read_text())['runs']}
        entries = []
        for spec in plan:
            saved = previous.get(spec.directory.name, {})
            entries.append({**spec.as_record(), 'status': 'queued', 'started_at': saved.get('started_at'),
                            'directory_owned': saved.get('directory_owned', False)})
        status = {'schema_version': 1, 'status': 'running', 'device': environment['device'], 'runs': entries,
                  **code_identity}

        def publish():
            with _atomic_output(status_path, 'w') as stream:
                json.dump(status, stream, indent=2, allow_nan=False)
                stream.write('\n')

        publish()
        try:
            for spec, entry in zip(plan, entries, strict=True):
                owned = entry['started_at'] is not None and entry['directory_owned']
                if entry['started_at'] is None:
                    entry['started_at'] = datetime.now(timezone.utc).isoformat()
                entry['status'] = 'running'
                try:
                    for attempt in range(1, 4):
                        print(f"{entry['run_id']} attempt {attempt}/3", flush=True)
                        entry['attempt'] = attempt
                        entry['directory_owned'] = owned or not spec.directory.exists()
                        publish()
                        try:
                            if not (spec.directory / 'run_manifest.json').exists():
                                if capture_code_identity(_ROOT) != code_identity:
                                    raise ValueError('the checkout changed after the core plan started')
                                _train_or_resume(spec, resume_incomplete=resume_incomplete or attempt > 1, owned=owned)
                                with checkpoint_directory(spec.directory, resume=True):
                                    finalize_core_run(spec.directory, environment,
                                                      datetime.fromisoformat(entry['started_at']))
                            manifest = _completed_run(spec, code_identity, environment['environment_id'])
                            break
                        except (OSError, torch.OutOfMemoryError) as exc:
                            if isinstance(exc, FileExistsError) or attempt == 3:
                                raise
                            print(f'Retry after {type(exc).__name__}: {exc}', flush=True)
                        owned = entry['directory_owned']
                        gc.collect()
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        sleep(10)
                except Exception as exc:
                    entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    traceback.print_exc()
                else:
                    entry.update(status='complete', best_epoch=manifest['best_epoch'],
                                 optimizer_steps=manifest['total_steps'], finished_at=manifest['finished_at'])
                    print(f"{entry['run_id']} complete with {entry['optimizer_steps']} updates", flush=True)
                publish()
        except KeyboardInterrupt:
            for entry in entries:
                if entry['status'] == 'running':
                    entry['status'] = 'interrupted'
            status['status'] = 'interrupted'
            publish()
            raise
        status['status'] = 'complete' if all(row['status'] == 'complete' for row in entries) else 'failed'
        publish()
    return 0 if status['status'] == 'complete' else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', type=Path, nargs=2,
                        default=(Path('configs/clean_cnn.yaml'), Path('configs/robust_cnn.yaml')))
    parser.add_argument('--seeds', type=int, nargs=3, default=(42, 123, 2026))
    parser.add_argument('--profile', choices=('core',), default='core')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--runs-root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume-incomplete', action='store_true')
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.configs, args.seeds, device=args.device, runs_root=args.runs_root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps([spec.as_record() for spec in plan], indent=2))
        return 0

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        return run_plan(plan, resume_incomplete=args.resume_incomplete)
    except KeyboardInterrupt:
        print('Launcher interrupted. Restart with --resume-incomplete.', flush=True)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
