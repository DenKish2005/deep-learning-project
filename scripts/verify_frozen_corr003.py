"""Read-only CORR-003 replay at its recorded source, without a branch/worktree.

Only the four recorded implementation files and package marker are extracted
from Git into ignored tmp. Actual inputs/exports stay in place and are verified
by the original strict verifier. This is not a general provenance framework.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


REVISION = '21ede652d2d1638ec69e4637f4707bcebf3c8224'
REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / 'outputs/figures/corruption_examples.source.json'
MANIFEST_SHA256 = 'f5f09446078fc53d26f8d0e5df4d057c62042ff0540de409d363f26ae2ae5854'


def snapshot_source(destination: Path) -> dict:
    """Extract pinned blobs, authenticate against the frozen manifest, or fail."""
    payload = MANIFEST.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MANIFEST_SHA256:
        raise ValueError('frozen CORR-003 source manifest mismatch; preserve it')
    hashes = json.loads(payload)['source']['implementation_sha256_lf']
    for name in (*hashes, 'src/__init__.py'):
        result = subprocess.run(['git', 'show', f'{REVISION}:{name}'], cwd=REPO,
                                check=True, capture_output=True)
        blob = result.stdout.replace(b'\r\n', b'\n')
        if name in hashes and hashlib.sha256(blob).hexdigest() != hashes[name]:
            raise ValueError(f'recorded CORR-003 implementation mismatch: {name}')
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    return hashes


def verify_frozen(root: Path, split_path: Path) -> dict:
    """Verify actual data, pixels and all exports under authenticated old source."""
    temporary = REPO / 'tmp'
    temporary.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='corr003-source-', dir=temporary) as directory:
        snapshot_source(Path(directory))
        code = (
            'import json,sys; from pathlib import Path; '
            'sys.path.insert(0,sys.argv[1]); '
            'from scripts.make_plots import export_audit; '
            'print(json.dumps(export_audit(Path(sys.argv[2]),Path(sys.argv[3]),'
            'Path(sys.argv[4]),verify_only=True),sort_keys=True))'
        )
        result = subprocess.run([sys.executable, '-c', code, directory,
                                 str(root.resolve()), str(split_path.resolve()),
                                 str(MANIFEST.parent)], cwd=REPO, check=True,
                                capture_output=True, text=True, timeout=180)
    return {'revision': REVISION, **json.loads(result.stdout)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/raw'))
    parser.add_argument('--split-path', type=Path, default=Path('data/splits/cifar10_split_v1.npz'))
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify_frozen(args.root, args.split_path), sort_keys=True))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
