"""Bounded CORR-003 train-only pixel audit; no metrics or training interfaces."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import torch

from src import data
from src.corruptions import apply_corruption, severity_params
from src.utils import stable_seed


# Frozen before viewing actual outputs; separate from production train/test RNG.
AUDIT_SEED = 494
AUDIT_NAMESPACE = 'cifar10-python/original-train/corr003-audit'
SELECTION_RULE = 'smallest original ID per class among persisted train45k; class order 0..9'
CONDITIONS = (('clean', 0),) + tuple(
    (kind, severity)
    for kind in ('gaussian_noise', 'gaussian_blur', 'brightness')
    for severity in range(1, 6))
OUTPUT_NAMES = ('corruption_examples.png', 'corruption_examples.pdf',
                'corruption_examples.source.json')
RENDER_SETTINGS = {'figsize_inches': [22, 14.5], 'dpi': 160,
                   'interpolation': 'nearest', 'rgb_range': [0, 1],
                   'auto_contrast': False, 'font': 'DejaVu Sans'}


def select_train_ids(targets, train_indices) -> list[int]:
    """Select by labels/IDs only, never by image content; synthetic tests allowed.

    The caller must verify the actual raw partition and frozen split first.
    Input order cannot change the minimum original ID selected for each class.
    """
    labels = np.asarray(targets)
    if (labels.ndim != 1 or not labels.size or labels.dtype.kind not in 'iu'
            or np.any(labels < 0) or np.any(labels > 9)):
        raise ValueError('targets must be integer class labels 0..9')
    indices = data._validate_indices(train_indices, len(labels))
    selected = []
    for label in range(10):
        candidates = indices[labels[indices] == label]
        if not candidates.size:
            raise ValueError(f'persisted train indices contain no class {label}')
        selected.append(int(candidates.min()))
    return selected


def prepare_grid_pixels(raw_data, targets, train_indices) -> tuple[torch.Tensor, list[dict]]:
    """Return CPU float32 [10,16,3,32,32] raw [0,1] pixels and ordered rows.

    No geometry/normalization/uint8 round trip after dispatch. Recreate the local
    CPU generator per image/condition so all five noise levels share epsilon.
    No production seed helper or global RNG state is changed.
    """
    data._validate_pixels(raw_data)
    if len(targets) != len(raw_data):
        raise ValueError('targets and raw_data lengths must match')
    ids = select_train_ids(targets, train_indices)
    rows, images = [], []
    for label, original in enumerate(ids):
        x = torch.from_numpy(raw_data[original].copy()).permute(2, 0, 1).float() / 255
        seed = stable_seed(AUDIT_SEED, AUDIT_NAMESPACE, original, 'gaussian_noise', 'v1')
        conditions = []
        for kind, severity in CONDITIONS:
            generator = (torch.Generator(device='cpu').manual_seed(seed)
                         if kind == 'gaussian_noise' else None)
            conditions.append(apply_corruption(x, kind, severity, generator))
        images.append(torch.stack(conditions))
        rows.append({'original_id': original, 'label': label,
                     'class_name': data.CLASSES[label], 'noise_seed': seed})
    return torch.stack(images), rows


def load_audit_grid(root: Path, split_path: Path) -> tuple[torch.Tensor, dict]:
    """Offline verification before selection; only original train is decoded.

    All raw files (including test bytes) are hashed by the existing loader;
    official test pixels/labels are never decoded here. No stats are required.
    """
    base = data.load_cifar10(root, train=True, download=False)
    train, _ = data.load_split(split_path, base.targets)
    pixels, rows = prepare_grid_pixels(base.data, base.targets, train)
    source = {
        'dataset': 'cifar10-python/original-train',
        'selection_rule': SELECTION_RULE, 'rows': rows,
        'raw_integrity': data.verify_cifar10_files(root),
        'split': {'npz_sha256': data._hash(Path(split_path), 'sha256'),
                  'manifest_sha256': data._hash(Path(split_path).with_name('split_manifest.json'), 'sha256'),
                  'train_indices_sha256': data._array_sha256(train)},
        'conditions': [{'kind': k, 'severity': s,
                        'params': {} if k == 'clean' else severity_params(k, s)}
                       for k, s in CONDITIONS],
        'noise_key': {'seed': AUDIT_SEED, 'namespace': AUDIT_NAMESPACE,
                      'parts': [AUDIT_SEED, AUDIT_NAMESPACE, '<original_id>', 'gaussian_noise', 'v1'],
                      'reuse': 'fresh CPU generator with same key for each noise severity'},
        'preprocessing': 'uint8 RGB -> CPU float32 CHW /255 -> apply_corruption; no geometry or Normalize',
        'test_decoded': False,
    }
    return pixels, source


def _validate_grid(pixels, rows) -> None:
    if (not isinstance(pixels, torch.Tensor) or pixels.layout != torch.strided
            or pixels.device.type != 'cpu' or pixels.dtype != torch.float32
            or tuple(pixels.shape) != (10, 16, 3, 32, 32)
            or not torch.isfinite(pixels).all()
            or not ((pixels >= 0) & (pixels <= 1)).all()):
        raise ValueError('grid must be finite CPU float32 [10,16,3,32,32] RGB pixels in [0,1]')
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError('exactly ten ordered source rows required')
    ids = []
    for label, row in enumerate(rows):
        if (not isinstance(row, dict) or row.get('label') != label
                or type(row.get('label')) is not int
                or row.get('class_name') != data.CLASSES[label]
                or type(row.get('original_id')) is not int
                or not 0 <= row['original_id'] < 50000):
            raise ValueError('rows require class order 0..9, class names and original-train IDs')
        ids.append(row['original_id'])
    if len(set(ids)) != 10:
        raise ValueError('source original IDs must be unique')


def plot_corruption_grid(pixels: torch.Tensor, rows: list[dict]) -> Figure:
    """Reusable ten-row, sixteen-column figure from validated raw RGB pixels.

    Fixed RGB [0,1], nearest sampling and no cell-specific contrast/normalization.
    The returned Figure owns its canvas; callers can save it without pyplot or
    changing the process-wide matplotlib backend. Source tensor is not mutated.
    """
    _validate_grid(pixels, rows)
    with matplotlib.rc_context(rc=matplotlib.rcParamsDefault):
        figure = Figure(figsize=RENDER_SETTINGS['figsize_inches'], dpi=RENDER_SETTINGS['dpi'],
                        facecolor='white')
        FigureCanvasAgg(figure)
        axes = figure.subplots(10, 16)
        figure.subplots_adjust(left=.09, right=.992, bottom=.065, top=.865,
                               wspace=.055, hspace=.13)
        array = pixels.detach().numpy().transpose(0, 1, 3, 4, 2)
        for row in range(10):
            for col, (kind, severity) in enumerate(CONDITIONS):
                ax = axes[row, col]
                ax.imshow(array[row, col], vmin=0, vmax=1, interpolation='nearest', resample=False)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    if kind == 'clean': title = 'clean\ns0'
                    else:
                        params = severity_params(kind, severity)
                        value = params.get('sigma', params.get('factor'))
                        title = f's{severity}\n{value:g}'
                    ax.set_title(title, fontsize=10, color='#253044', pad=6)
            axes[row, 0].set_ylabel(f"{rows[row]['class_name']}\nID {rows[row]['original_id']}",
                                    rotation=0, ha='right', va='center', fontsize=11,
                                    labelpad=13, color='#253044')
        figure.suptitle('CIFAR-10 | Train-only corruption sanity grid',
                         fontsize=19, y=.982, color='#172033')
        figure.text(.5, .952, 'One fixed image per class from persisted train45k | '
                    'Raw RGB [0,1] | No geometry, Normalize or auto-contrast',
                    ha='center', fontsize=11, color='#465166')
        for first, last, text in [(0, 0, 'Reference'), (1, 5, 'Gaussian noise | sigma'),
                                  (6, 10, 'Reflect Gaussian blur | sigma'),
                                  (11, 15, 'Multiplicative darkening | factor')]:
            center = (axes[0, first].get_position().x0 + axes[0, last].get_position().x1) / 2
            figure.text(center, .916, text, ha='center', fontsize=11, color='#253044')
        figure.text(.5, .027, 'Columns: clean s0, then s1-s5 in each family. '
                    'Noise shares one keyed epsilon per image. Blur kernels: 3 / 7 / 9 / 11 / 13.\n'
                    'CORR-003 implementation audit; independent Person 5 / team visual review pending.',
                    ha='center', va='center', fontsize=10, color='#465166', linespacing=1.6)
    return figure


def _source_record(pixels, source) -> dict:
    _validate_grid(pixels, source['rows'])
    root = Path(__file__).resolve().parents[1]
    implementations = {}
    for name in ('scripts/make_plots.py', 'src/data.py', 'src/corruptions.py', 'src/utils.py'):
        # LF-normalized implementation identity survives checkout newline policy.
        implementations[name] = hashlib.sha256(
            (root / name).read_text(encoding='utf-8').replace('\r\n', '\n').encode('utf-8')).hexdigest()
    array = pixels.detach().numpy().astype('<f4', copy=False)
    return {**source, 'implementation_sha256_lf': implementations,
            'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                         'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
            'pixels': {'shape': list(array.shape), 'dtype': '<f4', 'device': 'cpu',
                       'range': [0, 1], 'observed_min': float(array.min()),
                       'observed_max': float(array.max()),
                       'sha256_c_bytes': hashlib.sha256(array.tobytes(order='C')).hexdigest()},
            'render': RENDER_SETTINGS}


def _file_record(path) -> dict:
    return {'bytes': path.stat().st_size, 'sha256': data._hash(path, 'sha256')}


def _check_exports(png, pdf) -> None:
    with Image.open(png) as image:
        if image.format != 'PNG' or image.size != (3520, 2320):
            raise ValueError('invalid PNG export dimensions or format')
        image.verify()
    content = pdf.read_bytes()
    if not content.startswith(b'%PDF-') or not content.rstrip().endswith(b'%%EOF'):
        raise ValueError('invalid PDF export boundary')


def export_audit(root: Path, split_path: Path, output_dir: Path, *, verify_only=False) -> dict:
    """Create once, or verify source/pixel/output hashes read-only on rerun.

    No overwrites/repair: incomplete or conflicting sets fail. Render all files
    in a temporary sibling before exclusive hard-link publication, manifest last.
    A publication failure may leave a partial set: preserve it and use a new empty
    output directory. This bounded audit is not general REPRO-001 infrastructure.
    """
    output_dir = Path(output_dir)
    paths = [output_dir / name for name in OUTPUT_NAMES]
    present = [p.exists() or p.is_symlink() for p in paths]
    if any(p.is_symlink() or (exists and not p.is_file()) for p, exists in zip(paths, present)):
        raise ValueError('outputs must be regular files, not symlinks/directories; preserve existing artifacts')
    if any(present) and not all(present):
        raise ValueError('incomplete output set; preserve artifacts and choose a new empty --output directory')
    if verify_only and not all(present):
        raise ValueError('verify-only requires existing PNG, PDF and source manifest')
    pixels, source = load_audit_grid(root, split_path)
    source = _source_record(pixels, source)
    if all(present):
        record = json.loads(paths[2].read_text(encoding='utf-8'), object_pairs_hook=data._unique_json_pairs)
        expected = {'schema_version': 1, 'source': source,
                    'outputs': {p.name: _file_record(p) for p in paths[:2]}}
        if record != expected:
            raise ValueError('source/implementation/pixel/output hash mismatch; preserve artifacts, no regeneration')
        _check_exports(*paths[:2])
        return {'status': 'verified', 'mode': 'verify-only' if verify_only else 'reuse',
                'outputs': record['outputs'], 'source_manifest': _file_record(paths[2])}
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.corr003-', dir=output_dir.parent) as temporary:
        staged = [Path(temporary) / name for name in OUTPUT_NAMES]
        with matplotlib.rc_context(rc=matplotlib.rcParamsDefault):
            figure = plot_corruption_grid(pixels, source['rows'])
            try:
                figure.savefig(staged[0], format='png', dpi=RENDER_SETTINGS['dpi'],
                               metadata={'Software': 'CSCI494 CORR-003'})
                figure.savefig(staged[1], format='pdf', dpi=RENDER_SETTINGS['dpi'],
                               metadata={'Creator': 'CSCI494 CORR-003', 'CreationDate': None, 'ModDate': None})
            finally:
                figure.clear()
        _check_exports(*staged[:2])
        record = {'schema_version': 1, 'source': source,
                  'outputs': {p.name: _file_record(p) for p in staged[:2]}}
        staged[2].write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
        for stage, destination in zip(staged, paths):
            os.link(stage, destination)  # Exclusive: never replace a file that appeared during rendering.
    return {'status': 'created', 'mode': 'create', 'outputs': record['outputs'],
            'source_manifest': _file_record(paths[2])}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corruption-grid', action='store_true', required=True,
                        help='Explicitly run only the fixed train-only CORR-003 audit')
    parser.add_argument('--root', type=Path, default=Path('data/raw'))
    parser.add_argument('--split-path', type=Path, default=Path('data/splits/cifar10_split_v1.npz'))
    parser.add_argument('--output', type=Path, default=Path('outputs/figures'),
                        help='Directory for the three fixed corruption_examples filenames')
    parser.add_argument('--verify-only', action='store_true', help='Read-only source and export verification')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(export_audit(args.root, args.split_path, args.output,
                                     verify_only=args.verify_only), sort_keys=True))
        return 0
    except (data.DataIntegrityError, OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
