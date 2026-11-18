"""Bounded CORR-003 train-only pixel audit; no metrics or training interfaces."""

from pathlib import Path

import numpy as np
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
