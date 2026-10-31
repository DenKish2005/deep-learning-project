"""Actual persisted-split checks, explicit RUN_CIFAR_SPLIT=1; no training/download.

An opted-in absent or corrupt input/artifact is a failure, never a skip.
"""

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import pytest

from scripts import download_data
from src import data
from src.utils import load_config, validate_controlled_configs


pytestmark = pytest.mark.skipif(os.environ.get('RUN_CIFAR_SPLIT') != '1',
                                reason='Actual persisted split requires RUN_CIFAR_SPLIT=1')
ROOT = Path(os.environ.get('CIFAR10_ROOT', 'data/raw'))
SPLITS = Path(os.environ.get('CIFAR10_SPLIT_DIR', 'data/splits'))
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline_train_only(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('Actual split verification must remain offline')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)
    original = data._read_partition
    def train_only(root, train):
        assert train is True, 'Split must never decode test images or targets'
        return original(root, train)
    monkeypatch.setattr(data, '_read_partition', train_only)


def snapshot(*roots):
    return {(i, p.relative_to(root).as_posix()):
            (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for i, root in enumerate(roots) for p in root.rglob('*') if p.is_file()}


def test_actual_persisted_split_contract_and_config_identity():
    dataset = data.load_cifar10(ROOT, train=True, download=False)
    assert dataset.data.shape == (50000, 32, 32, 3)
    labels = np.asarray(dataset.targets, dtype='<i8')
    assert np.bincount(labels).tolist() == [5000] * 10
    train, val = data.load_split(SPLITS / data.SPLIT_NAME, labels)
    assert train.shape == (45000,) and val.shape == (5000,)
    assert train.dtype == val.dtype == np.dtype('<i8')
    assert np.bincount(labels[train]).tolist() == [4500] * 10
    assert np.bincount(labels[val]).tolist() == [500] * 10
    assert not np.intersect1d(train, val).size
    np.testing.assert_array_equal(np.sort(np.concatenate((train, val))), np.arange(50000))
    oracle = np.random.default_rng(494)
    expected_val = np.sort(np.concatenate([
        oracle.permutation(np.flatnonzero(labels == k))[:500] for k in range(10)]))
    np.testing.assert_array_equal(val, expected_val)
    record = json.loads((SPLITS / data.SPLIT_MANIFEST).read_text(encoding='utf-8'))
    assert record['algorithm']['seed'] == 494
    assert record['original_train']['targets_sha256'] == data._array_sha256(labels)
    configs = [load_config(REPO / 'configs' / name) for name in ('clean_cnn.yaml', 'robust_cnn.yaml')]
    validate_controlled_configs(*configs)
    observed = hashlib.sha256((SPLITS / data.SPLIT_NAME).read_bytes()).hexdigest()
    for config in configs:
        assert config['dataset']['split_sha256'] == observed
        # DATA-003 resolves stats symmetrically; this split-only test does not read it.
        assert config['dataset']['normalization_sha256'] == configs[0]['dataset']['normalization_sha256']


def test_actual_api_cli_reuse_preserves_hashes_and_mtimes():
    dataset = data.load_cifar10(ROOT, train=True, download=False)
    before = snapshot(ROOT, SPLITS)
    data.create_split(dataset.targets, 494, SPLITS)
    data.load_split(SPLITS / data.SPLIT_NAME, dataset.targets)
    for mode in ('--prepare-split', '--verify-split'):
        assert download_data.main(['--root', str(ROOT), '--split-dir', str(SPLITS), mode]) == 0
    # Exercise the real module entry point in another process too.
    result = subprocess.run([sys.executable, '-m', 'scripts.download_data', '--root', str(ROOT),
                             '--split-dir', str(SPLITS), '--verify-split'],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['status'] == 'verified'
    assert snapshot(ROOT, SPLITS) == before
