"""Explicit local-data integrity checks; never download, never train/tune.

Run separately with RUN_CIFAR_INTEGRATION=1. Opted-in missing data is a failure.
"""

import hashlib
import os
from pathlib import Path
import socket

import pytest

from scripts.download_data import main
from src.data import ARCHIVE, ARCHIVE_MD5, CLASSES, inspect_cifar10, load_cifar10


pytestmark = pytest.mark.skipif(os.environ.get('RUN_CIFAR_INTEGRATION') != '1',
                                reason='Actual CIFAR integrity requires RUN_CIFAR_INTEGRATION=1')
ROOT = Path(os.environ.get('CIFAR10_ROOT', 'data/raw'))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('Actual local-data verification must remain offline')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)


@pytest.mark.parametrize('train,n', [(True, 50000), (False, 10000)])
def test_actual_original_partition(train, n):
    dataset = load_cifar10(ROOT, train, download=False)
    assert len(dataset) == n
    assert dataset.data.shape == (n, 32, 32, 3)
    assert dataset.classes == list(CLASSES)
    assert all(dataset.targets.count(i) == n // 10 for i in range(10))


def _snapshot():
    result = {}
    for path in ROOT.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            result[str(path.relative_to(ROOT))] = (checksum, path.stat().st_mtime_ns)
    return result


def test_actual_integrity_and_cli_idempotence():
    inspection = inspect_cifar10(ROOT)
    assert inspection['archive']['md5'] == ARCHIVE_MD5
    assert (ROOT / ARCHIVE).is_file()
    assert len(inspection['files']) == 7
    assert inspection['partitions']['train']['class_counts'] == [5000] * 10
    assert inspection['partitions']['test']['class_counts'] == [1000] * 10
    before = _snapshot()
    assert main(['--root', str(ROOT), '--verify-only']) == 0
    assert main(['--root', str(ROOT)]) == 0
    assert main(['--root', str(ROOT), '--verify-only']) == 0
    assert _snapshot() == before
