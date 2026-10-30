"""Offline raw-data tests use tiny synthetic batches, never actual CIFAR data."""

import hashlib
import io
import pickle
import socket
import tarfile

import numpy as np
import pytest
from torchvision.datasets import CIFAR10

from src import data


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('Network is forbidden in default data tests')
    monkeypatch.setattr(socket, 'create_connection', blocked)
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(CIFAR10, 'download', blocked)


@pytest.fixture
def batches(monkeypatch):
    """Replace only size/checksum references with an explicit synthetic oracle."""
    contents = {}
    for i in range(5):
        labels = [j % 10 for j in range(i * 4, (i + 1) * 4)]
        contents[f'data_batch_{i + 1}'] = pickle.dumps({
            'data': np.full((4, 3072), i, dtype=np.uint8), 'labels': labels})
    contents['test_batch'] = pickle.dumps({
        'data': np.full((10, 3072), 101, dtype=np.uint8), 'labels': list(range(10))})
    contents['batches.meta'] = pickle.dumps({'label_names': list(data.CLASSES)})
    checksums = {name: hashlib.md5(value).hexdigest() for name, value in contents.items()}
    monkeypatch.setattr(data, 'FILE_MD5', checksums)
    monkeypatch.setattr(data, 'PARTITION_SIZE', {True: 20, False: 10})
    monkeypatch.setattr(CIFAR10, 'train_list', [[n, checksums[n]] for n in contents if n.startswith('data_')])
    monkeypatch.setattr(CIFAR10, 'test_list', [['test_batch', checksums['test_batch']]])
    monkeypatch.setattr(CIFAR10, 'meta', {'filename': 'batches.meta', 'key': 'label_names',
                                       'md5': checksums['batches.meta']})
    return contents


@pytest.fixture
def raw(tmp_path, batches):
    base = tmp_path / data.BASE_FOLDER
    base.mkdir()
    for name, contents in batches.items():
        (base / name).write_bytes(contents)
    return tmp_path


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def replace_batch(raw, monkeypatch, name, **updates):
    path = raw / data.BASE_FOLDER / name
    entry = pickle.loads(path.read_bytes())
    entry.update(updates)
    path.write_bytes(pickle.dumps(entry))
    checksum = hashlib.md5(path.read_bytes()).hexdigest()
    monkeypatch.setitem(data.FILE_MD5, name, checksum)
    for collection in (CIFAR10.train_list, CIFAR10.test_list):
        for row in collection:
            if row[0] == name:
                monkeypatch.setitem(data.FILE_MD5, name, checksum)
                row[1] = checksum
    if name == 'batches.meta':
        monkeypatch.setitem(CIFAR10.meta, 'md5', checksum)


def archive_bytes(batches):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as bundle:
        for name, value in batches.items():
            member = tarfile.TarInfo(f'{data.BASE_FOLDER}/{name}')
            member.size = len(value)
            bundle.addfile(member, io.BytesIO(value))
    return output.getvalue()


def set_archive_reference(monkeypatch, value):
    checksum = hashlib.md5(value).hexdigest()
    monkeypatch.setattr(data, 'ARCHIVE_MD5', checksum)
    monkeypatch.setattr(CIFAR10, 'tgz_md5', checksum)


@pytest.mark.parametrize('train', [True, False])
def test_offline_partition_contract(raw, train):
    before = snapshot(raw)
    dataset = data.load_cifar10(raw, train)
    image, label = dataset[0]
    assert len(dataset) == (20 if train else 10)
    assert dataset.data.shape == (len(dataset), 32, 32, 3)
    assert dataset.data.dtype == np.uint8
    assert image.mode == 'RGB' and image.size == (32, 32)
    assert type(label) is int and label == 0
    assert dataset.classes == list(data.CLASSES)
    assert dataset.class_to_idx == dict(zip(data.CLASSES, range(10)))
    assert np.bincount(dataset.targets).tolist() == [len(dataset) // 10] * 10
    assert snapshot(raw) == before


def test_missing_root_is_read_only(tmp_path):
    root = tmp_path / 'absent'
    with pytest.raises(data.DataIntegrityError, match='scripts.download_data'):
        data.load_cifar10(root, True)
    assert not root.exists()


@pytest.mark.parametrize('name', list(data.FILE_MD5))
@pytest.mark.parametrize('damage', ['missing', 'corrupt'])
def test_missing_or_corrupt_files_preserved(raw, name, damage):
    path = raw / data.BASE_FOLDER / name
    if damage == 'missing':
        path.unlink()
    else:
        path.write_bytes(b'corrupt')
    before = snapshot(raw)
    for download in (False, True):
        with pytest.raises(data.DataIntegrityError, match=name):
            data.load_cifar10(raw, True, download=download)
    assert snapshot(raw) == before


def test_corrupt_archive_not_ignored_or_overwritten(raw):
    (raw / data.ARCHIVE).write_bytes(b'damaged archive')
    before = snapshot(raw)
    with pytest.raises(data.DataIntegrityError, match='expected MD5.*observed'):
        data.load_cifar10(raw, False, download=True)
    assert snapshot(raw) == before


@pytest.mark.parametrize('updates,match', [
    ({'labels': [0] * 10}, 'per-class counts'),
    ({'labels': list(range(9)) + [10]}, 'labels'),
    ({'labels': [float(i) for i in range(10)]}, 'labels'),
    ({'labels': list(range(9))}, 'labels'),
    ({'data': np.zeros((9, 3072), dtype=np.uint8)}, 'image contract'),
    ({'data': np.zeros((10, 3072), dtype=np.float32)}, 'image contract'),
])
def test_semantic_contract_after_checksum(raw, monkeypatch, updates, match):
    replace_batch(raw, monkeypatch, 'test_batch', **updates)
    with pytest.raises(data.DataIntegrityError, match=match):
        data.load_cifar10(raw, False)


def test_class_mapping_rejected(raw, monkeypatch):
    replace_batch(raw, monkeypatch, 'batches.meta', label_names=list(reversed(data.CLASSES)))
    with pytest.raises(data.DataIntegrityError, match='class mapping'):
        data.load_cifar10(raw, True)


def test_train_does_not_inspect_test_distribution(raw, monkeypatch):
    replace_batch(raw, monkeypatch, 'test_batch', labels=[0] * 10)
    assert len(data.load_cifar10(raw, True)) == 20
    with pytest.raises(data.DataIntegrityError, match='per-class counts'):
        data.load_cifar10(raw, False)


def test_torchvision_reference_drift_rejected(raw, monkeypatch):
    monkeypatch.setattr(CIFAR10, 'tgz_md5', '0' * 32)
    with pytest.raises(data.DataIntegrityError, match='references differ'):
        data.load_cifar10(raw, True)


def test_explicit_download_and_repeat(tmp_path, batches, monkeypatch):
    value = archive_bytes(batches)
    set_archive_reference(monkeypatch, value)
    calls = []
    def fake_download(destination):
        calls.append(destination)
        destination.write_bytes(value)
        return data.SOURCE_URL
    monkeypatch.setattr(data, '_download_archive', fake_download)
    assert len(data.load_cifar10(tmp_path, True, download=True)) == 20
    before = snapshot(tmp_path)
    assert len(data.load_cifar10(tmp_path, False, download=True)) == 10
    assert len(calls) == 1
    assert snapshot(tmp_path) == before


def test_existing_archive_no_network(tmp_path, batches, monkeypatch):
    value = archive_bytes(batches)
    set_archive_reference(monkeypatch, value)
    (tmp_path / data.ARCHIVE).write_bytes(value)
    result = data.ensure_cifar10(tmp_path)
    assert result['method'] == 'preexisting_archive'
    assert len(data.load_cifar10(tmp_path, False)) == 10


def test_failed_download_retains_evidence(tmp_path, batches, monkeypatch):
    def fail(destination):
        destination.write_bytes(b'partial response')
        raise OSError('connection interrupted')
    monkeypatch.setattr(data, '_download_archive', fail)
    with pytest.raises(data.DataIntegrityError, match='diagnostic staging retained.*connection interrupted'):
        data.ensure_cifar10(tmp_path)
    assert not (tmp_path / data.BASE_FOLDER).exists()
    assert not (tmp_path / data.ARCHIVE).exists()
    assert list(tmp_path.glob('.cifar-acquire-*/*'))[0].read_bytes() == b'partial response'


def test_bad_download_checksum_not_published(tmp_path, batches, monkeypatch):
    monkeypatch.setattr(data, '_download_archive', lambda path: path.write_bytes(b'wrong'))
    with pytest.raises(data.DataIntegrityError, match='checksum mismatch'):
        data.ensure_cifar10(tmp_path)
    assert not (tmp_path / data.BASE_FOLDER).exists()
    assert not (tmp_path / data.ARCHIVE).exists()


@pytest.mark.parametrize('url,allowed', [
    (data.SOURCE_URL, True),
    ('https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz', True),
    ('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', False),
    ('https://mirror.example/cifar-10-python.tar.gz', False),
    ('https://www.cs.toronto.edu.evil.example/~kriz/cifar-10-python.tar.gz', False),
])
def test_official_https_only(url, allowed):
    assert data._allowed_url(url) is allowed


def test_redirect_to_mirror_refused():
    with pytest.raises(data.DataIntegrityError, match='redirect'):
        data._OfficialRedirects().redirect_request(None, None, 302, '', {}, 'https://mirror.example/file')


def test_bad_bool_arguments(tmp_path):
    with pytest.raises(TypeError, match='bool'):
        data.load_cifar10(tmp_path, 1)


def test_foreign_dataset_directory_preserved(tmp_path):
    base = tmp_path / data.BASE_FOLDER
    base.mkdir()
    (base / 'personal.txt').write_text('keep me')
    before = snapshot(tmp_path)
    with pytest.raises(data.DataIntegrityError, match='Missing'):
        data.ensure_cifar10(tmp_path)
    assert snapshot(tmp_path) == before


def test_archive_missing_member_not_extracted(tmp_path, batches, monkeypatch):
    del batches['batches.meta']
    value = archive_bytes(batches)
    set_archive_reference(monkeypatch, value)
    (tmp_path / data.ARCHIVE).write_bytes(value)
    with pytest.raises(data.DataIntegrityError, match='member: batches.meta'):
        data.ensure_cifar10(tmp_path)
    assert not (tmp_path / data.BASE_FOLDER).exists()
    assert (tmp_path / data.ARCHIVE).read_bytes() == value
