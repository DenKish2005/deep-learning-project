"""Offline raw-data tests use tiny synthetic batches, never actual CIFAR data."""

import hashlib
import io
import json
import pickle
import random
import socket
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torchvision.datasets import CIFAR10

from src import data
from scripts import download_data


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


def test_inspection_has_no_unverified_archive_claim(raw):
    result = data.inspect_cifar10(raw)
    assert result['archive'] is None
    assert len(result['files']) == 7
    assert result['partitions']['train']['class_counts'] == [2] * 10
    assert result['partitions']['test']['class_counts'] == [1] * 10
    assert result['files']['test_batch']['sha256'] == hashlib.sha256(
        (raw / data.BASE_FOLDER / 'test_batch').read_bytes()).hexdigest()


def test_cli_verify_only_no_manifest_write(raw, capsys):
    before = snapshot(raw)
    assert download_data.main(['--root', str(raw), '--verify-only']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'verified'
    assert result['manifest'] == 'not_created'
    assert result['archive_verified'] is False
    assert snapshot(raw) == before


def test_cli_manifest_and_idempotence(raw, capsys):
    assert download_data.main(['--root', str(raw)]) == 0
    capsys.readouterr()
    record = json.loads((raw / download_data.MANIFEST).read_text())
    assert record['schema_version'] == 1
    assert record['provenance']['method'] == 'preexisting_files'
    assert record['provenance']['download_url'] is None
    assert 'unknown' in record['provenance']['note']
    assert record['inspection']['archive'] is None
    assert set(record['verifier']['code_sha256_lf']) == {'src/data.py', 'scripts/download_data.py'}
    before = snapshot(raw)
    for flags in ([], ['--verify-only']):
        assert download_data.main(['--root', str(raw), *flags]) == 0
        assert json.loads(capsys.readouterr().out)['manifest'] == 'unchanged'
    assert snapshot(raw) == before


def test_cli_download_records_actual_provenance(tmp_path, batches, monkeypatch):
    value = archive_bytes(batches)
    set_archive_reference(monkeypatch, value)
    def fake_download(path):
        path.write_bytes(value)
        return data.SOURCE_URL
    monkeypatch.setattr(data, '_download_archive', fake_download)
    assert download_data.main(['--root', str(tmp_path)]) == 0
    record = json.loads((tmp_path / download_data.MANIFEST).read_text())
    assert record['provenance']['method'] == 'official_https_download'
    assert record['provenance']['download_url'] == data.SOURCE_URL
    assert record['inspection']['archive']['md5'] == data.ARCHIVE_MD5


@pytest.mark.parametrize('flags', [[], ['--verify-only']])
def test_manifest_cannot_hide_corrupt_data(raw, flags, capsys):
    assert download_data.main(['--root', str(raw)]) == 0
    capsys.readouterr()
    (raw / data.BASE_FOLDER / 'test_batch').write_bytes(b'damaged after verification')
    before = snapshot(raw)
    assert download_data.main(['--root', str(raw), *flags]) == 1
    assert 'checksum mismatch' in json.loads(capsys.readouterr().err)['error']
    assert snapshot(raw) == before


@pytest.mark.parametrize('contents', ['{invalid', '{}', '[]', 'null'])
def test_conflicting_manifest_preserved(raw, contents, capsys):
    (raw / download_data.MANIFEST).write_text(contents)
    before = snapshot(raw)
    for flags in ([], ['--verify-only']):
        assert download_data.main(['--root', str(raw), *flags]) == 1
        assert 'manifest' in json.loads(capsys.readouterr().err)['error']
    assert snapshot(raw) == before


def test_stale_manifest_not_overwritten(raw, capsys):
    assert download_data.main(['--root', str(raw)]) == 0
    capsys.readouterr()
    path = raw / download_data.MANIFEST
    record = json.loads(path.read_text())
    record['inspection']['partitions']['train']['count'] = 123
    path.write_text(json.dumps(record))
    before = snapshot(raw)
    assert download_data.main(['--root', str(raw)]) == 1
    assert 'fresh inspection mismatch' in capsys.readouterr().err
    assert snapshot(raw) == before


@pytest.mark.parametrize('field,value', [('provenance', {}), ('verifier', {}),
                                       ('verified_utc', 'yesterday')])
def test_missing_provenance_rejected(raw, capsys, field, value):
    assert download_data.main(['--root', str(raw)]) == 0
    capsys.readouterr()
    path = raw / download_data.MANIFEST
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    before = snapshot(raw)
    assert download_data.main(['--root', str(raw), '--verify-only']) == 1
    assert snapshot(raw) == before


def test_cli_missing_verify_does_not_create_root(tmp_path, capsys):
    root = tmp_path / 'missing'
    assert download_data.main(['--root', str(root), '--verify-only']) == 1
    assert json.loads(capsys.readouterr().err)['status'] == 'failed'
    assert not root.exists()


@pytest.mark.parametrize('flags,code', [(['--help'], 0), (['--prepare-split', '--verify-only'], 2),
                                     (['--root'], 2), (['--unknown'], 2)])
def test_cli_argument_contract(flags, code):
    with pytest.raises(SystemExit) as result:
        download_data.main(flags)
    assert result.value.code == code


def test_cli_module_invocation_failure(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'scripts.download_data',
                             '--root', str(tmp_path / 'absent'), '--verify-only'],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert json.loads(result.stderr)['status'] == 'failed'
    assert not (tmp_path / 'absent').exists()


@pytest.mark.parametrize('failure', [None, 'range', 'short', 'host'])
def test_bounded_official_download(tmp_path, monkeypatch, failure):
    payload = b'a' * (1024 * 1024) + b'final fragment'
    monkeypatch.setattr(data, 'ARCHIVE_BYTES', len(payload))
    requests = []
    class Response(io.BytesIO):
        status = 206
        def __init__(self, start, end):
            contents = payload[start:end + 1]
            super().__init__(contents[:-1] if failure == 'short' else contents)
            self.headers = {'Content-Range': f'bytes {start}-{end}/{len(payload)}'}
            if failure == 'range':
                self.headers['Content-Range'] = 'wrong range'
        def geturl(self):
            return 'https://mirror.example/file' if failure == 'host' else data.SOURCE_URL
    class Opener:
        def open(self, request, timeout):
            assert timeout == 60
            assert request.full_url == data.SOURCE_URL
            header = request.get_header('Range')
            requests.append(header)
            start, end = map(int, header.removeprefix('bytes=').split('-'))
            return Response(start, end)
    monkeypatch.setattr(data, 'build_opener', lambda *args: Opener())
    path = tmp_path / 'download'
    if failure:
        with pytest.raises(data.DataIntegrityError):
            data._download_archive(path)
    else:
        assert data._download_archive(path) == data.SOURCE_URL
        assert path.read_bytes() == payload
        assert requests == [f'bytes=0-{1024 * 1024 - 1}', f'bytes={1024 * 1024}-{len(payload) - 1}']


@pytest.fixture
def split_targets():
    # Interleaved labels make original-index/label alignment observable.
    return np.tile(np.arange(10, dtype=np.int64), 5000)


def test_exact_stratified_algorithm_and_invariants(split_targets):
    labels = data._validate_split_targets(split_targets)
    train, val = data._stratified_indices(labels, 494)
    oracle = np.random.default_rng(494)
    expected_val = np.sort(np.concatenate([
        oracle.permutation(np.arange(k, 50000, 10))[:500] for k in range(10)]))
    np.testing.assert_array_equal(val, expected_val)
    np.testing.assert_array_equal(train, np.setdiff1d(np.arange(50000), expected_val))
    data._validate_split_indices(train, val, labels)
    assert train.dtype == val.dtype == np.dtype('<i8')
    assert np.bincount(labels[train]).tolist() == [4500] * 10
    assert np.bincount(labels[val]).tolist() == [500] * 10
    assert not np.intersect1d(train, val).size
    assert np.unique(np.concatenate((train, val))).size == 50000


@pytest.mark.parametrize('seed', [42, 123, 2026, 495, -1, True, 494.0, '494', np.int64(494), None])
def test_split_wrong_seed(split_targets, seed):
    with pytest.raises(data.DataIntegrityError, match='integer 494'):
        data._stratified_indices(split_targets, seed)


@pytest.mark.parametrize('kind', ['short', 'test', 'matrix', 'float', 'bool', 'object',
                                 'mixed_bool', 'negative', 'eleven', 'unbalanced', 'generator'])
def test_split_invalid_targets(split_targets, kind):
    targets = split_targets.copy()
    if kind == 'short': targets = targets[:-1]
    elif kind == 'test': targets = targets[:10000]
    elif kind == 'matrix': targets = targets.reshape(5000, 10)
    elif kind in ('float', 'bool', 'object'): targets = targets.astype(kind)
    elif kind == 'mixed_bool': targets = targets.tolist(); targets[0] = False
    elif kind == 'negative': targets[0] = -1
    elif kind == 'eleven': targets[0] = 10
    elif kind == 'unbalanced': targets[0] = 1
    elif kind == 'generator': targets = iter(targets)
    with pytest.raises(data.DataIntegrityError):
        data._validate_split_targets(targets)


@pytest.mark.parametrize('kind', ['duplicate', 'overlap', 'negative', 'range', 'missing',
                                 'dtype', 'shape', 'unsorted', 'alignment'])
def test_split_rejects_invalid_indices(split_targets, kind):
    train, val = data._stratified_indices(split_targets, 494)
    if kind == 'duplicate': train[1] = train[0]
    elif kind == 'overlap': val[0] = train[0]; val.sort()
    elif kind == 'negative': train[0] = -1
    elif kind == 'range': val[-1] = 50000
    elif kind == 'missing': train = train[:-1]
    elif kind == 'dtype': train = train.astype(np.int32)
    elif kind == 'shape': val = val.reshape(500, 10)
    elif kind == 'unsorted': val = val[::-1]
    elif kind == 'alignment':
        a = np.flatnonzero(split_targets[train] == 0)[0]
        b = np.flatnonzero(split_targets[val] == 1)[0]
        train[a], val[b] = val[b], train[a]
        train.sort(); val.sort()
    with pytest.raises(data.DataIntegrityError):
        data._validate_split_indices(train, val, split_targets)


def test_split_global_rng_isolation(split_targets):
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    labels = data._validate_split_targets(split_targets)
    data._validate_split_indices(*data._stratified_indices(labels, 494), labels)
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert numpy_state[0] == after[0] and numpy_state[2:] == after[2:]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_split_persistence_hashes_reuse_and_rng(tmp_path, split_targets):
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    record = data.create_split(split_targets.tolist(), 494, tmp_path)
    before = snapshot(tmp_path)
    train, val = data.load_split(tmp_path / data.SPLIT_NAME, split_targets)
    assert data.create_split(split_targets, 494, tmp_path) == record
    assert snapshot(tmp_path) == before
    assert set(before) == {data.SPLIT_NAME, data.SPLIT_MANIFEST}
    assert record == json.loads((tmp_path / data.SPLIT_MANIFEST).read_text())
    assert record['npz']['sha256'] == hashlib.sha256(before[data.SPLIT_NAME][0]).hexdigest()
    for name, array in [('train_indices', train), ('val_indices', val), ('targets', split_targets)]:
        header = json.dumps({'dtype': '<i8', 'shape': [len(array)]}, sort_keys=True, separators=(',', ':'))
        expected_hash = hashlib.sha256(header.encode() + b'\n' + array.astype('<i8').tobytes()).hexdigest()
        observed = (record['original_train']['targets_sha256'] if name == 'targets'
                    else record['arrays'][name]['sha256'])
        assert observed == expected_hash
    assert record['original_train']['namespace'] == 'cifar10-python/original-train'
    assert 'test_batch' not in record['dataset']['train_files_md5_reference']
    assert random.getstate() == python_state
    after = np.random.get_state()
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert after[0] == numpy_state[0] and after[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_split_fresh_process_determinism(tmp_path, split_targets):
    first = data.create_split(split_targets, 494, tmp_path / 'first')
    code = '''
import json, sys
from pathlib import Path
import numpy as np
from src.data import create_split
print(json.dumps(create_split(np.tile(np.arange(10), 5000), 494, Path(sys.argv[1]))))
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path / 'fresh')],
                            capture_output=True, text=True, timeout=60, check=True)
    second = json.loads(result.stdout)
    assert first['arrays'] == second['arrays']
    assert first['original_train'] == second['original_train']
    # Observed byte determinism in the locked NumPy/ZIP implementation only.
    assert first['npz'] == second['npz']


@pytest.mark.parametrize('field,value', [
    ('schema_version', True), ('schema_version', 1.0), ('schema_version', 2),
    ('algorithm.seed', 42), ('algorithm.version', 2), ('algorithm.class_order', list(range(9, -1, -1))),
    ('dataset.name', 'other'), ('dataset.archive_md5_reference', '0' * 32),
    ('dataset.source_url', 'https://example.com'), ('dataset.classes', list(reversed(data.CLASSES))),
    ('original_train.namespace', 'cifar10-python/original-test'),
    ('original_train.targets_sha256', '0' * 64), ('original_train.count', 10000),
    ('arrays.train_indices.sha256', '0' * 64), ('arrays.val_indices.class_counts', [5000]),
    ('npz.sha256', '0' * 64), ('npz.filename', '../other.npz'),
    ('creator.sha256_lf', '0' * 64), ('creator.numpy', 'unknown'), ('extra', 1),
])
def test_split_manifest_tampering_rejected(tmp_path, split_targets, field, value):
    record = data.create_split(split_targets, 494, tmp_path)
    current = record
    parts = field.split('.')
    for key in parts[:-1]:
        current = current[key]
    current[parts[-1]] = value
    (tmp_path / data.SPLIT_MANIFEST).write_text(json.dumps(record), encoding='utf-8')
    before = snapshot(tmp_path)
    for operation in (lambda: data.load_split(tmp_path / data.SPLIT_NAME, split_targets),
                      lambda: data.create_split(split_targets, 494, tmp_path)):
        with pytest.raises(data.DataIntegrityError, match='manifest'):
            operation()
        assert snapshot(tmp_path) == before


@pytest.mark.parametrize('contents', ['{', '[]', 'null', '{"schema_version":1,"schema_version":1}',
                                     '{"x":NaN}', '{}'])
def test_split_malformed_manifest(tmp_path, split_targets, contents):
    data.create_split(split_targets, 494, tmp_path)
    (tmp_path / data.SPLIT_MANIFEST).write_text(contents, encoding='utf-8')
    with pytest.raises(data.DataIntegrityError):
        data.load_split(tmp_path / data.SPLIT_NAME, split_targets)


@pytest.mark.parametrize('kind', ['bytes', 'duplicate', 'overlap', 'negative', 'range', 'missing',
                                 'dtype', 'object', 'unsorted', 'extra', 'same_class_swap'])
def test_split_npz_tampering_rejected_even_with_updated_file_hash(tmp_path, split_targets, kind):
    record = data.create_split(split_targets, 494, tmp_path)
    path = tmp_path / data.SPLIT_NAME
    train, val = data.load_split(path, split_targets)
    if kind == 'bytes':
        path.write_bytes(b'not a zip')
    else:
        if kind == 'duplicate': train[1] = train[0]
        elif kind == 'overlap': val[0] = train[0]; val.sort()
        elif kind == 'negative': train[0] = -1
        elif kind == 'range': val[-1] = 50000
        elif kind == 'missing': train = train[:-1]
        elif kind == 'dtype': val = val.astype(np.int32)
        elif kind == 'object': val = val.astype(object)
        elif kind == 'unsorted': train = train[::-1]
        elif kind == 'same_class_swap':
            a = np.flatnonzero(split_targets[train] == 0)[0]
            b = np.flatnonzero(split_targets[val] == 0)[0]
            train[a], val[b] = val[b], train[a]
            train.sort(); val.sort()
            # Even self-consistent semantic hashes cannot bypass protocol verification.
            record['arrays']['train_indices']['sha256'] = data._array_sha256(train)
            record['arrays']['val_indices']['sha256'] = data._array_sha256(val)
        extra = {'unexpected': np.zeros(1)} if kind == 'extra' else {}
        np.savez_compressed(path, train_indices=train, val_indices=val, **extra)
    record['npz']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    record['npz']['bytes'] = path.stat().st_size
    (tmp_path / data.SPLIT_MANIFEST).write_text(json.dumps(record), encoding='utf-8')
    before = snapshot(tmp_path)
    with pytest.raises(data.DataIntegrityError):
        data.load_split(path, split_targets)
    with pytest.raises(data.DataIntegrityError):
        data.create_split(split_targets, 494, tmp_path)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('kind', ['reorder', 'alter'])
def test_split_rejects_changed_target_order(tmp_path, split_targets, kind):
    data.create_split(split_targets, 494, tmp_path)
    labels = split_targets.copy()
    if kind == 'reorder': labels = labels[::-1]
    else: labels[0], labels[1] = labels[1], labels[0]
    before = snapshot(tmp_path)
    with pytest.raises(data.DataIntegrityError):
        data.load_split(tmp_path / data.SPLIT_NAME, labels)
    with pytest.raises(data.DataIntegrityError):
        data.create_split(labels, 494, tmp_path)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('kind', ['absent', 'npz_only', 'manifest_only', 'directory', 'lock', 'stage'])
def test_split_missing_partial_and_interrupted_states(tmp_path, split_targets, kind):
    root = tmp_path / 'split'
    if kind != 'absent':
        data.create_split(split_targets, 494, root)
        if kind == 'npz_only': (root / data.SPLIT_MANIFEST).unlink()
        elif kind == 'manifest_only': (root / data.SPLIT_NAME).unlink()
        elif kind == 'directory': (root / data.SPLIT_NAME).unlink(); (root / data.SPLIT_NAME).mkdir()
        elif kind == 'lock': (root / '.split-write.lock').write_text('interrupted')
        elif kind == 'stage': (root / '.split-stage-orphan').mkdir()
    before = snapshot(tmp_path)
    with pytest.raises(data.DataIntegrityError):
        data.load_split(root / data.SPLIT_NAME, split_targets)
    if kind != 'absent':
        with pytest.raises(data.DataIntegrityError):
            data.create_split(split_targets, 494, root)
    assert snapshot(tmp_path) == before
    assert root.exists() is (kind != 'absent')


@pytest.mark.parametrize('failure', ['npz', 'first_publish', 'second_publish', 'partial_publish', 'fsync'])
def test_split_write_failure_preserved_and_explicit_recovery(tmp_path, split_targets, monkeypatch, failure):
    broken = tmp_path / 'broken'
    original_publish = data._publish_split_file
    calls = []
    def fail(*args, **kwargs):
        raise OSError('injected write failure')
    def publish(source, destination):
        calls.append(destination.name)
        if failure == 'partial_publish':
            destination.write_bytes(b'partial')
            fail()
        if len(calls) == (1 if failure == 'first_publish' else 2):
            fail()
        original_publish(source, destination)
    with monkeypatch.context() as patch:
        if failure == 'npz': patch.setattr(np, 'savez_compressed', fail)
        elif failure == 'fsync': patch.setattr(data.os, 'fsync', fail)
        else: patch.setattr(data, '_publish_split_file', publish)
        with pytest.raises(data.DataIntegrityError, match='diagnostic artifacts retained'):
            data.create_split(split_targets, 494, broken)
    assert (broken / '.split-write.lock').is_file()
    before = snapshot(broken)
    with pytest.raises(data.DataIntegrityError, match='lock'):
        data.load_split(broken / data.SPLIT_NAME, split_targets)
    with pytest.raises(data.DataIntegrityError, match='lock'):
        data.create_split(split_targets, 494, broken)
    assert snapshot(broken) == before
    recovered = tmp_path / 'recovered'
    data.create_split(split_targets, 494, recovered)
    assert len(data.load_split(recovered / data.SPLIT_NAME, split_targets)[0]) == 45000
    assert snapshot(broken) == before


def test_split_bad_api_arguments_do_not_write(tmp_path, split_targets):
    missing = tmp_path / 'missing'
    for targets, seed in [(split_targets, 42), (split_targets[:10000], 494), ([[1], 2], 494)]:
        with pytest.raises(data.DataIntegrityError):
            data.create_split(targets, seed, missing)
        assert not missing.exists()
    with pytest.raises(data.DataIntegrityError, match='filename'):
        data.load_split(tmp_path / 'wrong.npz', split_targets)


def test_split_publication_is_exclusive_and_lock_protected(tmp_path, split_targets, monkeypatch):
    original = data._publish_split_file
    def collision(source, destination):
        with pytest.raises(data.DataIntegrityError, match='lock'):
            data.load_split(tmp_path / data.SPLIT_NAME, split_targets)
        destination.write_bytes(b'foreign concurrent file')
        original(source, destination)
    monkeypatch.setattr(data, '_publish_split_file', collision)
    with pytest.raises(data.DataIntegrityError, match='diagnostic artifacts retained'):
        data.create_split(split_targets, 494, tmp_path)
    assert (tmp_path / data.SPLIT_NAME).read_bytes() == b'foreign concurrent file'
    assert (tmp_path / '.split-write.lock').exists()


@pytest.fixture
def split_cli_source(monkeypatch, split_targets):
    calls = []
    def load(root, train, download=False):
        assert train is True and download is False
        calls.append(root)
        return SimpleNamespace(targets=split_targets.tolist())
    monkeypatch.setattr(download_data, 'load_cifar10', load)
    def forbidden(*args, **kwargs):
        pytest.fail('Split mode must not acquire data, inspect test targets, or write raw manifest')
    for name in ('ensure_cifar10', 'inspect_cifar10', '_write_manifest'):
        monkeypatch.setattr(download_data, name, forbidden)
    return calls


def test_split_cli_prepare_verify_and_idempotence(tmp_path, split_cli_source, capsys):
    out_dir = tmp_path / 'splits'
    flags = ['--root', str(tmp_path / 'raw'), '--split-dir', str(out_dir)]
    assert download_data.main([*flags, '--prepare-split', '--split-seed', '494']) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared['train'] == 45000 and prepared['validation'] == 5000
    assert prepared['seed'] == 494 and prepared['status'] == 'verified'
    before = snapshot(tmp_path)
    for mode in ('--verify-split', '--prepare-split'):
        assert download_data.main([*flags, mode]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result['split_sha256'] == prepared['split_sha256']
        assert snapshot(tmp_path) == before
    assert len(split_cli_source) == 3
    assert not (tmp_path / 'raw').exists()


def test_split_cli_defaults_are_explicit(tmp_path, split_cli_source, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert download_data.main(['--prepare-split']) == 0
    assert json.loads(capsys.readouterr().out)['seed'] == 494
    assert (tmp_path / 'data/splits' / data.SPLIT_NAME).is_file()
    assert not (tmp_path / 'data/raw').exists()


@pytest.mark.parametrize('mode', ['--prepare-split', '--verify-split'])
def test_split_cli_missing_raw_fails_without_writes(tmp_path, mode, capsys):
    root, out_dir = tmp_path / 'raw', tmp_path / 'split'
    assert download_data.main(['--root', str(root), '--split-dir', str(out_dir), mode]) == 1
    assert json.loads(capsys.readouterr().err)['status'] == 'failed'
    assert not root.exists() and not out_dir.exists()


def test_split_cli_missing_and_tampered_pair_fail(tmp_path, split_cli_source, capsys):
    out_dir = tmp_path / 'split'
    flags = ['--split-dir', str(out_dir)]
    assert download_data.main([*flags, '--verify-split']) == 1
    capsys.readouterr()
    assert not out_dir.exists()
    assert download_data.main([*flags, '--prepare-split']) == 0
    capsys.readouterr()
    (out_dir / data.SPLIT_MANIFEST).write_text('{}')
    before = snapshot(out_dir)
    for mode in ('--verify-split', '--prepare-split'):
        assert download_data.main([*flags, mode]) == 1
        assert json.loads(capsys.readouterr().err)['status'] == 'failed'
        assert snapshot(out_dir) == before


def test_raw_verify_does_not_require_or_touch_split(raw, tmp_path, monkeypatch, capsys):
    # A broken split next to the chosen cwd must not affect the raw-only mode.
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'data/splits').mkdir(parents=True)
    (tmp_path / 'data/splits' / data.SPLIT_MANIFEST).write_text('broken')
    before = snapshot(tmp_path)
    assert download_data.main(['--root', str(raw), '--verify-only']) == 0
    assert json.loads(capsys.readouterr().out)['mode'] == 'verify-only'
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('flags', [
    ['--split-seed', '494'], ['--split-dir', 'other'], ['--verify-only', '--split-seed', '494'],
    ['--prepare-split', '--verify-split'], ['--verify-only', '--verify-split'],
    ['--prepare-split', '--split-seed', '42'], ['--prepare-split', '--split-seed', '494.0'],
    ['--verify-split', '--split-seed', 'true'], ['--prepare-split', '--split-dir'],
])
def test_split_cli_invalid_arguments(flags):
    with pytest.raises(SystemExit) as result:
        download_data.main(flags)
    assert result.value.code == 2


def test_split_cli_module_help_and_invalid_seed(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'scripts.download_data', '--help'],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    for option in ('--prepare-split', '--verify-split', '--split-dir', '--split-seed'):
        assert option in result.stdout
    result = subprocess.run([sys.executable, '-m', 'scripts.download_data', '--prepare-split',
                             '--root', str(tmp_path), '--split-seed', '42'],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2
    assert 'invalid choice' in result.stderr
    assert snapshot(tmp_path) == {}
