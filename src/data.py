"""Verified raw CIFAR-10 and fixed original-train split; no transforms/training.

MD5 references come from the official dataset page and torchvision v0.24.0.
Files are checked before torchvision unpickles them. MD5 is an upstream
integrity reference, not a cryptographic authenticity guarantee.
"""

import hashlib
from collections.abc import Sequence
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import zipfile
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
from torchvision.datasets import CIFAR10


SOURCE_PAGE = 'https://cave.cs.toronto.edu/kriz/cifar.html'
SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
BASE_FOLDER = 'cifar-10-batches-py'
ARCHIVE = 'cifar-10-python.tar.gz'
ARCHIVE_MD5 = 'c58f30108f718f92721af3b95e74349a'
ARCHIVE_BYTES = 170498071  # Official HTTPS Content-Length, checked 2026-10-30.
FILE_MD5 = {
    'data_batch_1': 'c99cafc152244af753f735de768cd75f',
    'data_batch_2': 'd4bba439e000b95fd0a9bffe97cbabec',
    'data_batch_3': '54ebc095f3ab1f0389bbae665268c751',
    'data_batch_4': '634d18415352ddfa80567beed471001a',
    'data_batch_5': '482c414d41f54cd18b22e5b47cb7c3cb',
    'test_batch': '40351d587109b95175f43aff81a1287e',
    'batches.meta': '5ff9c542aee3614f3951f8cda6e48888',
}
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck')
PARTITION_SIZE = {True: 50000, False: 10000}


class DataIntegrityError(RuntimeError):
    """Missing, incompatible or damaged raw dataset; never silently repaired."""


def _error(detail: str) -> DataIntegrityError:
    return DataIntegrityError(
        f'{detail}. No existing artifacts were replaced. For missing data run '
        'python -m scripts.download_data --root <root>. For damaged/conflicting '
        'artifacts preserve them and retry with a new empty --root directory.'
    )


def _hash(path: Path, algorithm: str) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def _verify_file(path: Path, expected: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise _error(f'Missing or non-regular CIFAR-10 file: {path}')
    actual = _hash(path, 'md5')
    if actual != expected:
        raise _error(f'CIFAR-10 checksum mismatch: {path}; expected MD5 '
                     f'{expected}, observed {actual}')
    return {'md5': actual, 'sha256': _hash(path, 'sha256'),
            'bytes': path.stat().st_size}


def _check_torchvision_contract() -> None:
    references = dict(CIFAR10.train_list + CIFAR10.test_list)
    references[CIFAR10.meta['filename']] = CIFAR10.meta['md5']
    if (references != FILE_MD5 or CIFAR10.tgz_md5 != ARCHIVE_MD5
            or CIFAR10.base_folder != BASE_FOLDER or CIFAR10.filename != ARCHIVE
            or CIFAR10.url != SOURCE_URL):
        raise _error('Installed torchvision CIFAR10 references differ from the verified contract')


def verify_cifar10_files(root: Path) -> dict:
    """Read-only full file inspection, including archive if one is present."""
    root = Path(root)
    _check_torchvision_contract()
    base = root / BASE_FOLDER
    if base.is_symlink() or not base.is_dir():
        raise _error(f'Missing or non-directory CIFAR-10 dataset: {base}')
    files = {name: _verify_file(base / name, checksum)
             for name, checksum in FILE_MD5.items()}
    archive = root / ARCHIVE
    archive_info = (_verify_file(archive, ARCHIVE_MD5)
                    if archive.exists() or archive.is_symlink() else None)
    return {'files': files, 'archive': archive_info}


def _validate_partition(dataset: CIFAR10, train: bool) -> dict:
    n = PARTITION_SIZE[train]
    name = 'train' if train else 'test'
    if (dataset.data.shape != (n, 32, 32, 3)
            or dataset.data.dtype != np.uint8 or len(dataset) != n):
        raise _error(f'Invalid {name} image contract: expected uint8 [{n},32,32,3]')
    if (tuple(dataset.classes) != CLASSES
            or dataset.class_to_idx != dict(zip(CLASSES, range(10)))):
        raise _error(f'Invalid {name} CIFAR-10 class mapping')
    labels = np.asarray(dataset.targets)
    if (labels.shape != (n,) or labels.dtype.kind not in 'iu'
            or np.any(labels < 0) or np.any(labels > 9)):
        raise _error(f'Invalid {name} labels: expected {n} integers in 0..9')
    counts = np.bincount(labels, minlength=10)
    if not np.array_equal(counts, np.full(10, n // 10)):
        raise _error(f'Invalid {name} per-class counts: {counts.tolist()}')
    image, label = dataset[0]
    if image.mode != 'RGB' or image.size != (32, 32) or type(label) is not int:
        raise _error(f'Invalid {name} raw PIL/label interface')
    return {'count': n, 'shape': [n, 32, 32, 3], 'dtype': 'uint8',
            'class_counts': counts.tolist()}


def _read_partition(root: Path, train: bool) -> CIFAR10:
    try:
        dataset = CIFAR10(root=root, train=train, download=False)
        _validate_partition(dataset, train)
        return dataset
    except DataIntegrityError:
        raise
    except Exception as exc:
        raise _error(f'Cannot decode verified CIFAR-10 partition: {exc}') from exc


def load_cifar10(root: Path, train: bool, download=False) -> CIFAR10:
    """Return raw PIL RGB images and integer labels; offline unless explicit.

    Both partitions' files and metadata are hashed before loading. Only the
    requested partition is decoded/inspected; train loading does not inspect
    test pixels or label distributions.
    """
    if type(train) is not bool or type(download) is not bool:
        raise TypeError('train and download must be bool')
    root = Path(root)
    if download:
        ensure_cifar10(root)
    verify_cifar10_files(root)
    return _read_partition(root, train)


def inspect_cifar10(root: Path) -> dict:
    """Fresh read-only integrity evidence for both original partitions.

    Test partition inspection here is solely for integrity, never model design.
    A missing archive is explicitly null, not an inferred checksum verification.
    """
    files = verify_cifar10_files(root)
    partitions = {}
    for train, name in ((True, 'train'), (False, 'test')):
        dataset = _read_partition(Path(root), train)
        partitions[name] = _validate_partition(dataset, train)
        del dataset
    return {'dataset': 'cifar10-python', 'source_page': SOURCE_PAGE,
            'source_url': SOURCE_URL, 'archive_name': ARCHIVE,
            'expected_archive_md5': ARCHIVE_MD5,
            'checksum_reference': 'https://github.com/pytorch/vision/blob/v0.24.0/torchvision/datasets/cifar.py',
            **files, 'classes': list(CLASSES),
            'class_to_idx': dict(zip(CLASSES, range(10))), 'partitions': partitions}


def _allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    return (parsed.scheme == 'https' and parsed.hostname in
            {'www.cs.toronto.edu', 'cave.cs.toronto.edu'}
            and parsed.port in (None, 443) and parsed.username is None
            and parsed.path in {'/~kriz/cifar-10-python.tar.gz',
                                '/kriz/cifar-10-python.tar.gz'})


class _OfficialRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _allowed_url(newurl):
            raise _error(f'Refusing non-official or non-HTTPS download redirect: {newurl}')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_archive(destination: Path) -> str:
    # Bounded requests make large transfers inspectable in validated pieces.
    # Each response must cover exactly its requested byte range; the complete
    # assembled archive still has to match the upstream MD5 before publication.
    opener = build_opener(_OfficialRedirects())
    final_url = SOURCE_URL
    chunk_bytes = 1024 * 1024
    with destination.open('xb') as output:
        for start in range(0, ARCHIVE_BYTES, chunk_bytes):
            end = min(start + chunk_bytes, ARCHIVE_BYTES) - 1
            request = Request(SOURCE_URL, headers={'Range': f'bytes={start}-{end}'})
            with opener.open(request, timeout=60) as response:
                final_url = response.geturl()
                if not _allowed_url(final_url):
                    raise _error(f'Refusing download source: {final_url}')
                expected_range = f'bytes {start}-{end}/{ARCHIVE_BYTES}'
                if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                    raise _error(f'Official source returned an unexpected byte range: '
                                 f'{response.headers.get("Content-Range")} (expected {expected_range})')
                contents = response.read(end - start + 2)
                if len(contents) != end - start + 1:
                    raise _error(f'Incomplete official archive range {start}-{end}: {len(contents)} bytes')
                output.write(contents)
                output.flush()
    return final_url


def _extract_verified_archive(archive: Path, stage: Path) -> None:
    """Extract only expected regular files, never links or arbitrary paths."""
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        for name in FILE_MD5:
            matches = [m for m in members if m.name == f'{BASE_FOLDER}/{name}']
            if len(matches) != 1 or not matches[0].isfile():
                raise _error(f'Archive missing/duplicate/non-regular member: {name}')
            destination = stage / BASE_FOLDER / name
            destination.parent.mkdir(exist_ok=True)
            with bundle.extractfile(matches[0]) as source, destination.open('xb') as out:
                shutil.copyfileobj(source, out)


def ensure_cifar10(root: Path) -> dict:
    """Explicit acquisition. Reuse verified data/archive; refuse damaged state.

    Failed new downloads/extractions remain in a uniquely named staging folder
    for diagnosis. Existing user artifacts are never removed or overwritten.
    """
    root = Path(root)
    _check_torchvision_contract()
    base, archive = root / BASE_FOLDER, root / ARCHIVE
    if base.exists() or base.is_symlink():
        verify_cifar10_files(root)
        return {'method': 'preexisting_files', 'download_url': None,
                'note': 'Historical acquisition is unknown; files verified locally.'}
    if archive.exists() or archive.is_symlink():
        _verify_file(archive, ARCHIVE_MD5)
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.cifar-acquire-', dir=root))
    provenance = {'method': 'preexisting_archive', 'download_url': None,
                  'note': 'Historical acquisition is unknown; archive verified locally.'}
    try:
        if not archive.exists():
            temporary_archive = stage / ARCHIVE
            final_url = _download_archive(temporary_archive)
            _verify_file(temporary_archive, ARCHIVE_MD5)
            # Exclusive creation works without Windows hard-link privileges.
            # If copying fails, retain both artifacts and refuse automatic repair.
            with temporary_archive.open('rb') as source, archive.open('xb') as output:
                shutil.copyfileobj(source, output)
            _verify_file(archive, ARCHIVE_MD5)
            provenance = {'method': 'official_https_download', 'download_url': final_url,
                          'note': 'Downloaded and checksum-verified by this acquisition.'}
        _extract_verified_archive(archive, stage)
        verify_cifar10_files(stage)
        for train in (True, False):
            _read_partition(stage, train)
        if base.exists() or base.is_symlink():
            raise _error(f'Dataset destination appeared during acquisition: {base}')
        (stage / BASE_FOLDER).rename(base)
        verify_cifar10_files(root)
    except Exception as exc:
        raise _error(f'CIFAR-10 acquisition failed; diagnostic staging retained at {stage}: {exc}') from exc
    # Verify the absolute deletion target stays within the explicit dataset root.
    stage.resolve().relative_to(root.resolve())
    if stage.is_symlink() or not stage.name.startswith('.cifar-acquire-'):
        raise _error(f'Unsafe staging cleanup target: {stage}')
    shutil.rmtree(stage)  # Only the uniquely created, owned staging directory.
    return provenance


# BEGIN DATA-002 v1 (LF-normalized source identity; keep DATA-003 outside this region)
SPLIT_NAME = 'cifar10_split_v1.npz'
SPLIT_MANIFEST = 'split_manifest.json'
SPLIT_SEED = 494


def _split_error(detail: str) -> DataIntegrityError:
    return DataIntegrityError(
        f'Split verification failed: {detail}. Preserve existing split/diagnostic '
        'artifacts; inspect the mismatch or use a new empty split directory. '
        'No automatic repair or replacement is performed.'
    )


def _validate_split_targets(targets: Sequence[int]) -> np.ndarray:
    """Only original-train-shaped labels; actual source is verified by the CLI."""
    if not isinstance(targets, (Sequence, np.ndarray)):
        raise _split_error('targets must be a sequence of 50000 integer labels')
    try:
        labels = np.asarray(targets)
    except (TypeError, ValueError) as exc:
        raise _split_error('targets must be a flat integer sequence') from exc
    if (labels.shape != (50000,) or labels.dtype.kind not in 'iu'
            or (not isinstance(targets, np.ndarray)
                and any(isinstance(x, (bool, np.bool_)) for x in targets))):
        raise _split_error('targets must be 50000 integer labels (no bool/float/object)')
    if np.any(labels > 9) or np.any(labels < 0):
        raise _split_error('targets must contain only classes 0..9')
    labels = labels.astype('<i8', copy=True)
    if not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 5000)):
        raise _split_error('original train requires exactly 5000 targets per class')
    return labels


def _stratified_indices(labels: np.ndarray, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Protocol v1: one local RNG, classes 0..9, first 500/class to validation."""
    if type(split_seed) is not int or split_seed != SPLIT_SEED:
        raise _split_error('split_seed must be the Python integer 494, independent of model seed')
    rng = np.random.default_rng(split_seed)
    train, val = [], []
    for label in range(10):
        candidates = np.flatnonzero(labels == label)  # Ascending original indices.
        permuted = rng.permutation(candidates)
        val.append(permuted[:500])
        train.append(permuted[500:])
    return (np.sort(np.concatenate(train)).astype('<i8'),
            np.sort(np.concatenate(val)).astype('<i8'))


def _validate_split_indices(train: np.ndarray, val: np.ndarray, labels: np.ndarray) -> None:
    for name, indices, n, per_class in (('train', train, 45000, 4500), ('val', val, 5000, 500)):
        if indices.dtype != np.dtype('<i8') or indices.shape != (n,):
            raise _split_error(f'{name} indices must be little-endian int64 [{n}]')
        if np.any(indices < 0) or np.any(indices >= 50000):
            raise _split_error(f'{name} indices outside original-train namespace 0..49999')
        if np.any(np.diff(indices) <= 0):
            raise _split_error(f'{name} indices must be sorted and unique')
        if not np.array_equal(np.bincount(labels[indices], minlength=10), np.full(10, per_class)):
            raise _split_error(f'{name} label/index alignment requires {per_class} per class')
    if not np.array_equal(np.sort(np.concatenate((train, val))), np.arange(50000)):
        raise _split_error('train/val must be disjoint and exhaustive; overlap or missing IDs')


def _array_sha256(array: np.ndarray) -> str:
    """Semantic identity: compact sorted JSON dtype/shape, LF, C-order bytes."""
    header = json.dumps({'dtype': array.dtype.str, 'shape': list(array.shape)},
                        sort_keys=True, separators=(',', ':')).encode('ascii')
    return hashlib.sha256(header + b'\n' + array.tobytes(order='C')).hexdigest()


def _split_code_sha256() -> str:
    # Scope the creation identity to this versioned implementation, so adding
    # unrelated DATA-003 functions outside the region does not invalidate v1.
    source = Path(__file__).read_text(encoding='utf-8')
    region = source.split('\n# BEGIN DATA-002 v1 ', 1)[1].split('\n# END DATA-002 v1', 1)[0]
    return hashlib.sha256(region.encode('utf-8')).hexdigest()


def _split_record(labels: np.ndarray, train: np.ndarray, val: np.ndarray, path: Path) -> dict:
    return {
        'schema_version': 1,
        'algorithm': {'name': 'cifar10-stratified-default_rng', 'version': 1,
                      'seed': SPLIT_SEED, 'class_order': list(range(10)),
                      'validation_per_class': 500, 'final_order': 'ascending'},
        'dataset': {'name': 'cifar10-python', 'source_page': SOURCE_PAGE,
                    'source_url': SOURCE_URL, 'archive_name': ARCHIVE,
                    'archive_md5_reference': ARCHIVE_MD5,
                    'train_files_md5_reference': {k: v for k, v in FILE_MD5.items()
                                                if k.startswith('data_batch_') or k == 'batches.meta'},
                    'classes': list(CLASSES)},
        'original_train': {'namespace': 'cifar10-python/original-train', 'count': 50000,
                           'class_counts': [5000] * 10,
                           'targets_sha256': _array_sha256(labels)},
        'arrays': {name: {'dtype': '<i8', 'shape': [len(indices)],
                          'sha256': _array_sha256(indices),
                          'class_counts': np.bincount(labels[indices], minlength=10).tolist()}
                   for name, indices in (('train_indices', train), ('val_indices', val))},
        'npz': {'filename': SPLIT_NAME, 'sha256': _hash(path, 'sha256'),
                'bytes': path.stat().st_size},
        'creator': {'source_region': 'src/data.py:DATA-002-v1',
                    'sha256_lf': _split_code_sha256(), 'numpy': np.__version__},
    }


def _split_state(out_dir: Path) -> bool:
    """Return whether the complete pair exists; never regard diagnostics as ready."""
    if out_dir.is_symlink() or (out_dir.exists() and not out_dir.is_dir()):
        raise _split_error(f'non-directory/symlink split directory: {out_dir}')
    if (out_dir / '.split-write.lock').exists() or (out_dir / '.split-write.lock').is_symlink():
        raise _split_error(f'interrupted or active write lock in {out_dir}')
    if list(out_dir.glob('.split-stage-*')):
        raise _split_error(f'interrupted diagnostic staging in {out_dir}')
    paths = [out_dir / SPLIT_NAME, out_dir / SPLIT_MANIFEST]
    present = [p.exists() or p.is_symlink() for p in paths]
    if any(present) and not all(present):
        raise _split_error(f'partial artifact pair in {out_dir}')
    for path, exists in zip(paths, present):
        if exists and (path.is_symlink() or not path.is_file()):
            raise _split_error(f'non-regular split artifact: {path}')
    return all(present)


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate manifest key: {key}')
        result[key] = value
    return result


def _read_split_pair(path: Path, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fresh verification, including a protocol replay oracle; never replace arrays."""
    try:
        # Bound decoding and reject duplicate/extra ZIP members before NumPy.
        if path.stat().st_size > 1000000:
            raise ValueError('NPZ exceeds fixed split size limit')
        with zipfile.ZipFile(path) as bundle:
            members = bundle.infolist()
            if (sorted(m.filename for m in members) != ['train_indices.npy', 'val_indices.npy']
                    or any(m.file_size > 400000 for m in members)):
                raise ValueError('unexpected NPZ members or uncompressed size')
        with np.load(path, allow_pickle=False) as bundle:
            train, val = bundle['train_indices'], bundle['val_indices']
        _validate_split_indices(train, val, labels)
        expected = _stratified_indices(labels, SPLIT_SEED)
        if any(not np.array_equal(a, b) for a, b in zip((train, val), expected)):
            raise ValueError('indices differ from exact seed494 protocol algorithm')
        manifest_path = path.parent / SPLIT_MANIFEST
        if manifest_path.stat().st_size > 65536:
            raise ValueError('manifest exceeds schema size limit')
        record = json.loads(manifest_path.read_text(encoding='utf-8'),
                            object_pairs_hook=_unique_json_pairs)
        expected_record = _split_record(labels, train, val, path)
        # Canonical JSON comparison distinguishes bool/int/float, unknown keys,
        # altered source/seed/label order, hashes and creation implementation.
        if json.dumps(record, sort_keys=True, allow_nan=False) != json.dumps(expected_record, sort_keys=True):
            raise ValueError('manifest schema/provenance/hash mismatch against verified inputs and code')
        return train, val, record
    except DataIntegrityError:
        raise
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise _split_error(f'invalid split pair at {path}: {exc}') from exc


def load_split(path: Path, targets: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Read-only v1 NPZ+manifest verification against supplied original-train labels.

    Returns sorted little-endian int64 train/val arrays, not labels or test IDs.
    Source images must first be verified by load_cifar10; arbitrary balanced
    synthetic labels are accepted for offline unit tests, not certified as CIFAR.
    """
    path = Path(path)
    labels = _validate_split_targets(targets)
    if path.name != SPLIT_NAME:
        raise _split_error(f'expected canonical NPZ filename {SPLIT_NAME}')
    if not _split_state(path.parent):
        raise _split_error(f'missing split pair in {path.parent}; prepare it explicitly')
    train, val, _ = _read_split_pair(path, labels)
    return train, val


def _publish_split_file(source: Path, destination: Path) -> None:
    """Exclusive publication; a failed copy remains partial and lock-protected."""
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())


def create_split(targets: Sequence[int], split_seed: int, out_dir: Path) -> dict:
    """Create once or fully verify/reuse; return the exact schema-v1 manifest dict.

    The write lock is the readiness guard for this two-file transaction. Failure
    retains lock/staging/partials for diagnosis; recovery requires preserving
    that directory and explicitly preparing a new empty one. No silent repair.
    """
    labels = _validate_split_targets(targets)
    train, val = _stratified_indices(labels, split_seed)
    _validate_split_indices(train, val, labels)
    out_dir = Path(out_dir)
    if _split_state(out_dir):
        return _read_split_pair(out_dir / SPLIT_NAME, labels)[2]
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / '.split-write.lock'
    try:
        with lock.open('xb') as stream:
            stream.write(b'DATA-002 incomplete until this lock is removed by successful creation\n')
            stream.flush()
            os.fsync(stream.fileno())
        stage = Path(tempfile.mkdtemp(prefix='.split-stage-', dir=out_dir))
        # A second creator may have won before lock acquisition. Never replace it.
        if any((out_dir / name).exists() or (out_dir / name).is_symlink()
               for name in (SPLIT_NAME, SPLIT_MANIFEST)):
            raise _split_error('destination appeared during creation')
        staged_npz = stage / SPLIT_NAME
        with staged_npz.open('xb') as stream:
            np.savez_compressed(stream, train_indices=train, val_indices=val)
            stream.flush()
            os.fsync(stream.fileno())
        record = _split_record(labels, train, val, staged_npz)
        with (stage / SPLIT_MANIFEST).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        _read_split_pair(staged_npz, labels)
        for name in (SPLIT_NAME, SPLIT_MANIFEST):
            _publish_split_file(stage / name, out_dir / name)
        _read_split_pair(out_dir / SPLIT_NAME, labels)
        # Remove only our own known files and empty staging directory.
        for name in (SPLIT_NAME, SPLIT_MANIFEST):
            (stage / name).unlink()
        stage.rmdir()
        lock.unlink()  # Last action: pair is now ready for read-only verification.
        return record
    except (OSError, ValueError, DataIntegrityError) as exc:
        raise _split_error(f'creation interrupted in {out_dir}; diagnostic artifacts retained: {exc}') from exc
# END DATA-002 v1


# BEGIN DATA-003 normalization v1
NORMALIZATION_NAME = 'normalization.json'


def _stats_error(detail: str) -> DataIntegrityError:
    return DataIntegrityError(f'Normalization verification failed: {detail}. Preserve '
                              'existing stats/diagnostics; no automatic replacement. '
                              'Inspect the mismatch or choose a new stats path.')


def _validate_pixels(raw_data: np.ndarray) -> None:
    if (not isinstance(raw_data, np.ndarray) or raw_data.dtype != np.uint8
            or raw_data.ndim != 4 or raw_data.shape[1:] != (32, 32, 3)
            or len(raw_data) == 0):
        raise _stats_error('pixels must be nonempty uint8 [N,32,32,3] RGB')


def _validate_indices(indices, size: int) -> np.ndarray:
    try:
        values = np.asarray(indices)
    except (ValueError, TypeError) as exc:
        raise _stats_error('indices must be a flat integer sequence') from exc
    if (values.ndim != 1 or not values.size or values.dtype.kind not in 'iu'
            or (not isinstance(indices, np.ndarray)
                and any(isinstance(i, (bool, np.bool_)) for i in indices))
            or np.any(values < 0) or np.any(values >= size)
            or np.unique(values).size != values.size):
        raise _stats_error('indices must be nonempty unique integers within the partition')
    return values.astype('<i8', copy=True)


def validate_normalization(stats: dict) -> None:
    """Validate the complete numeric contract returned by compute_normalization."""
    keys = {'mean', 'std', 'channels', 'scale', 'ddof', 'samples',
            'pixels_per_channel', 'train_indices_sha256', 'algorithm'}
    if not isinstance(stats, dict) or set(stats) != keys:
        raise _stats_error('invalid statistics schema')
    if (stats['channels'] != ['R', 'G', 'B'] or type(stats['scale']) is not float
            or stats['scale'] != 1.0 / 255 or type(stats['ddof']) is not int
            or stats['ddof'] != 0 or type(stats['samples']) is not int
            or stats['samples'] <= 0 or type(stats['pixels_per_channel']) is not int
            or stats['pixels_per_channel'] != stats['samples'] * 1024
            or stats['algorithm'] != 'float64-two-pass-chunks256-v1'):
        raise _stats_error('invalid channel/scale/count/algorithm contract')
    checksum = stats['train_indices_sha256']
    if (not isinstance(checksum, str) or len(checksum) != 64
            or any(c not in '0123456789abcdef' for c in checksum)):
        raise _stats_error('invalid train array semantic hash')
    for key in ('mean', 'std'):
        values = stats[key]
        if (not isinstance(values, list) or len(values) != 3
                or any(type(x) is not float or not np.isfinite(x) for x in values)
                or any(x < 0 or x > 1 for x in values)
                or (key == 'std' and any(x <= 0 for x in values))):
            raise _stats_error(f'invalid finite {key}; std must be positive')


def compute_normalization(raw_data: np.ndarray, train_indices) -> dict:
    """Train-only raw uint8/255 population moments, fixed float64 two-pass order.

    This numerical primitive accepts tiny offline fixtures. Official provenance
    and the persisted 45000 train IDs are enforced by prepare/load_normalization.
    No augmented, validation or test pixels enter the accumulation.
    """
    _validate_pixels(raw_data)
    indices = _validate_indices(train_indices, len(raw_data))
    total = np.zeros(3, dtype=np.float64)
    for start in range(0, len(indices), 256):
        pixels = raw_data[indices[start:start + 256]].astype(np.float64) / 255.0
        total += pixels.sum(axis=(0, 1, 2), dtype=np.float64)
    count = len(indices) * 1024
    mean = total / count
    squared = np.zeros(3, dtype=np.float64)
    for start in range(0, len(indices), 256):
        pixels = raw_data[indices[start:start + 256]].astype(np.float64) / 255.0
        squared += ((pixels - mean) ** 2).sum(axis=(0, 1, 2), dtype=np.float64)
    stats = {'mean': mean.tolist(), 'std': np.sqrt(squared / count).tolist(),
             'channels': ['R', 'G', 'B'], 'scale': 1.0 / 255, 'ddof': 0,
             'samples': len(indices), 'pixels_per_channel': count,
             'train_indices_sha256': _array_sha256(indices),
             'algorithm': 'float64-two-pass-chunks256-v1'}
    validate_normalization(stats)
    return stats


def _normalization_code_sha256() -> str:
    source = Path(__file__).read_text(encoding='utf-8')
    region = source.split('\n# BEGIN DATA-003 normalization v1\n', 1)[1].split(
        '\n# END DATA-003 normalization v1', 1)[0]
    return hashlib.sha256(region.encode('utf-8')).hexdigest()


def _json_canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _normalization_record(raw_data, targets, split_path: Path) -> dict:
    _validate_pixels(raw_data)
    if len(raw_data) != len(targets):
        raise _stats_error('raw pixels/targets length mismatch')
    train, _ = load_split(split_path, targets)  # Returns persisted IDs, read-only.
    manifest = json.loads((split_path.parent / SPLIT_MANIFEST).read_text(encoding='utf-8'))
    stats = compute_normalization(raw_data, train)
    # Semantic identity of the selected raw pixels, without a full 138MB copy.
    header = _json_canonical({'dtype': '|u1', 'shape': [len(train), 32, 32, 3]})
    digest = hashlib.sha256(header.encode('ascii') + b'\n')
    for start in range(0, len(train), 256):
        digest.update(raw_data[train[start:start + 256]].tobytes(order='C'))
    return {'schema_version': 1, 'statistics': stats,
            'statistics_sha256': hashlib.sha256(_json_canonical(stats).encode('ascii')).hexdigest(),
            'provenance': {'dataset': manifest['dataset'],
                           'original_train': manifest['original_train'],
                           'split_file_sha256': _hash(split_path, 'sha256'),
                           'split_manifest_sha256': _hash(split_path.parent / SPLIT_MANIFEST, 'sha256'),
                           'train_pixels_sha256': digest.hexdigest()},
            'creator': {'source_region': 'src/data.py:DATA-003-normalization-v1',
                        'sha256_lf': _normalization_code_sha256(), 'numpy': np.__version__}}


def _stats_paths(path: Path) -> tuple[Path, Path]:
    if path.parent.is_symlink() or (path.parent.exists() and not path.parent.is_dir()):
        raise _stats_error('non-directory/symlink stats parent')
    return (path.with_name(f'.{path.name}.write.lock'),
            path.with_name(f'.{path.name}.stage'))


def _read_normalization(path: Path, expected: dict) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16384:
        raise _stats_error(f'missing/non-regular/oversized stats: {path}')
    try:
        record = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_json_pairs)
        if _json_canonical(record) != _json_canonical(expected):
            raise ValueError('strict schema/numeric/provenance/code identity mismatch')
        return record
    except (ValueError, TypeError, OSError) as exc:
        raise _stats_error(f'invalid stats {path}: {exc}') from exc


def _normalization_state(path: Path) -> bool:
    lock, stage = _stats_paths(path)
    if any(p.exists() or p.is_symlink() for p in (lock, stage)):
        raise _stats_error('active/interrupted lock or stage')
    return path.exists() or path.is_symlink()


def load_normalization(raw_data, targets, split_path: Path, path: Path) -> dict:
    """Read-only fresh numeric/provenance verification; never trusts saved hashes alone.

    Caller must obtain raw_data/targets from the verified original train loader.
    No original test partition is decoded or needed by this API.
    """
    path, split_path = Path(path), Path(split_path)
    if not _normalization_state(path):
        raise _stats_error(f'missing stats: {path}; prepare explicitly')
    return _read_normalization(path, _normalization_record(raw_data, targets, split_path))


def prepare_normalization(raw_data, targets, split_path: Path, path: Path) -> dict:
    """Create exclusively, or verify/reuse with byte/mtime preservation.

    Lock is removed last. Failures retain stage/partials/lock; no repair/overwrite.
    This is a readiness protocol, not a power-loss filesystem transaction.
    """
    path, split_path = Path(path), Path(split_path)
    exists = _normalization_state(path)
    expected = _normalization_record(raw_data, targets, split_path)
    if exists:
        return _read_normalization(path, expected)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock, stage = _stats_paths(path)
    try:
        with lock.open('xb') as stream:
            stream.write(b'DATA-003 incomplete until successful verification and lock removal\n')
            stream.flush()
            os.fsync(stream.fileno())
        with stage.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        _read_normalization(stage, expected)
        # Exclusive copy, including a final flush/fsync; no os.replace overwrite.
        _publish_split_file(stage, path)
        result = _read_normalization(path, expected)
        stage.unlink()
        lock.unlink()
        return result
    except (OSError, ValueError, DataIntegrityError) as exc:
        raise _stats_error(f'creation interrupted; diagnostic artifacts retained: {exc}') from exc
# END DATA-003 normalization v1
