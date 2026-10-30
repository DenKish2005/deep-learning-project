"""Verified raw CIFAR-10 only; no splits, transforms or training decisions.

MD5 references come from the official dataset page and torchvision v0.24.0.
Files are checked before torchvision unpickles them. MD5 is an upstream
integrity reference, not a cryptographic authenticity guarantee.
"""

import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile
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
