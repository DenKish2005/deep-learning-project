"""DATA-003 / TEST-D03: offline numeric and artifact oracles, no actual CIFAR."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from src import data


@pytest.fixture
def pixels():
    return np.random.default_rng(7).integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)


def test_stats_train_only_float64_population_oracle(pixels):
    pixels[0] = [0, 64, 128]
    pixels[1] = [255, 192, 200]
    stats = data.compute_normalization(pixels, [0, 1])
    oracle = pixels[:2].astype(np.float64) / 255
    np.testing.assert_allclose(stats['mean'], oracle.mean(axis=(0, 1, 2)), atol=1e-14, rtol=0)
    np.testing.assert_allclose(stats['std'], oracle.std(axis=(0, 1, 2), ddof=0), atol=1e-14, rtol=0)
    assert stats['mean'][0] == 0.5 and stats['std'][0] == 0.5
    assert stats['samples'] == 2 and stats['pixels_per_channel'] == 2048
    before = pixels.copy()
    pixels[2:] = 255  # Validation outliers; a test partition is never passed at all.
    assert stats == data.compute_normalization(pixels, [0, 1])
    np.testing.assert_array_equal(pixels[:2], before[:2])


def test_two_pass_chunk_boundary_and_precision():
    raw = np.random.default_rng(9).integers(0, 256, (259, 32, 32, 3), dtype=np.uint8)
    actual = data.compute_normalization(raw, np.arange(259))
    oracle = raw.astype(np.float64) / 255
    np.testing.assert_allclose(actual['mean'], oracle.mean(axis=(0, 1, 2)), atol=1e-13, rtol=0)
    np.testing.assert_allclose(actual['std'], oracle.std(axis=(0, 1, 2)), atol=1e-13, rtol=0)


@pytest.mark.parametrize('indices', [[], [0, 0], [0, -1], [5], [True, 1], [0.0, 1.0],
                                     [[0]], ['0'], [2**64 - 1], None])
def test_invalid_indices(pixels, indices):
    with pytest.raises(data.DataIntegrityError, match='indices'):
        data.compute_normalization(pixels, indices)


@pytest.mark.parametrize('raw', [np.zeros((2, 32, 32, 3), dtype=np.float32),
                                 np.zeros((0, 32, 32, 3), dtype=np.uint8),
                                 np.zeros((2, 3, 32, 32), dtype=np.uint8),
                                 np.full((2, 32, 32, 3), np.nan), None])
def test_invalid_pixels(raw):
    with pytest.raises(data.DataIntegrityError, match='pixels'):
        data.compute_normalization(raw, [0])


def test_constant_channel_rejected():
    with pytest.raises(data.DataIntegrityError, match='std'):
        data.compute_normalization(np.zeros((2, 32, 32, 3), dtype=np.uint8), [0, 1])


@pytest.mark.parametrize('field,value', [('mean', [float('nan')] * 3), ('std', [0.0] * 3),
    ('mean', [True] * 3), ('std', [float('inf')] * 3), ('mean', [0.2]),
    ('scale', 255.0), ('ddof', True), ('samples', True), ('pixels_per_channel', 1),
    ('channels', ['B', 'G', 'R']), ('train_indices_sha256', 'x' * 64), ('algorithm', 'foreign')])
def test_invalid_stats(pixels, field, value):
    stats = data.compute_normalization(pixels, [0, 1])
    stats[field] = value
    with pytest.raises(data.DataIntegrityError):
        data.validate_normalization(stats)


@pytest.fixture
def artifact(pixels, tmp_path, monkeypatch):
    # Tiny persisted identity fixture replaces ONLY the frozen-split boundary.
    # Actual full-sized split/provenance replay is covered separately opt-in.
    split = tmp_path / data.SPLIT_NAME
    split.write_bytes(b'offline split identity')
    manifest = {'dataset': {'name': 'synthetic-only'}, 'original_train': {'count': 5}}
    (tmp_path / data.SPLIT_MANIFEST).write_text(json.dumps(manifest), encoding='utf-8')
    def frozen(path, targets):
        assert path == split and list(targets) == [0, 1, 2, 3, 4]
        return np.array([0, 1], dtype='<i8'), np.array([2, 3, 4], dtype='<i8')
    monkeypatch.setattr(data, 'load_split', frozen)
    return pixels, [0, 1, 2, 3, 4], split, tmp_path / data.NORMALIZATION_NAME


def snapshot(path):
    return path.read_bytes(), path.stat().st_mtime_ns


def test_roundtrip_full_precision_reuse_readonly(artifact):
    expected = data.prepare_normalization(*artifact)
    path = artifact[-1]
    before = snapshot(path)
    assert data.load_normalization(*artifact) == expected
    assert data.prepare_normalization(*artifact) == expected
    assert snapshot(path) == before
    stats = expected['statistics']
    assert expected['statistics_sha256'] == hashlib.sha256(data._json_canonical(stats).encode()).hexdigest()
    assert expected['creator']['sha256_lf'] == data._normalization_code_sha256()
    assert not list(path.parent.glob('.normalization*'))


@pytest.mark.parametrize('kind', ['mean', 'hash', 'provenance', 'creator', 'unknown',
                                  'version_type', 'missing', 'duplicate', 'partial', 'nan'])
def test_tamper_preserved(artifact, kind):
    record = data.prepare_normalization(*artifact)
    path = artifact[-1]
    if kind == 'mean': record['statistics']['mean'][0] += .01
    if kind == 'hash': record['statistics_sha256'] = '0' * 64
    if kind == 'provenance': record['provenance']['split_file_sha256'] = '0' * 64
    if kind == 'creator': record['creator']['sha256_lf'] = '0' * 64
    if kind == 'unknown': record['extra'] = 1
    if kind == 'version_type': record['schema_version'] = True
    if kind == 'missing': del record['statistics']
    if kind == 'nan': record['statistics']['mean'][0] = float('nan')
    text = json.dumps(record)
    if kind == 'duplicate': text = text[:-1] + ',"schema_version":1}'
    if kind == 'partial': text = text[:23]
    path.write_text(text, encoding='utf-8')
    before = snapshot(path)
    for operation in (data.load_normalization, data.prepare_normalization):
        with pytest.raises(data.DataIntegrityError): operation(*artifact)
        assert snapshot(path) == before


def test_selected_pixels_and_split_identity_rechecked(artifact):
    data.prepare_normalization(*artifact)
    raw, targets, split, path = artifact
    raw[0, 0, 0, 0] ^= 255
    with pytest.raises(data.DataIntegrityError, match='mismatch'): data.load_normalization(*artifact)
    raw[0, 0, 0, 0] ^= 255
    split.write_bytes(b'changed split')
    with pytest.raises(data.DataIntegrityError, match='mismatch'): data.load_normalization(*artifact)
    with pytest.raises(data.DataIntegrityError, match='length'):
        data.prepare_normalization(raw[:2], targets, split, path)


@pytest.mark.parametrize('diagnostic', ['lock', 'stage'])
def test_interrupted_state_and_missing_readonly(artifact, diagnostic):
    path = artifact[-1]
    with pytest.raises(data.DataIntegrityError, match='missing'): data.load_normalization(*artifact)
    assert not path.exists()
    lock, stage = data._stats_paths(path)
    target = lock if diagnostic == 'lock' else stage
    target.write_bytes(b'preserved diagnostic')
    with pytest.raises(data.DataIntegrityError, match='interrupted'): data.prepare_normalization(*artifact)
    assert target.read_bytes() == b'preserved diagnostic'


@pytest.mark.parametrize('failure', ['fsync', 'publication', 'concurrent'])
def test_failed_creation_preserves_diagnostics(artifact, monkeypatch, failure):
    def fail(*args):
        if failure == 'concurrent': artifact[-1].write_bytes(b'foreign')
        raise OSError('injected failure')
    if failure == 'fsync': monkeypatch.setattr(data.os, 'fsync', fail)
    else: monkeypatch.setattr(data, '_publish_split_file', fail)
    with pytest.raises(data.DataIntegrityError, match='interrupted'): data.prepare_normalization(*artifact)
    assert data._stats_paths(artifact[-1])[0].exists()
    if failure == 'concurrent': assert artifact[-1].read_bytes() == b'foreign'
    with pytest.raises(data.DataIntegrityError, match='interrupted'): data.load_normalization(*artifact)


def test_numeric_primitive_global_rng_isolation(pixels):
    import random
    import torch
    py, np_state, torch_state = random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone()
    data.compute_normalization(pixels, [0, 1])
    assert random.getstate() == py
    assert np.random.get_state()[0] == np_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
