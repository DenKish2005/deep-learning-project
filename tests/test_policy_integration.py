"""Production policy through the existing train-only hook; synthetic offline QA."""

from pathlib import Path
import random

import numpy as np
import pytest
import torch

from src import data
from src.utils import load_config, stable_seed
from tests.test_loaders import base_stats, collect, dataset  # noqa: F401
from tests.test_policy import pixel_oracle, policy


def geometry_oracle(base, original, run_seed, epoch):
    """Independent NumPy integer pad/crop/flip before float conversion."""
    generator = torch.Generator().manual_seed(stable_seed(run_seed, epoch, original, 'geometry'))
    offsets = torch.randint(0, 9, (2,), generator=generator).tolist()
    image = np.pad(base.data[original], ((4, 4), (4, 4), (0, 0)))
    top, left = offsets
    image = image[top:top + 32, left:left + 32]
    if torch.rand((), generator=generator).item() < .5:
        image = image[:, ::-1]
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255


@pytest.mark.parametrize('run_seed', [42, 123, 2026])
@pytest.mark.parametrize('epoch', [0, 1])
def test_real_policy_paired_geometry_labels_and_normalize_order(base_stats, run_seed, epoch):
    base, stats = base_stats
    clean = dataset(base_stats, seed=run_seed, epoch=epoch)
    robust = dataset(base_stats, seed=run_seed, epoch=epoch, policy=policy())
    before = base.data.copy()
    seen, noncommuting = set(), False
    for original in range(len(base.data)):
        pixels = geometry_oracle(base, original, run_seed, epoch)
        expected, audit = pixel_oracle(pixels, torch.Generator().manual_seed(
            stable_seed(run_seed, epoch, original, 'train_corruption')))
        actual, label, index = robust[original]
        assert (label, index) == (int(base.targets[original]), original)
        assert torch.equal(clean[original][0], (pixels - clean.mean) / clean.std)
        torch.testing.assert_close(actual, (expected - clean.mean) / clean.std, atol=5e-6, rtol=0)
        seen.add(audit['event'])
        if audit['event'] == 'identity': assert torch.equal(actual, clean[original][0])
        if audit['event'] == 'brightness':
            wrong = clean[original][0] * (.9, .8, .7)[audit['severity'] - 1]
            assert not torch.allclose(actual, wrong)
            noncommuting = True
    assert noncommuting and 'identity' in seen
    np.testing.assert_array_equal(base.data, before)


def test_production_policy_epoch_batch_spawn_pairing_and_global_isolation(base_stats):
    clean, robust = dataset(base_stats), dataset(base_stats, policy=policy())
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    loaders = [data._make_loader(clean, 7, 0), data._make_loader(robust, 7, 0),
               data._make_loader(robust, 5, 2)]
    previous_order, previous_pixels = None, None
    for epoch in (0, 1):
        clean.set_epoch(epoch)
        robust.set_epoch(epoch)
        reference, serial, spawned = [collect(loader) for loader in loaders]
        assert all(torch.equal(a, b) for a, b in zip(serial, spawned))
        assert torch.equal(reference[1], serial[1]) and torch.equal(reference[2], serial[2])
        assert sorted(serial[2].tolist()) == list(range(17))
        assert [len(batch[0]) for batch in loaders[1]] == [7, 7, 3]
        assert serial[0].dtype == torch.float32 and serial[1].dtype == serial[2].dtype == torch.int64
        for robust_x, original in zip(serial[0], serial[2].tolist()):
            key = stable_seed(42, epoch, original, 'train_corruption')
            expected, _ = pixel_oracle(geometry_oracle(base_stats[0], original, 42, epoch),
                                       torch.Generator().manual_seed(key))
            torch.testing.assert_close(robust_x, (expected - robust.mean) / robust.std, atol=5e-6, rtol=0)
        if previous_order is not None:
            assert not torch.equal(previous_order, serial[2])
            assert not torch.equal(previous_pixels, serial[0][torch.argsort(serial[2])])
        previous_order = serial[2]
        previous_pixels = serial[0][torch.argsort(serial[2])]
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and after[2:] == before[1][2:]
    np.testing.assert_array_equal(after[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


@pytest.mark.parametrize('mode', ['val', 'test'])
def test_production_policy_rejected_outside_train(base_stats, mode):
    with pytest.raises(ValueError, match='train-only'):
        dataset(base_stats, mode=mode, policy=policy())


@pytest.mark.parametrize('config_name,injected', [
    ('robust_cnn.yaml', None), ('clean_cnn.yaml', policy()), ('robust_cnn.yaml', 42)])
def test_config_still_requires_explicit_matching_injection_before_io(monkeypatch, config_name, injected):
    config = load_config(Path('configs') / config_name)
    def forbidden(*args, **kwargs): pytest.fail('injection gate must precede dataset I/O')
    monkeypatch.setattr(data, 'load_cifar10', forbidden)
    with pytest.raises(ValueError, match='explicitly injected callable policy'):
        data.make_loaders(config, 42, 0, policy=injected, partitions=('train', 'val'))
