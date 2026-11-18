"""Independent synthetic TEST-M01/M03 oracles; no data or optimizer steps."""

import math

import pytest
import torch
from torch import nn

from src.models import CompactCNN
from src.utils import load_config


@pytest.fixture(autouse=True)
def preserve_cpu_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def test_literal_architecture_and_parameter_oracle():
    model = CompactCNN()
    assert model.architecture == 'compact_cnn_v1'
    assert [type(layer) for layer in model.features] == [
        nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
        nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d,
        nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
        nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d,
        nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
        nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d,
    ]
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert [(m.in_channels, m.out_channels) for m in convs] == [
        (3, 32), (32, 32), (32, 64), (64, 64), (64, 128), (128, 128)]
    assert [m.weight.numel() for m in convs] == [864, 9216, 18432, 36864, 73728, 147456]
    for m in convs:
        assert (m.kernel_size, m.stride, m.padding, m.dilation) == (
            (3, 3), (1, 1), (1, 1), (1, 1))
        assert m.bias is None and m.groups == 1 and m.padding_mode == 'zeros'
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert [m.num_features for m in bns] == [32, 32, 64, 64, 128, 128]
    for m in bns:
        assert (m.eps, m.momentum, m.affine, m.track_running_stats) == (1e-5, .1, True, True)
        assert torch.equal(m.weight, torch.ones_like(m.weight))
        assert torch.count_nonzero(m.bias) == torch.count_nonzero(m.running_mean) == 0
        assert torch.equal(m.running_var, torch.ones_like(m.running_var))
        assert m.num_batches_tracked.item() == 0
    for m in model.features:
        if isinstance(m, nn.MaxPool2d):
            assert (m.kernel_size, m.stride, m.padding, m.ceil_mode) == (2, 2, 0, False)
    assert type(model.avgpool) is nn.AdaptiveAvgPool2d and model.avgpool.output_size == 1
    assert type(model.flatten) is nn.Flatten and model.flatten.start_dim == 1
    assert type(model.dropout) is nn.Dropout and model.dropout.p == .2
    assert type(model.classifier) is nn.Linear
    assert (model.classifier.in_features, model.classifier.out_features) == (128, 10)
    assert model.classifier.bias is not None
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 288746
    assert sum(b.numel() for b in model.buffers()) == 902


@pytest.mark.parametrize('batch', [1, 4, 128])
def test_model_shapes_and_raw_logits(batch):
    model = CompactCNN().eval()
    shapes = []
    handles = [m.register_forward_hook(lambda m, args, out: shapes.append(tuple(out.shape)))
               for m in model.features if isinstance(m, nn.MaxPool2d)]
    x = torch.linspace(-2, 2, batch * 3 * 32 * 32).reshape(batch, 3, 32, 32)
    before = x.clone()
    # A literal head oracle forces signed values incompatible with probabilities.
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.arange(-5., 5.))
        logits = model(x)
    for handle in handles:
        handle.remove()
    assert shapes == [(batch, 32, 16, 16), (batch, 64, 8, 8), (batch, 128, 4, 4)]
    assert logits.shape == (batch, 10) and logits.dtype == torch.float32
    assert torch.equal(logits, torch.arange(-5., 5.).expand(batch, -1))
    assert torch.equal(x, before)


def test_initialization_matches_independent_rng_oracle():
    model = CompactCNN()
    torch.manual_seed(913)
    state = torch.get_rng_state().clone()
    model.initialize_weights()
    after = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            expected = torch.empty_like(m.weight).normal_(0., math.sqrt(2 / (m.out_channels * 9)))
            assert torch.equal(m.weight, expected)
        elif isinstance(m, nn.Linear):
            expected = torch.empty_like(m.weight).normal_(0., .01)
            assert torch.equal(m.weight, expected)
            assert torch.count_nonzero(m.bias) == 0
    assert torch.equal(torch.get_rng_state(), after)


@pytest.mark.parametrize('seed', [42, 123, 2026])
def test_caller_seed_pairs_clean_robust_full_initial_state(seed):
    states = []
    for path in ('configs/clean_cnn.yaml', 'configs/robust_cnn.yaml'):
        cfg = load_config(path)
        torch.manual_seed(seed)
        model = CompactCNN(num_classes=cfg['num_classes'], dropout=cfg['dropout'])
        assert cfg['architecture'] == model.architecture
        states.append({k: v.clone() for k, v in model.state_dict().items()})
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    torch.manual_seed(seed + 1)
    other = CompactCNN().state_dict()
    assert not torch.equal(states[0]['features.0.weight'], other['features.0.weight'])
    # Consecutive construction without a caller reseed must consume the stream.
    another = CompactCNN().state_dict()
    assert not torch.equal(other['classifier.weight'], another['classifier.weight'])


@pytest.mark.parametrize('kwargs', [
    {'num_classes': 0}, {'num_classes': -1}, {'num_classes': True}, {'num_classes': 10.},
    {'dropout': True}, {'dropout': -.1}, {'dropout': 1.1},
    {'dropout': float('nan')}, {'dropout': float('inf')}, {'dropout': '.2'},
])
def test_invalid_constructor_before_rng_draws(kwargs):
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        CompactCNN(**kwargs)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('shape', [(3, 32, 32), (2, 1, 32, 32), (2, 3, 31, 32),
                                  (2, 3, 32, 64), (0, 3, 32, 32)])
def test_invalid_shape_before_bn_or_dropout(shape):
    model = CompactCNN().train()
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='N x 3 x 32 x 32'):
        model(torch.empty(shape))
    assert torch.equal(before, torch.get_rng_state())
    assert all(m.num_batches_tracked == 0 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
