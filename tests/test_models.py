"""Independent synthetic TEST-M01/M03 oracles; no data or optimizer steps."""

import copy
import io
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
        states.append(copy.deepcopy(model.state_dict()))
    assert states[0].keys() == states[1].keys()
    assert_states_equal(states[0], states[1])
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


def assert_states_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key


@pytest.mark.parametrize('batch', [1, 4])
def test_finite_ce_backward_without_parameter_updates(batch):
    torch.manual_seed(42)
    model = CompactCNN().train()
    weights = {k: p.detach().clone() for k, p in model.named_parameters()}
    x = torch.linspace(-2, 2, batch * 3 * 32 * 32).reshape(batch, 3, 32, 32).requires_grad_()
    targets = torch.arange(batch) % 10
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, targets)
    expected = (logits.logsumexp(1) - logits[torch.arange(batch), targets]).mean()
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=0)
    loss.backward()
    assert torch.isfinite(loss) and x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert torch.equal(p, weights[name]), name
    assert all(m.num_batches_tracked == 1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))


def test_bn_training_numeric_buffers_and_eval_freeze():
    model = CompactCNN().train()
    x = torch.linspace(-1, 2, 4 * 3 * 32 * 32).reshape(4, 3, 32, 32)
    first_bn_input = []
    handle = model.features[1].register_forward_pre_hook(
        lambda m, args: first_bn_input.append(args[0].detach().clone()))
    with torch.no_grad():
        model(x)
    handle.remove()
    z = first_bn_input[0].double()
    mean = z.mean((0, 2, 3))
    variance = ((z - mean[None, :, None, None]) ** 2).sum((0, 2, 3)) / (4 * 32 * 32 - 1)
    torch.testing.assert_close(model.features[1].running_mean.double(), .1 * mean, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(model.features[1].running_var.double(), .9 + .1 * variance, atol=1e-7, rtol=1e-6)
    assert all(m.num_batches_tracked == 1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    model.eval()
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        first, second = model(x), model(x)
    assert torch.equal(first, second) and torch.isfinite(first).all()
    assert torch.equal(rng, torch.get_rng_state())
    assert_states_equal(before, model.state_dict())
    model.train()
    with torch.no_grad():
        model(x)
    assert all(m.num_batches_tracked == 2 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    model.initialize_weights()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            assert m.num_batches_tracked == 0 and torch.count_nonzero(m.running_mean) == 0
            assert torch.equal(m.running_var, torch.ones_like(m.running_var))


def test_dropout_in_forward_has_inverted_scaling_and_eval_identity():
    model = CompactCNN().train()
    observed = []
    handle = model.dropout.register_forward_hook(
        lambda m, args, out: observed.append((args[0].detach().clone(), out.detach().clone())))
    x = torch.linspace(-2, 2, 4 * 3 * 32 * 32).reshape(4, 3, 32, 32)
    torch.manual_seed(123)
    with torch.no_grad():
        model(x)
    before, after = observed.pop()
    assert before.shape == after.shape == (4, 128)
    assert torch.all((after == 0) | (after == before * 1.25))
    # Fixed seed/fixture, structural mask assertions rather than frequency bounds.
    assert torch.any((before != 0) & (after == 0)) and torch.any(after != 0)
    model.eval()
    with torch.no_grad():
        model(x)
    before, after = observed.pop()
    handle.remove()
    assert torch.equal(before, after)


def test_strict_state_dict_serialized_roundtrip_including_bn_and_identity():
    model = CompactCNN().train()
    x = torch.linspace(-1, 1, 4 * 3 * 32 * 32).reshape(4, 3, 32, 32)
    with torch.no_grad():
        model(x)  # Exercise non-initial BN buffers, without any parameter update.
    model.eval()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, map_location='cpu', weights_only=True)
    assert state['_extra_state'] == {'architecture': 'compact_cnn_v1', 'num_classes': 10, 'dropout': .2}
    restored = CompactCNN().eval()
    result = restored.load_state_dict(state, strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert_states_equal(model.state_dict(), restored.state_dict())
    with torch.inference_mode():
        assert torch.equal(model(x), restored(x))


@pytest.mark.parametrize('target', [{'num_classes': 11}, {'dropout': .3}])
def test_incompatible_model_config_rejected(target):
    with pytest.raises(RuntimeError, match='incompatible CompactCNN architecture/config'):
        CompactCNN(**target).load_state_dict(CompactCNN().state_dict(), strict=True)


@pytest.mark.parametrize('damage', ['architecture', 'metadata_missing', 'missing_weight',
                                    'extra_weight', 'wrong_shape'])
def test_strict_state_dict_rejects_incompatible_architecture_or_keys(damage):
    state = CompactCNN().state_dict()
    match = r'Error\(s\) in loading state_dict'
    if damage == 'architecture':
        state['_extra_state']['architecture'] = 'compact_cnn_v2'
        match = 'incompatible CompactCNN architecture/config'
    elif damage == 'metadata_missing':
        del state['_extra_state']
    elif damage == 'missing_weight':
        del state['features.0.weight']
    elif damage == 'extra_weight':
        state['unexpected.weight'] = torch.zeros(1)
    else:
        state['classifier.weight'] = torch.zeros(11, 128)
    with pytest.raises(RuntimeError, match=match):
        CompactCNN().load_state_dict(state, strict=True)
