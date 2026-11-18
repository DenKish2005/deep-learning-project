"""Executed CPU reference / explicit CUDA model QA on synthetic tensors only.

No CIFAR inference, optimizer, backend setting changes or training runs. Evidence
hashes describe this locked runtime; no cross-device output equality is required.
"""

import copy
import hashlib
import io
import json
import os
import platform

import pytest
import torch
from torch import nn

from src.models import CompactCNN
from src.utils import load_config
from tests.test_models import assert_states_equal


def state_sha256(state):
    """Test-only stream: sorted keys, compact JSON headers + LF, tensor C bytes."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if isinstance(value, torch.Tensor):
            a = value.detach().cpu().contiguous().numpy()
            header = {'key': key, 'dtype': a.dtype.str, 'shape': list(a.shape)}
            digest.update(json.dumps(header, sort_keys=True, separators=(',', ':')).encode() + b'\n')
            digest.update(a.tobytes())
        else:
            digest.update(json.dumps({'key': key, 'value': value}, sort_keys=True,
                                     separators=(',', ':')).encode() + b'\n')
    return digest.hexdigest()


@pytest.mark.parametrize('device_name', [
    'cpu',
    pytest.param('cuda:0', marks=pytest.mark.skipif(
        os.environ.get('RUN_MODEL_CUDA') != '1',
        reason='Actual model CUDA requires RUN_MODEL_CUDA=1')),
])
def test_executed_model_device_contract(device_name):
    cuda = device_name.startswith('cuda')
    if cuda:
        assert torch.cuda.is_available(), 'Opted-in MODEL CUDA must be available'
    device = torch.device(device_name)
    with torch.random.fork_rng(devices=[0] if cuda else []):
        initial = {}
        for seed in (42, 123, 2026):
            paired = []
            for path in ('configs/clean_cnn.yaml', 'configs/robust_cnn.yaml'):
                cfg = load_config(path)
                torch.manual_seed(seed)  # Caller seeds before CPU construction.
                model = CompactCNN(cfg['num_classes'], cfg['dropout'])
                paired.append(copy.deepcopy(model.state_dict()))
            assert_states_equal(*paired)
            initial[str(seed)] = state_sha256(paired[0])
        assert len(set(initial.values())) == 3
        model = model.to(device).eval()
        shapes = {}
        eval_hashes = {}
        initial_state = copy.deepcopy(model.state_dict())
        rng_before = (torch.get_rng_state().clone(),
                      torch.cuda.get_rng_state(device).clone() if cuda else None)
        for batch in (1, 4, 128):
            x = torch.linspace(-2, 2, batch * 3 * 32 * 32).reshape(batch, 3, 32, 32).to(device)
            original = x.clone()
            with torch.inference_mode():
                first, second = model(x), model(x)
            assert first.shape == (batch, 10) and first.device == device
            assert first.dtype == torch.float32 and torch.isfinite(first).all()
            assert torch.equal(first, second) and torch.equal(x, original)
            shapes[str(batch)] = list(first.shape)
            eval_hashes[str(batch)] = state_sha256({'logits': first})
        assert_states_equal(initial_state, model.state_dict())
        assert torch.equal(rng_before[0], torch.get_rng_state())
        if cuda:
            assert torch.equal(rng_before[1], torch.cuda.get_rng_state(device))
        params_before = {k: p.detach().clone() for k, p in model.named_parameters()}
        model.train()
        x = torch.linspace(-2, 2, 4 * 3 * 32 * 32).reshape(4, 3, 32, 32).to(device).requires_grad_()
        labels = torch.tensor([0, 3, 6, 9], device=device)
        loss = nn.CrossEntropyLoss()(model(x), labels)
        loss.backward()
        assert torch.isfinite(loss) and x.grad is not None and torch.isfinite(x.grad).all()
        for key, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), key
            assert torch.equal(p, params_before[key]), key
        assert all(m.num_batches_tracked == 1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = CompactCNN().to(device).eval()
        restored.load_state_dict(torch.load(buffer, map_location=device, weights_only=True), strict=True)
        model.eval()
        assert_states_equal(model.state_dict(), restored.state_dict())
        with torch.inference_mode():
            assert torch.equal(model(x), restored(x))
        if cuda:
            torch.cuda.synchronize(device)
        print('MODEL001_EVIDENCE ' + json.dumps({
            'device': device_name, 'python': platform.python_version(), 'torch': torch.__version__,
            'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(device) if cuda else None,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'initial_state_hashes': initial, 'eval_shapes': shapes, 'eval_hashes': eval_hashes,
            'ce_loss': loss.item(), 'finite_all_gradients': True,
            'parameters_unchanged_after_backward': True, 'strict_roundtrip': True,
            'eval_state_and_rng_unchanged': True, 'optimizer_steps': 0,
            'scope': 'synthetic model QA; no CIFAR inference or training',
        }, sort_keys=True))
