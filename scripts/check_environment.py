"""Offline dependency/backend smoke; synthetic tensors only, no project model/data."""

import importlib
from importlib.metadata import version
import json
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import sys


def check_backend(device):
    import torch
    from torchvision.ops import nms

    torch.manual_seed(42)
    x = torch.randn(2, 3, 8, 8, device=device, requires_grad=True)
    weight = torch.randn(4, 3, 3, 3, device=device, requires_grad=True)
    y = torch.nn.functional.conv2d(x, weight, padding=1)
    loss = y.square().mean()
    loss.backward()
    assert torch.isfinite(loss).item()
    assert all(t.grad is not None and torch.isfinite(t.grad).all().item()
               for t in (x, weight))
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 1., 1.]], device=device)
    scores = torch.tensor([0.9, 0.8], device=device)
    assert nms(boxes, scores, 0.5).numel() == 2
    if device == 'cuda':
        torch.cuda.synchronize()
    return {'forward_backward': 'PASS', 'torchvision_nms': 'PASS'}


def main():
    if sys.prefix == sys.base_prefix:
        raise SystemExit('Run this check with the isolated .venv interpreter.')
    import torch
    import torchvision
    from torchvision.io import decode_png, encode_png

    packages = {
        'torch': 'torch', 'torchvision': 'torchvision', 'numpy': 'numpy',
        'pandas': 'pandas', 'scikit-learn': 'sklearn', 'matplotlib': 'matplotlib',
        'tqdm': 'tqdm', 'PyYAML': 'yaml', 'pytest': 'pytest', 'Pillow': 'PIL',
    }
    for module in packages.values():
        importlib.import_module(module)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'ieee'
    image = torch.arange(3 * 8 * 8, dtype=torch.uint8).reshape(3, 8, 8)
    assert torch.equal(decode_png(encode_png(image)), image)
    result = {
        'schema_version': 1,
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.system(),
        'machine': platform.machine(),
        'packages': {name: version(name) for name in packages},
        'torch_cuda_build': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'png_roundtrip': 'PASS', 'cpu': check_backend('cpu'),
        'cuda_available': torch.cuda.is_available(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cuda_matmul_fp32_precision': torch.backends.cuda.matmul.fp32_precision,
        'cudnn_conv_fp32_precision': torch.backends.cudnn.conv.fp32_precision,
    }
    lock = Path(__file__).resolve().parents[1] / 'requirements-lock.txt'
    if lock.exists():
        normalized = lock.read_text(encoding='utf-8').replace('\r\n', '\n').encode('utf-8')
        result['requirements_lock_sha256_lf'] = hashlib.sha256(normalized).hexdigest()
    try:
        result['nvidia_driver'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            text=True, timeout=10).strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        result['nvidia_driver'] = None
    if result['cuda_available']:
        props = torch.cuda.get_device_properties(0)
        result['gpu'] = {
            'name': props.name, 'vram_bytes': props.total_memory,
            'capability': list(torch.cuda.get_device_capability(0)),
            'architectures': torch.cuda.get_arch_list(),
            'checks': check_backend('cuda'),
        }
    else:
        result['gpu'] = {'checks': 'SKIP: CUDA unavailable; CPU-only validation'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
