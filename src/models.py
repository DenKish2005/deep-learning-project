"""CompactCNN v1: raw CIFAR-sized logits, with caller-owned model RNG.

Construct on CPU after the caller sets its torch seed, then move to the chosen
device. Layer constructors consume their normal RNG draws; once all layers
exist, initialize_weights() applies the protocol initialization exactly once.
No seed, dataset, corruption policy, optimizer or training loop is owned here.
"""

import math

from torch import Tensor, nn


class CompactCNN(nn.Module):
    """Six Conv/BN/ReLU layers, three pools and a GAP/dropout/linear head.

    Config validation fixes the study to 10 classes and dropout .2. The module
    permits other positive class counts and dropout in [0,1] for model-level QA;
    those instances are not protocol-compatible experiment configurations.
    """

    architecture = 'compact_cnn_v1'

    def __init__(self, num_classes: int = 10, dropout: float = .2):
        super().__init__()
        if type(num_classes) is not int or num_classes <= 0:
            raise ValueError('num_classes must be a positive Python int')
        if (type(dropout) not in (int, float) or not math.isfinite(dropout)
                or not 0 <= dropout <= 1):
            raise ValueError('dropout must be a finite Python number in [0,1]')
        self.num_classes = num_classes
        layers = []
        channels = 3
        for width in (32, 64, 128):
            for _ in range(2):
                layers.extend([
                    nn.Conv2d(channels, width, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(width, eps=1e-5, momentum=.1, affine=True),
                    nn.ReLU(),
                ])
                channels = width
            layers.append(nn.MaxPool2d(2, stride=2))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes, bias=True)
        self.initialize_weights()

    def get_extra_state(self) -> dict:
        """Bind model identity to the standard PyTorch state_dict."""
        return {'architecture': self.architecture, 'num_classes': self.num_classes,
                'dropout': self.dropout.p}

    def set_extra_state(self, state) -> None:
        """Reject incompatible identity on load; use load_state_dict(strict=True).

        This is model-level compatibility only, not experiment provenance or a
        checkpoint/resume framework. Like PyTorch loading, failures are not a
        transactional rollback; load into a fresh compatible model instance.
        """
        expected = self.get_extra_state()
        if (type(state) is not dict or state.keys() != expected.keys()
                or any(type(state[k]) is not type(v) or state[k] != v
                       for k, v in expected.items())):
            raise RuntimeError(f'incompatible CompactCNN architecture/config: expected {expected}')

    def initialize_weights(self) -> None:
        """Reinitialize using the current torch RNG, including fresh BN buffers.

        Explicit calls consume more caller RNG draws. No hidden reseeding occurs.
        Initialization precedes any forward/backward check or future optimizer.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                layer.reset_running_stats()
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0., std=.01)
                nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Return raw logits for a nonempty N x 3 x 32 x 32 tensor."""
        if (not isinstance(x, Tensor) or x.ndim != 4 or x.shape[0] < 1
                or tuple(x.shape[1:]) != (3, 32, 32)):
            raise ValueError('CompactCNN input must have shape N x 3 x 32 x 32, N > 0')
        x = self.flatten(self.avgpool(self.features(x)))
        return self.classifier(self.dropout(x))
