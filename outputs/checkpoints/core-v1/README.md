# Core CIFAR-10 model checkpoints

These are the six validation-selected CompactCNN checkpoints from 26 November
2026. Each `best.pt` is 1,173,297 bytes. The six models total 7,039,782 bytes.

| Seed | Clean CNN | Robust CNN |
|---:|---|---|
| 42 | [best.pt](core-v1-clean_cnn-s42-20abe0df/best.pt) | [best.pt](core-v1-robust_cnn-s42-52bf0744/best.pt) |
| 123 | [best.pt](core-v1-clean_cnn-s123-20abe0df/best.pt) | [best.pt](core-v1-robust_cnn-s123-52bf0744/best.pt) |
| 2026 | [best.pt](core-v1-clean_cnn-s2026-20abe0df/best.pt) | [best.pt](core-v1-robust_cnn-s2026-52bf0744/best.pt) |

Each directory includes the original resolved config, environment, 40-epoch
training history, and run manifest. The manifest records SHA-256 for the bundled
files and the recovery checkpoint. Recovery checkpoints remain in the original
run directories under `outputs/runs/`. Use those complete directories when
resuming training or invoking the six-run evaluation launcher.

All six manifests record base commit
`5de701cf34fe6786a68db82b4fe36b905d0f7186` with local changes. The tree at
`4ac4ff7` matches the recorded training snapshot SHA-256
`1a8093d1dbca2bff5089bd949355769ec9b0778f8baba58528c0fc030747da5f`.
The selected checkpoint bytes and original manifests are preserved.

To score a bundled model with verified local CIFAR-10, run from the repository
root with that model's config and a fresh output directory.

```bash
python -m src.evaluate \
  --checkpoint outputs/checkpoints/core-v1/core-v1-clean_cnn-s42-20abe0df/best.pt \
  --config outputs/checkpoints/core-v1/core-v1-clean_cnn-s42-20abe0df/config_resolved.yaml \
  --output-dir outputs/replay/clean-s42 --device auto
```

This runs all 16 test conditions for the selected checkpoint. Repeat with the
corresponding directory and a fresh output path to score a different model or
backend. See the [reproducibility guide](../../../docs/REPRODUCIBILITY.md).
