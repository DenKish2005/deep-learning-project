# Robustness of Convolutional Neural Networks to Common Image Corruptions

CSCI 494 — Deep Learning, Fall 2026

Foundation prepared on 30 October 2026: isolated environment, exact dependency
versions, two complete YAML configs, strict schema/controlled comparison validation
and 116 passing offline tests. The planned experiment compares the same CNN trained
on clean images and with corruption-aware augmentation on CIFAR-10.

Data loading, corruptions, CNN, training, evaluation and plotting remain future
work. No CIFAR-10 download or training has been performed.

## Working installation and checks

Verified locally on Windows AMD64 with Python **3.12.10**, torch **2.9.0+cu126**,
torchvision **0.24.0+cu126**, pytest **9.2.0**. CPU and CUDA smoke passed on RTX
3060 Laptop / driver555.97 / CUDA12.6 / cuDNN91002. The lock includes38 packages;
it targets this environment, with CPU operations tested in the same CUDA wheel.

Run from the repository root in PowerShell, using Python3.12.10. Activation is
unnecessary. The local temporary directory avoids the sandbox's default-temp
permission error; only `.venv` receives installed packages.

```powershell
python --version
New-Item -ItemType Directory -Force tmp/setup | Out-Null
$env:TEMP=(Resolve-Path tmp/setup).Path
$env:TMP=$env:TEMP
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --disable-pip-version-check
.\.venv\Scripts\python.exe -m pip check
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:MPLCONFIGDIR=(Resolve-Path .venv).Path + '/mplconfig'
.\.venv\Scripts\python.exe -m scripts.check_environment
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.venv/pytest-tmp
.\.venv\Scripts\python.exe -m src.utils --configs configs/clean_cnn.yaml configs/robust_cnn.yaml --controlled-pair
```

The packages were installed in a new isolated environment and tested. Reinstalling
the resulting lock in that environment passed without changes; an independent
fresh-machine replay remains pending. This is a version lock without wheel hashes.
See [installation sources and limits](docs/REPRODUCIBILITY.md), the
[environment snapshot](report/setup_environment.json) and [check record](report/reproduction.md).

## Working config interface

`load_config`, `validate_config` and `validate_controlled_configs` live in
`src/utils.py`. Both YAMLs are standalone preparation configs using schema v1.
Missing/unknown fields, duplicate YAML keys, unsafe YAML tags, incorrect types,
nonfinite/out-of-range numbers, invalid probabilities and altered severity maps
fail with a field-specific `ConfigError`. The CLI returns exit2 for invalid config.

The proposed core protocol fixes scientific settings. In the controlled pair only
`model_name`, `augmentation.enabled`, `augmentation.probabilities` and output paths
may differ. Dataset paths/hashes and runtime controls must match. Scientific
changes require coordinated protocol/schema/tests updates.

Dataset paths are declarations: `split_sha256` and `normalization_sha256` remain
`null` until artifacts exist and are verified in DATA-002/003. Config validation
does not create, download, open or authenticate these artifacts, and does not
assert training readiness. Artifact verification is a future training gate.

## Project documents and next work

Start with [MASTER_PROJECT_PLAN](docs/MASTER_PROJECT_PLAN.md),
[CURRENT_STATUS](docs/CURRENT_STATUS.md) and [EXPERIMENT_PROTOCOL](docs/EXPERIMENT_PROTOCOL.md).
[Source registry](report/course_materials/README.md) records local guidelines,
CSR and proposal candidates; submitted version, current rubric/LMS confirmation,
AI-policy conflict and team approvals remain pending.

Next: DATA-001 integrity/offline loading, then DATA-002 persisted stratified split
and DATA-003 train-only normalization/loaders, alongside required REPRO-001 identity
utilities. No pipeline CLI is implemented yet. Commands in the future-pipeline
section of [REPRODUCIBILITY](docs/REPRODUCIBILITY.md) describe planned interfaces.
