# Robustness of Convolutional Neural Networks to Common Image Corruptions

CSCI 494 — Deep Learning, Fall 2026

Foundation prepared on 30 October 2026: isolated environment, exact dependency
versions, two complete YAML configs, strict schema/controlled comparison validation
and 116 passing offline tests. The planned experiment compares the same CNN trained
on clean images and with corruption-aware augmentation on CIFAR-10.

DATA-001 adds verified raw CIFAR-10 loading, explicit acquisition, a read-only
integrity CLI and a versioned local manifest. DATA-002 adds the persisted stratified
45k/5k split, seed494, verified NPZ/manifest and offline prepare/verify CLI.
Normalization, corruptions, CNN, training, evaluation and plotting remain future work.
No training has run.
Official CIFAR-10 is downloaded and verified locally: 50000 train / 10000 test,
RGB32×32, all archive/file checksums and class counts pass. Current checks:
282 offline tests pass (5 opt-in tests skipped); separately all 5 actual tests pass.

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

Both configs now contain the verified NPZ file SHA-256 in `split_sha256`;
`normalization_sha256` remains `null` until DATA-003. Config validation
does not create, download, open or authenticate these artifacts, and does not
assert training readiness. Artifact verification is a future training gate.

## Working raw CIFAR-10 interface (DATA-001)

Run from the repository root using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-only
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.venv/pytest-tmp
# Separate opt-in: missing/corrupt local data is a failure, never a download/skip.
$env:RUN_CIFAR_INTEGRATION='1'
.\.venv\Scripts\python.exe -m pytest tests/test_cifar10_local.py -q --basetemp=.venv/pytest-cifar-local
Remove-Item Env:RUN_CIFAR_INTEGRATION
```

`load_cifar10(root: Path, train: bool, download=False)` in `src/data.py` returns
torchvision CIFAR10 with raw PIL RGB 32×32 images and integer labels. Default
loading is offline. Acquisition uses only official Toronto HTTPS endpoints and
upstream archive/file MD5 references, with SHA-256 observations in the manifest.
Both partitions' files are checked before unpickling; only the requested partition
is decoded by the loader. Full integrity inspection checks 50000/10000 samples,
10-class mapping and 5000/1000 samples per class. Test inspection is solely integrity.

The acquire command reuses valid existing data/archive and creates
`data/raw/cifar10_integrity_v1.json` once. Historical acquisition of preexisting
artifacts is recorded as unknown. Repeated runs recheck data and preserve the
manifest. `--verify-only` performs no dataset/manifest writes and can verify valid
extracted files without an archive, explicitly reporting `archive_verified=false`.
It never infers verification from a previous manifest.

Missing/corrupt/conflicting artifacts cause a nonzero exit with the failing path
and recovery action. Preserve damaged artifacts and retry with a new empty root;
automatic repair/overwrite is deliberately absent. Failed acquisition staging is
retained for diagnosis. Raw data, archives and the manifest stay ignored by Git.
Actual check results and source limits: [reproduction record](report/reproduction.md).

## Working persisted split (DATA-002)

```powershell
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --prepare-split --split-seed 494
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-split
# Separate explicit opt-in; requires actual raw data and versioned split.
$env:RUN_CIFAR_INTEGRATION='1'
$env:RUN_CIFAR_SPLIT='1'
.\.venv\Scripts\python.exe -m pytest tests/test_cifar10_local.py tests/test_split_local.py -q --basetemp=.venv/pytest-data002-actual
Remove-Item Env:RUN_CIFAR_INTEGRATION
Remove-Item Env:RUN_CIFAR_SPLIT
```

Split modes use existing raw data offline; only original train labels determine
the split. Default `--split-dir data/splits`, shared seed494. The original test10k
is a separate partition. `--verify-only` still checks raw data alone. Prepare does
not create normalization/loaders. Invalid flags exit2; data/artifact errors exit1.

`create_split(targets, split_seed, out_dir)` returns the schema-v1 manifest;
`load_split(npz_path, targets)` returns sorted `<i8` train/val arrays. Actual counts
are45000/5000, exactly4500/500 per class, disjoint/exhaustive original IDs0..49999.
Existing pairs are fully verified and reused without changing hashes/file mtimes.
Tampered/partial pairs and interrupted-write locks/staging fail with preservation
and explicit recovery guidance. [Full artifact contract](docs/TECHNICAL_ARCHITECTURE.md#split-artifact-contract-v1).

Versioned NPZ: [cifar10_split_v1.npz](data/splits/cifar10_split_v1.npz),76550 bytes.
File SHA-256: `1dc8e61a237d7e8fd3ac3c2dea4787e6a82f8f235d74993386af9074f6e1d284`.
Separate semantic array/label hashes, provenance and before/after mtimes:
[manifest](data/splits/split_manifest.json), [actual evidence](report/data002_evidence.json).
The saved indices are canonical; cross-version/platform byte-identical NPZ
regeneration is not promised. Historical raw acquisition manifest is unchanged.

## Project documents and next work

Start with [MASTER_PROJECT_PLAN](docs/MASTER_PROJECT_PLAN.md),
[CURRENT_STATUS](docs/CURRENT_STATUS.md) and [EXPERIMENT_PROTOCOL](docs/EXPERIMENT_PROTOCOL.md).
[Source registry](report/course_materials/README.md) records local guidelines,
CSR and proposal candidates; submitted version, current rubric/LMS confirmation,
AI-policy conflict and team approvals remain pending.

Next: DATA-003 train-only normalization/loaders, with the unimplemented CORR-001
dependency explicit in [next-step prompt](docs/NEXT_STEP_PROMPT.md). Full DATA-001/002
DoD remains PARTIAL until independent human review; agent checks do not replace it.
All commits and ordinary pushes use `main`, per the user's current instruction.
Commands in the future-pipeline section of [REPRODUCIBILITY](docs/REPRODUCIBILITY.md)
describe planned interfaces beyond the working raw-data/split commands above.
