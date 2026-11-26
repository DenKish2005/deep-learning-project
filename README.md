# Robustness of Convolutional Neural Networks to Common Image Corruptions

CSCI 494 — Deep Learning, Fall 2026

Foundation prepared on 30 October 2026: isolated environment, exact dependency
versions, two complete YAML configs, strict schema/controlled comparison validation
and 116 passing offline tests. The planned experiment compares the same CNN trained
on clean images and with corruption-aware augmentation on CIFAR-10.

DATA-001 adds verified raw CIFAR-10 loading, explicit acquisition, a read-only
integrity CLI and a versioned local manifest. DATA-002 adds the persisted stratified
45k/5k split, seed494, verified NPZ/manifest and offline prepare/verify CLI.
DATA-003 adds train-only normalization, keyed transforms/order and deterministic
typed loaders. CORR-001 adds exact three-family severity mapping, explicit-generator
Gaussian noise and reusable original-test keyed CPU epsilon. CORR-002 adds reflect
Gaussian blur, progressive darkening and `apply_corruption`. CORR-003 adds the
train-only 10x16 corruption grid, PNG/PDF, source manifest and agent visual audit.
ROBUST-001 adds production CorruptionPolicy through the existing explicit
train-only hook. MODEL-001 adds CompactCNN v1 and independent synthetic model tests.
The core experiment completed on 26 November 2026 with six 40-epoch runs and
all 96 held-out test evaluations. Robust CNN improved mean corrupted accuracy
from **55.64% to 77.55%**. Clean test accuracy went from **87.17% to 86.60%**.
See the [results and figures](report/core_results.md), the
[six trained models](outputs/checkpoints/core-v1/README.md), and the
[training and evaluation commands](docs/REPRODUCIBILITY.md#core-training-and-evaluation).
Official CIFAR-10 is downloaded and verified locally: 50000 train / 10000 test,
RGB32×32, all archive/file checksums and class counts pass. Historical ROBUST-001 checks:
914 offline tests pass (19 opt-in tests skipped); separately 6 actual checks pass
with no skips: policy CPU/CUDA3, CORR references1, frozen grid2. Other13 historical
opt-ins were not rerun; earlier evidence retains its dates. Raw/split/stats and
config CLIs passed18.11.2026. ROBUST-001 technical PASS, full DoD PARTIAL;
human review, Person5/team visual review and formal approvals remain pending.

## Working CompactCNN (MODEL-001)

Synthetic technical **PASS**, full DoD **PARTIAL** (18.11.2026). Default950PASS/
20opt-in SKIP; separate executed CPU/CUDA2PASS/0SKIP. Exact288746 trainable params,
6Conv+BN+ReLU,3pools,GAP,Dropout.2,Linear128->10 raw logits. Caller sets torch seed
before CPU construction; protocol init runs at end of constructor without reseed.
Move to device afterward. Paired full initial states42/123/2026 match; BN/dropout,
finite CE/backward without parameter updates, strict identity/state roundtrip PASS.

```python
import torch
from src.models import CompactCNN

torch.manual_seed(42)  # Caller owns model RNG.
model = CompactCNN().eval()
with torch.inference_mode():
    logits = model(torch.zeros(4, 3, 32, 32))  # synthetic API example
```

[Evidence](report/model001_evidence.json), [replay commands](docs/REPRODUCIBILITY.md).
The MODEL-001 checks above were synthetic and scoped to the local backend.
The later paired training and test results are recorded in
[CURRENT_STATUS](docs/CURRENT_STATUS.md). Human review and formal approvals
remain pending.

## Working train-only policy (ROBUST-001)

`CorruptionPolicy(probabilities, max_train_severity=3)` accepts the frozen robust
probabilities `[.50,.17,.17,.16]` only. One event/image: identity, noise, blur or
brightness; corrupted severity uniform1..3, identity0. Caller supplies an explicit
same-device torch.Generator. Draw order: float64 scalar event uniform; corrupted
events only, int64 severity; noise only, float32 image-shaped Gaussian draw.
Validation precedes draws; possible reflect blur requires H,W>4. New storage,
no mutation/global RNG, metadata `event`/`severity`. Normalize afterward.

```python
from pathlib import Path
from src.corruptions import CorruptionPolicy
from src.data import make_loaders
from src.utils import load_config

cfg = load_config(Path('configs/robust_cnn.yaml'))
policy = CorruptionPolicy(cfg['augmentation']['probabilities'])
loaders = make_loaders(cfg, 42, 0, policy=policy, partitions=('train', 'val'))
# Call loaders['train'].dataset.set_epoch(epoch) before each new iterator.
```

No implicit policy is created by make_loaders; clean config requires None.
Numeric/frequency/RNG/pairing/spawn checks and actual limits:
[evidence](report/robust001_evidence.json), [commands](docs/REPRODUCIBILITY.md).
Run heavy checks sequentially: overlapping processes caused MemoryError/WinError1455;
unchanged sequential reruns passed. This does not establish training readiness.

## Working train-only corruption audit (CORR-003)

```powershell
.\.venv\Scripts\python.exe -m scripts.verify_frozen_corr003
$env:RUN_CORR_GRID='1'
.\.venv\Scripts\python.exe -m pytest tests/test_corruption_grid_local.py -q -s --basetemp=.venv/pytest-corr003-actual
Remove-Item Env:RUN_CORR_GRID
```

Actual [PNG](outputs/figures/corruption_examples.png), [PDF](outputs/figures/corruption_examples.pdf),
[source manifest](outputs/figures/corruption_examples.source.json) and
[evidence](report/corr003_evidence.json). Minimum original ID per class in persisted
train45k, class order:29/4/6/9/3/27/0/7/8/1. Raw CPU float32 [0,1], fixed local
audit epsilon per image across noise levels, no geometry/Normalize/auto-contrast.
The grid never decodes official test; the separate integrity CLI does.

CLI accepts only this explicit audit: `--root`, `--split-path`, `--output` directory,
`--verify-only`; no metrics or learning-curve API. Complete existing outputs are
verified read-only; partial/source/implementation/hash conflicts fail without repair.
Use a new empty output directory for a separately documented render. A publication
failure may leave a partial set; preserve it. Implementation fingerprints make
later source changes an intentional verification mismatch; do not rewrite this
frozen audit to conceal it. Full commands/limits: [reproduction](report/reproduction.md).
After ROBUST-001, current-source grid verification intentionally exits1 with an
implementation mismatch. The bounded frozen verifier authenticates source blobs
at21ede65 in ignored tmp and verifies the unchanged exports read-only. No branch,
worktree, artifact rewrite or relaxed source check is involved.

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
`normalization_sha256` contains the verified stats FILE SHA-256 after DATA-003.
Config validation
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

## Working normalization and loaders (DATA-003)

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_normalization --root data/raw
.\.venv\Scripts\python.exe -m scripts.prepare_normalization --root data/raw --verify-only
$env:RUN_CIFAR_INTEGRATION='1'
$env:RUN_CIFAR_SPLIT='1'
$env:RUN_CIFAR_NORMALIZATION='1'
.\.venv\Scripts\python.exe -m pytest tests/test_cifar10_local.py tests/test_split_local.py tests/test_normalization_local.py -q --basetemp=.venv/pytest-data003-actual
Remove-Item Env:RUN_CIFAR_INTEGRATION
Remove-Item Env:RUN_CIFAR_SPLIT
Remove-Item Env:RUN_CIFAR_NORMALIZATION
```

The CLI verifies existing raw original train and frozen split offline, then creates
or fully verifies/reuses [normalization.json](data/splits/normalization.json).
Verify-only writes nothing; opted-in missing/corrupt inputs fail. Stats use only
persisted45000 train IDs, raw RGB/255, float64 accumulation and population std.
Actual mean `[0.4913397271582242,0.48209963252314836,0.44655201729302807]`;
std `[0.2469009494849942,0.24334132835714897,0.2615066322799661]`.
FILE SHA-256 `332c2958a59036b68b6db2b71020dcd2aa0ab021f2a91a32cb351bf446095688`.
Both YAMLs declare this identity; frozen NPZ/manifest bytes and mtimes are unchanged.

`make_loaders(config, run_seed, epoch, *, policy=None, partitions=...)` verifies
declared FILE hashes, raw train, split and stats. Batches are float32 NCHW,
int64 labels/IDs, drop_last=False. Train uses keyed pad/crop/flip, optional injected
pixel-space policy, then Normalize; val/test are sequential and deterministic.
Call `loader.dataset.set_epoch(epoch)` before iterator creation. Future training
must request `partitions=('train','val')` to avoid decoding test. Augmentation-enabled
config requires an explicit policy; ROBUST-001 production integration is technically
verified, while full DoD/formal review and training readiness remain separate gates.
Workers0 reference and Windows spawn2 equality passed on synthetic fixtures and129
actual train IDs across epochs/batch sizes. Full coverage45k/5k/10k and label
alignment passed. [Contracts](docs/TECHNICAL_ARCHITECTURE.md),
[actual evidence](report/data003_evidence.json), [commands/results](report/reproduction.md).

## Working Gaussian noise and keyed epsilon (CORR-001)

`severity_params(kind, severity)` returns fresh parameter dicts for noise/blur/
brightness levels 1..5. Clean is separate.
`gaussian_noise(x, sigma, generator)` accepts dense CPU/CUDA float32 RGB CHW pixels
in [0,1], positive H/W and finite nonnegative Python sigma. It returns new pixels
with the same shape/device/dtype, without global RNG use. Invalid inputs fail
before generator draws; sigma0 clones without drawing. Normalize afterward.

```python
from src.corruptions import gaussian_noise, keyed_test_generator, keyed_test_epsilon

namespace = 'cifar10-python/original-test'
epsilon = keyed_test_epsilon(71, namespace=namespace)  # CPU float32 [3,32,32]
# x is the raw original-test image71 converted to CPU float32 CHW /255.
# Recreate this generator for every image/condition; do not reuse a consumed stream.
# noisy = gaussian_noise(x, .08, keyed_test_generator(71, namespace=namespace))
```

The full original-test namespace and Python ID0..9999 are required. The seed uses
existing `stable_seed(100494,'cifar10-test',id,'gaussian_noise','v1')`; severity and
model/run seed are absent. This is a primitive boundary; CorruptedDataset,
evaluator remain future tasks. CPU bitwise references cover
only the locked local environment. CUDA primitive PASS does not imply CPU/CUDA
equality; canonical test epsilon stays CPU.

```powershell
$env:RUN_CIFAR_CORRUPTIONS='1'
$env:RUN_CORR_CUDA='1'
.\.venv\Scripts\python.exe -m pytest tests/test_corruptions_local.py tests/test_corruptions_cuda.py -q -s --basetemp=.venv/pytest-corr001-actual
Remove-Item Env:RUN_CIFAR_CORRUPTIONS
Remove-Item Env:RUN_CORR_CUDA
```

Actual CIFAR checks require existing verified raw/split/stats; no downloads.
Missing/corrupt inputs fail when opted in. CUDA opt-in also fails if unavailable.
[Evidence](report/corr001_evidence.json) and [saved regression tensors](report/corr001_reference.npz)
record fixed train IDs/noise-before-Normalize and fixed test-key inputs, without
model runs or test-based tuning. Full CORR-001 DoD remains PARTIAL pending review
and approvals.

## Working blur, darkening and dispatcher (CORR-002)

`gaussian_blur(x, sigma, kernel_size)` uses normalized discrete separable weights,
reflect padding and independent RGB channels. Sigma0/kernel1 clone; a positive odd
kernel must have radius smaller than both image dimensions, even for identity.
`darken(x, factor)` multiplies and clamps; Python finite factor0..1 only, with
factor1 identity and factor0 black. Both share the noise pixel contract.

`apply_corruption(x, kind, severity, generator=None)` supports `clean` at Python
integer0 and the three protocol families at1..5. Noise requires an explicit matching
generator; use existing keyed_test_generator for fixed evaluation input identity.
Clean/blur/brightness accept None or an unused torch.Generator on any device,
without consuming it. Wrong generator types fail. Normalize after dispatch.

```python
from src.corruptions import apply_corruption
import torch

pixels = torch.full((3, 32, 32), .5)
blurred = apply_corruption(pixels, 'gaussian_blur', 3)
darkened = apply_corruption(pixels, 'brightness', 3)
clean = apply_corruption(pixels, 'clean', 0)  # new storage, validated pixels
```

Historical CORR-002 replay: use the CORR-001 opt-in environment above and add
`tests/test_corr002_local.py` to the pytest file list. Combined result18.11.2026:
6PASS/0FAIL/0SKIP; default777PASS/14opt-in SKIP. Tests cover all16conditions on
17persisted train IDs, epochs0/1, and spawn0/2 batch7/5 for clean+severity3 families.
CPU and CUDA NumPy-oracle blur error max1.78814e-7. Fixed probe callables do not
implement the production training policy. [Evidence](report/corr002_evidence.json),
[actual reference](report/corr002_reference.npz), [commands](report/reproduction.md).
Full CORR-002 DoD PARTIAL pending formal approvals and independent human review;
CORR-003 independent human visual review, formal ROBUST-001 acceptance and general
REPRO-001 remain open.

## Project documents and next work

Start with [MASTER_PROJECT_PLAN](docs/MASTER_PROJECT_PLAN.md),
[CURRENT_STATUS](docs/CURRENT_STATUS.md) and [EXPERIMENT_PROTOCOL](docs/EXPERIMENT_PROTOCOL.md).
[Source registry](report/course_materials/README.md) records local guidelines,
CSR and proposal candidates; submitted version, current rubric/LMS confirmation,
AI-policy conflict and team approvals remain pending.

The next report task is to analyze per-class failures and connect the measured
results to the four research questions. The final figures, independent replay,
and team review are pending. See the [analysis handoff](docs/NEXT_STEP_PROMPT.md)
and [final deliverables](docs/FINAL_DELIVERABLES.md).
All commits and ordinary pushes use `main`, per the user's current instruction.
