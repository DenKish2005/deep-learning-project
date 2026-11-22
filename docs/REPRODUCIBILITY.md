# Воспроизводимость: проверенный setup и будущий pipeline

## Working MODEL-001 checks — 18.11.2026

Existing .venv only, no installs/updates/backend changes. Heavy checks sequential.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_models.py -q
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.venv/pytest-model001-replay
$env:RUN_MODEL_CUDA='1'
try {
    .\.venv\Scripts\python.exe -m pytest tests/test_models_devices.py -q -s
} finally {
    Remove-Item Env:RUN_MODEL_CUDA
}
```

Observed950default PASS/20opt-in SKIP, separate2device PASS/0SKIP (CPU and RTX3060
Laptop). Omit flag for CPU-only1PASS/1SKIP; that is not CUDA PASS. Explicit CUDA
unavailable is a failure. Synthetic linspace inputs only, no CIFAR inference or
optimizer steps. Initial state hashes include BN buffers and architecture/config
extra state; test-only encoding in [evidence](../report/model001_evidence.json).

Caller torch.manual_seed(run_seed), then CompactCNN on CPU, then move to device.
Constructor layer draws precede final explicit protocol initialization; no reseed
inside model. Record initial hash before any checks. Strict load into a fresh
compatible model; state_dict `_extra_state` carries architecture/classes/dropout.
weights_only=True, explicit map_location; no general checkpoint/resume API added.
Device checks leave backend flags unchanged, observed determinism flags false;
repeat eval passed on each device, no deterministic-training/cross-device claim.
Historical ROBUST/DATA/CORR opt-ins not rerun. [Record](../report/reproduction.md).

## Работающие ROBUST-001 проверки — 18.11.2026

Use the existing .venv, without install/update/backend changes. Run heavy pytest
commands sequentially: overlapping processes caused NumPy MemoryError and spawn
WinError1455; unchanged sequential reruns passed. Create ignored tmp/robust001
if JUnit logs are desired. These commands verify inputs/policy only, not training.

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-only
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-split
.\.venv\Scripts\python.exe -m scripts.prepare_normalization --root data/raw --verify-only
.\.venv\Scripts\python.exe -m scripts.verify_frozen_corr003
.\.venv\Scripts\python.exe -m src.utils --configs configs/clean_cnn.yaml configs/robust_cnn.yaml --controlled-pair
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.venv/pytest-robust001-final
$env:RUN_ROBUST_POLICY='1'
$env:RUN_CORR_CUDA='1'
.\.venv\Scripts\python.exe -m pytest tests/test_policy_local.py tests/test_policy_cuda.py -q -s --basetemp=.venv/pytest-robust001-actual-serial
Remove-Item Env:RUN_ROBUST_POLICY
Remove-Item Env:RUN_CORR_CUDA
$env:RUN_CIFAR_CORRUPTIONS='1'
.\.venv\Scripts\python.exe -m pytest tests/test_corr002_local.py::test_actual_all_conditions_before_normalize_and_saved_references -q -s --basetemp=.venv/pytest-robust001-references-serial
Remove-Item Env:RUN_CIFAR_CORRUPTIONS
$env:RUN_CORR_GRID='1'
.\.venv\Scripts\python.exe -m pytest tests/test_corruption_grid_local.py -q -s --basetemp=.venv/pytest-robust001-grid-final
Remove-Item Env:RUN_CORR_GRID
```

Actual results:914default PASS/19opt-in SKIP; separately3policy CPU/CUDA +1old
reference +2grid PASS,0FAIL/0SKIP. Other13historical opt-ins not rerun; preserve
their dates. CPU-only policy replay omits CUDA flag/file and yields2actual cases;
this is not CUDA PASS. An opted-in missing/corrupt input or unavailable CUDA fails.
Policy tests accept a repository-relative CIFAR10_ROOT override; split/stats remain
the frozen repository paths. No download/repair or synthetic actual substitution.

Frozen CORR-003 verifier extracts only4authenticated implementation blobs and
src/__init__.py via git show at21ede652d2d1638ec69e4637f4707bcebf3c8224 into ignored
tmp, then invokes the original strict read-only verifier on actual inputs/exports.
No branch/worktree or copied dataset. Current `scripts.make_plots --corruption-grid
--verify-only` intentionally exits1 with implementation mismatch after policy
source edits; do not regenerate/rebaseline old exports to hide it. Actual grid
tests preserve strict mismatch and full recorded-source replay simultaneously.
[Evidence](../report/robust001_evidence.json), [record](../report/reproduction.md).

## Historical CORR-003 проверки — 18.11.2026

Existing .venv, no installation/update/backend changes. From repository root:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-only
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-split
.\.venv\Scripts\python.exe -m scripts.prepare_normalization --root data/raw --verify-only
.\.venv\Scripts\python.exe -m scripts.make_plots --corruption-grid --verify-only
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.venv/pytest-corr003-final --junitxml=tmp/corr003/default.xml
$env:RUN_CORR_GRID='1'
.\.venv\Scripts\python.exe -m pytest tests/test_corruption_grid_local.py -q -s --basetemp=.venv/pytest-corr003-actual --junitxml=tmp/corr003/actual.xml
Remove-Item Env:RUN_CORR_GRID
pdfinfo outputs/figures/corruption_examples.pdf
pdftoppm -f 1 -singlefile -scale-to 2200 -png outputs/figures/corruption_examples.pdf tmp/corr003/pdf-preview
```

Create ignored tmp/corr003 first if needed for local logs/PDF previews. Default
815PASS/0FAIL/16SKIP; actual2PASS/0FAIL/0SKIP. Existing14actual DATA/corruption/CUDA
cases not repeated; their earlier evidence retains its dates. Opt-in root override
`CIFAR10_ROOT`, split-directory override `CIFAR10_SPLITS`; output is the curated
repository grid. Missing/corrupt input fails, never skips or downloads.

Initial actual creation used `-m scripts.make_plots --corruption-grid --root data/raw
--output outputs/figures`. Complete default reruns verify actual inputs, source
implementation, pixel hash and export hashes read-only. `--verify-only` also
requires all three artifacts; neither path repairs partial/mismatched outputs.
An intentional separate render requires a new empty `--output` directory. Staging
precedes exclusive publication; failure during publication may leave a partial
set. Preserve it for diagnosis. Only curated PNG/PDF/source files are unignored.
JSON LF and binary attributes preserve tracked byte hashes across checkouts.

Minimum train IDs/class29/4/6/9/3/27/0/7/8/1, class order0..9. Audit key:
`stable_seed(494,'cifar10-python/original-train/corr003-audit',original_id,'gaussian_noise','v1')`.
It was frozen before viewing; fresh CPU generator per noise level, shared epsilon.
Raw CPU float32[10,16,3,32,32], no geometry/Normalize/auto-contrast. Source manifest
records input/implementation/params/key/pixel/output identities. LF-normalized
source hashes deliberately make later implementation edits a verification mismatch;
do not regenerate this historical audit merely to pass after ROBUST-001 edits.

Agent inspected PNG and rendered PDF; independent Person5/team review PENDING.
Numeric160cell replay max blur error1.7881393432617188e-7;480RGB PDF/PNG samples
identical, source quantization error<1/255. [Evidence](../report/corr003_evidence.json),
[record/hashes](../report/reproduction.md). All23protected hashes/bytes/mtimes and
frozen source regions preserved. No cross-platform/version renderer byte equality
or independent-machine reproduction claimed; general REPRO-001 remains open.

## Historical CORR-002 проверки — 18.11.2026

Use existing .venv without install/update/backend changes, from repository root:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-only
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-split
.\.venv\Scripts\python.exe -m scripts.prepare_normalization --root data/raw --verify-only
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.venv/pytest-corr002-final --junitxml=tmp/corr002/default.xml
$env:RUN_CIFAR_CORRUPTIONS='1'
$env:RUN_CORR_CUDA='1'
.\.venv\Scripts\python.exe -m pytest tests/test_corr002_local.py tests/test_corruptions_local.py tests/test_corruptions_cuda.py -q -s --basetemp=.venv/pytest-corr002-actual --junitxml=tmp/corr002/actual.xml
Remove-Item Env:RUN_CIFAR_CORRUPTIONS
Remove-Item Env:RUN_CORR_CUDA
```

Default777PASS/0FAIL/14SKIP; separate6PASS/0FAIL/0SKIP. CPU-only replay omits CUDA
environment variable and CUDA file:4CIFAR cases, not a GPU PASS. Opt-in missing/
corrupt inputs and unavailable CUDA fail; tests never download. Original8DATA
actual cases were not rerun. Actual root/splits env overrides match CORR-001 below.

Reference [corr002_reference.npz](../report/corr002_reference.npz),187438bytes,
SHA-256 `b4396faba198a1497758cd3478983b96a85bd272972cb6266e14d497437d1f03`.
Created once from verified actual persisted train IDs0/1/2, run42 epoch0, clean0
and severity3 of noise/blur/brightness. Float32 pixels/normalized arrays shape
[4,3,3,32,32]; IDs/severities int64, kinds Unicode. Pixel arrays from dispatcher
after integer geometry, normalized output from IndexedCIFAR10 fixed injection
probe. Actual tests replay independent NumPy2D reflect/direct randn/factor oracles
for all16conditions×17IDs×2epochs, compare saved subset exactly and preserve the
old CORR-001 reference. Fixed probes are not production robust policy.

Array semantic hashes use compact sorted ASCII JSON dtype/shape +LF+C-order
little-endian bytes (Unicode uses NumPy UTF-32LE). Tensor stream hashes use only
float32 C bytes; list-order streams and shuffled-loader streams differ in order.
See [evidence](../report/corr002_evidence.json) for per-array and per-stream hashes.
Do not regenerate a mismatched reference. No NPZ byte equality across platforms/
versions promised. Canonical keyed epsilon remains CPU even for CUDA models.

All23protected raw/split/stats/config/reference/source/protocol/lock files retain
bytes/hashes/mtimes. CPU and CUDA blur max oracle error1.7881393432617188e-7;
CPU tolerance1e-6, CUDA2e-6. No global RNG/backend changes or cross-device bitwise
claim. CORR-003, ROBUST-001, full DATA-003 gates and general REPRO-001 remain open.

SETUP-002 выполнен локально 30.10.2026; config validation относится к SETUP-003.
DATA-001 реализует raw-data loader, acquisition/verification CLI и local manifest;
DATA-002 — persisted split494; DATA-003 — train-only stats/keyed loaders/stats CLI.
Historical CORR-002 default777PASS/14SKIP; separate6actual corruption PASS/0SKIP,
including CUDA and CORR-001 regressions. [Evidence](../report/corr002_evidence.json).
Original8actual DATA tests from31.10.2026 not rerun; fresh raw/split/stats CLIs PASS.
CORR-001/002 technical PASS/full DoD PARTIAL; DATA-003 remains PARTIAL pending
ROBUST-001 full integration, formal gates and independent human review.
Раздел «Плановые команды pipeline» ниже остаётся будущим интерфейсом для
train/evaluate/metrics/plots. Успешный empty script не является выполненной задачей.

## Historical CORR-001 проверки — 18.11.2026

Use the existing environment; no reinstall/update/backend changes. From repo root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.venv/pytest-corr001-final
$env:RUN_CIFAR_CORRUPTIONS='1'
$env:RUN_CORR_CUDA='1'
.\.venv\Scripts\python.exe -m pytest tests/test_corruptions_local.py tests/test_corruptions_cuda.py -q -s --basetemp=.venv/pytest-corr001-actual
Remove-Item Env:RUN_CIFAR_CORRUPTIONS
Remove-Item Env:RUN_CORR_CUDA
```

Default453PASS/0FAIL/11SKIP; separate3PASS/0FAIL/0SKIP. Defaults do not read actual
raw/split/stats and do not require CUDA. Opted-in missing/corrupt actual inputs fail;
opted-in unavailable CUDA fails. For CPU-only actual replay omit RUN_CORR_CUDA and
its test file: the2CIFAR tests still require RUN_CIFAR_CORRUPTIONS=1.
Actual defaults: CIFAR10_ROOT=data/raw, CIFAR10_SPLIT_DIR=data/splits. The saved
reference is repository-relative report/corr001_reference.npz, read-only, pickle-free.

The reference stores float32 noisy/normalized pixels for fixed actual train IDs0/1/2,
run42 epoch0 sigma.08, and CPU keyed epsilon for test IDs0/71/9999. It was created
from verified actual IndexedCIFAR10 output with numpy.savez_compressed; actual tests
replay independent integer geometry and direct local-generator noise oracles.
Do not silently regenerate a mismatched reference. File SHA-256 and dtype/shape/
semantic array hashes are in [evidence](../report/corr001_evidence.json); array hash
encoding is compact sorted ASCII JSON dtype/shape + LF + little-endian C bytes.
Tensor stream hashes omit the header and use C-order little-endian float32 bytes.
NPZ ZIP-byte regeneration across versions/platforms is not promised.

Canonical test generator/epsilon remain CPU; CUDA primitive is only a separately
verified helper path. No global RNG, device/backend or scientific setting changes;
all16raw/split/stats/config hashes and mtimes preserved. General REPRO-001 and
production wrapper/evaluator/policy remain pending. Full DoD PARTIAL.

## Окружение

Проверено: **Python3.12.10, Windows AMD64**, отдельная `.venv` с
`include-system-site-packages=false`. torch2.9.0+cu126 / torchvision0.24.0+cu126.
[Official compatibility table](https://github.com/pytorch/vision) включает Python3.12
для пары2.9/0.24; [official install combinations](https://pytorch.org/get-started/previous-versions/)
содержит эту пару для Windows/CUDA12.6. Candidate принят только после реальной
установки, imports, compiled ops и CPU/CUDA forward/backward. Глобальное окружение
не изменялось. Это проверенный выбор, не утверждение о latest version.

Exact direct versions — [requirements.txt](../requirements.txt), все38 installed
packages включая pip/setuptools — [requirements-lock.txt](../requirements-lock.txt).
Источники: PyTorch cu126 index для torch/vision и их первой dependency resolution;
PyPI для остальных packages. Полный snapshot и результаты —
[setup_environment.json](../report/setup_environment.json). Snapshot намеренно
не содержит username/absolute paths. Lock — version lock, **без hashes wheel-файлов**;
не гарантирует сохранение binaries на index, bitwise cross-platform или offline install.

Проверенная bootstrap последовательность Windows PowerShell, из repo root;
`python` здесь должен быть Python3.12.10. Launcher `py -3.12` на этой машине
не находит interpreter, поэтому используется доступный `python`. TEMP/TMP внутри
ignored `tmp/` обходят подтверждённую sandbox ошибку ensurepip с default temp:

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
```

При первом создании `.venv` реально выполнены `pip install torch==2.9.0
torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126`, затем
`pip install numpy pandas scikit-learn matplotlib tqdm pyyaml pytest --index-url
https://pypi.org/simple`. Resolver freeze сохранён после успешных checks.
Повторный install из lock в этом env прошёл без изменений. Новый полный install
именно из lock на второй машине не выполнен: это REPRO-002. Не обновлять lock
автоматически при каждом запуске и не заменять backend suffix молча.

Проверенный GPU: NVIDIA GeForce RTX3060 Laptop,6144MiB, capability8.6;
driver555.97, torch CUDA12.6, cuDNN91002; sm_86 есть в wheel architectures.
`nvidia-smi` показывает CUDA12.5 — это driver capability, не версия runtime wheel.
[NVIDIA minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
допускает CUDA12.x на подходящем старшем driver с ограничениями (например PTX/new
features); actual CUDA smoke прошёл. Не обновляли driver/toolkit. Это не проверка
всех CUDA operations, project CNN или memory/time budget для full training.
CPU-only wheel и Linux/macOS/Colab на этом этапе **не проверены**; CPU ops
проверены внутри того же cu126 wheel. На машине без CUDA script явно пишет SKIP.

Colab: fresh runtime, version check до imports, согласованная install pair, restart runtime если packages заменены, code commit/copy hash и mounted persistent artifact storage. Пути через pathlib, workers 0 reference. Не полагаться на ephemeral `/content` для единственной копии checkpoints. Результаты разных platforms отмечать, paired models по возможности обучать на одном backend.

## Determinism

Перед CUDA init в PowerShell: `$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'`; затем
в будущем seed utilities: random.seed, NumPy seed, torch.manual_seed и cuda seeds.
В environment smoke уже проверены `torch.use_deterministic_algorithms(True)`,
cudnn.benchmark=False, cudnn.deterministic=True. TF32 отключён через новый API2.9:
`torch.backends.cuda.matmul.fp32_precision='ieee'` и
`torch.backends.cudnn.conv.fp32_precision='ieee'`. Не смешивать old/new TF32 API.
[Official CUDA semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html),
[reproducibility guidance](https://docs.pytorch.org/docs/2.9/notes/randomness.html).
Полная determinism/resume infrastructure остаётся REPRO-001/TRAIN-002.

PyTorch не обещает одинаковые results между releases/platforms и CPU/GPU; deterministic mode может замедлить вычисления. Reference claims ограничивать locked software/backend. CPU pixel corruption и stable per-image streams обеспечивают общие inputs; model outputs across hardware сравнивать с заранее записанными tolerances, не заявлять bitwise cross-platform.

Worker initialization и Windows spawn требуют top-level worker functions и main guard; seed иных библиотек берётся из torch.initial_seed. Это учитывается в data interfaces. [DataLoader docs](https://docs.pytorch.org/docs/main/data.html). Exact epoch resume проще при workers 0; stateless transforms исключают скрытый worker transform state.

DATA-003 implements stable_seed/seed_worker and local per-image/order/worker
generators only. Call dataset.set_epoch before iteration; persistent_workers must
be False. Actual Windows spawn2 equality passed against workers0 on129persisted
train IDs, two epochs, batch64vs37; full raw45k/5k/10k typed coverage also passed.
Future training requests only train/val partitions; DATA-003 test access was solely
the fixed loader/integrity contract. No checkpoint/resume/model determinism claim.

Не использовать global RNG для robust policy: отдельные keyed generators сохраняют одинаковые init, crop/flip и sample order между моделями. Полные seed namespaces: [EXPERIMENT_PROTOCOL §4](EXPERIMENT_PROTOCOL.md).

## Работающие команды DATA-001

Из root, без изменения dependencies:

```powershell
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-only
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.venv/pytest-tmp
$env:RUN_CIFAR_INTEGRATION='1'
.\.venv\Scripts\python.exe -m pytest tests/test_cifar10_local.py -q --basetemp=.venv/pytest-cifar-local
Remove-Item Env:RUN_CIFAR_INTEGRATION
```

Default tests используют synthetic batches с заменёнными reference hashes/counts,
запрещают network в data tests; три actual tests отдельно opt-in. При opt-in missing
данные — FAIL. `CIFAR10_ROOT` при необходимости задаёт другой local root.
`--verify-only` не создаёт root/manifest и не меняет dataset artifacts. Каждый
запуск заново проверяет files и semantic counts; stale manifest не заменяется.

Sources проверены по [official dataset page](https://cave.cs.toronto.edu/kriz/cifar.html)
и [torchvision v0.24.0 CIFAR source](https://github.com/pytorch/vision/blob/v0.24.0/torchvision/datasets/cifar.py),
сверенным с установленной0.24.0+cu126. Python archive MD5
`c58f30108f718f92721af3b95e74349a`, семь file MD5 в src/data.py. При наличии
archive он тоже проверяется; при его отсутствии observation=null, без claim PASS.
HTTPS redirects разрешены только для проверенных Toronto paths/hosts, без HTTP
fallback/mirrors. Acquisition запрашивает проверяемые ranges по1MiB, проверяет
206/Content-Range/длину, затем полный archive MD5; ожидаемый размер170498071 bytes
подтверждён official HTTPS headers30.10.2026. MD5 — upstream integrity reference, не доказательство
cryptographic authenticity; SHA-256 — фактическая identity локальных bytes.

Ignored `data/raw/cifar10_integrity_v1.json` (schema_version1) содержит fresh
inspection, source URLs/reference, archive/files MD5/SHA-256/size, оба counts/shape,
classes/mapping, UTC verification timestamp, acquisition method/final URL,
Python/torchvision и LF-normalized SHA-256 двух verifier source files. Existing
artifacts не выдаются за download этого запуска: historical acquisition unknown.
Manifest создаётся эксклюзивно, flush+fsync; при interrupted write partial record
остаётся диагностикой и следующий запуск отвергает его. Это не generic atomic
write/backup implementation REPRO-001; эта задача остаётся TODO.

Corrupt data/archive, foreign/incomplete dataset directory или conflicting manifest
не удаляются/не перезаписываются; ошибка указывает путь и предлагает новый empty
root с сохранением старого. Failed acquisition оставляет uniquely named staging.
Не хранить единственную копию будущих unique artifacts в ignored directories.
Test partition здесь инспектируется только ради integrity, без scientific decisions.
После DATA-002 оба YAML содержат verified NPZ file hash; stats hash остаётся null,
training готовность не заявлена. Raw acquisition record сохраняет historical code
hashes DATA-001 и не переписывается при добавлении split API/CLI.

## Работающие команды DATA-002

Existing `.venv`, из repo root. Split modes offline: требуют предварительно
полученный и verified original train; не вызывают acquisition и не декодируют
test images/targets. Отсутствие/повреждение данных — exit1. Raw acquisition остаётся
отдельной явной командой DATA-001 выше; historical raw manifest не переписывается.

```powershell
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --prepare-split --split-seed 494
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-split
```

Defaults: `--root data/raw`, `--split-dir data/splits`, split seed494.
`--split-dir` позволяет явно выбрать новый empty directory после сохранения failed
directory. `--prepare-split` создаёт только NPZ/manifest или полностью проверяет
existing pair; normalization/loaders не создаёт. `--verify-split` не создаёт root,
split directory или files, не меняет hashes/file mtimes. Original
`--verify-only` проверяет только raw data, не требует split и не читает его.
Три modes взаимоисключающие; `--split-dir`/`--split-seed` допустимы только в split
modes; любой seed кроме494 — exit2. `--help` описывает flags; success JSON/exit0,
data/artifact failure JSON stderr/exit1, invalid arguments argparse stderr/exit2.
Подробная [schema/hash/write/recovery спецификация](TECHNICAL_ARCHITECTURE.md#split-artifact-contract-v1).

Actual NPZ76550 bytes SHA-256
`1dc8e61a237d7e8fd3ac3c2dea4787e6a82f8f235d74993386af9074f6e1d284`;
manifest file SHA-256 `8742e5d48e4001e7f6a61f924bc524ab33f2276a256bc716407a98fc9bc86a21`.
Schema/creator/semantic hashes и before/after mtimes — в
[manifest](../data/splits/split_manifest.json)/[evidence](../report/data002_evidence.json).
NPZ binary и JSON LF enforced через `.gitattributes`; diagnostic locks/staging ignored.
Frozen indices canonical: API читает saved arrays, protocol replay проверяет их.
Signed source authenticity и bitwise regeneration across versions/platforms не заявлены.
Versioned DATA-002 source region и NumPy2.5.4 должны совпасть при verification;
DATA-003 code добавлять вне region, не менять source identity frozen artifacts молча.

```powershell
$env:RUN_CIFAR_INTEGRATION='1'
$env:RUN_CIFAR_SPLIT='1'
.\.venv\Scripts\python.exe -m pytest tests/test_cifar10_local.py tests/test_split_local.py -q --basetemp=.venv/pytest-data002-actual
Remove-Item Env:RUN_CIFAR_INTEGRATION
Remove-Item Env:RUN_CIFAR_SPLIT
```

Defaults actual tests: `CIFAR10_ROOT=data/raw`, `CIFAR10_SPLIT_DIR=data/splits`.
При opt-in missing/corrupt input — FAIL. 31.10.2026 combined5PASS/0SKIP;
default suite282PASS/5SKIP не требует actual inputs. CUDA smoke не повторялся:
dependencies/backend неизменны; training/evaluation не запускались.

## Работающие команды DATA-003

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_normalization --root data/raw
.\.venv\Scripts\python.exe -m scripts.prepare_normalization --root data/raw --verify-only
```

Offline, original train only. Defaults: `--split-path
data/splits/cifar10_split_v1.npz`, `--output data/splits/normalization.json`.
Existing frozen split is verified, never created. Prepare creates stats exclusively
or fully verifies/reuses; verify-only requires existing stats and writes nothing.
Success JSON `status=stats_verified`, exit0, records FILE and semantic hashes;
this does not assert training readiness. Invalid flags exit2; missing/corrupt raw,
split/stats or interrupted diagnostics exit1 with preservation guidance. Select a
new output only after preserving/inspecting the failed one. No raw manifests are
rewritten. Existing download_data modes retain raw-only / split-only semantics.

## Плановые команды pipeline

В примерах `python` обозначает interpreter проверенного env; на Windows можно заменить на `.\.venv\Scripts\python.exe`. Команды из root. Device cpu — рабочий fallback после реализации; cuda допустим только после validation. Parameters subset/profile относятся только к smoke/debug и запрещены для core profile.

```powershell
# TEST-001: offline synthetic unit/integration tests после написания
python -m pytest tests -q

# TRAIN-003: отдельный debug overfit, test set не читается
python -m src.train --config configs/clean_cnn.yaml --seed 42 --profile debug --overfit-samples 64 --max-steps 300 --device cpu

# TEST-001: actual CIFAR smoke, no core outputs
$env:RUN_CIFAR_INTEGRATION='1'
python -m pytest tests/test_smoke.py -q

# TRAIN-004: проверить именно 6 core run specs до вычислений
python -m scripts.run_experiments --configs configs/clean_cnn.yaml configs/robust_cnn.yaml --seeds 42 123 2026 --profile core --dry-run

# CLEAN-001 / ROBUST-002: пример пары; orchestration ниже охватывает все seeds
python -m src.train --config configs/clean_cnn.yaml --seed 42 --profile core --device cpu
python -m src.train --config configs/robust_cnn.yaml --seed 42 --profile core --device cpu

# Полные 6 training runs, completed verified runs пропускаются
python -m scripts.run_experiments --configs configs/clean_cnn.yaml configs/robust_cnn.yaml --seeds 42 123 2026 --profile core --device cpu --resume-incomplete

# EVAL-002..004: matrix planning и затем 96 evaluations
python -m scripts.run_all_evaluations --checkpoints-root outputs/checkpoints --require-complete --dry-run
python -m scripts.run_all_evaluations --checkpoints-root outputs/checkpoints --device cpu --require-complete

# METRIC-002 и PLOT-001..003
python -m src.metrics --input outputs/metrics/results.csv --output-dir outputs/metrics/aggregate/core-v1
python -m scripts.make_plots --metrics outputs/metrics/aggregate --runs outputs/runs --output outputs/figures
```

Для metrics CLI нужен заранее собранный полный core `results.csv`; этот файл
появится только после реализации evaluator и EVAL-004. Каталог `core-v1` должен
быть новым, CLI существующие результаты не перезаписывает.

`--profile debug` explicitly выключает geometry/corruption/dropout только для 64-image diagnostic; manifest содержит overrides, mode и max_steps; core validator rejects debug. `test_smoke` создаёт temporary outputs, profile smoke,2models×1 seed×16 conditions на 100 test images,1 epoch; full validator считает только core.

Single-condition evaluation после реализации:

```powershell
python -m src.evaluate --checkpoint outputs/checkpoints/RUN_ID/best.pt --corruption gaussian_noise --severity 3 --device cpu
```

`RUN_ID` — явный placeholder; взять точное имя из run manifest. Resume:

```powershell
python -m src.train --config configs/clean_cnn.yaml --seed 42 --profile core --resume outputs/checkpoints/RUN_ID/last.pt --device cpu
```

Аргументы всех CLI: `--help` со defaults, invalid argument nonzero exit, JSON status success/failed/interrupted. `run_experiments --resume-incomplete` проверяет hash/config/code identity и не запускает другой config поверх старого directory. `--require-complete` в eval требует все шесть manifests и core profile; pilot16 может запускать src.evaluate отдельно, без core aggregate.

Optional benchmark после BENCH-001: `python -m scripts.run_all_evaluations --checkpoints-root outputs/checkpoints --dataset cifar10c --data-root data/raw/CIFAR-10-C --device cpu --require-complete`; loader берет gaussian_noise/defocus_blur/brightness1..5, отдельные directories и outputs90. Этот flag отсутствует до extension task; он не меняет core 96.

## Provenance и verification

### Local synthetic run records (REPRO-001)

The config hash uses canonical JSON for settings and data identities and omits
data and output file locations. The raw dataset checksum is recorded separately
in the manifest. The checkout hash covers raw bytes of tracked and non-ignored
untracked files, so Git status and the checkout hash identify local changes
alongside `git_sha`. Capture `environment.json` on the actual device inside the
active training settings so its flags reflect the run.

`src.repro.write_run_record` writes a new synthetic run directory from a completed
trainer result, its exact resolved config, environment and checkout identities,
and artifact paths. The manifest appears after all listed files and links verify.
`verify_run_record` reads and checks the complete directory. `backup_run_record`
copies a verified run into a new local directory and checks its file inventory
against the source manifest. A directory left before the manifest is written is
incomplete. The backup retains the original run ID even if its folder name differs.

The training checkpoint uses `run_id`, `model_name`, `seed`, `profile`, `config_hash`,
`dataset_sha256`, `split_hash`, `normalization_hash`, `init_sha256`, `git_sha`,
`git_dirty`, `code_snapshot_sha256`, and `environment_id` to associate a checkpoint
with a run. The synthetic record and backup tests run with
`python -m pytest tests/test_repro.py -q`.

### Local checkpoints and resume (TRAIN-002)

Use `train_run(config, seed, run_dir=Path(...))` for a fresh checkpointed run and
`train_run(config, seed, resume=Path(...)/'last.pt')` to continue it. Use a new
directory under the ignored `outputs/runs/` tree or outside the checkout so that
generated files stay outside the checkout fingerprint. Keep the run directory
name when relocating the files because it is the run ID.

`last.pt` contains the completed epoch, history, optimizer and scheduler states,
Python, NumPy, Torch and loader random states, and the best model. Resume compares
the run identity, restores state, rebuilds `best.pt` and `history.csv`, and starts
at the next epoch. Load a trusted `best.pt` with `load_checkpoint(path, kind='best')`
and load its `model` into a compatible `CompactCNN` with `strict=True`. The synthetic
CPU acceptance checks run with `python -m pytest tests/test_checkpoints.py -q`.

Per epoch, per run и per condition schemas canonical в EXPERIMENT_PROTOCOL §6. Обязательные identity links: data checksum → split hash → stats hash → resolved config hash → code identity → checkpoint hash → condition hash → predictions hash → metric row → figure/table manifest. Git SHA без dirty status недостаточен; core release должен ссылаться на stable code revision и record any local patch hash.

Run outputs ignored в Git должны иметь artifact_manifest и backup, доступный всем six teammates. После каждого complete epoch last atomic; после run backup best/last/history/config/env, проверить hashes после copy. Каждую ночь активных runs проверять свободное место и archive status. Dataset можно повторно скачать по checksum; unique checkpoints нельзя заменить обещанием retrain.

REPRO-002: другой участник в fresh env делает install, unit tests и actual smoke по README, затем загружает один frozen best checkpoint и воспроизводит clean + noise 5 predictions. Для одинакового locked CPU backend ожидается exact predictions; при GPU численные logits tolerance объявить (например atol1e-5, rtol1e-5) и отдельно compare predicted labels/metrics. Если tolerance не соблюдена, исследовать cause, не увеличивать допуск без evidence. Все результаты и команды в report/reproduction.md; полный retrain second machine не обязателен, но claims о retraining reproducibility без него ограничены.

## Плановый artifact audit

| Check | Pass |
|---|---|
| Data | Original checksums, one split, train-only stats и balanced counts |
| Training | 6 core manifests,40 epochs each,14080 steps each, stable code/config |
| Checkpoints | All best/last load, architecture288746 params, SHA verified |
| Evaluation | 96 unique keys,10000 predictions each, common input hashes |
| Aggregation | No missing seeds, sampleSD, paired deltas, pp/fraction correct |
| Figures | Regenerated from frozen tables; manifest lists hashes |
| Recovery | CPU reference epoch resume equality + backup retrieval test |
| Reproduction | Independent fresh-env smoke and frozen checkpoint evaluation |

Все pass записи появятся только после фактического выполнения, не при наличии command snippets.
