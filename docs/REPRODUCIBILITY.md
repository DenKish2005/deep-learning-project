# Воспроизводимость: проверенный setup и будущий pipeline

SETUP-002 выполнен локально 30.10.2026; config validation относится к SETUP-003.
DATA-001 реализует raw-data loader, acquisition/verification CLI и local manifest;
DATA-002 — persisted split494 и prepare/verify CLI. Current default282PASS/5SKIP,
separate actual5PASS/0SKIP; [evidence](../report/data002_evidence.json).
Раздел «Плановые команды pipeline» ниже остаётся будущим интерфейсом для
stats/train/evaluate/metrics/plots. Успешный empty script не является выполненной задачей.

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
python -m src.metrics --input outputs/metrics --output outputs/metrics/aggregate
python -m scripts.make_plots --metrics outputs/metrics/aggregate --runs outputs/runs --output outputs/figures
```

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
