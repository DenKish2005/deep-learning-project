# Техническая архитектура

Это преимущественно проект будущих интерфейсов. SETUP-003 реализует только
config validation в utils, оба YAML и test_utils; DATA-001 реализует load_cifar10,
file/partition inspection и acquisition CLI/manifest. DATA-002 реализует persisted
create_split/load_split, provenance/reuse/failure safety и split CLI. DATA-003
реализует stats, keyed transforms/order, clean typed loaders и policy boundary;
CORR-001 реализует severity mapping, Gaussian noise и keyed CPU epsilon.
CORR-002 adds reflect blur, darkening and dispatch. CORR-003 implements only the
train-only sanity grid, PNG/PDF/source manifest and bounded CLI.
ROBUST-001 implements the production policy and verifies the existing explicit
train-only integration. MODEL-001 implements CompactCNN and synthetic model QA.
Human/Person5 visual review and formal acceptance remain pending. Остальные production APIs ниже
ещё не реализованы. Состояние: [CURRENT_STATUS.md](CURRENT_STATUS.md), уточнения
и причины: [SETUP_DECISIONS.md](SETUP_DECISIONS.md). Протокол —
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). Новые frameworks не вводятся.

## Рекомендуемое итоговое дерево

```text
robustness-to-noise/
├── README.md
├── ARCHITECTURE.txt
├── LICENSE
├── .gitignore
├── requirements.txt
├── requirements-lock.txt              # после проверки окружения
├── configs/
│   ├── clean_cnn.yaml
│   └── robust_cnn.yaml
├── docs/
│   ├── MASTER_PROJECT_PLAN.md
│   ├── START_HERE.md
│   ├── PROJECT_OVERVIEW.md
│   ├── REQUIREMENTS.md
│   ├── CURRENT_STATUS.md
│   ├── TECHNICAL_ARCHITECTURE.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── IMPLEMENTATION_ROADMAP.md
│   ├── TEAM_TASKS.md
│   ├── TESTING_PLAN.md
│   ├── REPRODUCIBILITY.md
│   ├── PROJECT_SCHEDULE.md
│   ├── FINAL_DELIVERABLES.md
│   └── RISKS_AND_LIMITATIONS.md
├── data/
│   ├── raw/                           # CIFAR archive/extracted files, ignored
│   ├── processed/                     # optional caches, ignored
│   └── splits/
│       ├── cifar10_split_v1.npz        # indices, сохранять с release
│       ├── split_manifest.json
│       └── normalization.json         # только 45k train
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── corruptions.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── metrics.py
├── scripts/
│   ├── check_environment.py           # implemented offline dependency/backend smoke
│   ├── download_data.py
│   ├── run_all_evaluations.py
│   ├── run_experiments.py              # обоснован: 6 training runs, status/resume
│   └── make_plots.py
├── notebooks/
│   └── 01_eda_and_sanity.ipynb
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # synthetic fixtures, no network default
│   ├── test_data_pipeline.py
│   ├── test_corruptions.py
│   ├── test_models.py
│   ├── test_utils.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   ├── test_metrics.py
│   ├── test_plots.py
│   ├── test_protocol.py
│   └── test_smoke.py
├── outputs/
│   ├── runs/<run_id>/                  # resolved config, env, history, manifests
│   ├── checkpoints/<run_id>/           # best.pt, last.pt, sha256.json
│   ├── predictions/<run_id>/           # <condition_id>.npz
│   ├── metrics/<run_id>/               # <condition_id>.json
│   ├── metrics/aggregate/              # results.csv, summaries, analysis tables
│   └── figures/                        # PNG + PDF, source manifest
└── report/
    ├── proposal.md                     # восстановленный настоящий proposal
    ├── course_materials/README.md      # provenance; originals если можно хранить
    ├── ai_usage.md
    ├── reproduction.md
    ├── rehearsals.md
    ├── final_report/
    │   ├── README.md
    │   ├── report.md                   # либо согласованный TeX/Word source
    │   ├── references.bib
    │   └── final_report.pdf
    └── presentation/
        ├── outline.md
        ├── slides.pptx                # либо согласованный slides source
        └── slides.pdf
```

`.gitkeep` в существующих data/output dirs сохраняются. Tree показывает будущий
end state, а не inventory готовых файлов. Сейчас реализованы configs/config utils,
их tests, environment checker и DATA-001..003 data APIs/CLI; полный pipeline ещё отсутствует. Report source
format можно выбрать командой. Дополнительные frameworks/services не нужны.

## Направление зависимостей

`utils` и `corruptions` не импортируют high-level modules. `models` зависит только от torch. `data → corruptions, utils`; `train → data, models, utils`; `evaluate → data, models, metrics, utils`; `metrics → numpy/pandas/stdlib`; scripts вызывают src. Metrics CLI использует lazy main и не импортирует train. Notebook только вызывает production functions, без их копирования. Ни train, ни evaluate не импортируют scripts.

Все CLI запускаются из root через `python -m ...`, main под `if __name__ == '__main__'`. Для scripts допустим namespace package, отдельный `scripts/__init__.py` не обязателен. Paths обрабатываются pathlib; cwd не зашивается в source.

## Контракты каждого production Python-файла

### `src/corruptions.py`

- `severity_params(kind: str, severity: int) -> dict`: только три типа и levels 1..5; clean severity 0 обрабатывается отдельно. Unknown kind/invalid severity → ValueError.
- `gaussian_noise(x: Tensor, sigma: float, generator: torch.Generator) -> Tensor`; `gaussian_blur(x: Tensor, sigma: float, kernel_size: int) -> Tensor`; `darken(x: Tensor, factor: float) -> Tensor`.
- `apply_corruption(x: Tensor, kind: str, severity: int, generator=None) -> Tensor`: RGB CHW float32 `[0,1]`, без in-place, output same shape/device/dtype. Noise generator обязан соответствовать device; main evaluation CPU. Для clean clone/identity без мутации.
- `CorruptionPolicy(probabilities, max_train_severity=3)` и `__call__(x, generator) -> (Tensor, dict)`: per-image выбор; dict содержит event/severity для pilot audit, не label.
- Нет зависимости на labels, normalization или model. Тесты TEST-C01..08; integration — unnormalized image → corruption → normalization → model. DoD: mapping, range, deterministic equality, CPU tests и визуальный grid; CUDA только если доступна.

#### CORR-001 operational boundary (implemented 18.11.2026)

`severity_params` returns fresh dicts: noise `sigma`, blur `sigma`/`kernel_size`,
brightness `factor`; exact protocol tables, Python severity int1..5, invalid
kind/type/severity → ValueError. CORR-002 implements blur/darken below.

`gaussian_noise` validates dense strided CPU/CUDA float32 RGB CHW, positive H/W,
finite [0,1] pixels, finite nonnegative Python sigma and explicit torch.Generator
on the exact same device. Invalid inputs fail before drawing. Sigma0 clones
without consuming RNG. Other sigma uses torch.randn with local generator, adds
scaled epsilon and clamps; finite sigma beyond float32 uses float64 arithmetic
before clamp/cast. New output storage, same shape/device/dtype; no mutation or
global RNG. Main CIFAR shape remains3×32×32; larger helper tensors support numeric tests.

`keyed_test_generator(original_index, *, namespace)` returns a fresh CPU generator;
`keyed_test_epsilon(original_index, *, namespace)` returns CPU float32 [3,32,32]
pre-clamp N(0,1). Full namespace must equal `cifar10-python/original-test`; ID is
Python int0..9999. The seed alias is `cifar10-test`:
`stable_seed(100494,'cifar10-test',original_index,'gaussian_noise','v1')`.
No model/run seed or severity argument. Each image/condition recreates its generator;
passing a consumed stream changes the noise. Train uses the existing separate
`train_corruption` key; test helpers reject train/val namespaces.

TEST-C01/C02/C04 cover the noise subset and all-family mapping; C03/U02 seed subset
covers common pre-clamp epsilon, batching/order/spawn/fresh processes and global RNG.
Actual train probe checks data→noise→Normalize with frozen artifacts; it is a fixed
test callable, not CorruptionPolicy. Actual CUDA primitive validation passed locally;
keyed reference stays CPU and no CPU/CUDA bitwise equality is claimed.

`CorruptionPolicy` is implemented in ROBUST-001 below. CORR-002 primitives follow.
Production CorruptedDataset/evaluator, condition manifests and full robust
acceptance are future tasks; CORR-003 visual grid is implemented below. Technical PASS/full DoD
PARTIAL; [evidence](../report/corr001_evidence.json).

#### CORR-002 operational boundary (implemented 18.11.2026)

`gaussian_blur(x, sigma, kernel_size)` shares the dense CPU/CUDA float32 RGB CHW,
positive H/W and finite [0,1] contract. Sigma: finite nonnegative Python int/float,
excluding bool. Kernel: positive odd Python int; reflect radius<both H,W required,
even when sigma0. Sigma0 or kernel1 clones; otherwise weights are exp(-.5*(q/sigma)^2)
in float64, normalized/cast to float32. Two grouped horizontal/vertical convolutions
use reflect padding, no channel mixing. Clamp contains [0,1] roundoff. New storage,
same shape/device/dtype; no RNG. Exact main sigma/kernel tables unchanged.

`darken(x, factor)` validates the same pixels and finite Python factor0..1 (not bool),
returns clamp(factor*x,0,1). Factor1 identity/new storage; factor0 black. Factors>1
rejected. No additive shift, mutation or RNG.

`apply_corruption(x, kind, severity, generator=None)`: clean only Python int0,
three families only1..5 via severity_params. Noise delegates to unchanged primitive
and requires matching explicit generator. Deterministic families accept None or a
torch.Generator on any device, ignore it without draws; other types fail. Clean
validates pixels then clones. Invalid arguments fail before drawing; no seed/policy/
dataset wrapper assignment. Callers normalize after dispatch. Evaluation callers
retain existing keyed_test_generator/CPU identity boundary.

TEST-C01/C02/C05/C06/C08 numeric/contract acceptance PASS; separate actual17train
IDs×16conditions×epochs0/1, spawn0/2 and actual CUDA tests PASS. Saved actual reference
covers train0/1/2, clean0 and severity3 families. [Evidence](../report/corr002_evidence.json).
CORR-003 technical audit and ROBUST-001 production integration are implemented;
formal DATA-003 dependent acceptance remains separate.
Full DoD PARTIAL pending formal/human review; general REPRO-001 remains TODO.

#### ROBUST-001 operational policy (implemented 18.11.2026)

`CorruptionPolicy(probabilities, max_train_severity=3)` copies a Python list/tuple
to immutable settings. Only exact robust `[.50,.17,.17,.16]` and Python int3 are
accepted, matching frozen config validation; arbitrary tuning/clean probabilities
are rejected. Clean training injects None. No labels, model, batch quotas or
compositions enter the policy.

`policy(x, generator) -> (pixels, {'event': str, 'severity': int})` validates raw
dense CPU/CUDA float32 RGB CHW finite[0,1], matching explicit torch.Generator and
H,W>4 before draws. The extra spatial constraint makes every possible radius4
training blur valid without consuming RNG on invalid input. Output new storage,
same shape/device/dtype; input untouched. Normalize only after policy.

Exact same-device draw order: scalar `torch.rand((), dtype=torch.float64)` chooses
identity/noise/blur/brightness at literal boundaries.50/.67/.84. Identity reports
severity0 without another draw. Each corruption draws scalar int64
`torch.randint(1,4,())`; only noise then draws float32 `randn(x.shape)` through
the existing dispatcher. No seed creation or global fallback. Existing hook owns
`stable_seed(run_seed,epoch,original_id,'train_corruption')`; geometry/order and
original-test seed helpers stay unchanged. Metadata is for audit, not labels.

Offline and actual numeric/frequency/RNG/pairing/spawn acceptance PASS;
[evidence](../report/robust001_evidence.json). No model initialization/budget,
training readiness, general REPRO-001 or full scientific gate claim. Frozen
CORR-003 uses authenticated read-only replay at21ede65 via
`scripts.verify_frozen_corr003`; strict current-source mismatch remains expected.

### `src/data.py`

- `load_cifar10(root: Path, train: bool, download=False) -> torchvision.datasets.CIFAR10`: raw RGB/PIL + labels, dataset integrity через torchvision; offline missing data → actionable error, не silent network call.
- **DATA-001 implemented:** explicit official-HTTPS acquisition, archive/seven-file checksum verification before unpickle, requested partition semantic contract; `inspect_cifar10(root)` проверяет оба original partitions только для integrity. Raw CLI: `--root`/`--verify-only`, schema-v1 local manifest ignored.
- **DATA-002 implemented:** `create_split(targets: Sequence[int], split_seed: int, out_dir: Path) -> dict` создаёт пару или полностью проверяет existing pair; возвращает точный schema-v1 manifest (без runtime status/path fields). Только Python int494; targets — integer sequence/array `[50000]`, classes0..9, ровно5000/class, без bool/float/object. Ошибки split contract — `DataIntegrityError`, без silent overwrite.
- `load_split(path: Path, targets) -> tuple[np.ndarray, np.ndarray]`: read-only; path — NPZ с canonical filename `cifar10_split_v1.npz`, manifest — sibling `split_manifest.json`. Возвращает sorted little-endian int64 (`<i8`) arrays `[45000]`, `[5000]`. Проверяет dtype/range/uniqueness/disjointness/exhaustiveness/class alignment, exact protocol replay и full manifest equality с независимо вычисленным expected record. Replay служит oracle проверки; возвращаются persisted arrays.
- `compute_normalization(raw_data: np.ndarray, train_indices) -> dict`: channel mean/std float64 accumulation по uint8/255, save full precision; только 45k.
- `IndexedCIFAR10(base, indices, mode, normalization, run_seed, epoch=0, policy=None)`: `__getitem__ -> (Tensor[3,32,32], int label, int original_index)`; `set_epoch(epoch)` до loader iteration. Train crop/flip и robust policy; val/test deterministic.
- `CorruptedDataset(base, condition, normalization, eval_seed)` — future: одинаковые test original_index и per-image noise для всех моделей.
- `make_loaders(config, run_seed, epoch) -> dict[str, DataLoader]`: отдельный epoch sampler generator, `drop_last=False`; zero workers reference. Target dtype collate int64 `[N]`; batch images `[N,3,32,32]` float32.
- Dependencies: torchvision, torch, numpy, corruptions, utils. Ошибка class counts, corrupted stats/std=0 или split mismatch → fail before training. Tests TEST-D01..05; DoD: 45k/5k/10k, 4500/500/1000 per class, same split hashes, clean eval deterministic.

#### Split artifact contract v1

NPZ содержит ровно `train_indices` и `val_indices`, без pickle. Manifest strict:
`schema_version`, `algorithm` (name/version/seed/class_order/validation_per_class/final_order),
`dataset` (name/source_page/source_url/archive_name/archive_md5_reference/
train_files_md5_reference/classes), `original_train` (namespace/count/class_counts/
targets_sha256), `arrays` (для каждого name: dtype/shape/sha256/class_counts),
`npz` (filename/sha256/bytes), `creator` (source_region/sha256_lf/numpy).
Unknown/missing/duplicate keys, wrong scalar types, altered source/seed/namespace/
labels/hashes/creator отвергаются. Source references сверяются с DATA-001 constants;
NPZ/semantic hashes пересчитываются. API проверяет связь с supplied targets;
сертификация actual images требует сначала `load_cifar10(root, train=True)`.
Synthetic labels в offline tests не являются official-data evidence.

Semantic SHA-256: ASCII compact sorted JSON `{"dtype":"<i8","shape":[N]}`,
один LF byte, затем C-order little-endian int64 bytes. Это identity массива с
dtype/shape, отдельно от SHA-256 NPZ bytes. Creator SHA-256 — UTF-8/LF source region
между маркерами DATA-002 v1 в `src/data.py` (после префикса BEGIN, до END); это
creation implementation identity, не подпись и не full repository Git identity.
Region и NumPy version при чтении должны совпасть. DATA-003 добавлять вне region;
изменение split implementation требует явной compatibility/migration проверки.

Creation: exclusive `.split-write.lock`, uniquely named `.split-stage-*`, flush/fsync,
проверка staged pair, exclusive copy обоих files, повторная verification, удаление
собственных stage files/empty directory и lock последним. Это readiness protocol
для пары, не atomic filesystem transaction/power-loss guarantee. Active/stale lock,
orphan staging или partial pair всегда FAIL, включая complete pair с lock.
Interrupted diagnostics сохраняются. Recovery: сохранить failed directory и явно
выбрать новый empty out_dir; автоматического repair/delete нет. Existing verified
pair переиспользуется без изменения bytes/hashes/file mtimes. NPZ ZIP bytes
проверяются отдельно; cross-version/platform byte-identical regeneration не обещана.
General checkpoint/RNG/backup infrastructure REPRO-001 остаётся TODO.

#### DATA-003 normalization artifact v1 (implemented)

`compute_normalization(raw_data, train_indices)` returns strict numeric statistics;
tiny synthetic arrays are allowed for offline oracles. `prepare_normalization` /
`load_normalization(raw_data, targets, split_path, path)` require a verified frozen
split via `load_split`; callers first use `load_cifar10(..., train=True)`. Neither
API reads the original test partition. The actual split supplies exactly45000 IDs.

Schema1 contains exactly `schema_version`, `statistics`, `statistics_sha256`,
`provenance`, `creator`. Statistics: RGB channels, scale1/255, ddof0, samples,
pixels_per_channel, mean/std (full float64 JSON precision), train_indices_sha256,
algorithm `float64-two-pass-chunks256-v1`. First pass sums uint8/255 in float64
over fixed256-image chunks; second sums squared deviations from that mean.
Population std uses N×1024. Finite positive std is mandatory. No augmentation.

Provenance includes verified split dataset/original-label identities, split NPZ
and manifest FILE hashes, and selected train pixel SEMANTIC hash (compact sorted
JSON dtype/shape + LF + C-order uint8 bytes, as with split array identities).
Statistics semantic hash is SHA-256 of ASCII compact sorted statistics JSON;
normalization FILE hash separately hashes complete saved bytes. Creator hashes
the LF-normalized DATA-003 normalization v1 source region and records NumPy.
No timestamps or machine paths are necessary in this deterministic record.

Read/reuse recalculates all numbers and identities from supplied verified train
pixels and persisted IDs, compares strict canonical JSON (including scalar types),
rejects duplicate/unknown/missing keys and leaves bytes/mtimes intact. Creation
uses sibling `.normalization.json.write.lock` and `.normalization.json.stage`,
exclusive fsynced publication and verification, then stage deletion and lock last.
Missing/tampered/partial/foreign stats or interrupted diagnostics fail and remain
preserved. Recovery: inspect/preserve them and explicitly select a new stats path;
no automatic repair. This readiness protocol is not a power-loss transaction.
General REPRO-001/checkpoint/backup work and formal dependent acceptance remain pending.

#### DATA-003 keyed loaders (implemented)

`IndexedCIFAR10` takes numeric `record['statistics']`; base must be raw and its
original partition must agree with mode. Indices are unique partition-relative
IDs; dataset.namespace distinguishes original train and original test. Geometry
uses a CPU torch.Generator: zero pad4, two randint(0,9) crop offsets, rand()<.5
horizontal flip, then float32 CHW/255. An optional train-only callable has
`policy(x, generator=local_generator) -> (Tensor, dict)`. Mutation, wrong dtype,
shape/device/nonfinite/range output fail; labels/IDs are always obtained from base.
Policy is injected by the caller; CORR-001/002 primitives and fixed actual injection
probes are verified. ROBUST-001 production policy/integration is technically
verified; full DoD/formal dependent acceptance and human review remain pending.

`EpochOrderSampler` creates a fresh local generator per iteration from
stable_seed(run_seed, epoch, 'order'); dataset.set_epoch updates order/geometry.
Call set_epoch before iterator creation, and finish/discard existing iterators
before changing epoch. `_make_loader` gives float32 NCHW, int64 labels/IDs,
drop_last=False, sequential val/test, workers0 reference, persistent_workers=False,
spawn workers with top-level seed_worker, dedicated worker generator. pin_memory
can only be enabled for CUDA; explicit False is respected. Workers2 is tested
with actual spawned processes across epochs, not inferred from workers0.

`make_loaders(config, run_seed, epoch, *, policy=None,
partitions=('train','val','test'))` verifies config and both declared FILE hashes,
raw original train, persisted split and fresh stats before building loaders. It
creates no artifacts. Augmentation-enabled config requires a supplied callable;
clean config requires None. No training-ready success is implied. A future trainer
must request `partitions=('train','val')`; this path never decodes original test.
Only explicit test requests load test images for the fixed loader contract.

### `src/models.py` — implemented MODEL-00118.11.2026

Technical PASS/full DoD PARTIAL; [evidence](../report/model001_evidence.json).
Caller torch seed before CPU construction; standard layer constructor draws then
one initialize_weights call, then caller moves to device. No hidden reseed.
Explicit reinitialization consumes RNG and resets BN buffers. Class attribute
architecture='compact_cnn_v1'. Forward rejects empty/wrong NCHW32x32 shapes before
BN/dropout. Config remains10classes/dropout.2; module accepts positive class counts
and finite Python dropout[0,1] for QA only. Standard state_dict `_extra_state`
contains architecture/num_classes/dropout, validated on strict load. Missing keys,
wrong shapes or incompatible identity fail; load into fresh instance (PyTorch load
is not transactional). weights_only=True/explicit map_location roundtrip verified;
general checkpoint/config/data provenance and resume remain future REPRO/TRAIN.
Default950PASS/20SKIP; separate synthetic CPU/CUDA2PASS/0SKIP, no optimizer steps.


- `CompactCNN(num_classes=10, dropout=0.2)`; `forward(x: Tensor[N,3,32,32]) -> Tensor[N,10]` logits; `initialize_weights()` либо internal init helper.
- Conv2d 3×3 stride 1 padding 1 bias=False; BN default eps=1e-5, momentum=0.1, affine=True; ReLU; pool 2×2 stride 2. AdaptiveAvgPool2d(1) → flatten → Dropout(0.2) → Linear(128,10, bias=True).
- Conv init Kaiming normal fan_out/relu; BN weight=1, bias=0; Linear normal std=.01, bias=0. Schema `compact_cnn_v1` в checkpoint. Нет softmax перед CrossEntropyLoss.

| Слой | Выход без batch | Trainable params |
|---|---|---:|
| Conv 3→32 + BN + ReLU | 32×32×32 | 864+64=928 |
| Conv 32→32 + BN + ReLU | 32×32×32 | 9216+64=9280 |
| MaxPool | 32×16×16 | 0 |
| Conv 32→64 + BN + ReLU | 64×16×16 | 18432+128=18560 |
| Conv 64→64 + BN + ReLU | 64×16×16 | 36864+128=36992 |
| MaxPool | 64×8×8 | 0 |
| Conv 64→128 + BN + ReLU | 128×8×8 | 73728+256=73984 |
| Conv 128→128 + BN + ReLU | 128×8×8 | 147456+256=147712 |
| MaxPool → GAP → flatten | 128×4×4 → 128×1×1 →128 | 0 |
| Dropout → Linear | 10 | 1290 |
| Всего | N×10 logits | **288746** |

BN running buffers не trainable parameters. `train()` обновляет BN stats и включает dropout; `eval()` замораживает stats и отключает dropout. Checkpoint load со strict=True, несовпадающий architecture/config → clear error. Tests TEST-M01..03. Архитектура достаточно компактна для from-scratch CIFAR-10 controlled study; GAP может терять spatial detail, три pooling уменьшают resolution до 4×4. Accuracy не обещается; перед core runs — pilot, bottleneck profiling и 64-image overfit.

### `src/utils.py`

- **DATA-003 implemented:** `stable_seed` и top-level `seed_worker`; encoding в protocol §4. `seed_everything` и general identity/checkpoint helpers ниже — future REPRO-001.
- `load_config(path: Path) -> dict`; `validate_config(cfg) -> None`: nonempty mapping, known keys, positive batch/epochs/lr, probabilities sum=1, exact severity maps; unknown schema → error.
- `seed_everything(seed: int, deterministic=True) -> None`; `seed_worker(worker_id: int) -> None`; `stable_seed(*parts) -> int`: SHA-256 UTF-8 canonical representation, low 63 bits; не Python hash().
- `config_hash(cfg: dict) -> str`; `sha256_file(path) -> str`; `capture_environment() -> dict`; `write_json_atomic(path, value)`; `save_checkpoint_atomic(path, state)`; `load_checkpoint(path, device) -> dict`.
- Для atomic writes — temporary file в том же filesystem и replace после flush; отказ overwrite чужого run, incomplete files не считаются complete.
- `validate_controlled_configs(clean, robust) -> None`: разрешены только model_name,
  augmentation.enabled/probabilities и output paths; остальные settings равны,
  включая dataset paths/hashes и runtime. Unknown identity metadata keys запрещены.
- `load_checkpoint` только для собственных trusted артефактов; явные map_location и serialization policy для version-locked PyTorch. RNG states сериализовать совместимыми primitive/tensor types и проверить round trip.
- Tests TEST-U01..04, integration checkpoint/resume. DoD: config errors до GPU work, repeatable hashes, environment provenance, safe output paths и recovery evidence.

### `src/train.py`

- `train_one_epoch(model, loader, optimizer, criterion, device) -> dict`: weighted loss sum/N, correct/N, elapsed seconds; finite checks loss/gradients.
- `validate(model, loader, criterion, device) -> dict`: eval + inference_mode, no state update, deterministic clean val.
- `train_run(config: dict, seed: int, resume: Path | None=None, profile='core') -> Path`: output run manifest/checkpoint path.
- `main(argv=None)`: flags в REPRODUCIBILITY. No test loader in train_run. Shared engine для обеих models, отличие только policy в train dataset.
- Inputs cfg + saved split/stats; outputs history.csv, manifests, best/last checkpoint. Epoch loop 1..40, scheduler.step один раз после epoch, logs содержат LR, использованный в текущем epoch. Best: maximum val_acc, ties lower val_loss, remaining ties earlier epoch.
- NaN/Inf → run failed, сохранить diagnostic manifest, не публиковать partial как complete. KeyboardInterrupt сохраняет status interrupted; resume только с последнего complete epoch. Tests TEST-T01..03; integration 2 epochs on fixture, checkpoint → eval. DoD: valid logs, unchanged val BN stats, 352 updates/epoch, resume equality reference CPU.

### `src/evaluate.py`

- `build_conditions() -> list[dict]`: clean-0, gaussian_noise-1..5, gaussian_blur-1..5, brightness-1..5 в фиксированном порядке.
- `evaluate_condition(checkpoint: Path, config: dict, condition: dict, device: str) -> dict`: verify provenance, eval inference_mode, load frozen normalization/split, iterate all 10k in index order. Outputs probabilities/logits per index and summaries.
- `save_predictions(path, indices, targets, logits) -> None`; `main(argv=None)`.
- Invalid checkpoint/hash/condition/range or fewer than expected samples → error. Не подставлять случайно инициализированную модель. Tests TEST-E01..03; integration checkpoint + loader + metrics. DoD: counts 10000, labels order correct, no missing/duplicate index, metrics reconstructable from NPZ, condition hash shared.
- OPTIONAL `load_cifar10c_condition(root, category, severity)` можно добавить в data.py, без отдельного framework; dataset ID и benchmark outputs отделены от custom.

### `src/metrics.py`

- `confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, num_classes=10) -> np.ndarray[10,10]` int64; rows=true.
- `classification_metrics(y_true, y_pred, num_classes=10) -> dict`; `robustness_metrics(results: pd.DataFrame) -> pd.DataFrame`; `aggregate_seeds(results) -> dict[str,pd.DataFrame]`.
- `validate_matrix(results, expected_seeds=(42,123,2026)) -> None`; `main(argv=None)` flags `--input outputs/metrics --output outputs/metrics/aggregate`.
- Sample counts positive, labels 0..9, empty/missing/duplicate condition → error. Accuracy stored fractions; drops/gaps additionally pp. RR zero denominator → null + flag. No implicit averaging of missing values. Optional ECE separate explicit path.
- Tests TEST-K01..04 with hand fixtures; integration evaluation JSONs → tables. DoD: 96 input rows, six complete mCA, 32 condition aggregates, three paired seed deltas, readable schema metadata.

### Скрипты

| Файл | Интерфейс и I/O | Ошибки / зависимости | Тест и DoD |
|---|---|---|---|
| scripts/download_data.py | **Working DATA-001/002:** `main(argv=None)`; --root, raw --verify-only, --prepare-split / --verify-split, --split-seed494, --split-dir; no stats side effects | src.data; raw/split mismatch → exit1, invalid flags → exit2; verify modes read-only | TEST-D01/02; repeated verification preserves artifacts |
| scripts/prepare_normalization.py | **Working DATA-003:** offline prepare/reuse; --root, --split-path, --output, read-only --verify-only | src.data; errors exit1, invalid flags exit2; original train only | TEST-D03/CLI + explicit actual opt-in |
| scripts/run_experiments.py (новый) | `main`; --configs [2 paths], --seeds [3 ints], --profile core/smoke, --device, --dry-run, --resume-incomplete; шесть train_run | utils/train; sequential default, no GPU oversubscription; wrong config/duplicate ID → error | TEST-T03/S01; dry run ровно 6 core entries |
| scripts/run_all_evaluations.py | `main`; --checkpoints-root, --device, --require-complete, --dry-run; calls evaluate; 96 entries | evaluate/utils; missing checkpoint or stale hash → nonzero; skip только verified complete result | TEST-E02; rerun idempotent |
| scripts/make_plots.py | `load_tables(root)`, `plot_corruption_grid(...)`, `plot_learning_curves(...)`, `plot_robustness(...)`, `plot_tradeoff(...)`, `plot_per_class(...)`, `plot_confusion(...)`, `main`; --metrics, --runs, --output | metrics/data/matplotlib; missing columns/NaNs/source → error, никакой fake data | TEST-P01; figure manifest + PNG/PDF |

### CORR-003 operational plotting boundary (implemented 18.11.2026)

`scripts/make_plots.py` implements only `select_train_ids`, `prepare_grid_pixels`,
`load_audit_grid`, reusable `plot_corruption_grid(pixels, rows) -> Figure`,
`export_audit` and explicit `--corruption-grid` CLI. The general metrics interfaces
in the table above remain planned. Raw verification and frozen split verification
precede selection: smallest original ID/class among train45k, class order0..9.
Only original train decoded; existing raw loader also hashes official test bytes.

Grid shape[10,16,3,32,32], CPU float32 raw[0,1]; clean0, noise1..5, blur1..5,
brightness1..5. No crop/flip/Normalize/auto-contrast. Local audit key frozen before
viewing: stable_seed(494,'cifar10-python/original-train/corr003-audit',original_id,
'gaussian_noise','v1'). Fresh CPU generator per noise condition shares epsilon.
Production train/test streams and stable_seed unchanged. Matplotlib Figure/Agg,
nearest RGB sampling,160dpi,3520x2320 PNG and22x14.5inch one-page PDF.

Source JSON records inputs, split, IDs/labels/classes, key, params, pixel hash,
LF-normalized implementation hashes, versions and PNG/PDF byte hashes. Exactly
three curated figure files tracked; JSON LF and binary exports protected by
scoped attributes. Complete reruns verify read-only; partial/mismatch fails.
All exports rendered in a temporary sibling, then exclusive hard links publish
PNG/PDF and manifest last. Publication failure can leave a partial set: preserve
it, use a new empty directory. This is bounded audit behavior, not REPRO-001.
Later implementation edits intentionally invalidate strict current-source
verification of this historical artifact; never regenerate it to hide mismatch.
Agent visual audit PASS; independent Person5/team review PENDING. [Evidence](../report/corr003_evidence.json).

## Вспомогательные Python-файлы и notebook

`src/__init__.py`, `tests/__init__.py` остаются без business logic. `tests/conftest.py`: `synthetic_images`, `tiny_dataset`, `tiny_config`, `tmp_run` fixtures; no implicit downloads. Каждый `tests/test_*.py` отвечает за одноимённый module; функции и ожидаемые fixtures перечислены в TESTING_PLAN. `test_protocol.py` проверяет controlled fields и completeness; `test_smoke.py` вызывает CLI в subprocess с explicit temp outputs и проверяет chain data→train→eval→plots. Notebook содержит headings/data counts/train-only samples и calls data module; не импортируется production pipeline. DoD: reproducible restart and run all, нет test tuning.

## Полная схема configuration v1

Оба файла самостоятельные YAML, не наследуют друг друга. Валидация сравнивает
common fields, чтобы файлы не разошлись. Значения ниже реализованы в YAML;
schema_version1 фиксирует core scientific constants из proposed protocol,
их изменение требует согласованного изменения protocol/schema/tests. Freeze
командой ещё pending. Это preparation schema, не schema для будущих smoke/debug
overrides: отдельные profiles/validators появятся в TRAIN/TEST tasks.

| Key | Default / смысл |
|---|---|
| schema_version | 1 |
| model_name | clean_cnn / robust_cnn |
| architecture | compact_cnn_v1 |
| num_classes, dropout | 10; 0.2 |
| dataset.name, dataset.root | cifar10; data/raw |
| dataset.split_path | data/splits/cifar10_split_v1.npz |
| dataset.normalization_path | data/splits/normalization.json |
| dataset.split_sha256, normalization_sha256 | Verified NPZ/stats FILE hashes симметричны в обоих YAML после DATA-002/003. Schema допускает null для unresolved preparation; config loader не verifies artifact, make_loaders verifies |
| dataset.split_seed | 494; общий, не run seed |
| dataset.train_size, val_size, test_size | 45000; 5000; 10000 |
| seeds | [42,123,2026] |
| training.epochs, batch_size | 40; 128 |
| training.optimizer, lr, weight_decay | AdamW; 0.001; 0.0001, все trainable params |
| training.betas, eps | [0.9,0.999]; 1e-8 |
| training.loss | cross_entropy, mean reduction, no label smoothing |
| training.scheduler, t_max, eta_min | cosine; 40; 0.0 |
| training.early_stopping, amp, gradient_clip | false; false; null |
| training.selection | clean_val_accuracy_then_loss_then_earliest |
| training.crop_size, crop_padding, crop_padding_mode | 32; 4; constant zero |
| training.horizontal_flip_p | 0.5 |
| augmentation.enabled | false / true |
| augmentation.probabilities | Clean: [1,0,0,0]; Robust: [0.50,0.17,0.17,0.16], порядок clean/noise/blur/brightness |
| augmentation.train_severities | [1,2,3] uniform conditional |
| corruptions.noise_sigma | [0.02,0.05,0.08,0.12,0.18] |
| corruptions.blur_sigma | [0.3,0.7,1.1,1.5,2.0] |
| corruptions.blur_kernel_rule, blur_padding | 2*ceil(3*sigma)+1; reflect |
| corruptions.brightness_factor | [0.90,0.80,0.70,0.55,0.40] |
| evaluation.severities, batch_size, seed | [1,2,3,4,5]; 256; 100494 |
| runtime.device, num_workers, pin_memory | auto; 0; auto (=CUDA only) |
| runtime.deterministic, persistent_workers | true; false |
| output.root, checkpoint_root, metrics_root | outputs/runs; outputs/checkpoints; outputs/metrics |
| output.predictions_root, figures_root | outputs/predictions; outputs/figures |

Device/paths могут отличаться между машинами и сохраняются в execution metadata.
В пределах одной controlled пары dataset paths/hashes и runtime должны совпадать;
output paths могут различаться. Config v1 принимает переносимые repository-relative
пути с `/`, без `..`; cwd не зашивается в config. Отсутствующие файлы допускаются
при чтении config, но не являются verified data. Paired core training выбирает
один backend/hardware; trainer позже проверяет artifact existence и hashes.

Normalization values вычисляются до обучения только на 45k train, затем замораживаются;
в preparation config путь и null hash, позже hash сверяется с реальным artifact.
Это устраняет неоднозначность разных «стандартных CIFAR stats». Будущий hash identity
считается по resolved config со stats/split/architecture/policy; operational
timestamp/path excluded, а seed и profile добавляются в run identity (REPRO-001).

## Артефакты, имена, retention

Run ID: `<profile>-v1-<model_name>-s<seed>-<config_hash8>`; condition ID: `clean-0`, `gaussian_noise-1`, `gaussian_blur-5`, `brightness-3` и т. п. Повтор с другим code/config hash — новый run ID; overwrite запрещён, unless проверенный epoch resume того же run. UUID attempt в manifest различает failed/restarted attempts без смешивания histories.

- `outputs/runs/<run_id>/`: `config_resolved.yaml`, `environment.json`, `run_manifest.json`, `history.csv`, `status.json`. Raw logs сохранять до сдачи и как минимум до конца grade review.
- `checkpoints/<run_id>/`: `best.pt`, `last.pt`, hashes; best нужен для результатов, last — recovery. Хранить все шесть best и metadata независимо от Git. Не удалять last до verified backup.
- `predictions/<run_id>/<condition>.npz`: indices int64, targets int64, logits float32 [10000,10]; ~0.56 MB uncompressed/condition, ~54 MB/96. Optional confidence derived, no redundant image copies.
- `metrics/<run_id>/<condition>.json`: schema + scalars + counts + CM/per-class + checkpoint/condition hashes. Aggregate CSV/JSON, source manifest и figures можно включить в release bundle/curated report artifacts.
- `data/splits/` не попадает под текущий ignore raw/processed; small indices/stats можно хранить в Git с provenance. Normalized image cache по умолчанию выключен.
- `.gitignore`: future outputs/runs и outputs/predictions обязательно игнорировать; не помещать checkpoints в Git. Внешний shared storage выбирает команда; `artifact_manifest.json` хранит location, bytes, SHA-256, owner и backup status. Нужна вторая копия на другом устройстве/хранилище.

Raw CIFAR uint8 ~184.32 MB для 60k; один test float32 массив ~122.88 MB; 15 corrupted float32 test copies ~1.84 GB. Поэтому corruption генерируется CPU per image, один condition за раз; images не дублируются в RAM для каждой модели. Cache optional, canonical float32 + hash, без повторного uint8 quantization. Checkpoint model weights ~1.16 MB, Adam states увеличивают размер; реальный размер измерить, не считать прогноз GPU memory доказанным.
