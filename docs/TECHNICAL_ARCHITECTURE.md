# Техническая архитектура

Это преимущественно проект будущих интерфейсов. SETUP-003 реализует только
config validation в utils, оба YAML и test_utils; DATA-001 реализует load_cifar10,
file/partition inspection и acquisition CLI/manifest. DATA-002 реализует persisted
create_split/load_split, provenance/reuse/failure safety и split CLI. Остальные production APIs ниже
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
их tests и environment checker; полный pipeline ещё отсутствует. Report source
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

### `src/data.py`

- `load_cifar10(root: Path, train: bool, download=False) -> torchvision.datasets.CIFAR10`: raw RGB/PIL + labels, dataset integrity через torchvision; offline missing data → actionable error, не silent network call.
- **DATA-001 implemented:** explicit official-HTTPS acquisition, archive/seven-file checksum verification before unpickle, requested partition semantic contract; `inspect_cifar10(root)` проверяет оба original partitions только для integrity. Raw CLI: `--root`/`--verify-only`, schema-v1 local manifest ignored. Stats/transforms APIs ниже остаются future.
- **DATA-002 implemented:** `create_split(targets: Sequence[int], split_seed: int, out_dir: Path) -> dict` создаёт пару или полностью проверяет existing pair; возвращает точный schema-v1 manifest (без runtime status/path fields). Только Python int494; targets — integer sequence/array `[50000]`, classes0..9, ровно5000/class, без bool/float/object. Ошибки split contract — `DataIntegrityError`, без silent overwrite.
- `load_split(path: Path, targets) -> tuple[np.ndarray, np.ndarray]`: read-only; path — NPZ с canonical filename `cifar10_split_v1.npz`, manifest — sibling `split_manifest.json`. Возвращает sorted little-endian int64 (`<i8`) arrays `[45000]`, `[5000]`. Проверяет dtype/range/uniqueness/disjointness/exhaustiveness/class alignment, exact protocol replay и full manifest equality с независимо вычисленным expected record. Replay служит oracle проверки; возвращаются persisted arrays.
- `compute_normalization(raw_data: np.ndarray, train_indices) -> dict`: channel mean/std float64 accumulation по uint8/255, save full precision; только 45k.
- `IndexedCIFAR10(base, indices, mode, normalization, run_seed, epoch=0, policy=None)`: `__getitem__ -> (Tensor[3,32,32], int label, int original_index)`; `set_epoch(epoch)` до loader iteration. Train crop/flip и robust policy; val/test deterministic.
- `CorruptedDataset(base, condition, normalization, eval_seed)`: одинаковые test original_index и per-image noise для всех моделей.
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

### `src/models.py`

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
| scripts/download_data.py | **Working DATA-001/002:** `main(argv=None)`; --root, raw --verify-only, --prepare-split / --verify-split, --split-seed494, --split-dir. Stats/loaders future DATA-003 | src.data; raw/split mismatch → exit1, invalid flags → exit2; verify modes read-only | TEST-D01/02; repeated verification preserves artifacts |
| scripts/run_experiments.py (новый) | `main`; --configs [2 paths], --seeds [3 ints], --profile core/smoke, --device, --dry-run, --resume-incomplete; шесть train_run | utils/train; sequential default, no GPU oversubscription; wrong config/duplicate ID → error | TEST-T03/S01; dry run ровно 6 core entries |
| scripts/run_all_evaluations.py | `main`; --checkpoints-root, --device, --require-complete, --dry-run; calls evaluate; 96 entries | evaluate/utils; missing checkpoint or stale hash → nonzero; skip только verified complete result | TEST-E02; rerun idempotent |
| scripts/make_plots.py | `load_tables(root)`, `plot_corruption_grid(...)`, `plot_learning_curves(...)`, `plot_robustness(...)`, `plot_tradeoff(...)`, `plot_per_class(...)`, `plot_confusion(...)`, `main`; --metrics, --runs, --output | metrics/data/matplotlib; missing columns/NaNs/source → error, никакой fake data | TEST-P01; figure manifest + PNG/PDF |

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
| dataset.split_sha256, normalization_sha256 | verified NPZ file SHA-256 в обоих YAML после DATA-002; normalization null до DATA-003. Schema допускает null для unresolved preparation; config loader не verifies artifact |
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
