# Подробный implementation roadmap

Дата: 30.10.2026. AUDIT-001 DONE; SETUP-001/002/003, PROTO-001 и TEAM-001
**PARTIAL** по полному DoD. У SETUP-002/003 вся локальная техническая реализация
и acceptance выполнены, human reviewer approval ещё pending. DATA-001: raw
loader/acquisition/CLI/manifest и offline tests реализованы; полный DoD PARTIAL
до independent human review. DATA-002 также technical acceptance PASS/full DoD
PARTIAL: actual split494/45k/5k, manifest/hash/reuse verified;282default PASS/5SKIP,
отдельно5actual PASS/0SKIP. Actual evidence — в reproduction record/data002_evidence.json.
Остальные tasks TODO. Training/evaluation не запускались. Evidence: [CURRENT_STATUS](CURRENT_STATUS.md),
[reproduction record](../report/reproduction.md). Канонические interfaces:
[TECHNICAL_ARCHITECTURE](TECHNICAL_ARCHITECTURE.md); methodology:
[EXPERIMENT_PROTOCOL](EXPERIMENT_PROTOCOL.md); tests: [TESTING_PLAN](TESTING_PLAN.md).

## Как выполнять карточки

Task dependencies имеют приоритет над номерами phase. Независимые API/fixtures/report
drafts делаются параллельно; core compute начинается только после gates. Person1..6 —
роли, mapping имён pending. Оценки — active person-hours, не GPU wall time/promises.
В каждом task20 полей; Input/Output/Artifacts разделены. Полный DoD = acceptance +
test evidence + reviewer approval + updated status/artifact paths. Foundation
исторически выполнялся в codex/setup-foundation; теперь работа и публикация
идут только в main по актуальному указанию пользователя. Self-review не считается reviewer approval,
поэтому текущие tasks PARTIAL даже при technical acceptance PASS. Это не блокирует
независимую техническую подготовку; external approvals нужны до соответствующих gates.

Актуальное указание пользователя после foundation: все изменения, маленькие
проверенные commits и обычный push выполняются только в `main`; feature branches
не создаются. Поля Branch и Commit/PR scope ниже — исторические рекомендации
первоначального плана; их branch names заменены этим правилом, границы задач и
требования проверок сохранены. DONE не ставится по наличию filename или push.
Диапазоны task IDs обозначают существующие IDs, не скрытые задачи.

## Phases и evidence gates

| Phase | Task IDs | Выход / ownership / проверка |
|---|---|---|
| 0. Аудит и freeze | AUDIT-001; SETUP-001; PROTO-001; TEAM-001 | Документы и source registry, Person 1; проверка official conflicts и review protocol. |
| 1. Repository/environment | SETUP-002; SETUP-003; REPRO-001 | Общее env/config/identity, Persons1/6; import+config tests и manifest evidence. |
| 2. Dataset/split/EDA | DATA-001; DATA-002; DATA-003; DATA-004 | Raw→saved split/stats/loaders, Person 2; integrity/counts/order tests. |
| 3. Corruptions | CORR-001; CORR-002; CORR-003 | Pixel-space module и grid, Person 2; numeric tests+visual review. |
| 4. CNN | MODEL-001 | Model contract, Person 3; shapes/288746params/modes tests. |
| 5. Training infrastructure | TRAIN-001; TRAIN-002; TRAIN-004 | Shared train loop, checkpoint, resume, orchestrator; Persons3/6; tiny integration. |
| 6. Baseline sanity и end-to-end | TRAIN-003; TEST-001; TEST-002 | Debug overfit, smoke и scientific gate; Persons1/3/6; не core metrics. |
| 7. Clean full training | CLEAN-001 | Frozen seed 42, Person 3; history 40 и validated best. |
| 8. Baseline corruption evaluation | EVAL-001; EVAL-002 | 16 conditions firstbaseline, Person 5; no post-test policy tuning. |
| 9. Robust policy | ROBUST-001 | Person 4 работает ДО core training/test; это phase label, dependencies важнее номера. |
| 10. Robust full training | ROBUST-002 | Frozen counterpart42, Person 4; equal budget and identity evidence. |
| 11. Repeated seeds | CLEAN-002; CLEAN-003; ROBUST-003; EVAL-003; EVAL-004 | Sixruns/core96,Persons3/4/5; registry/hash/cardinality audit. |
| 12. Metrics/statistics | METRIC-001; METRIC-002; METRIC-003 | Person 5; API заранее на fixtures, final aggregate после 96results. Optional отдельно. |
| 13. Figures/failures | PLOT-001; PLOT-002; PLOT-003; ANALYSIS-001 | Persons4/6; exporter заранее, final plots от frozen tables. |
| 14. Optional benchmark | BENCH-001 | Person 4; только после core+repro,90extra и отдельный dataset ID. |
| 15. QA/reproducibility | TEST-003; REPRO-002 | Person 6 + independent reviewer; suite и fresh-env smoke. |
| 16. Final report | REPORT-001; REPORT-002; REPORT-003 | Persons1/4 ведут; все пишут свои chapters; draft начинается рано. |
| 17. Presentation/rehearsals | PRESENT-001; PRESENT-002; PRESENT-003 | Person 4 координирует, all6 speak;2full rehearsals, offline backups. |
| 18. Release/submission | RELEASE-001; RELEASE-002 | Persons6/1; candidate до independent reproduction, final package после report/rehearsals. |

## Dependency graph и critical path

Основной путь: SETUP-001 → PROTO-001 → SETUP-003 → DATA-002/003 + CORR-001/002 + MODEL-001 → TRAIN-001/002 → ROBUST-001 + TRAIN-003/004 + TEST-001/002 → CLEAN-001/002/003 и ROBUST-002/003 → EVAL-004 → METRIC-002 → ANALYSIS-001/PLOT-003 → REPORT-002/003 → PRESENT-002/003 → RELEASE-002. SETUP-002 и REPRO-001 входят в ранние gates. RELEASE-001 candidate → TEST-003 → REPRO-002 → RELEASE-002. Формальные prerequisite edges каждой карточки ниже — источник истины; graph здесь агрегирован.

Параллельно: Person 2 data/corruptions; Person 3 model/trainer на synthetic fixtures; Person 4 policy tests после corruption API и writing/failure protocol; Person 5 metrics/eval fixtures; Person 6 env/metadata/plots; Person 1 requirements/config/report. Не ждать 6 runs для написания Methods или plot functions. Core runs могут идти sequential на одном GPU; между seeds писать docs/review, не переписывать frozen scientific config.

## Карточки задач

### AUDIT-001 — Аудит и planning package

- **1. ID:** AUDIT-001.
- **2. Title:** Аудит и planning package.
- **3. Purpose:** Зафиксировать факты до кода.
- **4. Current status:** DONE.
- **5. Priority:** MUST.
- **6. Owner:** Person 1 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** —.
- **8. Steps:** 1) read-only Git/file inventory; 2) отделить source от предположений; 3) написать 14 docs.
- **9. Files:** docs/*.md.
- **10. Functions/classes:** нет production functions.
- **11. Inputs:** 29 файлов и S3.
- **12. Outputs:** 14planning docs.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Проверить links/IDs/matrix/tree.
- **15. Acceptance:** Все 14 docs непустые, original files unchanged.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 2-3 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Не принять scaffold за реализацию.
- **19. Branch:** текущая planning worktree, без commits.
- **20. Commit/PR scope:** Аудит и planning package; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 1.

### SETUP-001 — Получить course sources и proposal

- **1. ID:** SETUP-001.
- **2. Title:** Получить course sources и proposal.
- **3. Purpose:** Установить официальные ограничения.
- **4. Current status:** PARTIAL — local source search/register выполнены; latest LMS/rubric, submitted version/receipt, AI-policy precedence, cutoff/slot и independent source review pending. Evidence: report/course_materials/README.md, report/proposal.md, report/ai_usage.md.
- **5. Priority:** MUST.
- **6. Owner:** Person 1 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** AUDIT-001.
- **8. Steps:** 1) получить originals от команды/LMS; 2) записать version/page/date и AI policy; 3) сверить rubric/deadlines.
- **9. Files:** report/proposal.md; report/course_materials/README.md; report/ai_usage.md; docs/REQUIREMENTS.md.
- **10. Functions/classes:** source registry.
- **11. Inputs:** S3 и доступ команды к LMS.
- **12. Outputs:** verified requirements register.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Другой участник сверяет original locator.
- **15. Acceptance:** Все A01..A11 подтверждены либо явно pending с owner.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 2-3 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Не выдумать submitted proposal или permission.
- **19. Branch:** codex/setup-001.
- **20. Commit/PR scope:** Получить course sources и proposal; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 1.

### PROTO-001 — Review и freeze protocol v1

- **1. ID:** PROTO-001.
- **2. Title:** Review и freeze protocol v1.
- **3. Purpose:** Предотвратить post-test tuning.
- **4. Current status:** PARTIAL — technical consistency review и corrections выполнены (docs/SETUP_DECISIONS.md); actual team acceptance, Persons2/5 review, source conflicts и freeze pending.
- **5. Priority:** MUST.
- **6. Owner:** Person 1 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** AUDIT-001; SETUP-001.
- **8. Steps:** 1) проверить seeds/split/selection; 2) принять severity и ≤2pp ориентир; 3) записать freeze/date и change policy.
- **9. Files:** docs/EXPERIMENT_PROTOCOL.md; docs/PROJECT_OVERVIEW.md.
- **10. Functions/classes:** protocol version.
- **11. Inputs:** planning docs и official corrections.
- **12. Outputs:** frozen protocol decisions.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Review Persons2/5, no test data consulted.
- **15. Acceptance:** Scope6runs/96conditions принят, конфликты источников разобраны.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 2-3 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Превратить recommendation в official rule.
- **19. Branch:** codex/proto-001.
- **20. Commit/PR scope:** Review и freeze protocol v1; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 1.

### TEAM-001 — Назначить owners и reviews

- **1. ID:** TEAM-001.
- **2. Title:** Назначить owners и reviews.
- **3. Purpose:** Обеспечить параллельную работу.
- **4. Current status:** PARTIAL — role/file plan проверен, candidate roster найден; submitted roster, names-to-roles mapping, availability/backup и participant/reviewer acceptance pending.
- **5. Priority:** MUST.
- **6. Owner:** Person 1 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** AUDIT-001.
- **8. Steps:** 1) заменить Person 1..6 именами; 2) согласовать availability; 3) распределить files/PR и reviewers.
- **9. Files:** docs/TEAM_TASKS.md.
- **10. Functions/classes:** ownership table.
- **11. Inputs:** 6 участников/нагрузка.
- **12. Outputs:** named owners и review pairs.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Все участники подтверждают responsibilities.
- **15. Acceptance:** Каждый file/task имеет одного primary owner.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 1-2 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Два человека одновременно правят train.py.
- **19. Branch:** codex/team-001.
- **20. Commit/PR scope:** Назначить owners и reviews; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 1.

### SETUP-002 — Проверить и зафиксировать environment

- **1. ID:** SETUP-002.
- **2. Title:** Проверить и зафиксировать environment.
- **3. Purpose:** Получить общий CPU/GPU runtime.
- **4. Current status:** PARTIAL — local technical acceptance PASS: isolated Python3.12.10,38 exact versions, pip check/imports/CPU+CUDA smoke, install recipe/sanitized evidence. Остался formal human reviewer approval по DoD; independent fresh-machine replay относится к REPRO-002.
- **5. Priority:** MUST.
- **6. Owner:** Person 6 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** AUDIT-001.
- **8. Steps:** 1) создать venv Python 3.12; 2) проверить согласованную torch/vision pair и pytest; 3) pip check/import/CPU smoke, затем GPU/driver profile.
- **9. Files:** requirements.txt; requirements-lock.txt; docs/REPRODUCIBILITY.md.
- **10. Functions/classes:** environment capture.
- **11. Inputs:** candidate versions и hardware.
- **12. Outputs:** lock+install recipe+env metadata.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** pip check; tiny forward/backward CPU и used GPU.
- **15. Acceptance:** Versions exact; supported backend реально проверен.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 4-6 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Metadata не равны working imports; CUDA mismatch.
- **19. Branch:** codex/setup-002.
- **20. Commit/PR scope:** Проверить и зафиксировать environment; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 6.

### SETUP-003 — Schema и common configs

- **1. ID:** SETUP-003.
- **2. Title:** Schema и common configs.
- **3. Purpose:** Сделать параметры однозначными.
- **4. Current status:** PARTIAL — local technical acceptance PASS: оба YAML, schema/config CLI, controlled validation,116 passing tests. Остались formal human reviewer approval и team protocol acceptance; data artifacts относятся к DATA-002/003 и будущему training gate.
- **5. Priority:** MUST.
- **6. Owner:** Person 1 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** PROTO-001; SETUP-002.
- **8. Steps:** 1) реализовать validate_config; 2) заполнить все defaults; 3) controlled equality только разрешённых differences.
- **9. Files:** configs/clean_cnn.yaml; configs/robust_cnn.yaml; src/utils.py; tests/test_utils.py.
- **10. Functions/classes:** load_config; validate_config; validate_controlled_configs.
- **11. Inputs:** architecture config table.
- **12. Outputs:** 2valid configs + validation errors.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-U01.
- **15. Acceptance:** Comment-only/unknown/bad probabilities fail; common fields equal.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Случайно разные optimizer/dropout.
- **19. Branch:** codex/setup-003.
- **20. Commit/PR scope:** Schema и common configs; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 1.

### DATA-001 — Download и integrity CIFAR-10

- **1. ID:** DATA-001.
- **2. Title:** Download и integrity CIFAR-10.
- **3. Purpose:** Получить проверенный dataset.
- **4. Current status:** PARTIAL, local technical acceptance PASS;180offline PASS/3default opt-in SKIP, separately3actual PASS/0SKIP. Official archive/files/counts verified; evidence in report/reproduction.md; independent human artifact/code/test reviewer approval pending.
- **5. Priority:** MUST.
- **6. Owner:** Person 2 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** SETUP-002.
- **8. Steps:** 1) load official archive; 2) verify checksum/files; 3) count labels/images и offline errors.
- **9. Files:** src/data.py; scripts/download_data.py; tests/test_data_pipeline.py.
- **10. Functions/classes:** load_cifar10; download main.
- **11. Inputs:** official CIFAR source.
- **12. Outputs:** verified raw dataset + manifest.
- **13. Artifacts:** ignored data/raw/cifar-10-python.tar.gz, cifar-10-batches-py/ и cifar10_integrity_v1.json (schema1); tests/test_cifar10_local.py; report/reproduction.md. Actual verification evidence recorded separately; no dataset artifacts in Git.
- **14. Testing:** TEST-D01.
- **15. Acceptance:** 50k/10k RGB32×32,10 classes, checksums pass.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Network failure/неверный archive.
- **19. Branch:** main only (current explicit user workflow).
- **20. Commit/PR scope:** Small verified commits: loader/acquisition + tests; CLI/manifest + tests; bounded source requests + tests; evidence/docs. Ordinary push after each commit; no feature branch/PR required by current user workflow.

### DATA-002 — Persist stratified split

- **1. ID:** DATA-002.
- **2. Title:** Persist stratified split.
- **3. Purpose:** Один split для всех seeds.
- **4. Current status:** PARTIAL — local technical acceptance PASS31.10.2026; actual45k/5k seed494,4500/500class, disjoint/exhaustive, provenance/hash/reuse checks PASS; independent human code/artifact/test reviewer approval pending.
- **5. Priority:** MUST.
- **6. Owner:** Person 2 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** DATA-001; SETUP-003.
- **8. Steps:** 1) split per class seed 494; 2) save indices+hashes; 3) reject mismatch/tamper.
- **9. Files:** src/data.py; data/splits/cifar10_split_v1.npz; data/splits/split_manifest.json; tests/test_data_pipeline.py.
- **10. Functions/classes:** create_split; load_split.
- **11. Inputs:** original train targets.
- **12. Outputs:** 45k/5k indices + manifest.
- **13. Artifacts:** data/splits/cifar10_split_v1.npz (76550bytes, SHA-2561dc8e61a237d7e8fd3ac3c2dea4787e6a82f8f235d74993386af9074f6e1d284), split_manifest.json; report/data002_evidence.json и report/reproduction.md; оба YAML с тем же file hash, normalization null.
- **14. Testing:** TEST-D02 implemented; tests/test_data_pipeline.py + tests/test_split_local.py, связанные config tests. Default282PASS/5opt-in SKIP; actual5PASS/0SKIP (вместе с DATA-001). Hashes/file mtimes сохраняются при повторе; invalid/partial/tampered pair FAIL.
- **15. Acceptance:** 4500/500class; disjoint/exhaustive; rerun identical.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Создать split по training seed.
- **19. Branch:** main only (explicit persistent user workflow).
- **20. Commit/PR scope:** Small working algorithm/tests, persistence/tests, CLI/tests, actual artifacts/configs/tests/evidence, docs blocks; diff/staged files/whitespace/relevant checks before each commit, immediately ordinary push origin main. No feature branch/PR required.

### DATA-003 — Normalization и loaders

- **1. ID:** DATA-003.
- **2. Title:** Normalization и loaders.
- **3. Purpose:** Правильный deterministic input.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 2 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** DATA-002; CORR-001.
- **8. Steps:** 1) compute45k-only stats; 2) keyed crop/flip и epoch order; 3) deterministic val/test, drop_lastFalse.
- **9. Files:** src/data.py; data/splits/normalization.json; tests/test_data_pipeline.py.
- **10. Functions/classes:** compute_normalization; IndexedCIFAR10; make_loaders.
- **11. Inputs:** raw/split/config.
- **12. Outputs:** stats + typed DataLoaders.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-D03..05; TEST-C08.
- **15. Acceptance:** N×3×32×32 float32, int64 labels, paired geometry/order equal.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 5-7 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Leakage stats, global RNG collision, Windows spawn.
- **19. Branch:** codex/data-003.
- **20. Commit/PR scope:** Normalization и loaders; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 2.

### DATA-004 — Train-only EDA и visual inventory

- **1. ID:** DATA-004.
- **2. Title:** Train-only EDA и visual inventory.
- **3. Purpose:** Проверить labels и input quality.
- **4. Current status:** TODO.
- **5. Priority:** SHOULD.
- **6. Owner:** Person 2 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** DATA-003; CORR-003.
- **8. Steps:** 1) counts/classes/sample rows; 2) show normalized inverse correctly; 3) restart/run all без production copy.
- **9. Files:** notebooks/01_eda_and_sanity.ipynb.
- **10. Functions/classes:** calls src.data and grid functions.
- **11. Inputs:** train/val + fixed train sample IDs.
- **12. Outputs:** executed EDA notebook.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Restart&Run All, visual review Person 5.
- **15. Acceptance:** Counts и labels совпадают manifest, test tuning отсутствует.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 2-3 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Скрытый notebook state/auto contrast.
- **19. Branch:** codex/data-004.
- **20. Commit/PR scope:** Train-only EDA и visual inventory; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 2.

### CORR-001 — Noise и stable RNG primitives

- **1. ID:** CORR-001.
- **2. Title:** Noise и stable RNG primitives.
- **3. Purpose:** Общие pixel corruptions.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 2 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** PROTO-001; SETUP-002.
- **8. Steps:** 1) stable_seed SHA256; 2) severity mapping; 3) CPU noise local generator+clamp без mutation.
- **9. Files:** src/corruptions.py; src/utils.py; tests/test_corruptions.py; tests/test_utils.py.
- **10. Functions/classes:** stable_seed; severity_params; gaussian_noise.
- **11. Inputs:** float32 pixel tensors.
- **12. Outputs:** deterministic noise API.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-C01..04; TEST-U02.
- **15. Acceptance:** Same index→same ε across6models; valid shape/range.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 4-5 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Seed зависит от run seed или batch order.
- **19. Branch:** codex/corr-001.
- **20. Commit/PR scope:** Noise и stable RNG primitives; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 2.

### CORR-002 — Gaussian blur и darkening

- **1. ID:** CORR-002.
- **2. Title:** Gaussian blur и darkening.
- **3. Purpose:** Завершить 3 семейства.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 2 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** CORR-001.
- **8. Steps:** 1) odd kernels3/7/9/11/13; 2) reflect Gaussian; 3) factors.9/.8/.7/.55/.4; 4) errors/identity helpers.
- **9. Files:** src/corruptions.py; tests/test_corruptions.py.
- **10. Functions/classes:** gaussian_blur; darken; apply_corruption.
- **11. Inputs:** pixel tensors + severity.
- **12. Outputs:** 3-family corruption module.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-C05..06, C08.
- **15. Acceptance:** Constant blur preserved, correct factors, CPU tests pass.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Неверная padding/brightening вместо darkening.
- **19. Branch:** codex/corr-002.
- **20. Commit/PR scope:** Gaussian blur и darkening; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 2.

### CORR-003 — Sanity examples и corruption audit

- **1. ID:** CORR-003.
- **2. Title:** Sanity examples и corruption audit.
- **3. Purpose:** Проверить визуальный смысл.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 2 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** CORR-002; DATA-001.
- **8. Steps:** 1) выбрать first train ID/class; 2) export clean+15 conditions; 3) review padding/clipping/range с Person 5.
- **9. Files:** scripts/make_plots.py; outputs/figures/corruption_examples.png.
- **10. Functions/classes:** plot_corruption_grid.
- **11. Inputs:** 10fixed train images.
- **12. Outputs:** grid PNG/PDF + source IDs.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Visual review + numeric TEST-C01..08.
- **15. Acceptance:** Visible increasing degradation, no normalize-before-corrupt bug.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 2-3 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Принимать красивую картинку за numeric proof.
- **19. Branch:** codex/corr-003.
- **20. Commit/PR scope:** Sanity examples и corruption audit; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 2.

### MODEL-001 — CompactCNN v1

- **1. ID:** MODEL-001.
- **2. Title:** CompactCNN v1.
- **3. Purpose:** Одна архитектура с нуля.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 3 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** PROTO-001; SETUP-002.
- **8. Steps:** 1) 6Conv+BN+ReLU/3pool; 2) GAP/dropout.2/linear; 3) explicit init и architecture ID.
- **9. Files:** src/models.py; tests/test_models.py.
- **10. Functions/classes:** CompactCNN; initialize_weights; forward.
- **11. Inputs:** N×3×32×32.
- **12. Outputs:** N×10 logits and state_dict.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-M01..03.
- **15. Acceptance:** 288746 params; shapes/modes/finite backward correct.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Softmax до CE, лишние conv biases.
- **19. Branch:** codex/model-001.
- **20. Commit/PR scope:** CompactCNN v1; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 3.

### TRAIN-001 — Shared training/validation engine

- **1. ID:** TRAIN-001.
- **2. Title:** Shared training/validation engine.
- **3. Purpose:** Одинаковый budget и selection.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 3 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** SETUP-003; DATA-003; MODEL-001.
- **8. Steps:** 1) weighted epoch metrics; 2) AdamW/cosine40; 3) clean-val selection/ties; 4) no test loader.
- **9. Files:** src/train.py; tests/test_training.py.
- **10. Functions/classes:** train_one_epoch; validate; train_run.
- **11. Inputs:** loaders/model/config.
- **12. Outputs:** history+best score+status.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-T01; TEST-S02.
- **15. Acceptance:** 40epochs/14080steps; val eval mode; loss weighted byN.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 5-7 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Batch mean averaging/BN update during val.
- **19. Branch:** codex/train-001.
- **20. Commit/PR scope:** Shared training/validation engine; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 3.

### TRAIN-002 — Checkpoint и epoch resume

- **1. ID:** TRAIN-002.
- **2. Title:** Checkpoint и epoch resume.
- **3. Purpose:** Защита от потери training.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 3 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** TRAIN-001; REPRO-001.
- **8. Steps:** 1) atomic best/last; 2) RNG+optimizer/scheduler/config identity; 3) epoch-boundary resume и append history.
- **9. Files:** src/train.py; src/utils.py; tests/test_training.py; tests/test_utils.py.
- **10. Functions/classes:** save_checkpoint_atomic; load_checkpoint.
- **11. Inputs:** run state + last checkpoint.
- **12. Outputs:** recoverable best.pt/last.pt.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-U03..04; TEST-T02.
- **15. Acceptance:** CPU uninterrupted2 epochs ==1+resume кроме timing.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 4-5 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Не восстановить LR/RNG; partial overwrite.
- **19. Branch:** codex/train-002.
- **20. Commit/PR scope:** Checkpoint и epoch resume; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 3.

### TRAIN-003 — 64-image overfit и pilot profiling

- **1. ID:** TRAIN-003.
- **2. Title:** 64-image overfit и pilot profiling.
- **3. Purpose:** Не тратить budget на broken pipeline.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 3 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** TRAIN-002; CORR-003.
- **8. Steps:** 1) отдельный debug no aug/dropout; 2) ≥95%train on64 до 300 steps; 3) pilot epoch timings/memory.
- **9. Files:** src/train.py; outputs/runs/debug-*; outputs/runs/pilot-*.
- **10. Functions/classes:** debug profile; train_run.
- **11. Inputs:** train-only subset.
- **12. Outputs:** diagnostic curves + runtime estimate.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-M02, T01; manual curve review.
- **15. Acceptance:** Gate overfit pass или cause fixed; measured t_clean/t_robust.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 2-3 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Debug overrides просочились в core.
- **19. Branch:** codex/train-003.
- **20. Commit/PR scope:** 64-image overfit и pilot profiling; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 3.

### TRAIN-004 — Six-run orchestration

- **1. ID:** TRAIN-004.
- **2. Title:** Six-run orchestration.
- **3. Purpose:** Управлять 6 runs без ручных ошибок.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 6 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** TRAIN-002; SETUP-003.
- **8. Steps:** 1) dry-run 6 specs; 2) sequential dispatch; 3) verified complete skip и incomplete resume.
- **9. Files:** scripts/run_experiments.py; tests/test_training.py.
- **10. Functions/classes:** main; train_run dispatcher.
- **11. Inputs:** 2configs×3 seeds.
- **12. Outputs:** run registry, status6.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-T03; TEST-S03.
- **15. Acceptance:** Ровно 6 core entries, no overwrite/pilot mixing.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** GPU oversubscription, false success пустого script.
- **19. Branch:** codex/train-004.
- **20. Commit/PR scope:** Six-run orchestration; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 6.

### REPRO-001 — Identity/metadata/backup infrastructure

- **1. ID:** REPRO-001.
- **2. Title:** Identity/metadata/backup infrastructure.
- **3. Purpose:** Связать outputs с inputs/code.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 6 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** SETUP-002; SETUP-003.
- **8. Steps:** 1) environment/config/code hashes; 2) run/condition manifests; 3) ignore new raw outputs, define backup and verify copy.
- **9. Files:** src/utils.py; .gitignore; outputs/runs/; tests/test_utils.py.
- **10. Functions/classes:** config_hash; capture_environment; write_json_atomic.
- **11. Inputs:** config+Git/device+paths.
- **12. Outputs:** identity functions+artifact manifest.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-U02..04; git check-ignore.
- **15. Acceptance:** All provenance fields present, collision rejected, backup checksum verified.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 4-6 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Ignored outputs исчезли/dirty SHA скрыт.
- **19. Branch:** codex/repro-001.
- **20. Commit/PR scope:** Identity/metadata/backup infrastructure; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 6.

### TEST-001 — Integration и actual smoke gate

- **1. ID:** TEST-001.
- **2. Title:** Integration и actual smoke gate.
- **3. Purpose:** Проверить complete mechanics до core.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 6 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** TRAIN-004; ROBUST-001; EVAL-001; METRIC-001; PLOT-001.
- **8. Steps:** 1) synthetic offline chain; 2) CIFAR128/64/100 smoke2models1 epoch; 3) export32 conditions and figures.
- **9. Files:** tests/conftest.py; tests/test_smoke.py; tests/test_training.py; tests/test_evaluation.py; tests/test_plots.py.
- **10. Functions/classes:** fixtures; test_smoke_pipeline.
- **11. Inputs:** verified local CIFAR + tiny configs.
- **12. Outputs:** passing smoke evidence.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-END01; unit/integration suite.
- **15. Acceptance:** 32smoke rows×100 samples; core validator rejects smoke.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 4-5 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Smoke performance принимают за final results.
- **19. Branch:** codex/test-001.
- **20. Commit/PR scope:** Integration и actual smoke gate; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 6.

### CLEAN-001 — Clean CNN seed 42 full training

- **1. ID:** CLEAN-001.
- **2. Title:** Clean CNN seed 42 full training.
- **3. Purpose:** Первый frozen baseline run.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 3 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** TRAIN-003; TEST-001; PROTO-001.
- **8. Steps:** 1) run 40 epochs; 2) monitor finite metrics; 3) verify best/last/hash and backup.
- **9. Files:** outputs/runs/core-v1-clean_cnn-s42-*; outputs/checkpoints/core-v1-clean_cnn-s42-*.
- **10. Functions/classes:** train_run.
- **11. Inputs:** frozen clean config, seed 42.
- **12. Outputs:** 1complete core run.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** History40, steps14080, checkpoint reload.
- **15. Acceptance:** best clean-val selected, no test access.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 1-2 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Interrupted run/не сохранён best.
- **19. Branch:** codex/clean-001.
- **20. Commit/PR scope:** Clean CNN seed 42 full training; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 3.

### CLEAN-002 — Clean CNN seed 123 full training

- **1. ID:** CLEAN-002.
- **2. Title:** Clean CNN seed 123 full training.
- **3. Purpose:** Второй independent training seed.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 3 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** CLEAN-001.
- **8. Steps:** 1) repeat frozen config seed 123; 2) capture env/hash; 3) backup complete artifacts.
- **9. Files:** outputs/runs/core-v1-clean_cnn-s123-*; outputs/checkpoints/core-v1-clean_cnn-s123-*.
- **10. Functions/classes:** train_run.
- **11. Inputs:** same split/config seed 123.
- **12. Outputs:** 2nd baseline run.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Same gates CLEAN-001.
- **15. Acceptance:** 40 epochs+best/last+metadata, seed recorded.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 1-2 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Менять config между seeds.
- **19. Branch:** codex/clean-002.
- **20. Commit/PR scope:** Clean CNN seed 123 full training; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 3.

### CLEAN-003 — Clean CNN seed 2026 full training

- **1. ID:** CLEAN-003.
- **2. Title:** Clean CNN seed 2026 full training.
- **3. Purpose:** Третий baseline repeat.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 3 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** CLEAN-001.
- **8. Steps:** 1) runseed2026; 2) validate logs/checkpoint; 3) register all3baseline runs.
- **9. Files:** outputs/runs/core-v1-clean_cnn-s2026-*; outputs/checkpoints/core-v1-clean_cnn-s2026-*.
- **10. Functions/classes:** train_run.
- **11. Inputs:** same split/config seed 2026.
- **12. Outputs:** 3rd baseline run + registry3.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Hash/run count audit.
- **15. Acceptance:** 3complete clean seeds без best-seed selection.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 1-2 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Cherry-pick лучшего seed.
- **19. Branch:** codex/clean-003.
- **20. Commit/PR scope:** Clean CNN seed 2026 full training; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 3.

### ROBUST-001 — Per-image corruption-aware policy

- **1. ID:** ROBUST-001.
- **2. Title:** Per-image corruption-aware policy.
- **3. Purpose:** Единственное experimental treatment.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 4 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** CORR-002; DATA-003; SETUP-003.
- **8. Steps:** 1) policy.50/.17/.17/.16; 2) severity 1..3uniform; 3) separate keyed stream, train-only hook.
- **9. Files:** src/corruptions.py; src/data.py; tests/test_corruptions.py; tests/test_protocol.py.
- **10. Functions/classes:** CorruptionPolicy; IndexedCIFAR10 policy hook.
- **11. Inputs:** clean train pixels+key.
- **12. Outputs:** robust training transform.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-C07; TEST-S01.
- **15. Acceptance:** Frequency/seed isolation pass, geometry and init paired equal.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 4-5 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Policy потребляет global RNG/меняет budget.
- **19. Branch:** codex/robust-001.
- **20. Commit/PR scope:** Per-image corruption-aware policy; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 4.

### ROBUST-002 — Robust seed 42 pilot и full run

- **1. ID:** ROBUST-002.
- **2. Title:** Robust seed 42 pilot и full run.
- **3. Purpose:** Первый robust counterpart.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 4 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** CLEAN-001; ROBUST-001; TEST-001.
- **8. Steps:** 1) train-only pilot frequencies/runtime; 2) frozen40 epoch full; 3) check paired init/order/steps.
- **9. Files:** outputs/runs/core-v1-robust_cnn-s42-*; outputs/checkpoints/core-v1-robust_cnn-s42-*.
- **10. Functions/classes:** train_run.
- **11. Inputs:** robust config seed 42.
- **12. Outputs:** 1complete robust run.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-S01 plus40epoch/14080step audit.
- **15. Acceptance:** Paired controlled fields equal, best selected clean-val.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 2-3 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Поменять p после baseline test.
- **19. Branch:** codex/robust-002.
- **20. Commit/PR scope:** Robust seed 42 pilot и full run; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 4.

### ROBUST-003 — Robust seeds 123 и 2026

- **1. ID:** ROBUST-003.
- **2. Title:** Robust seeds 123 и 2026.
- **3. Purpose:** Завершить six instances.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 4 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** ROBUST-002; CLEAN-002; CLEAN-003.
- **8. Steps:** 1) repeat2 seeds unchanged; 2) verify counterparts; 3) backup all6best/last.
- **9. Files:** outputs/runs/core-v1-robust_cnn-s123-*; outputs/runs/core-v1-robust_cnn-s2026-*; outputs/checkpoints/.
- **10. Functions/classes:** train_run.
- **11. Inputs:** frozen config, same seeds.
- **12. Outputs:** 3robust/6total complete runs.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-S01, S03; checkpoint audit.
- **15. Acceptance:** 6 runs complete, no missing/selected seed.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 2-3 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Config drift or lost checkpoint.
- **19. Branch:** codex/robust-003.
- **20. Commit/PR scope:** Robust seeds 123 и 2026; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 4.

### EVAL-001 — Single-condition evaluator

- **1. ID:** EVAL-001.
- **2. Title:** Single-condition evaluator.
- **3. Purpose:** Детерминированная inference/metrics.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 5 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** DATA-003; MODEL-001; CORR-002; METRIC-001; TRAIN-002.
- **8. Steps:** 1) build16 conditions; 2) strict checkpoint load; 3) eval/inference; 4) save indexed logits+summary/hashes.
- **9. Files:** src/evaluate.py; tests/test_evaluation.py.
- **10. Functions/classes:** build_conditions; evaluate_condition; save_predictions.
- **11. Inputs:** checkpoint+raw test+condition.
- **12. Outputs:** JSON/NPZ per condition.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-E01, E03.
- **15. Acceptance:** Counts/order/metrics reconstruct, no stochastic eval.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 4-6 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Label order mismatch/normalization repeated.
- **19. Branch:** codex/eval-001.
- **20. Commit/PR scope:** Single-condition evaluator; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 5.

### EVAL-002 — Baseline seed 42 corruption evaluation

- **1. ID:** EVAL-002.
- **2. Title:** Baseline seed 42 corruption evaluation.
- **3. Purpose:** Проверить first complete16.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 5 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** CLEAN-001; EVAL-001; TEST-002.
- **8. Steps:** 1) evaluate clean+15 conditions; 2) audit n/hash/units; 3) freeze files.
- **9. Files:** outputs/metrics/core-v1-clean_cnn-s42-*; outputs/predictions/core-v1-clean_cnn-s42-*.
- **10. Functions/classes:** evaluate_condition.
- **11. Inputs:** validation-selected clean 42 best.
- **12. Outputs:** 16 core evaluation pairs.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-E01, E03; raw logits reconstruction.
- **15. Acceptance:** 16unique conditions×10000, allstatus complete.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 1-2 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Использовать result для tuning policy.
- **19. Branch:** codex/eval-002.
- **20. Commit/PR scope:** Baseline seed 42 corruption evaluation; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 5.

### EVAL-003 — Матрица и остальные baseline evaluations

- **1. ID:** EVAL-003.
- **2. Title:** Матрица и остальные baseline evaluations.
- **3. Purpose:** Довести clean side до 48.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 5 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** EVAL-002; CLEAN-002; CLEAN-003.
- **8. Steps:** 1) manifest-based expected keys; 2) eval remaining32baseline; 3) dry-run 96specs.
- **9. Files:** scripts/run_all_evaluations.py; outputs/metrics/; outputs/predictions/; tests/test_evaluation.py.
- **10. Functions/classes:** matrix main; validate_matrix.
- **11. Inputs:** 3clean checkpoints.
- **12. Outputs:** 48baseline results + orchestrator.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-E02, T03.
- **15. Acceptance:** 3×16complete; absent robust runs explicit pending.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 2-3 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Silent skip missing condition.
- **19. Branch:** codex/eval-003.
- **20. Commit/PR scope:** Матрица и остальные baseline evaluations; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 5.

### EVAL-004 — Robust evaluations и core 96 audit

- **1. ID:** EVAL-004.
- **2. Title:** Robust evaluations и core 96 audit.
- **3. Purpose:** Полный paired experiment.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 5 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** EVAL-003; ROBUST-003.
- **8. Steps:** 1) evaluate robust48; 2) compare condition hashes across6; 3) verify 96keys/sample counts.
- **9. Files:** outputs/metrics/; outputs/predictions/.
- **10. Functions/classes:** evaluate_condition; validate_matrix.
- **11. Inputs:** 6 best checkpoints.
- **12. Outputs:** 96complete core JSON/NPZ pairs.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-E02, S01, S03.
- **15. Acceptance:** 960000 predictions,96unique rows, zero missing/stale.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 2-3 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Другие noise realizations для Robust.
- **19. Branch:** codex/eval-004.
- **20. Commit/PR scope:** Robust evaluations и core 96 audit; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 5.

### TEST-002 — Scientific validity audit до test release

- **1. ID:** TEST-002.
- **2. Title:** Scientific validity audit до test release.
- **3. Purpose:** Остановить leakage и drift.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 1 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** PROTO-001; TRAIN-003; ROBUST-001; REPRO-001.
- **8. Steps:** 1) test common split/stats/init/order/budget; 2) trace training data access; 3) freeze treatment before test.
- **9. Files:** tests/test_protocol.py; docs/EXPERIMENT_PROTOCOL.md.
- **10. Functions/classes:** test_controlled_comparison; test_no_test_loader.
- **11. Inputs:** paired pilot manifests+code review.
- **12. Outputs:** signed protocol QA gate.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-S01..03.
- **15. Acceptance:** No test tuning; identities same; bugs blocking test resolved.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Бумажный freeze после просмотра test.
- **19. Branch:** codex/test-002.
- **20. Commit/PR scope:** Scientific validity audit до test release; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 1.

### METRIC-001 — Classification и robustness metrics

- **1. ID:** METRIC-001.
- **2. Title:** Classification и robustness metrics.
- **3. Purpose:** Правильные формулы и единицы.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 5 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** PROTO-001; SETUP-002.
- **8. Steps:** 1) CM/F1/perclass; 2) drop/RR/mCA/gap; 3) denominator/error rules.
- **9. Files:** src/metrics.py; tests/test_metrics.py.
- **10. Functions/classes:** classification_metrics; robustness_metrics.
- **11. Inputs:** indexed labels/predictions and rows.
- **12. Outputs:** tested metrics API.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-K01..02, K04.
- **15. Acceptance:** Hand fixtures exact; pp != percent;no empty mean.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Неправильный macro-F1/clean included in mCA.
- **19. Branch:** codex/metric-001.
- **20. Commit/PR scope:** Classification и robustness metrics; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 5.

### METRIC-002 — Seed aggregation и result tables

- **1. ID:** METRIC-002.
- **2. Title:** Seed aggregation и result tables.
- **3. Purpose:** Ответить RQ с uncertainty.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 5 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** EVAL-004; METRIC-001.
- **8. Steps:** 1) validate96 rows; 2) per-seed mCA/gap; 3) group mean/sampleSD; 4) pairedΔ and6 tables.
- **9. Files:** src/metrics.py; outputs/metrics/aggregate/; tests/test_metrics.py.
- **10. Functions/classes:** aggregate_seeds; validate_matrix; metrics main.
- **11. Inputs:** 96verified summaries.
- **12. Outputs:** results/summary/paired/perclass tables.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-K03; matrix count assertions.
- **15. Acceptance:** n=3 everywhere;32 condition groups;allseeds represented.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** 15 conditions приняты independent repeats.
- **19. Branch:** codex/metric-002.
- **20. Commit/PR scope:** Seed aggregation и result tables; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 5.

### METRIC-003 — Optional calibration/CI

- **1. ID:** METRIC-003.
- **2. Title:** Optional calibration/CI.
- **3. Purpose:** Дополнить uncertainty без overclaim.
- **4. Current status:** TODO.
- **5. Priority:** OPTIONAL.
- **6. Owner:** Person 5 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** METRIC-002; REPORT-002.
- **8. Steps:** 1) фиксировать 15ECE bins; 2) test empty bins; 3) optionaltCI with n 3 caveat.
- **9. Files:** src/metrics.py; tests/test_metrics.py; report/final_report/report.md.
- **10. Functions/classes:** expected_calibration_error; optional_ci.
- **11. Inputs:** frozen logits +3 seed summaries.
- **12. Outputs:** separate optional analysis.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Hand ECE fixture/CI formula review.
- **15. Acceptance:** Clearly exploratory, no temperature tuning test.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 2-3 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** False significance from tiny n.
- **19. Branch:** codex/metric-003.
- **20. Commit/PR scope:** Optional calibration/CI; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 5.

### PLOT-001 — Plot infrastructure и learning curves

- **1. ID:** PLOT-001.
- **2. Title:** Plot infrastructure и learning curves.
- **3. Purpose:** Общий reproducible exporter.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 6 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** REPRO-001; METRIC-001.
- **8. Steps:** 1) table schema checks; 2) PNG300dpi/PDF; 3) loss/accuracy histories plus source manifest; 4) synthetic fixture smoke.
- **9. Files:** scripts/make_plots.py; tests/test_plots.py.
- **10. Functions/classes:** load_tables; plot_learning_curves.
- **11. Inputs:** history.csv, synthetic metrics fixture.
- **12. Outputs:** learning curves exporter.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-P01.
- **15. Acceptance:** Axes/legend/epochs/source paths verified; no fake values in final outputs.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Plot берет неполные histories.
- **19. Branch:** codex/plot-001.
- **20. Commit/PR scope:** Plot infrastructure и learning curves; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 6.

### PLOT-002 — Robustness и trade-off figures

- **1. ID:** PLOT-002.
- **2. Title:** Robustness и trade-off figures.
- **3. Purpose:** Показать RQ1/3/4.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 6 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** PLOT-001; METRIC-002.
- **8. Steps:** 1) family severity curves; 2) mean±SD ribbons; 3) clean-vs-mCA scatter; 4) annotate units and3 seeds.
- **9. Files:** scripts/make_plots.py; outputs/figures/.
- **10. Functions/classes:** plot_robustness; plot_tradeoff.
- **11. Inputs:** aggregate+per-seed tables.
- **12. Outputs:** required figures4..6.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-P01 + table/point cross-check.
- **15. Acceptance:** Every plotted value matches frozen tables.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Сгладить nonmonotonic data/SD назвать CI.
- **19. Branch:** codex/plot-002.
- **20. Commit/PR scope:** Robustness и trade-off figures; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 6.

### PLOT-003 — Per-class и confusion figures

- **1. ID:** PLOT-003.
- **2. Title:** Per-class и confusion figures.
- **3. Purpose:** Показать failures без cherry-pick.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 6 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** PLOT-002; ANALYSIS-001.
- **8. Steps:** 1) perclassdrop; 2) fixed seed 42 clean/noise5/blur5/brightness5 CM; 3) failure example IDs.
- **9. Files:** scripts/make_plots.py; outputs/figures/.
- **10. Functions/classes:** plot_per_class; plot_confusion.
- **11. Inputs:** counts/perclass tables, saved predictions.
- **12. Outputs:** required figures7..8 + failure grid.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** CMsum/support, visual review.
- **15. Acceptance:** Rows true/columns pred, source IDs,counts/normalization labelled.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Выбрать только выигрышные условия.
- **19. Branch:** codex/plot-003.
- **20. Commit/PR scope:** Per-class и confusion figures; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 6.

### ANALYSIS-001 — Failure analysis и RQ evidence map

- **1. ID:** ANALYSIS-001.
- **2. Title:** Failure analysis и RQ evidence map.
- **3. Purpose:** Интерпретировать actual results.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 4 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** METRIC-002; DATA-004.
- **8. Steps:** 1) review per-seed/family/class; 2) fixed lowest-index failure sampling; 3) distinguish observation/cause; 4) negative-result narrative.
- **9. Files:** report/final_report/report.md; outputs/metrics/aggregate/failures.csv.
- **10. Functions/classes:** index-keyed analysis from predictions.
- **11. Inputs:** 96results and curves.
- **12. Outputs:** RQ evidence table + diagnostic notes.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Peer review Person 5, claims→artifact check.
- **15. Acceptance:** Все 4RQ answered, no fabricated causal explanation.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 4-6 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Объяснить noise sensitivity без evidence.
- **19. Branch:** codex/analysis-001.
- **20. Commit/PR scope:** Failure analysis и RQ evidence map; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 4.

### BENCH-001 — Optional CIFAR-10-C evaluation

- **1. ID:** BENCH-001.
- **2. Title:** Optional CIFAR-10-C evaluation.
- **3. Purpose:** Отдельная standardized check.
- **4. Current status:** TODO.
- **5. Priority:** OPTIONAL.
- **6. Owner:** Person 4 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** EVAL-004; METRIC-002; REPRO-002.
- **8. Steps:** 1) verify2.9GBarchive; 2) loadblocks/categories; 3) evaluate90extra; 4) separate summaries/limitations.
- **9. Files:** src/data.py; src/evaluate.py; scripts/run_all_evaluations.py; tests/test_data_pipeline.py.
- **10. Functions/classes:** load_cifar10c_condition; benchmark flags.
- **11. Inputs:** six frozen checkpoints+CIFAR10C.
- **12. Outputs:** 90additional condition outputs.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Integrity/block labels/no train access.
- **15. Acceptance:** No core mixing; blur/darkening distinction explicit.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 4-6 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Storage/time; принять scales эквивалентными.
- **19. Branch:** codex/bench-001.
- **20. Commit/PR scope:** Optional CIFAR-10-C evaluation; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 4.

### TEST-003 — Полная suite и artifact QA

- **1. ID:** TEST-003.
- **2. Title:** Полная suite и artifact QA.
- **3. Purpose:** Закрыть regressions до release.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 6 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** EVAL-004; METRIC-002; PLOT-003.
- **8. Steps:** 1) rununit/integration/scientific suite; 2) verify artifacts/counts; 3) list skips/known limits.
- **9. Files:** tests/; report/reproduction.md.
- **10. Functions/classes:** test suite and matrix validator.
- **11. Inputs:** final source/metrics.
- **12. Outputs:** QA report with real commands.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** All required TEST cases.
- **15. Acceptance:** No critical scientific failures, actual smoke pass.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Формальный pass без actual integration.
- **19. Branch:** codex/test-003.
- **20. Commit/PR scope:** Полная suite и artifact QA; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 6.

### REPRO-002 — Independent fresh-machine reproduction

- **1. ID:** REPRO-002.
- **2. Title:** Independent fresh-machine reproduction.
- **3. Purpose:** Проверить инструкции и retrieval.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 6 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** TEST-003; RELEASE-001.
- **8. Steps:** 1) second person clean install+smoke; 2) retrieve onebest; 3) reproduce clean/noise5; 4) record environment/hashes.
- **9. Files:** README.md; report/reproduction.md; docs/REPRODUCIBILITY.md.
- **10. Functions/classes:** documented CLI chain.
- **11. Inputs:** release candidate+artifact backup.
- **12. Outputs:** independent reproduction log.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** TEST-END01 and frozen checkpoint comparison.
- **15. Acceptance:** Commands work independently, mismatches explained.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Only original developer can run project.
- **19. Branch:** codex/repro-002.
- **20. Commit/PR scope:** Independent fresh-machine reproduction; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 6.

### REPORT-001 — Report skeleton и verified background

- **1. ID:** REPORT-001.
- **2. Title:** Report skeleton и verified background.
- **3. Purpose:** Начать writing до результатов.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 1 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** SETUP-001; PROTO-001.
- **8. Steps:** 1) 10 chapters; 2) intro/background/data/method draft; 3) verify citations/full text; 4) map figures/tables.
- **9. Files:** report/final_report/report.md; report/final_report/references.bib; report/final_report/README.md.
- **10. Functions/classes:** report outline/build recipe.
- **11. Inputs:** rubric+protocol+primary sources.
- **12. Outputs:** methods draft + verified bibliography.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Chapter checklist FINAL_DELIVERABLES.
- **15. Acceptance:** Methods permit reproduction; pending results explicit.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 5-7 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Invent citation или write conclusion early.
- **19. Branch:** codex/report-001.
- **20. Commit/PR scope:** Report skeleton и verified background; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 1.

### REPORT-002 — Results/discussion/limitations

- **1. ID:** REPORT-002.
- **2. Title:** Results/discussion/limitations.
- **3. Purpose:** Сформировать scientific narrative.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 4 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** REPORT-001; ANALYSIS-001; PLOT-003.
- **8. Steps:** 1) insert frozen figures/tables; 2) answerRQ1..4; 3) uncertainty/tradeoffs/negative findings; 4) limitations.
- **9. Files:** report/final_report/report.md.
- **10. Functions/classes:** claims→evidence map.
- **11. Inputs:** actual results +allseed summaries.
- **12. Outputs:** complete report draft.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Person1/5 technical review.
- **15. Acceptance:** Все claims supported, no positive-result dependency.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 5-7 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Selection bias, synthetic→realworld overclaim.
- **19. Branch:** codex/report-002.
- **20. Commit/PR scope:** Results/discussion/limitations; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 4.

### REPORT-003 — Report review/export и AI disclosure

- **1. ID:** REPORT-003.
- **2. Title:** Report review/export и AI disclosure.
- **3. Purpose:** Готовый submission document.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 1 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** REPORT-002; TEST-003.
- **8. Steps:** 1) rubric cross-check; 2) six-person read-through; 3) citations/captions/numbers; 4) exportPDF and inspect.
- **9. Files:** report/final_report/; report/ai_usage.md.
- **10. Functions/classes:** chosen report build/export.
- **11. Inputs:** complete draft +policy.
- **12. Outputs:** final_report.pdf +source/disclosure.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Visual PDF review + requirements matrix.
- **15. Acceptance:** No placeholders/broken refs, numbers equaltables, format perS1.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 4-6 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Позднее несоответствие rubric/AI policy.
- **19. Branch:** codex/report-003.
- **20. Commit/PR scope:** Report review/export и AI disclosure; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 1.

### PRESENT-001 — Slides и speaker notes

- **1. ID:** PRESENT-001.
- **2. Title:** Slides и speaker notes.
- **3. Purpose:** Объяснить experiment ясно.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 4 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** REPORT-001; PLOT-002.
- **8. Steps:** 1) 14slide outline; 2) insert actual figures when ready; 3) assign6 speakers and backups.
- **9. Files:** report/presentation/outline.md; report/presentation/slides.pptx; report/presentation/slides.pdf.
- **10. Functions/classes:** slide evidence map.
- **11. Inputs:** report draft +figures.
- **12. Outputs:** deck PDF/source/notes.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Claim/data legibility review.
- **15. Acceptance:** All sections/RQs covered; timings fit verified slot.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 4-6 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Слишком много slides, unverified date.
- **19. Branch:** codex/present-001.
- **20. Commit/PR scope:** Slides и speaker notes; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 4.

### PRESENT-002 — Первая полная rehearsal

- **1. ID:** PRESENT-002.
- **2. Title:** Первая полная rehearsal.
- **3. Purpose:** Проверить understanding и timing.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 4 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** PRESENT-001; REPORT-002.
- **8. Steps:** 1) all6 speakers timed run; 2) ask hard questions; 3) log fixes+owners.
- **9. Files:** report/rehearsals.md; report/presentation/.
- **10. Functions/classes:** defence question checklist.
- **11. Inputs:** deck+code+results.
- **12. Outputs:** rehearsal1 log.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Every person explains shared pipeline.
- **15. Acceptance:** Within slot, critical questions answered, fixes assigned.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 2-3 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Один человек знает только свои slides.
- **19. Branch:** codex/present-002.
- **20. Commit/PR scope:** Первая полная rehearsal; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 4.

### PRESENT-003 — Финальная rehearsal и backup

- **1. ID:** PRESENT-003.
- **2. Title:** Финальная rehearsal и backup.
- **3. Purpose:** Готовность к presentation.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 4 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** PRESENT-002; REPORT-003.
- **8. Steps:** 1) applyfixes; 2) second timed run; 3) offlinePDF/backup device/demo plan.
- **9. Files:** report/rehearsals.md; report/presentation/.
- **10. Functions/classes:** final rehearsal checklist.
- **11. Inputs:** revised deck/final results.
- **12. Outputs:** rehearsal2+offline package.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Second-device open and Q&A.
- **15. Acceptance:** Two full sessions, verified date/slot, all6ready.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 2-3 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Зависимость от internet/GPU demo.
- **19. Branch:** codex/present-003.
- **20. Commit/PR scope:** Финальная rehearsal и backup; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 4.

### RELEASE-001 — Release candidate и artifact bundle

- **1. ID:** RELEASE-001.
- **2. Title:** Release candidate и artifact bundle.
- **3. Purpose:** Собрать доступный complete product.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 6 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** METRIC-002; PLOT-003; EVAL-004.
- **8. Steps:** 1) updateinstructions/tree; 2) bundle manifests6best/logs96results; 3) backup hash verification; 4) stage candidate identity.
- **9. Files:** README.md; ARCHITECTURE.txt; artifact_manifest.json; outputs/; report/.
- **10. Functions/classes:** artifact inventory and checksums.
- **11. Inputs:** frozen sources+results.
- **12. Outputs:** release candidate+retrievable bundle.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** All FINAL_DELIVERABLES technical checklist.
- **15. Acceptance:** All required artifacts available to team, code identity stable.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 3-4 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Checkpoints only on lost laptop.
- **19. Branch:** codex/release-001.
- **20. Commit/PR scope:** Release candidate и artifact bundle; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 6.

### RELEASE-002 — Final submission verification

- **1. ID:** RELEASE-002.
- **2. Title:** Final submission verification.
- **3. Purpose:** Закрыть сдачу без потерь.
- **4. Current status:** TODO.
- **5. Priority:** MUST.
- **6. Owner:** Person 1 (reviewer — TEAM_TASKS).
- **7. Prerequisites:** REPRO-002; REPORT-003; PRESENT-003.
- **8. Steps:** 1) independent checklist; 2) verify deadline/timezone/files; 3) submit per course instructions and record receipt.
- **9. Files:** artifact_manifest.json; report/final_report/; report/presentation/; README.md.
- **10. Functions/classes:** final acceptance checklist.
- **11. Inputs:** complete release+PDF/deck+LMS rules.
- **12. Outputs:** submission package+receipt.
- **13. Artifacts:** перечисленные outputs в указанных files/dirs + test/review evidence; paths/hash записать в task/PR.
- **14. Testing:** Second-person reopen/download and checksum.
- **15. Acceptance:** All MUST pass or explicit disclosed scope exception; receipt verified.
- **16. DoD:** acceptance выполнен; тестовые результаты и artifacts проверены reviewer; status обновлён. Для OPTIONAL допустимо SKIPPED с причиной, без объявления PASS.
- **17. Effort:** 2-3 person-hours, без full-training wait time.
- **18. Risks/common mistakes:** Wrong export/deadline/timezone or missing upload.
- **19. Branch:** codex/release-002.
- **20. Commit/PR scope:** Final submission verification; только файлы пункта 9, включая relevant tests/docs; один logical PR, owner Person 1.

## Правила сокращения scope

Сначала отказаться от BENCH-001/METRIC-003 и косметических дополнительных plots. Core6runs/96conditions/3seeds, необходимые metrics, report и 2rehearsals сохранять. Если compute не помещается, до test visibility согласованно сократить epochs для обеих моделей в новом protocol, rerun затронутые seeds, раскрыть фактический budget. Один seed вместо трёх — degraded partial delivery и не PASS полного acceptance. Нельзя заменять отсутствующие runs усреднением наличных или fabricated rows.
