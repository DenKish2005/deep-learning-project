# Промпт следующего этапа — DATA-002

Подготовлен 31.10.2026 после DATA-001 и Graphify. Скопируйте текст ниже в новый
чат проекта. Это задание на реализацию DATA-002, а не на повтор DATA-001.

```text
Продолжи проект CSCI 494 — Robustness to Noise.

Поставь цель: выполнить DATA-002 — создать, сохранить и проверить единственный
stratified train/validation split CIFAR-10. Выполни этап до проверяемого результата:
actual artifacts, tests, документация, маленькие commits и push в main.
Не останавливайся на плане. Работай самостоятельно в указанном объёме.

СНАЧАЛА ПРОВЕРЬ РЕАЛЬНОЕ СОСТОЯНИЕ

Прочитай AGENTS.md, если он существует, README.md и документы:
- docs/MASTER_PROJECT_PLAN.md
- docs/START_HERE.md
- docs/CURRENT_STATUS.md
- docs/REQUIREMENTS.md
- docs/TECHNICAL_ARCHITECTURE.md — create_split/load_split и artifact contracts
- docs/EXPERIMENT_PROTOCOL.md — exact split algorithm, test isolation, provenance
- docs/IMPLEMENTATION_ROADMAP.md — DATA-002, DATA-001, SETUP-003 и REPRO-001
- docs/REPRODUCIBILITY.md
- docs/TESTING_PLAN.md — TEST-D02 и существующие DATA-001 tests
- docs/SETUP_DECISIONS.md
- report/course_materials/README.md
- report/ai_usage.md
- report/reproduction.md

Проверь branch, git status, последние commits, fetch/remote history и origin/main.
Не захватывай чужие изменения. Graphify в graphify-out/ помогает навигации, но
является snapshot: сверяй его сведения с текущими файлами и source revision.

На момент подготовки промпта:
- DATA-001 technical acceptance PASS; full DoD PARTIAL до independent human review.
- Реализованы load_cifar10(root: Path, train: bool, download=False), official
  acquisition, read-only --verify-only и ignored integrity manifest schema1.
- CIFAR-10 реально получен: original train50000/test10000, RGB32x32,10classes,
  5000/1000per class, archive/seven-file checksums проверены.
- Default suite:180PASS/3actual opt-in SKIP; отдельно actual checks:3PASS/0SKIP.
- Есть отдельная .venv, exact dependency lock, strict config validation и два YAML.
- Split ещё не создан; split_sha256 и normalization_sha256 в обоих YAML null.
- Все предыдущие блоки опубликованы в main. Этот snapshot не заменяет fetch
  и фактические checks текущих tests/data. Не обновляй versions и scientific settings.

ОБЪЁМ DATA-002

1. Используй существующую .venv. Перед изменениями выполни:
   .\.venv\Scripts\python.exe -m pip check
   .\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.venv/pytest-data002-baseline
   .\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-only
   Default pytest должен оставаться offline. Не переустанавливай dependencies,
   не меняй global environment/backend. CUDA smoke повторяй только при их изменении.
   Если raw data недоступны, не выдавай skip за PASS; при необходимости используй
   существующий explicit acquisition CLI. Сохрани diagnostic partial artifacts.

2. Реализуй в src/data.py согласованные API:
   create_split(targets: Sequence[int], split_seed: int, out_dir: Path) -> dict
   load_split(path: Path, targets) -> tuple[np.ndarray, np.ndarray]
   Return schema, validation errors, integer dtype и manifest fields зафиксируй
   в docs/tests. Сохрани работающий DATA-001 API и его offline default.

3. Используй exact algorithm из EXPERIMENT_PROTOCOL, не выбирай новый:
   - вход: ТОЛЬКО original CIFAR-10 train targets длины50000, классы0..9;
   - один local np.random.default_rng(494), class order0..9;
   - для каждого класса sorted candidate indices -> permutation тем же RNG;
   - первые500 indices класса -> validation, остальные4500 -> train;
   - final indices каждой partition сортируются;
   - результат45000/5000, ровно4500/500 per class, indices0..49999;
   - uniqueness, disjointness, exhaustive union и label/index alignment обязательны.
   Seed494 общий для model seeds42/123/2026. Global NumPy/Python/torch RNG state
   не меняется. Сохранённые indices после freeze являются canonical source.
   Не используй test images/targets, model seed или качество моделей для split.
   Original test10000 не включается в train/validation split.

4. Сохрани actual artifacts:
   data/splits/cifar10_split_v1.npz
   data/splits/split_manifest.json
   Manifest: schema/algorithm version, seed, dataset/source/archive checksum,
   original-train partition namespace, counts/per-class counts, identity исходного
   train и порядка labels, arrays SHA-256, NPZ file SHA-256, creation code identity.
   Различай file-byte hash и semantic array identity. Проверяй provenance, а не
   доверяй произвольному JSON. Без absolute local paths, credentials и test labels.

5. Existing artifacts сначала полностью проверяй и переиспользуй. Успешный повтор
   create/load/verify сохраняет indices, hashes и existing file mtimes.
   Wrong seed/dataset/label order, malformed arrays/manifest, tampering или partial
   pair -> actionable error; не пересоздавай/не перезаписывай молча. Продумай safe
   writes и восстановление после interrupted write: incomplete state не verified.
   Из REPRO-001 добавляй только необходимые hash/write helpers с relevant tests;
   не закрывай general identity/checkpoint/RNG/backup infrastructure этим этапом.
   Учти ZIP metadata/timestamps NPZ: не обещай byte-identical regeneration без проверки.

6. Расширь scripts/download_data.py рабочим split CLI. Целевой prepare command:
   .\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --prepare-split --split-seed 494
   Добавь явный read-only способ проверить existing split; документируй точную
   команду и defaults. Existing --verify-only без split flags по-прежнему проверяет
   только raw data, не создаёт artifacts и не требует split. Новые flags должны
   иметь --help, понятные errors и nonzero exit при invalid arguments.
   --prepare-split готовит только split, не normalization/loaders.

7. Добавь meaningful offline tests в tests/test_data_pipeline.py и при необходимости
   соответствующий CLI/actual test module:
   - balanced synthetic50000 targets, exact4500/500 counts, dtype/range;
   - uniqueness/disjointness/exhaustiveness и label alignment;
   - exact protocol algorithm, determinism в fresh process и global RNG isolation;
   - round trip, stable existing artifacts/mtime и корректные hashes;
   - wrong seed/length/types/classes, duplicates/overlap/out-of-range/missing IDs;
   - altered/reordered targets, tampered NPZ/manifest, missing/partial artifact pair;
   - failure/recovery safety, CLI prepare/verify success и nonzero failures;
   - сохранение DATA-001 offline/default и read-only contracts.
   Default suite не скачивает CIFAR-10 и не зависит от actual split/data.

8. Отдельно с explicit opt-in проверь actual original train50000, создание actual
   split, обе partition counts, все invariants/provenance, повторное чтение и CLI.
   Сравни hashes/mtimes до и после повторного run. Missing/corrupt actual data —
   честный FAIL, не synthetic PASS/skip. Не запускай обучение.
   Не перезаписывай ignored raw integrity manifest: это historical acquisition
   record; новые split provenance и evidence сохраняются отдельно.

9. После actual verification запиши фактический NPZ SHA-256 симметрично в
   dataset.split_sha256 ОБОИХ YAML. normalization_sha256 оставь null.
   Другие scientific settings не меняй. Уточнения artifact identity в schema/tests/
   docs вноси согласованно, не ослабляя validation. Сохрани компактные split NPZ
   и manifest в Git (это indices, не raw CIFAR images), проверь ignore rules и
   portable paths. Наличие split ещё не означает training readiness.

10. Обнови README, CURRENT_STATUS, roadmap, TESTING_PLAN, REPRODUCIBILITY,
    architecture при изменении contracts, report/reproduction.md и report/ai_usage.md.
    Запиши actual commands, PASS/FAIL/SKIP, hashes/provenance и ограничения.
    Не приписывай agent checks команде. Full DATA-002 DoD оставь PARTIAL, если
    independent human reviewer approval отсутствует. После завершения подготовь
    NEXT_STEP_PROMPT для DATA-003 по фактам, сохранив main-only workflow.

ГРАНИЦЫ

Не реализуй normalization/statistics, transforms/training loaders, corruptions,
CNN, training, evaluation, metrics, plotting или EDA. Не запускай scientific runs.
Не меняй protocol/augmentation/model budget. Из configs меняется только verified
split identity, симметрично для обеих моделей; normalization hash unresolved.
Не добавляй frameworks/services. Policy/LMS/submitted proposal/team uncertainties
остаются pending и не блокируют независимую техническую работу. Не объявляй course
compliance, human review или независимое воспроизведение подтверждёнными.

GIT — ЯВНОЕ И ПОСТОЯННОЕ ПРАВИЛО ПОЛЬЗОВАТЕЛЯ

Работай, коммить и пушь ВСЕГДА ТОЛЬКО в main. Не создавай и не используй codex/*
или любые другие feature branches. Старые branch/PR рекомендации planning docs
не действуют. Пользователь разрешает маленькие commits и обычный push в origin main.

- Перед изменениями проверь status и fetch/remote history. Если текущая ветка не
  main, не редактируй её; переходи в main только без потери/переноса чужой работы.
- После КАЖДОГО небольшого завершённого логического блока проверь полный diff,
  relevant tests/checks, staged diff и ТОЧНЫЙ список staged files; выполни
  git diff --check и git diff --cached --check. Stage только явные task paths,
  не используй git add . или git add -A.
- Реализацию и относящиеся к ней tests коммить вместе. Каждый commit должен быть
  рабочим. Не коммить заведомо broken промежуточное состояние и не откладывай весь
  этап в один большой финальный commit.
- Сразу после каждого прошедшего проверки commit выполняй git push origin main.
  Не копи commits до конца этапа. При ошибке push выясни причину до следующего блока;
  не объявляй локальный commit опубликованным.
- Не используй force push, amend, reset или переписывание истории. Если remote
  изменился, сначала прочитай чужие commits. Safe fast-forward допустим; при
  несовместимом divergence/конфликте сохрани работу и сообщи точную блокировку,
  не перезаписывай remote и не делай конфликтный merge автоматически.
- Не коммить raw CIFAR archives/images, .venv, caches, credentials, absolute local
  paths, temporary outputs и diagnostic partials. Versioned split indices/manifest
  и компактное evidence должны быть reviewable и связаны с проверенными inputs.

Пример границ commits (не обязательная квота):
1) split construction/invariant validation + offline tests;
2) persistence/manifest/hash/reuse/tamper safety + tests;
3) working CLI + tests;
4) verified actual split artifacts + config identities + relevant checks;
5) проверенные docs/evidence и prompt следующего этапа.
Выбирай границы по завершённости; не дроби код искусственно ради числа commits.
Показывай короткие progress updates с checks и commit/push status.

КРИТЕРИИ ГОТОВНОСТИ

create_split/load_split и оба split CLI режима работают; actual45k/5k artifacts
созданы и verified; seed494,4500/500 per class, disjoint/exhaustive indices,
hashes/provenance подтверждены; повторный run ничего не портит; tampering/partial
state отвергаются; default tests offline и foundation/DATA-001 tests сохранены.
Оба YAML ссылаются на один verified split hash; normalization hash null.
Все завершённые блоки отдельно committed и pushed только в main. В конце проверь
clean status, staged files и main=origin/main либо точно перечисли блокировки.

В финале по-русски сообщи: что реализовано, PASS/FAIL/SKIP, actual split evidence,
paths/hashes, commits и результат каждого push, pending human/source/team сведения
и следующий конкретный DATA-003. Вопросы задавай только при действительно блокирующей
неопределённости. Продолжай самостоятельно до завершения этого объёма.
```
