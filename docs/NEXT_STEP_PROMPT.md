# Промпт следующего этапа — DATA-003

Подготовлен31.10.2026 после technical acceptance DATA-002. Скопируйте текст ниже
в новый чат. Это DATA-003, не повтор split и не запуск training.

```text
Продолжи проект CSCI 494 — Robustness to Noise.

Поставь цель: выполнить независимый технический объём DATA-003 — train-only
normalization и deterministic typed data loaders поверх frozen DATA-002 split.
Работай самостоятельно до actual artifacts/tests/evidence, маленьких commits
и push в main. Не ограничивайся планом. Не обучай модели.

СНАЧАЛА ПРОВЕРЬ СОСТОЯНИЕ

Прочитай AGENTS.md, если существует, README и документы:
- docs/MASTER_PROJECT_PLAN.md, START_HERE.md, CURRENT_STATUS.md, REQUIREMENTS.md;
- TECHNICAL_ARCHITECTURE.md — split region/manifest, stats/dataset/loaders APIs;
- EXPERIMENT_PROTOCOL.md — stats, transform order, RNG namespaces/test isolation;
- IMPLEMENTATION_ROADMAP.md — DATA-003, CORR-001, REPRO-001;
- REPRODUCIBILITY.md, TESTING_PLAN.md, SETUP_DECISIONS.md (все выше в docs/);
- report/reproduction.md, report/data002_evidence.json, report/ai_usage.md,
  report/course_materials/README.md и data/splits/split_manifest.json.
Проверь branch/status/log, fetch/remote history и origin/main. Не захватывай чужие
изменения. Graphify — historical snapshot до DATA-002; сверяй с current source.

Факты на момент подготовки, не замена новым checks:
- DATA-001/002 technical PASS; full DoD PARTIAL до independent human review.
- Working raw loader/acquire/read-only verify; split create/load/prepare/verify.
- Actual original train50000→45000/5000, seed494,4500/500class, sorted <i8>
  IDs0..49999, disjoint/exhaustive/protocol replay PASS; original test отдельно.
- NPZ data/splits/cifar10_split_v1.npz,76550bytes, SHA-256
  1dc8e61a237d7e8fd3ac3c2dea4787e6a82f8f235d74993386af9074f6e1d284.
- Manifest SHA-2568742e5d48e4001e7f6a61f924bc524ab33f2276a256bc716407a98fc9bc86a21.
- Оба YAML с verified NPZ file hash; normalization_sha256 null.
- Default282PASS/5actual opt-in SKIP; отдельно5actual PASS/0SKIP.
- Existing .venv/lock/backend unchanged, NumPy2.5.4. Training не запускался.
- Published4044d0c/fd0a272/f4d7b42/cf8e684 плюс final docs commit в истории.

CORR-001 prerequisite DATA-003 ещё TODO. Не объявляй prerequisite/full robust
integration закрытыми. Сделай независимые stats, keyed geometry/order, clean
train/val/test loaders и policy injection boundary согласно contracts. Boundary
tests — synthetic callable, без реализации corruptions. Actual robust integration
и dependent acceptance явно pending до CORR-001/002; dependency graph не меняй,
задачу молча на corruptions не расширяй.

ПЕРЕД ИЗМЕНЕНИЯМИ

Используй existing .venv; не переустанавливай dependencies/global env/backend:
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.venv/pytest-data003-baseline
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-only
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-split
Missing/corrupt actual inputs — честная диагностика, не synthetic PASS/skip.
Split нельзя молча пересоздать; historical raw manifest/diagnostics не менять.
CUDA smoke повторять только при изменении dependencies/backend.

ОБЪЁМ DATA-003

1. Сохрани frozen split bytes/hashes/file mtimes и DATA-001/002 APIs. DATA-003
   code добавляй ВНЕ DATA-002 v1 source region: manifest проверяет LF region hash
   4ec051baa81ab0c70bfd2ca41f7a6750ca3b13f61930cd832b2a84dbd9244f96.
   Если split correction необходима: диагностика, preservation original artifacts
   и explicit compatibility/migration evidence, без обхода validation.

2. Реализуй compute_normalization/IndexedCIFAR10/make_loaders из architecture.
   Missing operational schema details согласованно зафиксируй в docs/tests,
   scientific settings не меняй. Stats: raw uint8 RGB /255, только persisted45k
   train indices, channel mean/population std, float64 accumulation, full
   precision, finite/std>0. Val/test не участвуют в stats.

3. Сохрани data/splits/normalization.json: strict schema/version, mean/std,
   channel order/scale/ddof/count, verified dataset/split/train-array identities,
   algorithm/creation code identity. Existing stats verify/reuse; wrong provenance/
   tamper/partial/interruption → error и preservation. Различай semantic/file
   hashes. Только необходимые helpers REPRO-001, без general checkpoint/RNG/backup.

4. Train: zero padding4 → crop32 → flip p=.5 → float32 CHW /255 → optional
   injected pixel-space policy → Normalize. Val/test: float32/255 → Normalize,
   без crop/flip/shuffle. Policy сохраняет labels/IDs; namespaces train/test разные.
   Shuffle stable_seed(run_seed, epoch, 'order'); geometry
   stable_seed(run_seed, epoch, original_index, 'geometry'); separate policy
   train_corruption stream. SHA-256 stable seed encoding документируй/тестируй;
   не built-in hash/global NumPy/Python RNG. Paired geometry/order одинаковы,
   policy не сдвигает geometry/global model streams.

5. Batches: float32 images[N,3,32,32], int64 labels[N], original IDs;
   train shuffle local epoch generator, val/test shuffle=False, drop_last=False,
   workers0 reference, pin_memory только CUDA. set_epoch до iteration. Windows
   spawn/worker settings документируй; workers2 reproducibility требует actual
   test. Model seeds42/123/2026 не меняют split/stats.

6. При необходимости добавь явный stats prepare/read-only verify CLI с help/
   defaults/errors. Existing --prepare-split только split, --verify-only только
   raw, --verify-split только split. Не объявляй training-ready success.

7. Meaningful offline tests TEST-D03..05 и boundary TEST-C08: known pixel oracle,
   float64/population stats, val/test outliers не меняют stats; invalid pixels/
   indices/stats/provenance, hashes/mtimes/reuse/interruption/tamper; transform
   order/dtype/shape/range/labels, deterministic eval, same seed/epoch/index
   geometry, paired order, epoch changes, nonfull batch, global RNG isolation.
   Default suite offline, без зависимости от actual raw/stats artifacts.

8. Explicit actual opt-in: verify raw+frozen split, stats только45k, save/load/
   reuse, typed batches/index coverage/label alignment, hashes/mtimes. Test можно
   читать только для fixed loader contract, не scientific tuning. Без model/
   training/evaluation/scientific runs. Missing/corrupt actual inputs — FAIL.
   Agent checks отдельно от human/team evidence.

9. После actual stats verification впиши normalization FILE SHA-256 симметрично
   в оба YAML; split hash/остальные settings неизменны. Versioned small stats/
   evidence, portable paths, без raw images/absolute local paths. Обнови README/
   status/roadmap/testing/repro/architecture/report/ai_usage и следующий prompt.
   Full DATA-003 DoD PARTIAL при missing human review/dependent robust integration;
   отдельно перечисли technical PASS и pending acceptance.

ГРАНИЦЫ

Без corruptions implementations, CNN/training/evaluation/metrics/plots/EDA,
frameworks/services и scientific runs. Protocol/augmentation/budget не меняй.
Policy/LMS/submitted proposal/team uncertainties pending, не блокируют independent
technical work и не позволяют объявить course compliance.

GIT — ПОСТОЯННОЕ ПРАВИЛО

Работай, коммить и пушь ВСЕГДА ТОЛЬКО main. Не создавай codex/* или feature
branches. Старые branch/PR рекомендации planning docs не действуют.
Перед изменениями status/fetch/history. После КАЖДОГО небольшого рабочего блока:
полный diff, relevant tests, точный staged list/staged diff, git diff --check
и git diff --cached --check. Stage explicit task paths, без add . / -A.
Implementation и related tests вместе; отдельный working commit и НЕМЕДЛЕННЫЙ
обычный git push origin main. Не копи commits до конца; push failure сначала
исследуй. Без force/amend/reset/history rewrite. Чужие remote commits прочитай;
safe fast-forward допустим, divergence/conflict не затирай/не merge автоматически.
Не захватывай чужие changes; raw/.venv/caches/credentials/diagnostic partials не
коммить. Блоки: stats/contracts+tests; transforms/loaders+tests; CLI+tests;
actual stats/configs/evidence; docs. Перед каждым commit checks, сразу push.

В финале по-русски: PASS/FAIL/SKIP, actual stats/loader evidence/hashes, unchanged
split identity, commits/push каждого блока, clean status/staged files/main=origin/main,
limitations и next concrete task. Вопросы только при блокирующей неопределённости.
```
