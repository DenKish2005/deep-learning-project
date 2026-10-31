# Промпт следующего этапа — CORR-001

Обновлён31.10.2026 после DATA-003 independent technical acceptance и publication.

```text
Продолжи проект CSCI 494 — Robustness to Noise.
Выполни независимый технический объём CORR-001: severity mapping и keyed Gaussian
noise поверх existing stable_seed. Не обучай модели; не расширяй этап на CORR-002,
ROBUST-001, EDA или scientific evaluation. Работай до actual artifacts/tests/evidence.

СНАЧАЛА ПРОВЕРЬ АКТУАЛЬНОЕ СОСТОЯНИЕ

Прочитай AGENTS.md при наличии, README и docs/MASTER_PROJECT_PLAN.md,
START_HERE.md, CURRENT_STATUS.md, REQUIREMENTS.md, TECHNICAL_ARCHITECTURE.md,
EXPERIMENT_PROTOCOL.md, IMPLEMENTATION_ROADMAP.md (CORR-001/002, DATA-003,
ROBUST-001, REPRO-001), REPRODUCIBILITY.md, TESTING_PLAN.md, SETUP_DECISIONS.md;
report/reproduction.md, data003_evidence.json, ai_usage.md,
report/course_materials/README.md, data/splits/split_manifest.json и normalization.json.
Проверь main/status/staged list/log, fetch и remote history/origin/main.
Graphify — historical snapshot до DATA-002; сверяй с current source, не rebuild.

Факты на момент handoff, не замена fresh checks:
- DATA-001/002 technical PASS/full DoD PARTIAL до independent human review.
- DATA-003 independent technical scope PASS: stats, keyed geometry/order,
  clean typed loaders/CLI и synthetic policy boundary. Full DoD PARTIAL:
  CORR-001 prerequisite, CORR-002 actual robust integration/dependent acceptance,
  human review pending. Не меняй dependency edges ради DONE.
- Default357PASS/0FAIL/8actual opt-in SKIP, отдельно8actual PASS/0FAIL/0SKIP.
- Actual45k-only mean [0.4913397271582242,0.48209963252314836,0.44655201729302807],
  population std [0.2469009494849942,0.24334132835714897,0.2615066322799661].
- Stats FILE SHA-256332c2958a59036b68b6db2b71020dcd2aa0ab021f2a91a32cb351bf446095688;
  semantic4450836de3384df4286535985ed0b56b9a411f3033a60ecf57bbfe5edcb01877.
- Frozen NPZ1dc8e61a237d7e8fd3ac3c2dea4787e6a82f8f235d74993386af9074f6e1d284;
  manifest8742e5d48e4001e7f6a61f924bc524ab33f2276a256bc716407a98fc9bc86a21.
- Actual full45k/5k/10k typed coverage/labels; train352batches(last72).
  Spawn2/0 equality on129actual train IDs across epochs0/1, batches64/37 PASS.
- stable_seed/seed_worker уже реализованы в src/utils.py. Encoding v1:
  prefix csci494-stable-seed-v1 + LF + compact JSON list, ensure_ascii=False,
  UTF-8 SHA-256, low63bits big-endian; только Python int (не bool) и str.
- Geometry stable_seed(run_seed,epoch,original_index,'geometry'); order
  stable_seed(run_seed,epoch,'order'); policy отдельный 'train_corruption'.
- Test noise seed stable_seed(100494,'cifar10-test',original_index,
  'gaussian_noise','v1'), без run/model seed или severity: общий epsilon.
- Published DATA-003 cdffe3d/df6745b/60464bd/de303d0; docs commit в Git history.
- Existing .venv/lock/backend неизменны. General REPRO-001 остаётся TODO;
  DATA-003 не реализует global seeding/checkpoints/resume/backup.

ПЕРЕД ИЗМЕНЕНИЯМИ

Используй existing .venv, без reinstall/update/global env/backend changes:
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.venv/pytest-corr001-baseline
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-only
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-split
.\.venv\Scripts\python.exe -m scripts.prepare_normalization --root data/raw --verify-only
Missing/corrupt actual inputs — FAIL с диагностикой, не synthetic PASS/skip.
Сохрани bytes/hashes/mtimes raw/split/stats; не regenerate их молча. DATA-002 v1
source-region hash4ec051baa81ab0c70bfd2ca41f7a6750ca3b13f61930cd832b2a84dbd9244f96
и DATA-003 normalization v1 region hash1cbf01e93d4b1a838daf222c7e1672316117ad65ec1559c4e3075151dae8327c
не менять. При необходимой correction сначала preservation/diagnostics и explicit
compatibility/migration evidence; validation не обходить.

ОБЪЁМ CORR-001

1. Сохрани existing stable_seed encoding/golden tests; не реализуй второй RNG
   protocol. Добавь только необходимые seed identity regression tests.
2. Реализуй severity_params(kind,severity) по exact protocol tables. Все3families
   mapping допустимы, но actual gaussian_blur/darken/apply_corruption/CorruptionPolicy
   implementation остаются CORR-002/ROBUST-001. Invalid types/kind/severity fail.
3. gaussian_noise(x,sigma,generator): float32 RGB CHW [0,1], same shape/device/dtype,
   finite, no mutation; clip(x+sigma*epsilon,0,1). sigma>=0 finite, sigma0 identity
   clone допустима. Local explicit generator/device validation, no global RNG.
4. Зафиксируй reusable keyed epsilon boundary согласно architecture/protocol:
   original test namespace/IDs, eval100494, общий epsilon across severity/models,
   независимо от batch/worker/order. Не делай CorruptedDataset/evaluator или policy
   scope молча; missing operational API согласованно документируй/tests.
5. TEST-C01..04 и нужный TEST-U02: exact mapping/errors, input mutation/range,
   same index/namespace identity, common pre-clamp epsilon, statistical oracle
   на synthetic large tensor, clipping, global Python/NumPy/torch RNG isolation,
   batch/order/fresh-process determinism. CUDA только если actual used/tested;
   CPU reference обязателен. Не выдавай skipped CUDA за PASS.
6. Meaningful actual opt-in на заранее фиксированных train IDs для data→noise→
   Normalize boundary и сохранности frozen identities; no test-based tuning,
   no model/scientific runs. Actual test access допустим только как fixed keyed
   input contract по protocol, если для проверки нужен original-test namespace.
7. Отдельно technical PASS и pending full DoD: PROTO-001/team freeze/policy/source
   approvals/human review остаются pending. DATA-003 robust integration нельзя
   закрыть одним noise API; CORR-002 и ROBUST-001 ещё отсутствуют.
8. Обнови relevant docs/report/ai_usage, compact actual evidence и следующий prompt
   для CORR-002. Не создавай неподтверждённые results/plots/course compliance.

ГРАНИЦЫ

Без blur/darkening/policy implementations, моделей/training/evaluation/metrics,
EDA/plots/scientific runs, новых frameworks/services. Scientific protocol,
augmentation probabilities/budget и оба YAML identities неизменны. Human/course
uncertainties не блокируют independent technical work, но остаются явными gates.

GIT — ПОСТОЯННОЕ ПРАВИЛО

Пользователь разрешил самостоятельные commits и обычный push origin main без
повторного согласования. Работай/коммить/пушь только в точной ветке main.
Не создавай codex/*, feature branches или worktrees. Старые branch/PR рекомендации
planning docs заменены этим правилом; границы задач и prerequisites сохранены.

После КАЖДОГО небольшого завершённого логического блока:
- останови editing, проверь ПОЛНЫЙ diff и relevant tests;
- implementation и связанные tests публикуй вместе;
- git diff --check; stage только explicit task paths, без git add . / -A;
- git diff --cached --name-only, ПОЛНЫЙ git diff --cached,
  git diff --cached --check, отсутствие посторонних files;
- отдельный рабочий commit в main; СРАЗУ обычный git push origin main;
- проверь push и sync, кратко сообщи и продолжай следующий блок самостоятельно.
Не копи commits локально и не объединяй всё в финальный commit. Failed push
разбери до дальнейших edits; remote changes сначала прочитай. Safe fast-forward
допустим; divergence/conflicts не затирай и не merge автоматически. Без force,
amend/reset/history rewrite. Не коммить raw/archives/env/caches/credentials,
absolute machine paths и diagnostic partials. Compact verified evidence versioned.
В конце clean status/staged empty и main=origin/main после fresh fetch.

Финал по-русски: PASS/FAIL/SKIP, actual evidence/hashes/limits, каждый commit и
результат push, clean/staged/main sync, remaining gates и next concrete task.
```
