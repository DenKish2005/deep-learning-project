# Промпт следующего этапа — CORR-003

Обновлён18.11.2026 после CORR-002 independent technical PASS/full DoD PARTIAL.
Implementation/tests: d339179 blur,72c933a darkening,2586a67 dispatcher;
actual tests/reference/evidence:20724ac. Каждый блок сразу push origin main PASS.
Final docs/status/ai_usage/handoff опубликованы в fa5a619; fresh fetch подтвердил
clean main=origin/main на этом commit перед текущим уточнением промпта.
Все пять блоков CORR-002 опубликованы ordinary push origin main PASS.
Ниже handoff для нового чата; сохранение этого файла не запускает CORR-003.
Counts ниже actual18.11.2026; original8DATA opt-in cases не повторялись.

```text
Продолжи проект CSCI494 — Robustness to Noise.
Работай, коммить и пушь всегда только в существующую exact branch main.
Не создавай codex/*, другие feature branches или worktrees. Исторические branch/PR
советы в planning docs не переопределяют это правило. Разрешаю самостоятельные
commits и ordinary git push origin main после каждого проверенного блока без
повторного согласования. Это постоянное разрешение на весь CORR-003.

Выполни independent technical объём CORR-003: train-only corruption sanity grid,
source manifest и честный visual audit поверх CORR-001/002. Работай до проверяемого
результата с actual PNG/PDF, tests и evidence. Не расширяй этап на DATA-004 EDA,
ROBUST-001 policy, models/training/evaluator/metrics или scientific evaluation.

СНАЧАЛА ПРОВЕРЬ АКТУАЛЬНЫЙ РЕПОЗИТОРИЙ

Прочитай AGENTS.md при наличии, README, docs/MASTER_PROJECT_PLAN.md, START_HERE.md,
CURRENT_STATUS.md, REQUIREMENTS.md, TECHNICAL_ARCHITECTURE.md, EXPERIMENT_PROTOCOL.md,
IMPLEMENTATION_ROADMAP.md (CORR-001/002/003, DATA-003/004, ROBUST-001, REPRO-001),
REPRODUCIBILITY.md, TESTING_PLAN.md, SETUP_DECISIONS.md, report/reproduction.md,
report/corr001_evidence.json, report/corr002_evidence.json, report/ai_usage.md,
report/course_materials/README.md, data/splits/split_manifest.json и normalization.json.
Проверь branch/status/staged/log; fresh fetch, origin/main и remote history.
Graphify — historical pre-DATA-002 snapshot; сверяй с current source, не rebuild.

Existing .venv, без reinstall/update/global/backend changes. Перед edits:
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.venv/pytest-corr003-baseline
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-only
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-split
.\.venv\Scripts\python.exe -m scripts.prepare_normalization --root data/raw --verify-only

Missing/corrupt actual inputs — FAIL с диагностикой, не synthetic PASS/skip.
Сохрани raw/split/stats/config и оба CORR reference bytes/hashes/mtimes. Не regenerate
на mismatch. DATA-002 region4ec051baa81ab0c70bfd2ca41f7a6750ca3b13f61930cd832b2a84dbd9244f96
и DATA-003 normalization region1cbf01e93d4b1a838daf222c7e1672316117ad65ec1559c4e3075151dae8327c
не менять. Validation/gates не ослаблять ради DONE.

СОХРАНЁННЫЙ CORR-001/002 CONTRACT

- severity_params fresh dict, Python int1..5; noise sigma.02/.05/.08/.12/.18;
  blur sigma.3/.7/1.1/1.5/2.0, kernels3/7/9/11/13; brightness factors.9/.8/.7/.55/.4.
- Dense CPU/CUDA float32 RGB CHW finite pixels[0,1], positive H/W, new output storage,
  same shape/device/dtype, no mutation. Main CIFAR3x32x32. Normalize AFTER corruption.
- gaussian_noise finite nonnegative Python sigma(not bool), explicit same-device
  torch.Generator; sigma0 clones/no draws; invalid args fail before draws.
- gaussian_blur normalized discrete separable Gaussian, float64 weights normalized
  then cast float32, reflect padding and independent channels, clamp for roundoff.
  Positive odd Python kernel, radius<both H,W even sigma0. Sigma0/kernel1 clone;
  invalid sigma/kernel/reflect shape fail. No RNG. Do not replace padding/backend.
- darken finite Python factor0..1, clip(factor*x,0,1); factor1 equal/new storage,
  factor0 black; no brightening/additive shift, invalid input fails. No RNG.
- apply_corruption clean ONLY Python int0; families ONLY1..5. Noise requires matching
  explicit generator; deterministic families accept None or unused torch.Generator
  on any device, leave it untouched; invalid generator types fail. No seed invention.
- keyed_test_generator/epsilon: full namespace cifar10-python/original-test,
  Python original_index0..9999, fresh CPU stream/float32[3,32,32] epsilon from
  stable_seed(100494,'cifar10-test',id,'gaussian_noise','v1'). No severity/model/run
  seed. Recreate each image/condition; consumed generator not reused. Canonical CPU.
- stable_seed encoding/golden tests, geometry/order/train_corruption streams unchanged.
  CorruptionPolicy/CorruptedDataset/evaluator/condition manifests still absent.
- Fresh18.11default777PASS/0FAIL/14opt-in SKIP; separately6actual PASS/0FAIL/0SKIP:
  4CIFAR corruption cases RUN_CIFAR_CORRUPTIONS=1 +2CUDA RUN_CORR_CUDA=1. Original8DATA
  opt-in results historical31.10; raw/split/stats CLI checks fresh. CUDA0 RTX3060Laptop
  all16conditions/device/global RNG PASS; no CPU/CUDA bitwise equality claim.
- Fixed17train IDs [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17], epochs0/1,
  all16conditions=544image-conditions; clean+severity3 spawn0/2 batch7/5. Probes
  are fixed test callables, not production robust policy. CPU/CUDA max blur NumPy2D
  oracle error1.7881393432617188e-7. Test IDs0/71/9999 only fixed identity regression.
- report/corr001_reference.npz97580bytes SHA-256
  40179be75281bdd06dadd266f565e70ad0f7779e81e31fdc6bd1818b45084b33.
- report/corr002_reference.npz187438bytes SHA-256
  b4396faba198a1497758cd3478983b96a85bd272972cb6266e14d497437d1f03.
  Actual train IDs0/1/2 run42 epoch0; clean0+severity3families, float32 pixel/normalized
  arrays[4,3,3,32,32], int64IDs/severities/Unicode kinds. Independent read-only replay.
- Frozen split1dc8e61a237d7e8fd3ac3c2dea4787e6a82f8f235d74993386af9074f6e1d284;
  manifest8742e5d48e4001e7f6a61f924bc524ab33f2276a256bc716407a98fc9bc86a21;
  stats FILE332c2958a59036b68b6db2b71020dcd2aa0ab021f2a91a32cb351bf446095688;
  stats semantic4450836de3384df4286535985ed0b56b9a411f3033a60ecf57bbfe5edcb01877.

ОБЪЁМ CORR-003

1. Inspect scripts/make_plots.py/current interfaces first. Implement only reusable
   plot_corruption_grid plus a bounded explicit CLI for this audit, without fake
   general metrics/learning-curve APIs. Use existing plotting dependencies.
2. Offline verify actual raw train and frozen split before selecting exactly one
   image/class: smallest original ID of each class among persisted train45k.
   Save ordered original IDs/labels/class names and selection rule. Never decode
   official test, select hard cases, tune severities or expand to EDA.
3. Render10rows x16columns: clean0 +noise1..5 +blur1..5 +brightness1..5 in exact order.
   Raw pixel-space dispatcher output, shared explicit local per-image Gaussian
   epsilon across noise levels via a documented train-only audit namespace/key.
   Never misuse keyed_test_generator on train IDs. Freeze the audit key before
   viewing outputs; no new production seed API. No geometry/Normalize required
   for raw pixel grid; do not change train loader. No auto-contrast or per-cell
   normalization; RGB fixed0..1, readable labels and consistent PNG/PDF colors.
4. Write actual outputs/figures/corruption_examples.png and .pdf plus compact
   source/provenance manifest (curated paths explicitly tracked as needed; do not
   unignore all outputs). Include input/implementation/params/IDs/key/output hashes,
   raw[0,1]/float32 contract and enough information to reproduce. Preserve existing
   artifacts on conflict/failure; document bounded rerun/verification behavior.
5. Meaningful offline tests for selection/condition order/key reuse/RNG isolation,
   numeric source boundary/no input mutation, no test decoding, output/CLI errors;
   separate actual opt-in, missing/corrupt data FAIL. Verify exported PNG/PDF opens
   and visually inspect rendered artifact(s): reflect edges/channel artifacts,
   clipping, multiplicative darkening, fixed ranges and readability. Record exact
   agent observations with honest limits. Required independent Person5/team visual
   review stays PENDING until real human evidence; do not invent approval.
6. Compact actual evidence, relevant docs/status/ai_usage and next handoff for a
   bounded technical task (ROBUST-001 can be considered after current review of
   prerequisites). Technical PASS separate from full DoD; no dependency/gate edits.
   General REPRO-001, PROTO-001/team freeze/source-policy approvals, full DATA-003
   robust integration/human review and second-machine reproduction remain open.

БЛОКИ И ПУБЛИКАЦИЯ

Раздели работу на небольшие завершённые логические блоки. Ориентир:
1. Выбор fixed train IDs, подготовка pixels/conditions/keyed noise и связанные tests.
2. PNG/PDF rendering, bounded CLI и связанные tests.
3. Actual opt-in checks, PNG/PDF/source manifest, визуальная проверка и evidence.
4. Итоговые docs/status/ai_usage и следующий prompt.

Адаптируй размер блоков к current source. Implementation и связанные tests коммить
вместе. Если блок слишком большой — раздели на меньшие рабочие части. Не делай
commits с заглушками, падающими tests или искусственными косметическими частями.
Не накапливай реализацию до одного финального commit.

После КАЖДОГО завершённого блока:
- останови editing и проверь ПОЛНЫЙ diff и relevant tests;
- выполни git diff --check; stage только явные task paths, без git add . / -A;
- проверь git diff --cached --name-only и ПОЛНЫЙ git diff --cached: staged list
  должен содержать только файлы текущего блока;
- выполни git diff --check и git diff --cached --check;
- сделай отдельный рабочий commit в exact branch main;
- СРАЗУ выполни ordinary git push origin main;
- проверь успешный push, main=origin/main и отсутствие неопубликованных changes;
- сообщи checks, commit hash и результат push.

Только после успешной публикации блока переходи к editing следующего. После push
продолжай самостоятельно без повторного согласования до CORR-003 technical result;
промежуточный commit не является завершением задачи.

При failed push сначала разберись с причиной и прочитай remote history, до новых
edits. Safe fast-forward допустим; divergence/conflicts не затирай и не merge
автоматически. Без force, amend/reset и history rewrite.
Не коммить raw/archives/env/caches/credentials/absolute machine paths/diagnostic
partials. Versioned artifacts/evidence должны быть actual, compact и verified.
В конце fresh fetch, clean status, пустой staged list и exact main=origin/main.
Финал по-русски: PASS/FAIL/SKIP с причинами, actual artifacts/evidence/hashes,
каждый commit и результат push, оставшиеся ограничения/gates и next concrete task.
Training readiness не заявляй.
```
