# Промпт следующего этапа — CORR-002

Обновлён 18.11.2026 после CORR-001 independent technical PASS/full DoD PARTIAL.
Повторный fresh fetch подтвердил clean main=origin/main на da63fa2.
Implementation/evidence опубликованы в main7d47569/b73536b/f89d70f/5d12a06;
documentation — da63fa2. Каждый блок сразу push origin main PASS.
Ниже handoff, он не запускает CORR-002 сам по себе. Counts — actual18.11.2026;
original8DATA opt-in checks здесь не повторялись (их31.10evidence historical).

```text
Продолжи проект CSCI494 — Robustness to Noise.
Работай, коммить и пушь всегда только в существующую ветку main.
Не создавай codex/*, другие feature branches или worktrees.
Разрешаю самостоятельные commits и обычный git push origin main после каждого
проверенного блока без повторного согласования. Это постоянное правило этапа.

Выполни независимый технический объём CORR-002: reflect Gaussian blur, progressive
darkening и apply_corruption поверх existing CORR-001. Работай самостоятельно до
проверяемого результата с actual artifacts, tests и evidence. Не обучай модели,
не расширяй этап на ROBUST-001/CORR-003/EDA/scientific evaluation и не останавливайся
после промежуточного commit.

СНАЧАЛА ПРОВЕРЬ АКТУАЛЬНЫЙ РЕПОЗИТОРИЙ

Прочитай AGENTS.md при наличии, README, docs/MASTER_PROJECT_PLAN.md, START_HERE.md,
CURRENT_STATUS.md, REQUIREMENTS.md, TECHNICAL_ARCHITECTURE.md, EXPERIMENT_PROTOCOL.md,
IMPLEMENTATION_ROADMAP.md (CORR-001/002/003, DATA-003, ROBUST-001, REPRO-001),
REPRODUCIBILITY.md, TESTING_PLAN.md, SETUP_DECISIONS.md, report/reproduction.md,
report/corr001_evidence.json, report/ai_usage.md, report/course_materials/README.md,
data/splits/split_manifest.json и normalization.json. Проверь main/status/staged
list/log; fetch и remote history/origin/main. Graphify — historical snapshot до
DATA-002; сверяй с current source, не rebuild. Source/gates не менять ради DONE.

ПЕРЕД ИЗМЕНЕНИЯМИ — EXISTING ENV, FRESH CHECKS

Используй existing .venv без reinstall/update/global/backend changes:
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.venv/pytest-corr002-baseline
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-only
.\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-split
.\.venv\Scripts\python.exe -m scripts.prepare_normalization --root data/raw --verify-only

Missing/corrupt actual inputs — FAIL с диагностикой, не synthetic PASS/skip.
Сохрани raw/split/stats/config и CORR-001 reference bytes/hashes/mtimes; не regenerate
молча. DATA-002 source-region hash4ec051baa81ab0c70bfd2ca41f7a6750ca3b13f61930cd832b2a84dbd9244f96
и DATA-003 normalization region1cbf01e93d4b1a838daf222c7e1672316117ad65ec1559c4e3075151dae8327c
не менять. Necessary correction требует preservation/diagnostics и explicit
compatibility/migration evidence; validation не обходить.

СОХРАНЁННЫЙ CORR-001 CONTRACT — ПРОВЕРЬ, НЕ ПЕРЕПИСЫВАЙ

- Exact severity_params: gaussian_noise→sigma .02/.05/.08/.12/.18;
  gaussian_blur→sigma .3/.7/1.1/1.5/2.0 и kernel_size3/7/9/11/13;
  brightness→factor .90/.80/.70/.55/.40. Fresh dict, Python int1..5, invalid→ValueError.
- gaussian_noise: dense float32 RGB CHW [0,1], positive H/W, CPU/CUDA, explicit
  same-device torch.Generator, finite sigma>=0 (Python int/float, not bool),
  no mutation/global RNG, new same shape/device/dtype. Sigma0 clones/no draw;
  invalid inputs fail before generator draws. Main CIFAR shape3x32x32.
- keyed_test_generator(original_index, *, namespace) и keyed_test_epsilon(...):
  full namespace cifar10-python/original-test, Python ID0..9999, fresh CPU stream /
  float32[3,32,32] epsilon. Existing stable_seed(100494,'cifar10-test',id,
  'gaussian_noise','v1'); no severity/model/run seed. Recreate each image/condition,
  consumed generator не reuse. Canonical keyed epsilon CPU, даже при CUDA model.
- Existing stable_seed encoding v1/golden tests unchanged: prefix
  csci494-stable-seed-v1 + LF + compact ordered JSON list, ensure_ascii=False,
  UTF-8 SHA-256 low63bits big-endian; только Python int(not bool)/str.
- Geometry/order/train_corruption streams unchanged. Training probe in tests is
  fixed sigma.08 callable, not production policy. CorruptedDataset/evaluator и
  condition manifests отсутствуют; не добавляй их в CORR-002 молча.
- 18.11.2026 default453PASS/0FAIL/11opt-in SKIP; separate new actual3PASS/0FAIL/0SKIP:
  2CIFAR cases RUN_CIFAR_CORRUPTIONS=1 +1CUDA RUN_CORR_CUDA=1. Actual CUDA primitive
  на RTX3060Laptop проверен same-device и wrong-device generators both ways;
  CPU/CUDA bitwise equality не заявлена. Original8actual DATA tests historical.
- Actual train IDs [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17], epochs0/1, batch7/5,
  workers0/2; original-test IDs0/71/9999 only fixed keyed input contract, no tuning.
- report/corr001_reference.npz97580bytes SHA-256
  40179be75281bdd06dadd266f565e70ad0f7779e81e31fdc6bd1818b45084b33.
  Contains train IDs0/1/2 noisy/normalized tensors(epoch0), test IDs0/71/9999 epsilon;
  independent actual regression tests read-only. Do not regenerate on mismatch.
- Frozen split1dc8e61a237d7e8fd3ac3c2dea4787e6a82f8f235d74993386af9074f6e1d284;
  manifest8742e5d48e4001e7f6a61f924bc524ab33f2276a256bc716407a98fc9bc86a21;
  stats FILE332c2958a59036b68b6db2b71020dcd2aa0ab021f2a91a32cb351bf446095688;
  semantic4450836de3384df4286535985ed0b56b9a411f3033a60ecf57bbfe5edcb01877.

ОБЪЁМ CORR-002

1. gaussian_blur(x,sigma,kernel_size): normalized separable/discrete Gaussian,
   no channel mixing, reflect padding; exact tables and odd kernels. Explicit
   finite/types/range/channel/device checks, no mutation, same shape/device/dtype.
   Reject invalid kernels/sigma and reflect radius incompatible with image size.
   Согласованно зафиксируй helper identity sigma0 при поддержке; main map1..5.
2. darken(x,factor): clip(factor*x,0,1), main factors<=1, boundary helpers factor1
   identity/factor0 black, invalid factors/types fail. Не превращать в brightening
   или additive shift. Input validation согласуй с existing noise contract.
3. apply_corruption(x,kind,severity,generator=None): clean отдельно severity0;
   noise/blur/brightness severity1..5; no silent global RNG fallback. Noise требует
   explicit matching generator; evaluation caller использует existing keyed_test_generator.
   Dispatcher не должен изобретать model/severity seed, policy или dataset wrapper.
   Validate clean arguments и no mutation. Документируй unused-generator behavior
   deterministic families; preserve RNG isolation and error-before-draw contract.
4. TEST-C05/C06/C08 и all-family C01/C02 completion: kernels3/7/9/11/13, normalized
   weights/symmetry, constant preserve atol1e-6, independent impulse/ramp/checkerboard
   oracle/high-frequency reduction, factors exact atol1e-7, input/range/errors,
   noise golden/common epsilon regression, Normalize boundary и RNG isolation.
   CPU reference обязателен; actual-used CUDA path проверяй отдельно без ложного PASS
   skipped GPU. Не вводи dependencies/frameworks.
5. Meaningful actual opt-in на заранее фиксированных persisted train IDs для
   data→dispatcher→Normalize и frozen identity/reference preservation, без model
   runs/EDA/plots. Можно использовать fixed test inputs только для regression
   existing keyed identity, без tuning/scientific evaluation. Failure actual
   inputs — FAIL. Все reference outputs реальны, не synthetic под видом actual.
6. Compact verified evidence, relevant docs/status/ai_usage и next handoff для
   следующего согласованного технического этапа. Не закрывай ROBUST-001/full DATA-003
   robust integration одним dispatcher. CORR-003 visual audit остаётся отдельным.
   PROTO-001/team freeze/source-policy approvals/human review pending; full DoD
   отделяй от independent technical PASS. General REPRO-001 остаётся TODO.

НЕБОЛЬШИЕ ЗАВЕРШЁННЫЕ БЛОКИ

Раздели этап на небольшие завершённые логические блоки. Ориентир:
1. Gaussian blur вместе с numeric/contract tests.
2. Darkening вместе с exact-factor/identity/error tests.
3. apply_corruption вместе с dispatcher/all-family/RNG regression tests.
4. Actual opt-in checks, проверенные reference artifacts/evidence и связанные tests.
5. Итоговые docs/status/ai_usage и промпт следующего этапа.

Адаптируй границы к коду: слишком большой блок раздели на меньшие рабочие части;
тесно связанные implementation и tests коммить вместе. Не делай commits с
заглушками, падающими tests или искусственными косметическими частями ради количества.
Закончил блок — останови editing, проверь и опубликуй его, только затем начинай
следующий. Не накапливай всю реализацию до одного финального commit и не оставляй
промежуточные commits без push. После каждого успешного push продолжай самостоятельно
до завершения CORR-002; промежуточный commit не является завершением задачи.
Scientific protocol/maps/budget/probabilities и оба YAML identities неизменны.
Без models/training/evaluator/metrics/EDA/plots/policy/new services.

GIT — ПОСТОЯННОЕ ПРАВИЛО

Пользователь разрешил самостоятельные commits и ordinary git push origin main
без повторных согласований. Работай/коммить/пушь ТОЛЬКО в exact branch main.
Не создавай codex/*, feature branches или worktrees. Planning branch/PR советы
исторические; границы задач и prerequisites сохранены. До edits проверь branch
и чужие changes; main переключай только без потери/переноса чужой работы.

После КАЖДОГО небольшого finished logical block:
- останови editing; проверь ПОЛНЫЙ diff и relevant tests;
- git diff --check; stage только explicit task paths (без git add . / -A);
- git diff --cached --name-only, ПОЛНЫЙ git diff --cached,
  git diff --cached --check; exact staged list без посторонних files;
- отдельный рабочий commit в main; СРАЗУ ordinary git push origin main;
- проверь успешный push и sync main/origin/main; сообщи block/checks/hash/push,
  затем продолжай следующий блок без повторного согласования.
Не копи всё в финальный commit. Failed push разбери до дальнейших edits; remote
changes сначала прочитай. Safe fast-forward допустим; divergence/conflicts не
затирай и не merge автоматически. Без force/amend/reset/history rewrite.
Не коммить raw/archives/env/caches/credentials/absolute machine paths/diagnostic
partials. Versioned evidence compact и verified. В конце fresh fetch, clean status,
пустой staged list и exact main=origin/main.

Финал по-русски: PASS/FAIL/SKIP с причинами, actual evidence/hashes, каждый commit
и push, ограничения/gates, clean/staged empty/main sync и next concrete task.
```
