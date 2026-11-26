# AI assistance record

## Per-class analysis — 26.11.2026

At the user's request, Codex wrote the saved-count analysis and seven focused
tests, compared both models across all three seeds, generated per-class recall
drop and seed-42 confusion figures, and drafted the findings. Codex checked
the matrices against saved test class counts, rebuilt the existing summaries,
and verified the six checkpoint hashes. Original prediction files are absent
from this checkout; no training or inference was performed. Independent
human review remains pending. [Verification record](reproduction.md)

## Core training and evaluation — 26.11.2026

At the user's request, Codex implemented and tested the training launcher,
fixed the evaluator's normalization input, added matrix orchestration, and
ran the six CUDA training jobs and 96 evaluations. Codex verified the
checkpoints and metrics, produced preliminary charts, prepared the model
archive, and drafted the results and project documentation. Independent
team review is pending. [Verification record](reproduction.md)

## TRAIN-003 — 22.11.2026

At the user's request, Codex implemented and tested the diagnostic and pilot
profiles. Codex ran the local 64-image learning check and paired CUDA pilots,
verified the run records, and documented the measured results and time estimates.
Independent review is pending. [Verification record](reproduction.md)

## TRAIN-002 — 22.11.2026

At the user's request, Codex planned and implemented epoch checkpoint saving,
resume, and history recovery. Codex wrote and ran synthetic tests for exact CPU
resume, checkpoint identity, interrupted writes, and best-model inference.
The CIFAR-10 pilot and independent human review are pending.
[Verification record](../docs/CURRENT_STATUS.md)

## REPRO-001 — 22.11.2026

At the user's request, Codex planned and implemented local experiment identity,
run records, and backup verification. Codex wrote and ran synthetic tests,
including a run record and backup from two epochs of CompactCNN training.
Production integration and independent human review are pending.
[Verification record](../docs/CURRENT_STATUS.md)

## TRAIN-001 — 22.11.2026

At the user's request, Codex planned and implemented the shared training and
validation loop, wrote and ran synthetic tests for both CompactCNN regimes,
and ran a separate CUDA synthetic check. Codex documented the test evidence.
Checkpoint-backed training and independent human review are pending.
[Verification record](../docs/CURRENT_STATUS.md)

## EVAL-001 — 22.11.2026

At the user's request, Codex planned and implemented the synthetic milestone for
single-condition evaluation, wrote and ran the evaluator tests, and documented
the verification results. The tests exercised known logits, prediction export
and metric reconstruction, and a CompactCNN state-dict round trip on synthetic
images. Production checkpoint integration, official CIFAR-10 evaluation, and
independent human review are pending. [Verification record](../docs/CURRENT_STATUS.md)

## METRIC-001 — 19.11.2026

At the user's request, Codex planned and implemented the classification and
per-run robustness metrics, wrote and ran synthetic tests, reviewed the changes,
and documented the APIs and verification results. Codex diagnosed a strict frozen
CORR-003 replay failure that occurred before code changes. The verification record
lists the mismatch and the matching hashes for the frozen files. Human review and
formal prerequisite sign-off are pending.
[Verification record](reproduction.md)

## MODEL-001 — 18.11.2026

По прямому запросу пользователя Codex прочитал current repo/saved prompt/docs,
проверил prerequisites и clean main=origin/main2e29bdd после fresh fetch. AGENTS.md
не найден; Graphify historical, без rebuild. Existing .venv, pip/raw/split/stats/
frozen replay/config checks PASS; baseline914PASS/19SKIP. Formal PROTO-001 и
SETUP-002 approvals не объявлены закрытыми; bounded preparation разрешена.

Агент реализовал CompactCNN v1, explicit constructor initialization и стандартную
model-level state_dict identity,35independent synthetic cases + CPU/CUDA device
checks. Init timing и caller ownership определены до probes: seed→CPU constructor
с default layer draws→final protocol init→device; без reseed. Paired full states
seeds42/123/2026 включают BN buffers/identity. Shapes/raw logits, independent init
RNG/BN/CE oracles, dropout/modes, finite backward без обновления параметров и strict
serialized roundtrip PASS. Device inputs — fixed synthetic linspace; no CIFAR
inference/quality claims. Backend flags unchanged; no cross-device/full training
reproducibility claim. [Evidence](model001_evidence.json), [record](reproduction.md).

Final950PASS/0FAIL/20SKIP; separately executed CPU/CUDA2PASS/0FAIL/0SKIP. Previous
19actual opt-ins не повторялись, прежние evidence не передатированы. All36protected
bytes/hashes/mtimes и DATA regions сохранены. Training/optimizer steps, EDA,
metrics/evaluator/figures, TRAIN-* и general REPRO-001 не выполнялись.

Blocks489eb04/11cffd5/6dad624: каждый full diff/tests, explicit staged list/full
diff, оба diff checks, commit exact main и immediate ordinary push origin main
PASS, fresh-fetch clean0/0/staged empty. Final docs/handoff отдельно. Все code/tests/
review выполнены агентом; human understanding/code/artifact review, formal
prerequisites, course AI-policy conflict/LMS/proposal/team/source approvals и
second-machine reproduction PENDING. Technical PASS/full DoD PARTIAL. Новая model
init QA не заменяет budget/full scientific gate и не переписывает ROBUST evidence.

## ROBUST-001 — 18.11.2026

По прямому запросу пользователя Codex проверил fresh main/origin/history/docs,
prerequisites и existing .venv. Initial main de20097 clean/synced; baseline
815PASS/16SKIP; pip/raw/split/stats/grid/config checks PASS. Historical Graphify
прочитан как stale navigation snapshot, без rebuild. Technical preparation
разрешена; formal prerequisites и course permission не объявлены закрытыми.

Агент сначала сохранил bounded authenticated CORR-003 source replay at21ede65,
затем реализовал CorruptionPolicy с immutable frozen settings и explicit generator.
Draw order определён до actual outputs: float64 event, int64 severity только для
corruption, float32 Gaussian tensor только для noise. src/data.py existing hook,
src/utils.py/stable_seed/geometry/order/test streams и прежние8corruption functions
не изменены. Добавлены99offline и3actual opt-in cases; CORR-003 tests сохраняют
строгий current-source mismatch и recorded-source verification, artifacts unchanged.

Actual IDs0..11,13..17, seeds42/123/2026, epochs0/1 объявлены до outputs. Numeric
102conditions, clean val17, production make_loaders, spawn0/2 batch7/5 наseed42,
CUDA RTX3060Laptop PASS. No test decoding в actual policy/grid/reference checks;
official test только в required raw integrity CLI. Frequency audit100000selections
использует production selector с dispatcher clone вместо дорогих pixels; реальные
pixels/RNG отдельно проверены независимыми NumPy2D/direct noise/factor oracles.

Default914PASS/0FAIL/19SKIP; отдельно6actual PASS/0FAIL/0SKIP. Other13historical
opt-ins not rerun. Initial concurrent reference attempt failed NumPy MemoryError;
first actual spawn check stopped after WinError1455. Failed logs preserved ignored;
unchanged sequential reruns passed. Missing/corrupt negative controls expected
DataIntegrityError/exit1/0SKIP. All28protected files bytes/hashes/mtimes preserved.
[Evidence](robust001_evidence.json), [commands/hashes](reproduction.md).

Published ef78e05 frozen replay,0c3d3ca policy/tests,d4d56b0 integration tests,
01bcfb7 actual/evidence; each full diff/tests, explicit staged paths/list/full diff,
оба diff checks, commit exact main и immediate ordinary push origin main PASS.
Final docs/handoff отдельным блоком; no branches/worktrees/history rewrite.

Все implementation/tests/reviews — агентские. Independent human understanding/
code/artifact review, CORR-003 Person5/team visual review, course AI-policy conflict,
LMS/proposal/source approvals, team/protocol freeze, general REPRO-001 и independent
second-machine reproduction **PENDING**. ROBUST-001 technical PASS/full DoD PARTIAL;
DATA-003 integration technically verified, formal dependent acceptance not replaced.
No training/model/evaluator/metrics/EDA/core runs, no test tuning or training
readiness/full scientific init/budget claim. Next candidate MODEL-001 after review.

## CORR-003 — 18.11.2026

По прямому запросу пользователя Codex проверил current main/origin/history/docs,
existing env и historical Graphify без rebuild. Initial main0490833 clean/synced;
baseline777PASS/14SKIP, pip/raw/split/stats verification PASS. Агент реализовал
train-only selection/pixels/keyed audit noise, reusable10x16 Figure, bounded
PNG/PDF/source CLI;38offline cases и2actual opt-in cases. Final815PASS/0FAIL/
16SKIP; отдельно2actual PASS/0FAIL/0SKIP. Existing14actual DATA/corruption/CUDA
cases не повторялись; прежние даты/results не переобозначены как fresh.

Ключ audit зафиксирован до просмотра actual outputs; минимальный original ID/class
среди persisted45k, без content selection или tuning. Actual IDs29/4/6/9/3/27/0/7/8/1,
raw pixel grid без geometry/Normalize. Official test только в baseline integrity
CLI, никогда в grid или её actual tests. Агент создал actual PNG/PDF/source
manifest, проверил160cells независимыми numeric oracles и480RGB export samples,
открыл PNG и Poppler-rendered PDF. Max blur1.7881393432617188e-7, PDF/PNG sampled
RGB difference0. Exact observations/limits, hashes и expected-failure controls:
[evidence](corr003_evidence.json), [record](reproduction.md).

Published blocks3c7a46e pixels/tests, b57a3c7 renderer/CLI/tests,21ede65 actual
artifacts/tests/evidence. Каждый блок: full diff/relevant tests, explicit staged
paths/list/full diff, оба diff --check, commit main и immediate ordinary push
origin main PASS. Final docs/handoff — отдельный блок. All23protected files
bytes/hashes/mtimes и frozen regions сохранены; dependencies/backend/protocol/
configs не менялись. Models/training/evaluator/metrics/EDA/ROBUST-001 не реализованы.

Все implementation/tests/visual checks выполнил агент. Independent Person5/team
visual/code/artifact review **PENDING**; technical PASS/full DoD PARTIAL. Human
understanding, unresolved course AI policy/LMS/proposal/source approvals, team/
protocol freeze, general REPRO-001 и second-machine reproduction остаются open.
Course compliance и training readiness не заявляются. Next bounded candidate —
ROBUST-001 после current prerequisite review; dependency/gates не изменены.

## CORR-002 — 18.11.2026

По прямому запросу пользователя Codex проверил clean main/origin/fetch/history,
перечисленные документы/current source, historical Graphify без rebuild и existing
env. Baseline453PASS/11SKIP; pip/raw/split/stats verification PASS. Агент реализовал
reflect separable Gaussian blur, progressive darkening и apply_corruption,324offline
cases и3new actual opt-in cases. Default777PASS/0FAIL/14SKIP; отдельно6actual PASS/
0FAIL/0SKIP, включая3existing CORR-001 regressions. Original8DATA actual tests не
повторялись;31.10evidence остаётся historical, fresh CLIs отдельно PASS.

Actual17fixed persisted train IDs×16conditions×epochs0/1, data→dispatcher→Normalize,
labels/IDs и spawn0/2 batch7/5 PASS; fixed injection probes не production policy.
Actual CUDA0 RTX3060Laptop all-family/device/RNG checks PASS; max blur NumPy2D
oracle error CPU/CUDA1.7881393432617188e-7. Test IDs0/71/9999 — только существующий
keyed input regression, без tuning. Saved187438byte NPZ на actual train0/1/2,
run42 epoch0, clean0 иseverity3 каждого семейства; independent replay PASS.
Все23protected raw/split/stats/config/reference/source/protocol/lock files сохранили
bytes/hashes/mtimes; frozen source regions/stable_seed/noise/keyed functions unchanged.
Dependencies/backend/scientific protocol не менялись.

Implementation+tests опубликованы по блокам d339179/72c933a/2586a67; actual artifacts/
evidence —20724ac. Каждый блок: full diff/tests, explicit staged paths/list/full
diff, оба diff --check, commit main и immediate ordinary push origin main PASS.
Final documentation/handoff — отдельный блок. [Evidence](corr002_evidence.json),
[record](reproduction.md). Следующий этап CORR-003 visual audit; ROBUST-001/full
DATA-003 dependent acceptance и general REPRO-001 остаются pending. Models/training/
evaluator/metrics/EDA/plots не реализованы/не запускались. Все проверки/review —
агентские; independent human understanding/review, course policy/LMS/proposal,
team/protocol freeze и independent reproduction pending. Technical PASS/full DoD
PARTIAL; course compliance не заявляется, prerequisite edges сохранены.

## CORR-001 — 18.11.2026

По прямому запросу пользователя Codex проверил актуальные clean main/origin/docs,
исторический Graphify без rebuild, existing env и fresh baseline357PASS/8SKIP;
pip/raw/split/stats verification PASS. Реализованы exact severity mapping всех3
семейств, explicit-generator Gaussian noise и original-test keyed CPU generator/
epsilon поверх unchanged stable_seed. Добавлены96offline cases,2actual CIFAR
cases и1actual CUDA primitive case. Default453PASS/0FAIL/11opt-in SKIP; отдельно
новые actual3PASS/0FAIL/0SKIP. Original8actual data cases не повторялись в этом этапе.

Actual17fixed train IDs проверяют geometry→noise→Normalize, epochs0/1, batch7/5,
spawn0/2 и labels/IDs. Test IDs0/71/9999 проверяют только keyed input contract,
без model/scientific evaluation или test-based tuning. Saved97580byte regression
NPZ и compact evidence versioned. Все16raw/split/stats/config bytes/hashes/mtimes,
frozen source regions, dependencies/backend и scientific settings сохранены.
CUDA primitive реально проверен на RTX3060Laptop; canonical epsilon остаётся CPU,
bitwise CPU/CUDA/cross-platform equality не заявлена.

Commits7d47569/b73536b/f89d70f/5d12a06: полный diff, tests, explicit staged paths,
staged diff и оба diff --check, затем отдельный commit и immediate ordinary push
origin main PASS. Final docs/handoff публикуются отдельным блоком. Blur/darkening/
apply_corruption остаются CORR-002, production policy ROBUST-001, visual audit
CORR-003, full DATA-003 robust integration/dependent acceptance pending. General
REPRO-001 не закрыта. [Evidence](corr001_evidence.json), [record](reproduction.md).
Все implementation/review/checks выполнил агент; technical PASS/full DoD PARTIAL.
Human understanding/review, team/protocol freeze, course policy/LMS/proposal
confirmations и independent reproduction остаются PENDING; course compliance
не заявляется.

## DATA-003 — 31.10.2026

По прямому запросу пользователя Codex проверил main/status/fetch/docs, existing
env, baseline282PASS/5SKIP и actual raw/frozen split, затем реализовал train-only
normalization/provenance/read-only reuse/failure safety, keyed geometry/order,
typed loaders, SHA-256 stable_seed/worker hook и отдельный stats CLI с75offline
cases. Actual stats сохранены по persisted45k; оба YAML получили FILE hash.
Default357PASS/8SKIP; separately8actual PASS/0SKIP. Independent uint64 stats oracle,
full45k/5k/10k coverage/labels, synthetic-policy pairing и spawn2/0 на129actual
train IDs across epochs/batch sizes PASS. Test access — fixed loader/integrity
contract only, без scientific tuning. Frozen split/raw bytes и mtimes сохранены.

Блоки cdffe3d/df6745b/60464bd/de303d0 проверены и каждый сразу push origin main
PASS. Final documentation/handoff — отдельный commit. Dependency/backend/protocol
settings и excluded modules сохранены; corruptions/model/training/eval/plots/EDA
не реализованы и не запускались. Policy boundary использует только synthetic
callables; CORR-001 prerequisite/CORR-002 integration/dependent acceptance pending.
General REPRO-001 не закрыта. Graphify historical snapshot сверен с current source,
rebuild не выполнялся. [Evidence](data003_evidence.json), [record](reproduction.md).
Все checks выполнил агент; independent human understanding/review, team freeze,
second-machine reproduction, course policy/LMS/submitted proposal confirmations
pending. Independent DATA-003 technical PASS, full DoD PARTIAL.

## DATA-002 — 31.10.2026

По прямому запросу пользователя Codex прочитал сохранённый prompt/docs и проверил
clean main, fetch/origin history, dependencies, baseline tests и actual raw integrity.
Агент реализовал exact stratified split494 (one local NumPy RNG), create/load API,
strict manifest/provenance/hash validation, exclusive publication/readiness lock,
preserved failure diagnostics, split prepare/read-only verify CLI и102additional
offline cases; добавил2actual opt-in cases и обновил config tests.

Actual original train50000→45000/5000,4500/500per class verified; NPZ/manifest
versioned, оба YAML содержат один verified NPZ hash, normalization hash null.
Default282PASS/5SKIP; separately5actual PASS/0SKIP (включая3DATA-001).
Repeated calls сохраняют split/raw hashes+file mtimes и historical acquisition
manifest. Split decisions используют original train; test decoding только в
отдельном DATA-001 integrity check. Training/normalization/corruptions не реализованы.

Commits4044d0c/fd0a272/f4d7b42/cf8e684 опубликованы ordinary push в main после
каждого checked block; evidence/docs завершаются отдельным commit. Dependencies,
backend и scientific settings сохранены. Graphify использован как navigation
snapshot и сверен с current files; rebuild не выполнялся.
[Commands/hashes/results/limits](reproduction.md), [compact evidence](data002_evidence.json).
Все checks и review этого этапа выполнил агент; human review, понимание кода
командой, policy/source/team approval и second-machine reproduction PENDING.
Full DATA-002 DoD PARTIAL при technical acceptance PASS.

## DATA-001 — 30–31.10.2026

По прямому запросу пользователя Codex реализовал raw offline CIFAR-10 loader,
official-HTTPS acquisition с проверкой archive/seven-file checksums до unpickle,
оба partition contracts, CLI acquire/verify-only, ignored schema-v1 manifest,
64 synthetic offline cases и3 separate opt-in actual tests. Сверены official
dataset page, torchvision v0.24.0 source и установленная0.24.0+cu126. Данные реально
получены и инспектированы только для integrity; test images/labels не применялись
к scientific decisions. Configs, dependencies/backend и training scope не менялись.

Agent checks, команды, actual results и ограничения — [reproduction.md](reproduction.md).
Код/tests опубликованы небольшими проверенными commits5fd4088/f159fd7/50a9fc4
на main; обычный push после каждого. Отдельный documentation/evidence commit
фиксирует текущий этап. Review выполнял агент; independent human reviewer,
course-policy clarification и подтверждение понимания кода командой остаются
PENDING. Полный DATA-001 DoD — PARTIAL; технические PASS не являются team sign-off.

## Исторический foundation record

30.10.2026: Codex использован по прямому запросу пользователя для source audit,
проверки protocol/architecture, isolated environment, configs/schema validation,
tests и обновления инструкций. Изменения доступны как локальные Git commits.
Data pipeline, CNN, corruptions, training и evaluation на этом этапе не реализуются.

Затронутые файлы: configs/clean_cnn.yaml, configs/robust_cnn.yaml, src/utils.py,
tests/test_utils.py; scripts/check_environment.py, requirements*.txt, .gitignore;
README, planning docs и source/reproduction records в report/. Exact diff и авторство
помощи сохраняются в commits. Local machine verification:116 config tests,
pip check, dependency imports, CPU/CUDA forward/backward, torchvision NMS/PNG
round-trip — PASS. Эти проверки выполнял агент; human understanding/review не
подтверждены ими автоматически.

## Policy status: CONFLICT_PENDING

[Реестр](course_materials/README.md): S1a p17–20 допускает debugging, explanations,
writing improvements и small code suggestions, требует ответственности за код
и понимания pipeline. S1b p7 запрещает code generation tools на graded assignments.
Applicability/precedence для course project и допустимость текущего объёма помощи
не установлены. Авторизация пользователя не является course permission.

Перед сдачей команда должна уточнить policy у course staff, проверить и понимать
каждый изменённый файл, определить требуемый disclosure. Human review,
подтверждение policy и disclosure format **PENDING**; AI self-review не заменяет их.
