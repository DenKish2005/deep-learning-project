# MASTER PROJECT PLAN — CSCI 494 Robustness to Noise

**Главная точка входа для команды.** Дата:30 октября 2026, Asia/Qyzylorda.
Обновление31.10.2026: DATA-001/002 raw loader и frozen split494 сохранены;
DATA-003 independent technical scope PASS: train-only stats, keyed transforms/order,
typed loaders/CLI, actual45k/5k/10k coverage и spawn2 equality. Default357PASS/8SKIP;
отдельно8actual PASS/0SKIP. Evidence — CURRENT_STATUS/report/reproduction.md и
report/data003_evidence.json. Full DATA-003 DoD PARTIAL: CORR-001 prerequisite,
CORR-002 robust integration и human review pending. Следующий этап CORR-001;
dependency graph не изменён.
Работа/commits/push только main по указанию пользователя.
Local environment/config acceptance PASS; training/measured results отсутствуют.

## Что нужно сделать и зачем

Исследовать, как Gaussian noise, Gaussian blur и progressive darkening ухудшают классификацию CIFAR-10, и помогает ли corruption-aware training при той же CNN и том же budget. Вклад — контролируемое воспроизводимое сравнение. Отрицательный результат допустим.

Core: один stratified split45k/5k исходного train и original10k test; CompactCNN с 288746 параметрами, from scratch; Clean и Robust regimes; seeds42/123/2026, по 40 epochs. Шесть training runs, по 16 conditions на best checkpoint: clean +3families×5severity = **96 evaluations**. Все модели видят одинаковые corrupted test images. Corruption применяется в pixels [0,1] **до Normalize**. Checkpoint выбирается только по clean validation accuracy; ties — меньший val loss, затем более ранний epoch.

Полные RQ1–RQ4 и hypotheses: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md). Гипотезы ещё не проверены. Ориентир clean loss≤2pp — предложенный practical criterion, не official requirement и не significance test.

## Исторический foundation snapshot

Исторический аудит5ce6b1a обнаружил scaffold:29 tracked files,20zero-byte и14пустых
Python files; planning package сохранён в0633f67. Foundation продолжен с clean main
на ветке codex/setup-foundation. Теперь есть отдельная `.venv`, exact38-package
lock, CPU/CUDA smoke и sanitized evidence,2полных YAML, strict config validation
и116passing tests. Notebook, data/model/train/eval/plots остаются placeholders;
данные не скачивались, test set не читался.

Локальные commits сделаны по логическим блокам; push/merge не выполнялись.
Полный task DoD остаётся PARTIAL у SETUP-001/002/003, PROTO-001, TEAM-001 из-за
неполученного human review/source/team evidence, при закрытом local technical
acceptance SETUP-002/003. [CURRENT_STATUS](CURRENT_STATUS.md) и
[reproduction record](../report/reproduction.md) разделяют эти уровни готовности.

## Что требует курс и что выбрала команда

Найдены local guidelines Fall2026, CSR и2proposal candidates. Guidelines подтверждают
problem/dataset/baseline/team-trained NN/controlled quantitative comparison/honest
discussion/report/presentation, weights30%/70% и dates Sep20/Week14/Dec12. CSR
содержит project25%course, но semester applicability pending. Точные cutoff/timezone
и presentation slot не указаны. Latest LMS/rubric и submitted version/receipt pending.
Есть конфликт AI policy guidelines/CSR; разрешение курса на объём помощи не
объявлено. [Реестр с pages/hashes](../report/course_materials/README.md).

Два regimes/три families согласуются с обоими proposal candidates; exact3seeds,
96evaluations,40epochs, severity maps и train policy — последующий scope пользователя
и protocol, не текст submitted proposal. Tests/hashes/backups/paired statistics —
engineering/research requirements плана. CIFAR-10-C, ECE и дополнительные CI optional.

Матрица требований с source, priority, tasks, evidence и status: [REQUIREMENTS.md](REQUIREMENTS.md). Person 1 закрывает source gaps в SETUP-001 и уточняет actual AI policy до последующей реализации и сдачи.

## Карта документации: где источник истины

| Документ | Что искать |
|---|---|
| [START_HERE.md](START_HERE.md) | Первые 15 действий, очередь каждого teammate, milestones |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Цель, scope, RQs, hypotheses, научный вклад |
| [REQUIREMENTS.md](REQUIREMENTS.md) | Source hierarchy, A/B/C/D traceability, course uncertainty, AI policy |
| [CURRENT_STATUS.md](CURRENT_STATUS.md) | Read-only audit29 исходных файлов, Git и environment |
| [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) | Полное end-state tree, каждый module/API, types/shapes, config defaults, artifacts |
| [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) | Methodology, severity/RNG, selection, schemas, metrics, явная 96-row matrix |
| [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) |49task cards с 20 полями,19phases, dependencies, tests, DoD, branches, effort |
| [TEAM_TASKS.md](TEAM_TASKS.md) |6owners, файлы, reviews, workload, parallel work и PR strategy |
| [TESTING_PLAN.md](TESTING_PLAN.md) | Named unit/integration/smoke/scientific cases, oracles, diagnostics |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Environment, planned CLI, determinism, resume, artifact verification |
| [PROJECT_SCHEDULE.md](PROJECT_SCHEDULE.md) | Weekly milestones30.10–12.12, critical path, compute estimate и contingency |
| [FINAL_DELIVERABLES.md](FINAL_DELIVERABLES.md) | Tables/figures,10report chapters,14 slides, defence questions, AC01..18 |
| [RISKS_AND_LIMITATIONS.md](RISKS_AND_LIMITATIONS.md) |19risks, owners/fallbacks, assumptions, negative-result policy |
| Этот MASTER_PROJECT_PLAN.md | Сквозная последовательность проекта и общая готовность |

Порядок чтения: этот документ → START_HERE → CURRENT_STATUS → EXPERIMENT_PROTOCOL → TECHNICAL_ARCHITECTURE → свои task cards. Person 1 и reviewers дополнительно читают REQUIREMENTS и TESTING_PLAN. Относительные ссылки внутри docs позволяют переносить repository без привязки к компьютеру.

## Полный путь реализации

1. **Sources/protocol/ownership:** найти originals, уточнить dates/policy, принять RQs и protocol, назначить реальные имена.
2. **Environment/config/identity:** проверить isolated Python 3.12 и torch/vision, записать exact lock; заполнить configs, schema, stable hashes, manifests и backup.
3. **Data:** проверить official CIFAR, сохранить split 494 один раз, вычислить stats только на 45k train; сделать keyed crop/flip/order и deterministic val/test.
4. **Corruptions/model:** реализовать 15 conditions, fixed CPU noise, reflect blur и darkening; CNN, init, modes и tests.
5. **Shared training:** AdamW lr.001/wd.0001, CE, cosine40, batch 128,40 epochs без early stopping; clean-val selection, atomic best/last и resume.
6. **Sanity:** overfit64 train images в debug mode без augmentation; pilots обеих policies, synthetic integration и actual CIFAR smoke; scientific gate.
7. **Core training:** Clean42/123/2026 и Robust42/123/2026; отличается только train corruption policy; logs/checkpoints/backups проверены.
8. **Evaluation:** ровно 96 conditions×10000 predictions, одинаковые inputs, raw logits/counts/provenance, без пропущенных seeds.
9. **Statistics/plots:** per-seed mCA/gap, paired Δ, mean±sample SD;8 figure types,6 table types и failure analysis.
10. **QA/reproduction:** completeness/hash audit, suite, fresh-env smoke и checkpoint comparison другим участником, release candidate.
11. **Report/presentation:** Methods заранее, Results по фактам, честные limitations; все 6 speakers, две полные rehearsals, offline backup.
12. **Submission:** verified rubric/date/format, stable code/artifact bundle, final PDFs, receipt и повторное открытие загруженных файлов.

Точные prerequisites, файлы, функции, steps и acceptance — в roadmap. Phases не запрещают параллельную работу: policy, metric fixtures, plot functions и Methods готовятся до full training.

## Protocol: quick reference

| Decision | Planning candidate |
|---|---|
| Split / normalization | One stratified45k/5k, seed 494; stats только 45k train |
| Model |6Conv+BN+ReLU,3pool, GAP, Dropout.2, Linear128→10;288746 params |
| Training | AdamW lr.001, wd.0001, batch 128,40 epochs, cosine T_max 40, CE, no AMP/early stopping |
| Seeds / budget |42,123,2026;14080updates/run,6 runs |
| Robust policy | Identity.50 / noise.17 / blur.17 / darkening.16; per image, uniform severity 1..3 |
| Evaluation | Clean0 + noise/blur/darkening1..5; fixed noise seed 100494 keyed by original index |
| Noise σ |.02,.05,.08,.12,.18 |
| Blur σ / kernels |.3,.7,1.1,1.5,2.0 /3,7,9,11,13, reflect |
| Darkening factors |.90,.80,.70,.55,.40 |
| Selection | Best clean val accuracy, затем lower val loss, затем earliest epoch |
| Primary comparison | Per-seed mCA_B−mCA_A, все 3 paired Δ; sample SD ddof 1 |
| Test isolation | No test-based policy/hyperparameter/checkpoint selection |

После PROTO-001 решения сохраняются в configs/manifests. Если pilot показывает нехватку budget, менять его симметрично для обеих моделей до final test и обновить docs, steps и acceptance вместе.

## Команда и даты

Person 1 — requirements/protocol/config/report/integration;2 — data/corruptions;3 — CNN/shared trainer/baseline;4 — policy/robust/analysis/presentation;5 — evaluation/metrics/Results;6 — env/repro/plots/QA.

План core workload:186–261active person-hours, включая coauthor/review reserve, без GPU wait; ориентир 5–8h/person/week. Actual availability и hardware пока неизвестны.

Realistic gates:02.11sources/roles;08.11data/model/corruption APIs;15.11smoke;22.11core96;29.11MVP+report draft;06.12final report/reproduction/rehearsals;09.12internal submission;10–12.12buffer. Presentation schedule пересчитать сразу после получения actual date. Детали: PROJECT_SCHEDULE.

Критический путь: sources/protocol → env/config/data/corruptions/model → trainer/policy/smoke →6 runs →96eval → aggregation/plots → report/review/reproduction/rehearsals → submission. Пока идёт compute, команда пишет Methods и reviews.

## Definition of Done и negative results

Full DoD:6runs/96eval, shared inputs/training controls, raw artifacts/hashes, correct metrics/SD/units, required figures/tables, suite+actual smoke, independent reproduction, report/PDF, slides/2rehearsals, retrievable backup и verified submission. Объективный checklist AC01..18 — FINAL_DELIVERABLES.

Если ΔmCA≤0, проверить bugs/control/underfit, показать все seeds и actual trade-off, продолжить report по графику. Negative finding допустим. Missing run, leakage и fabricated data — незавершённый scientific core. OPTIONAL benchmark/ECE можно SKIP; core нельзя сокращать молча.

## Согласованность planning package

Canonical constants:6 runs,96 conditions, seeds42/123/2026, split 494, eval 100494,40 epochs,14080updates/run,288746 params, clean-val selection и pixel-space corruption. Task dependencies должны быть без cycles, local links — действительными; future tree не означает working implementation.

Историческая проверка planning package:14required docs,96unique matrix rows,
49task cards с20полями, links/dependency graph и original29files unchanged тогда.
Текущая foundation проверка:116tests PASS, schema/controlled-pair CLI PASS,
pip check/imports/CPU+CUDA environment smoke PASS. Data/training/evaluation не
запускались. При изменении protocol обновлять requirements/tasks/acceptance вместе.

## Что ещё неизвестно

Неизвестны latest LMS/rubric, AI-policy precedence, submitted proposal version,
presentation date/slot/language/demo, report cutoff/timezone, confirmed roster/role
mapping/availability, artifact storage. GPU/driver/VRAM проверены локально, но full
training runtime/memory не измерены. Независимая техническая работа продолжается;
team/protocol/course compliance остаются pending до соответствующих gates.
