# MASTER PROJECT PLAN — CSCI 494 Robustness to Noise

**Главная точка входа для команды.** Дата:30 октября 2026, Asia/Qyzylorda. Этап: аудит и planning. Этот пакет не реализует ML pipeline и не содержит измеренных результатов.

## Что нужно сделать и зачем

Исследовать, как Gaussian noise, Gaussian blur и progressive darkening ухудшают классификацию CIFAR-10, и помогает ли corruption-aware training при той же CNN и том же budget. Вклад — контролируемое воспроизводимое сравнение. Отрицательный результат допустим.

Core: один stratified split45k/5k исходного train и original10k test; CompactCNN с 288746 параметрами, from scratch; Clean и Robust regimes; seeds42/123/2026, по 40 epochs. Шесть training runs, по 16 conditions на best checkpoint: clean +3families×5severity = **96 evaluations**. Все модели видят одинаковые corrupted test images. Corruption применяется в pixels [0,1] **до Normalize**. Checkpoint выбирается только по clean validation accuracy; ties — меньший val loss, затем более ранний epoch.

Полные RQ1–RQ4 и hypotheses: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md). Гипотезы ещё не проверены. Ориентир clean loss≤2pp — предложенный practical criterion, не official requirement и не significance test.

## Что уже есть

Аудит commit5ce6b1a показал 29 tracked files,20 zero-byte files и **14 пустых Python files**. README описывает repository scaffold. YAML содержат только comments, notebook без cells, proposal пуст. Нет данных, training/evaluation pipeline, работающих tests, checkpoints, metrics и figures. Dependency names есть, versions/lock/pytest отсутствуют.

Созданные 14 docs — завершённый planning package, не реализация компонентов. Git commits, push и branches не менялись. Подробный inventory всех файлов, Git/env facts и blockers: [CURRENT_STATUS.md](CURRENT_STATUS.md).

## Что требует курс и что выбрала команда

Оригиналов guidelines, rubric, AI policy и submitted proposal нет. По пересказу в MASTER PROMPT нужны: чёткая problem, dataset, baseline, ≥1team-trained NN, controlled comparison, quantitative evaluation, honest limitations, report и presentation. Weights30%presentation/70%report/project25%course **не подтверждены**. Report12.12.2026 и presentation Week 14 также требуют сверки; cutoff/timezone неизвестны. Proposal20.09 уже прошёл.

Два training regimes, три families, три seeds и 96evaluations — явный scope пользователя. Нельзя приписывать эти commitments submitted proposal, пока он не найден. Tests/hashes/backups/paired statistics — engineering и research quality requirements этого плана. CIFAR-10-C, ECE и дополнительные CI — optional.

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

Проверки planning package:14required docs,96unique matrix rows,49task cards с 20 полями, links и dependency graph, original29 tracked files unchanged. Код, обучение и ML tests/evaluation не запускались. Итоговый результат проверок указан в ответе чата. При изменении protocol обновлять затронутые requirements/tasks/schedule/acceptance одним согласованным PR.

## Что ещё неизвестно

Нужны actual course guidelines/rubric/AI policy, submitted proposal, presentation date/slot/language/demo, report cutoff/timezone, имена и availability команды, GPU/driver/VRAM, artifact storage. Эти неизвестные не блокируют planning; Persons1/6 закрывают их до соответствующих execution/submission gates. До проверки status — VERIFY.
