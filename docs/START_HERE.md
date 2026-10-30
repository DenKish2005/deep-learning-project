# С чего начать

Foundation этап выполнен в локальном техническом объёме: отдельная `.venv`, exact
versions, CPU/CUDA environment smoke, complete YAMLs и strict validation с116
passing tests. ML pipeline ещё не реализован; данные не скачаны, модели не обучены.
Найдены guidelines/CSR и proposal candidates; submitted version, policy precedence
и human approvals pending. Актуальные статусы полного DoD — в CURRENT_STATUS,
проверенные команды — README, evidence — report/reproduction.md.

Сначала прочитайте [MASTER_PROJECT_PLAN.md](MASTER_PROJECT_PLAN.md), затем [CURRENT_STATUS.md](CURRENT_STATUS.md). Перед кодом — [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md); перед своим PR — [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) и карточку задачи в [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md).

## Первые 15 действий команды

Это reference очередь implementation phase. SETUP-001/PROTO-001/TEAM-001 частично
закрыты; у SETUP-002/003 local technical acceptance PASS, formal human review
pending. Следующая реализация — DATA-001/002/003 и REPRO-001 primitives. Независимая
техническая подготовка не ждёт missing source/team information; full training
начинается только после policy/protocol и smoke/scientific gates.

| # / Task ID | Конкретное действие и файлы | Кто | Evidence завершения |
|---|---|---|---|
|1 SETUP-001 | Найти rubric/calendar/AI policy и proposal; report/course_materials/README.md, report/proposal.md, report/ai_usage.md |1 | Sources с version/date/locator; A01..A11 confirmed либо owned pending |
|2 TEAM-001 | Указать имена, availability и reviewers в docs/TEAM_TASKS.md |1+все | Critical tasks имеют owner/backup, files разделены |
|3 PROTO-001 | Review и freeze docs/EXPERIMENT_PROTOCOL.md до просмотра final test |1+2+5 | Дата/version; selection, severity и budget приняты |
|4 SETUP-002 | Проверить venv, Python/torch/vision/pytest и CPU/GPU; requirements.txt и будущий requirements-lock.txt |6 | Реальные imports/pip check/smoke, exact lock |
|5 SETUP-003 | Заполнить оба YAML; schema/common-field validation; configs/*.yaml, src/utils.py, tests/test_utils.py |1+6 | Invalid configs rejected, только treatment отличается |
|6 REPRO-001 | Identity/metadata/atomic writes/backup; src/utils.py, .gitignore |6 | Stable hashes, collision errors, backup copy verified |
|7 DATA-001 | Download/integrity CIFAR; src/data.py, scripts/download_data.py |2 | Checksum, counts50k/10k, RGB32×32,10 classes |
|8 CORR-001 | Keyed noise, stable seed, severity map; src/corruptions.py, src/utils.py, tests |2+6 | Same index→same ε, range/shape/mutation tests |
|9 DATA-002 | Сохранить stratified split 494; data/splits/, src/data.py |2 |45k/5k,4500/500per class, disjoint/exhaustive, hashes |
|10 MODEL-001 | CompactCNN и shape tests; src/models.py, tests/test_models.py |3 |N×10logits,288746 params, finite backward/modes |
|11 METRIC-001 | CM/F1/mCA/drop/RR/gap на hand fixtures; src/metrics.py, tests/test_metrics.py |5 | Exact synthetic oracles, pp/fraction units |
|12 CORR-002 | Reflect Gaussian blur и darkening; src/corruptions.py и tests |2 | Kernels3/7/9/11/13, exact factors, input unchanged |
|13 DATA-003 | Train-only stats, typed loaders, keyed crop/flip/order; src/data.py, data/splits/normalization.json |2 | Paired geometry/order equal, val/test deterministic |
|14 TRAIN-001 | Shared training/validation; src/train.py, tests/test_training.py |3 | Tiny2 epoch integration, weighted metrics, no test access |
|15 ROBUST-001 | Per-image policy с отдельным RNG; src/corruptions.py, src/data.py, tests/test_protocol.py |4+2 | Frequencies/severity1..3 correct, global RNG unchanged |

Следом: TRAIN-002 checkpoint/resume → TRAIN-003 overfit/pilot; TRAIN-004 orchestrator и EVAL-001/PLOT-001 → TEST-001/002 smoke/scientific gate → core training. Не выбирать probabilities, epochs или checkpoint по test accuracy.

Install/check_environment/config validation/pytest команды в README и начальном
разделе [REPRODUCIBILITY.md](REPRODUCIBILITY.md) проверены. Раздел «Плановые команды
pipeline» остаётся будущим интерфейсом, его CLI ещё не работают.

## Очередь каждого участника

- **Person 1:** sources → protocol/config → scientific review → report integration. Methods начать до результатов.
- **Person 2:** data/corruptions → split/stats/loaders → visual QA/EDA → Dataset/Corruptions главы.
- **Person 3:** model → shared trainer/checkpoint → overfit/pilot →3baseline runs → Training/Architecture главы.
- **Person 4:** policy tests → data hook →3robust runs → failure analysis/Discussion/slides/rehearsals.
- **Person 5:** metric fixtures → evaluator →96evaluations → aggregation → Results и statistical review.
- **Person 6:** env/lock → metadata/backup → orchestrator/plots → smoke/QA → independent reproduction/release candidate.

Primary module owners и последовательность PR в общих файлах описаны в TEAM_TASKS. Полные task details с 20 полями находятся в roadmap.

## Milestones без неоднозначности

| Gate | Evidence |
|---|---|
| План готов |14 docs, полный audit,96-row matrix,49task cards, links/consistency checked |
| Pipeline готов к долгим runs | Unit/integration tests,64-image overfit, actual CIFAR smoke, paired controls |
| Core experiment готов |6complete40 epoch runs,96 conditions×10k, hashes/backups verified |
| Analysis готов |32 condition groups с n 3,3 paired deltas,6 table types/8figure types, RQ evidence map |
| Submission готова | Final report/slides,2rehearsals, reproduction log, course checklist, artifact retrieval и receipt |

Targets: smoke15.11, core 22.11, MVP29.11, internal submission09.12.2026. Report12.12 предварительно; presentation Week 14 без подтверждённой даты. Уточнения и fallback: [PROJECT_SCHEDULE.md](PROJECT_SCHEDULE.md).

Если Robust не выиграла, проверить correctness, показать фактические all-seed findings и продолжить report. Неподтверждённая hypothesis не оправдывает cherry-picking или настройку augmentation по test.
