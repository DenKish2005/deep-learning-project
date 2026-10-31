# Требования и traceability

Дата: 30.10.2026. [Главный план](MASTER_PROJECT_PLAN.md). MUST/SHOULD/OPTIONAL обозначают приоритет **в проектном плане**, а не автоматически университетское требование.

Обновление SETUP-001: найдены локальные guidelines Fall2026, CSR и две версии
proposal. [Реестр с hashes/pages/status](../report/course_materials/README.md)
уточняет строки A/B и разделяет source verification и implementation readiness.
Текст локальных источников проверен; LMS
freshness, submitted version и AI-policy precedence остаются pending.

## Источники и иерархия

| Код | Источник | Доступность и достоверность |
|---|---|---|
| S1 | CSCI 494 guidelines/rubric/calendar/AI policy | S1a guidelines Fall2026 и S1b CSR найдены локально; отдельная rubric/LMS freshness pending; AI policy conflict |
| S2 | Submitted project proposal | Две локальные DOCX candidates, содержание сверено; факт сдачи и версия pending, см. report/proposal.md |
| S3 | MASTER PROMPT пользователя, приложенный текст в чате 30.10.2026 | Доступен, задаёт текущий project scope; пересказ course rules предварительный |
| S4 | README.md и ARCHITECTURE.txt на 5ce6b1a | Подтверждают skeleton и тему; не содержат rubric |
| S5 | Настоящие planning docs | Рекомендации команды для реализации; не официальный источник |

При получении S1/S2 Person 1 ведёт запись: имя документа, версия, дата, locator/страница, точный текст требования, решение конфликта и затронутые tasks. Более свежие официальные указания выше пересказа S3. Разницу scope с proposal нужно обсудить с командой и при необходимости преподавателем, не переписывать историю.

## A. Предварительно известные требования курса

Формулировки таблицы происходят из S3; их фактическая сверка с локальными S1a/S1b
и текущие пределы подтверждения вынесены в реестр. Колонка «Сейчас» разделяет
проверку текста источника и готовность реализации.

| ID | Требование / источник | Приоритет | Tasks | Файлы | Evidence завершения | Проверка | Сейчас |
|---|---|---|---|---|---|---|---|
| A01 | Чёткая проблема; S1a p2–7 | MUST | PROTO-001, REPORT-001 | docs/PROJECT_OVERVIEW.md, report/final_report/report.md | RQ, scope, motivation | Report review vs S1 | Local source checked; RQ draft |
| A02 | Доступный usable dataset; S1a p2–7 | MUST | DATA-001..003 | src/data.py, data/splits/ | Checksum/counts/split manifest | Data tests | DATA-001/002 technical PASS; DATA-003/reviewer pending |
| A03 | Baseline и ≥1 NN, обученная командой; S1a p2–7 | MUST | MODEL-001, CLEAN-001..003 | src/models.py, src/train.py | Checkpoints + logs + attribution | Provenance audit | Local source checked; models TODO |
| A04 | Controlled comparison; S1a p2–7 | MUST | PROTO-001, ROBUST-001..003, TEST-002 | configs/, src/train.py | Equality fields, paired seeds | Scientific validity tests | Config controls implemented; run controls TODO |
| A05 | Quantitative evaluation; S1a p2–7 | MUST | EVAL-001..004, METRIC-001..002 | src/evaluate.py, src/metrics.py | 96 results + tables | Matrix audit | Local source checked; evaluation TODO |
| A06 | Честное обсуждение successes/failures/limitations; S1a p2–7 | MUST | ANALYSIS-001, REPORT-002 | report/final_report/report.md | Все RQ, uncertainty, negative results | Evidence review | Local source checked; discussion TODO |
| A07 | Final presentation; S1a p16 | MUST | PRESENT-001..003 | report/presentation/ | Slides + 2 rehearsal logs | Rubric + rehearsal | Week14 указан; slot pending |
| A08 | Final report; S1a p16 | MUST | REPORT-001..003 | report/final_report/ | Export PDF + source | Rubric/completeness | Local source checked; report TODO |
| A09 | Presentation30%/report70% (S1a p22), project25% курса (S1b p4) | VERIFY_CURRENT | SETUP-001 | report/course_materials/README.md | LMS version confirmation | Проверка весов и версии | Local text checked; CSR applicability pending |
| A10 | Proposal Sep20; presentation Week14; report Dec12 (Fall2026); S1a p16 | VERIFY_DETAILS | SETUP-001, PRESENT-003, RELEASE-002 | docs/PROJECT_SCHEDULE.md | Calendar + timezone | Сверка LMS | Local text checked; cutoff/timezone/slot pending |
| A11 | Limited AI suggestions (S1a p17–20) vs code-generation prohibition (S1b p7) | CONFLICT_PENDING | SETUP-001, REPORT-003 | report/ai_usage.md | Policy precedence + disclosure + понимание кода | Human review | Conflict recorded; permission unresolved |

## B. Project scope и предполагаемые commitments proposal

Это scope S3/planning. Локальные S2a/S2b подтверждают общую постановку,
но submitted version pending. Нельзя приписывать им exact seeds/budget/maps из
плана; [proposal provenance](../report/proposal.md) разделяет эти commitments.

| ID | Требование / источник | Приоритет | Tasks | Файлы | Evidence | Проверка | Сейчас |
|---|---|---|---|---|---|---|---|
| B01 | CIFAR-10 45k/5k/10k, S3 | MUST | DATA-001..003 | src/data.py, data/splits/ | Disjoint stratified indices + stats | TEST-D02 | Actual split/counts/provenance PASS; stats/loaders/reviewer pending |
| B02 | Одна compact custom CNN с нуля, S3 | MUST | MODEL-001 | src/models.py | 288746 params, logits N×10 | TEST-M01..03 | Нет |
| B03 | Noise, blur, darkening ×5 pixel-space severity, S3 | MUST | CORR-001..003 | src/corruptions.py | Severity table + example grid | TEST-C01..08 | Нет |
| B04 | Clean vs Robust; только train policy различается, S3 | MUST | TRAIN-001..003, ROBUST-001 | configs/, src/train.py | Paired initialization/order/budget logs | TEST-S01 | Config validation PASS; runtime identity TODO |
| B05 | Seeds 42,123,2026, S3 | MUST | CLEAN-001..003, ROBUST-002..003 | outputs/runs/, checkpoints/ | 6 complete runs | Manifest audit | Нет |
| B06 | 16 условий/run, 96 оценок, одни test images, S3 | MUST | EVAL-002..004 | scripts/run_all_evaluations.py | 96 keys, 960000 predictions | TEST-E02 | Нет |
| B07 | Accuracy/drop/RR/mCA/gap/F1/per-class/CM/mean±SD, S3 | MUST | METRIC-001..002 | src/metrics.py | Formula fixtures + exports | TEST-K01..04 | Нет |
| B08 | Figures и tables из raw data, S3 | MUST | PLOT-001..003 | scripts/make_plots.py | PNG/PDF + provenance | TEST-P01 | Нет |
| B09 | 6 участников понимают pipeline, S3 | MUST | TEAM-001, PRESENT-002..003 | docs/TEAM_TASKS.md, report/rehearsals.md | Review assignments, defence rehearsal | Peer explanation | План |
| B10 | Полный пакет planning docs до кода, S3 | MUST | AUDIT-001 | docs/*.md | 14 файлов, ссылки, consistency checks | Documentation audit | Выполнено этим пакетом |

## C. Engineering и research quality

| ID | Рекомендация / источник | Приоритет | Tasks | Файлы | Evidence | Проверка | Сейчас |
|---|---|---|---|---|---|---|---|
| C01 | Persist split, hashes, per-image RNG, S5 | MUST | DATA-002, CORR-001, REPRO-001 | data/splits/, src/utils.py | Manifests/hash equality | TEST-D02, C03 | DATA-002 technical PASS; per-image RNG/general REPRO TODO |
| C02 | Validation-only checkpoint selection; test isolation, S3/S5 | MUST | PROTO-001, TRAIN-001, TEST-002 | src/train.py | clean-val selection logs | Code + provenance audit | План |
| C03 | Tested env, version lock, CPU/GPU smoke, S5 | MUST | SETUP-002, TEST-001 | requirements-lock.txt, tests/ | pip check, import and smoke evidence | Independent run | Local env PASS; full pipeline/independent run pending |
| C04 | Atomic best/last checkpoint and epoch-boundary resume, S5 | MUST | TRAIN-002, REPRO-001 | src/utils.py, src/train.py | Resume vs uninterrupted fixture | TEST-U03, T02 | Нет |
| C05 | Unit/integration/scientific tests, S5 | MUST | TEST-001..003 | tests/ | Assertions and passing report | TESTING_PLAN |282offline PASS/5opt-in SKIP, separately5actual PASS; ML/scientific suite TODO |
| C06 | Paired deltas, sample SD, n=3 caveat, S5 | MUST | METRIC-002, ANALYSIS-001 | src/metrics.py, report/ | All seeds included | Aggregation fixtures | Нет |
| C07 | Predeclared ≤2 pp clean-loss interpretation, S5 | SHOULD | PROTO-001, ANALYSIS-001 | docs/EXPERIMENT_PROTOCOL.md | Threshold fixed before test | Protocol history | План |
| C08 | Independent reproduction + backups, S5 | MUST | REPRO-002, RELEASE-001 | report/reproduction.md, manifests | Checkpoint hashes and smoke on second machine | Person 6 + reviewer | Нет |
| C09 | Versioned PR/code review workflow, S5 | SHOULD | TEAM-001 | docs/TEAM_TASKS.md | Review evidence + checks | PR review | План |

## D. Optional extensions

| ID | Extension / источник | Приоритет | Tasks | Файлы | Evidence / проверка | Сейчас |
|---|---|---|---|---|---|---|
| D01 | CIFAR-10-C: noise, defocus_blur, brightness; S3/S5 | OPTIONAL | BENCH-001 | src/data.py, src/evaluate.py | Separate benchmark manifest + 90 corruption evaluations | Не начато |
| D02 | ECE с фиксированными bins; S3 | OPTIONAL | METRIC-003 | src/metrics.py | Hand-calculated bins fixture | Не начато |
| D03 | CI и дополнительные statistical tests; S3/S5 | OPTIONAL | METRIC-003 | src/metrics.py | Предпосылки, низкая power n=3, separate label | Не начато |

## AI-assisted work

Текущая техническая подготовка явно разрешена пользователем. Это не свидетельствует
о разрешении курса на данный объём сгенерированного кода. Найденные S1a и S1b
расходятся по AI policy; [ai_usage.md](../report/ai_usage.md) фиксирует конфликт,
помощь и pending human review/disclosure. Каждый участник обязан объяснить pipeline,
формулы, RNG и собственные изменения. Course compliance не объявляется verified.

Если policy строже S3, адаптировать последующий workflow и объём помощи; текущие документы можно использовать как planning material с необходимым disclosure. Не подменять официальное согласование предположением.
