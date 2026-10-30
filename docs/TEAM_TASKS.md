# Работа шести участников

Person 1..6 — placeholders до TEAM-001, реальные имена и availability неизвестны. Это распределение будущей работы; агенты и дополнительные чаты в ходе планирования не создавались. [Roadmap](IMPLEMENTATION_ROADMAP.md) содержит primary owners каждого task.

## Ответственность, файлы и результаты

| Участник | Primary tasks / файлы | Deliverables и зависимости | Тесты / DoD | Review responsibility |
|---|---|---|---|---|
| Person 1 — lead/research/integration | AUDIT-001,SETUP-001/003,PROTO-001,TEAM-001,TEST-002,REPORT-001/003,RELEASE-002; docs/, configs/,report/course_materials/, proposal.md, ai_usage.md, final_report/ | Verified requirements, frozen protocol/config, report integration, submission receipt. Зависит от sources, затем measured results | Config/scientific tests; requirement→evidence matrix закрыта; final report без unsupported claims | Eval/metrics/statistical claims, code/data identity и scope changes |
| Person 2 — dataset/corruptions | DATA-001..004,CORR-001..003; src/data.py,src/corruptions.py,scripts/download_data.py,data/splits/, notebook,tests/test_data_pipeline.py, test_corruptions.py | Verified dataset, one split, train-only stats, keyed inputs, corruption grid; depends env/protocol | TEST-D/C suite; same corrupted inputs для 6models; EDA restart/run all | Robust augmentation и pre-normalization order |
| Person 3 — model/training/baseline | MODEL-001,TRAIN-001..003,CLEAN-001..003; src/models.py,src/train.py,tests/test_models.py, test_training.py, clean run artifacts | CNN288746 params, shared trainer,best/last/resume,3clean runs; depends loaders/config | Model/epoch/checkpoint/overfit tests, all3clean seeds with40 epochs | Robust counterparts, learning curves и checkpoint validity |
| Person 4 — robust/analysis/presentation | ROBUST-001..003,ANALYSIS-001,REPORT-002,PRESENT-001..003; policy hook in corruptions/data via Person 2 review, robust artifacts, report final/presentation | Tested policy,3robust runs,failures/RQ narrative, slides+2rehearsals; depends corruption/train API; OPTIONAL BENCH-001 | Policy/RNG isolation, controlled comparisons; all6 speakers can explain code | Baseline/geometry equality, failure-case sampling, limitations |
| Person 5 — evaluation/metrics | EVAL-001..004,METRIC-001/002; src/evaluate.py,src/metrics.py,scripts/run_all_evaluations.py,tests/test_evaluation.py, test_metrics.py, aggregate tables | Core96evaluations, allmetrics, pairedΔ+SD, tables; depends checkpoints/common inputs; OPTIONAL METRIC-003 | TEST-E/K;96uniquekeys and correct units/aggregation | Corruption numeric tests, report Results, statistical interpretation и figure values |
| Person 6 — env/repro/plots/QA | SETUP-002,REPRO-001/002,TRAIN-004,TEST-001/003,PLOT-001..003,RELEASE-001; requirements*,.gitignore,src/utils.py,scripts/run_experiments.py, make_plots.py,tests/conftest.py, test_utils.py, test_plots.py, test_smoke.py, README, ARCHITECTURE.txt, artifact_manifest | Locked env,identity/backups, orchestration,8 figure types, independent reproduction/release candidate | Import/smoke/manifest/plot/full QA; retrievable backup and actual reproduction evidence | Trainer recovery, install portability, source-to-figure provenance |

`tests/test_protocol.py` primary Person 1; Person 4 supplies treatment fixtures, Person 5 completeness fixtures. `tests/__init__.py`, `src/__init__.py` need no feature owner changes. Source contract owners are fixed; task author may differ but must obtain that module owner's review.

## Баланс нагрузки

Roadmap effort — primary author active hours; GPU wait не включён. К ним нужно добавить coauthor/review/rehearsal время, указанное ниже. Это резерв workload, а не гарантированный runtime. Каждый делает presentation и читает весь report.

| Person | Primary tasks, hours | Coauthor/review/support reserve | Общий active план | Конкретная дополнительная работа |
|---|---:|---:|---:|---|
| 1 | 24–35 | 8–10 | 32–45 | Intro/background integration, final rubric cross-check, source verification |
| 2 | 22–30 | 8–12 | 30–42 | Dataset/Corruptions report sections, visual QA, policy review, defence |
| 3 | 17–25 | 12–16 | 29–41 | Method/training report sections, run monitoring, recovery test review, architecture slide |
| 4 | 25–36 | 6–8 | 31–44 | Discussion/limitations coediting, pairedrun diagnostics, speaking notes |
| 5 | 15–22 | 16–20 | 31–42 | Results/statistics report authorship, figure verification, negative-result analysis, independent eval review |
| 6 | 33–45 | 0–2 | 33–47 | Transfer routine artifact hash verification to3/5, delegate caption proofreading to2 |

Core total **186–261 person-hours**, ориентир 5–8hours/person/week до report deadline. OPTIONAL BENCH-001 дополнительно 4–6 h Person 4, METRIC-0032–3 h Person 5 плюс measured compute. Если фактическая availability ниже, Person 1 пересматривает график сразу, не в декабре. Person 6 не должен единолично писать report и slides; их ведут1/4, chapters пишут все.

## Как работать параллельно без конфликтов

1. После protocol/interface freeze шесть направлений идут параллельно на synthetic fixtures: data, model/training, robust policy design, metrics/eval, env/plot skeleton, requirements/report Methods. Full data/GPU не нужны для первых unit tests.
2. Один primary owner на module. `src/utils.py` owner6: Person 1 config validation, Person 2 stable_seed, Person 3 checkpoint functions добавляют последовательными small PR с review6; не переписывают файл целиком. `src/train.py` owner3, Person 4 **не копирует** trainer для robust model.
3. Person 2 первым merges corruption API. Person 4 потом добавляет policy и dataset hook отдельным PR; Person 2 в это время заканчивает EDA и docs. После hook merge Person 2 data module не меняет без coordination.
4. Person 6 предоставляет plot exporter skeleton; Person 2 добавляет grid в отдельном PR после согласования API. Person 5 пишет aggregate schema до final results; Person 6 строит plots на synthetic fixtures с explicit test labels.
5. Config pair owner1: treatment values правятся одновременно в одном PR; после freeze изменения научных полей только по protocol change. Не редактировать YAML вручную для каждого seed — seed CLI сохраняется в resolved config.
6. Report sections делить по headings:1intro/background/integration,2dataset/corruptions,3architecture/training,5results/statistics,4discussion/limitations,6reproduction/appendix. Один integrator1 merges edits; dataset/results raw tables не переписывать руками.

## Git/PR workflow для будущей реализации

`main` стабильная; short-lived feature branches `codex/<task-id>`. Отдельный `dev` сейчас не нужен: команда маленькая, sequential reviewed PR в main проще. Можно добавить dev только при подтверждённой потребности длительной интеграции, с теми же required checks.

PR границы: SETUP config/schema; DATA split/loaders; CORR primitives; MODEL; TRAIN engine; TRAIN checkpoint/resume; ROBUST policy; METRIC; EVAL; PLOT; REPORT отдельно. Code PR содержит tests и краткое описание behavior, validation, remaining limits. Не включать raw dataset, checkpoints, generated large outputs или чужие feature changes.

Merge gate: ≥1reviewer вне автора; relevant tests+synthetic integration; no unintended controlled-field differences; no secrets/large artifacts; docs/interfaces updated. Data/scientific PR дополнительно reviewer5 или 1; core training не запускать с локальным unreviewed scientific fix. После source freeze run SHA/config metadata совпадают с release identity; исправления корректности делают новый versioned run, а не меняют старые метрики.

В рамках текущей planning task commits, branches, push и remote changes не делались. Этот workflow описывает последующие действия команды.

## Общая готовность участников

До первой rehearsal каждый за 10 мин объясняет полный путь raw image→split→pixel corruption→Normalize→CNN→loss/checkpoint→metrics→figure. Reviewer задаёт вопросы про val/test separation, seed streams, mean±SD и negative result. Если участник не может ответить, назначить pair review конкретного module и повторить объяснение. Presentation распределение и defence questions: [FINAL_DELIVERABLES.md](FINAL_DELIVERABLES.md).
