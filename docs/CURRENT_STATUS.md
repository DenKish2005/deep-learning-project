# CURRENT PROJECT STATUS

## Текущее состояние после foundation — 30.10.2026

База продолжения: commit0633f67, исходная ветка main была clean; последующих
commits при старте не было. Создана `codex/setup-foundation`, локальные commits
по логическим блокам; push/merge/fetch не выполнялись. Remote freshness неизвестна.
Применимых AGENTS.md в repository/родительской цепочке нет.

| Task | Статус полного DoD | Фактически выполнено / осталось |
|---|---|---|
| SETUP-001 | PARTIAL | Найдены local guidelines2026, CSR, 2proposal candidates; реестр hashes/pages. Pending: LMS freshness, detailed rubric, receipt/version, policy precedence, точные cutoff/slot |
| PROTO-001 | PARTIAL | Technical architecture/protocol review и corrections записаны; team approval/freeze pending |
| TEAM-001 | PARTIAL | Role/file plan проверен, имена найдены в candidate; mapping/availability/backup/reviewer acceptance pending |
| SETUP-002 | PARTIAL, local technical acceptance PASS | `.venv`,38 exact versions, recipe, pip check/imports/CPU+CUDA smoke PASS; formal human review pending |
| SETUP-003 | PARTIAL, local technical acceptance PASS | 2YAML, strict schema/config CLI,116 tests PASS; formal human review/protocol acceptance pending |

Это завершённый **локальный технический объём текущего этапа**, не team sign-off.
[Evidence](../report/reproduction.md), [environment snapshot](../report/setup_environment.json),
[source registry](../report/course_materials/README.md), [decisions](SETUP_DECISIONS.md).

Реализованы только config functions в src/utils.py, test_utils и environment smoke.
Данные не скачивались, test set не читался, CNN/training/evaluation/plotting не
реализованы и не запускались. Split/stats paths декларативные, hashes null.
REPRO-001 TODO (ignore protection добавлена, identity/manifests/backup ещё нет).
Следующий технический этап: DATA-001 → DATA-002 → DATA-003 с REPRO-001 primitives;
course policy/team approvals уточняются отдельно до соответствующих gates.

## Исторический аудит до planning commit

Весь текст ниже сохраняет snapshot на 5ce6b1a до создания `docs/`; его размеры,
версии, статусы и команды не описывают текущую рабочую ветку. Обследованы все29
исходных tracked файлов. Текущее состояние и проверки перечислены выше.

## Наблюдаемые факты

- Ветка `main`, HEAD `5ce6b1a` — `Architecture created`, 22.09.2026 09:48 +0500. Предыдущий commit `db34ee3` — `Initial commit`, 17.09.2026 22:11 +0500.
- Из локальных веток только main; видны `origin/main` и `origin/HEAD`, совпадающие с HEAD. Fetch не выполнялся, актуальность remote не установлена.
- Remote: `https://github.com/DenKish2005/deep-learning-project.git`. До аудита `git status --short` пустой.
- 20 из 29 файлов имеют размер 0 bytes; из них пять — ожидаемые `.gitkeep`, два — допустимые package markers. Остальные 13 — отсутствующая реализация и пустой proposal.
- 14 Python-файлов: 12 source/script/test файлов и два `__init__.py`; все пустые. Нет функций, training loop, моделей, работающих CLI и assertions.
- Нет datasets, checkpoints, metrics, figures и course documents, включая PDF/DOCX/PPTX. В ignored каталогах тоже только `.gitkeep`.
- Notebook — валидная структура nbformat 4 с `cells: []`, без EDA или execution results.
- Оба YAML содержат только comment. YAML parser вернёт `None`, а не config mapping; команды обучения сейчас нельзя считать рабочими.
- `requirements.txt` содержит восемь имён без versions: torch, torchvision, numpy, pandas, scikit-learn, matplotlib, tqdm, pyyaml. pytest и lock отсутствуют.
- Найден Python 3.12.10. Через package metadata найдены torch 2.14.0, torchvision 0.29.0, numpy 2.5.3, PyYAML 6.0.3; pytest не найден. Это **не проверка imports, ABI, CUDA или пригодности окружения**. GPU/VRAM не проверялись.
- AGENTS.md в репозитории и проверенной цепочке родительских каталогов не обнаружен.

## Реестр каждого исходного файла

В столбце «Проверка/DoD» указан будущий acceptance criterion, а не уже пройденная проверка. Контракты и зависимости раскрыты в [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md).

| Путь | Bytes / текущее назначение и статус | Что должно содержаться / изменение | Зависимости | Проверка / DoD |
|---|---|---|---|---|
| README.md | 436; частично: описание scaffold | Setup → data → train → eval → plots, ссылки на docs и artifacts | Все CLI | Новый участник выполняет smoke по README |
| ARCHITECTURE.txt | 1114; реализовано как дерево scaffold | Синхронизировать с финальным деревом, без заявлений о готовности | Architecture doc | Все пути дерева существуют после реализации |
| LICENSE | 1095; полный MIT license | Сохранить; dataset/paper licenses учитывать отдельно | Нет | Текст сохранён, license не выдается за dataset license |
| requirements.txt | 67; частичный список | Версии и проверенный lock, pytest/dev dependencies | Env validation | pip check + imports + smoke |
| .gitignore | 460; базовые ignore rules реализованы | Добавить outputs/runs, predictions, cache, locks при необходимости; splits и curated tables не терять | Artifact layout | git check-ignore на данных и разрешённых manifests |
| configs/clean_cnn.yaml | 42; comment placeholder | Полный config с schema_version | utils/train/data | parse + invalid config tests |
| configs/robust_cnn.yaml | 43; comment placeholder | Тот же config, отличается только regime и policy | clean config/corruptions | Controlled fields equality test |
| data/raw/.gitkeep | 0; реализованный directory marker | Сохранить; raw CIFAR отдельно от Git | download_data | checksum + class counts |
| data/processed/.gitkeep | 0; directory marker | Сохранить; cache optional и versioned | split/corruptions | Cache hash соответствует manifest |
| outputs/checkpoints/.gitkeep | 0; directory marker | Сохранить; best.pt/last.pt per run | train | load round trip, hashes |
| outputs/metrics/.gitkeep | 0; directory marker | Сохранить; 96 summaries, aggregate tables | evaluate/metrics | Уникальность matrix keys, cardinality |
| outputs/figures/.gitkeep | 0; directory marker | Сохранить; PNG/PDF figures | plots/metrics | Rendering + source provenance |
| src/__init__.py | 0; допустимый package marker | Можно оставить пустым, без side effects | Нет | import src успешен |
| src/data.py | 0; не реализован | CIFAR loader, persisted split, transforms, datasets, loaders | torchvision/corruptions | test_data_pipeline |
| src/corruptions.py | 0; не реализован | Pixel-space corruption + training policy | torch/torchvision | test_corruptions |
| src/models.py | 0; не реализован | CompactCNN, init, architecture ID | torch.nn | test_models |
| src/train.py | 0; не реализован | Shared training engine + CLI, validation, resume | data/models/utils | test_training + smoke |
| src/evaluate.py | 0; не реализован | Fixed conditions inference, predictions, summary | data/models/metrics | test_evaluation |
| src/metrics.py | 0; не реализован | Metrics, paired aggregation, table exports + CLI | numpy/pandas | test_metrics |
| src/utils.py | 0; не реализован | Config validation, seeds, atomic artifacts, metadata | YAML/stdlib/torch | test_utils |
| scripts/download_data.py | 0; не реализован | Download/integrity/split/stats CLI | src.data | Offline error + idempotence |
| scripts/run_all_evaluations.py | 0; не реализован | Matrix validation, eval orchestration | src.evaluate | Exactly 96 unique outputs |
| scripts/make_plots.py | 0; не реализован | Reusable plots from saved tables | metrics/data/matplotlib | test_plots |
| tests/__init__.py | 0; допустимый marker | Можно оставить пустым | pytest | Collection succeeds |
| tests/test_data_pipeline.py | 0; тестов нет | Split, range, transform, loader tests | data fixtures | Реальные assertions, no downloads by default |
| tests/test_corruptions.py | 0; тестов нет | Identity/range/determinism/severity tests | corruptions fixtures | Все типы/severity и invalid inputs |
| notebooks/01_eda_and_sanity.ipynb | 72; пустой notebook | EDA только train, class balance, sample grids; без production logic | src.data | Restart & Run All без скрытого state |
| report/proposal.md | 0; отсутствует содержимое | Восстановить реальный submitted proposal с provenance, не сочинять прошлую сдачу | Teammates/LMS | Source/date/status подтверждены |
| report/final_report/README.md | 67; placeholder | Report plan, source/build/export links | FINAL_DELIVERABLES | Готовый report соответствует rubric |

## Классификация готовности

**Already implemented:** skeleton directory layout, Git repository, MIT text, базовые ignore rules, краткое описание проекта и package/directory markers. Это не рабочая ML-система.

**Partially implemented:** README, dependency declaration, architecture tree, место для report. YAML и notebook имеют только структуру, полезная функциональность отсутствует.

**Not implemented:** data pipeline, split persistence, corruptions, CNN, training, robust policy, evaluation, metrics, plots, tests, experiment tracking, report, slides и reproducibility evidence.

**Requires verification:** submitted proposal, latest rubric, deadlines/timezone, presentation date/duration, AI policy, teammates' names, GPU availability, import/CUDA compatibility, доступ к dataset, branch freshness.

**Out of scope:** training и реализация сейчас; frontend/cloud/deployment; обязательный SOTA; adversarial robustness. OPTIONAL extension CIFAR-10-C не блокирует core.

## Проблемы и блокеры

| ID | Наблюдение | Последствие | Следующая задача |
|---|---|---|---|
| AUD-01 | Нет официальных документов и пустой proposal | Нельзя подтвердить оценки/даты/AI permissions | SETUP-001 |
| AUD-02 | Пустые configs и source | Не существует исполняемого experiment | SETUP-003 → DATA/MODEL/TRAIN |
| AUD-03 | Unpinned dependencies; pytest отсутствует | Нет общего проверенного env | SETUP-002 |
| AUD-04 | Split, normalization, RNG и checkpoint schema отсутствуют | При наивной реализации риск leakage и несопоставимости | DATA-002, DATA-003, TRAIN-002 |
| AUD-05 | Raw outputs целиком ignored | Потеря артефактов без внешней копии | REPRO-001, RELEASE-001 |

Дублирующего рабочего кода, circular imports и фактической утечки данных не обнаружено, поскольку кода и данных нет. Это не доказательство корректности будущей реализации. Реальных runtime bugs и accuracy пока оценить нельзя.

## Выполненные и невыполненные проверки

Выполнены `rg --files --hidden`, recursive listing with sizes, чтение всех непустых файлов, проверка zero-byte файлов, `git status`, `git branch -avv`, `git log -8`, `git remote -v`, `git ls-files`, `python --version` и package metadata inspection. Эти операции не изменяли исходные файлы и Git.

Не выполнялись: installation, imports torch/torchvision, dataset download, notebook execution, pytest, training, evaluation и model benchmarks. Плановые команды в документации не являются отчётом об успешном запуске.

После планирования добавлены только документы `docs/*.md`. Исходные 29 файлов должны остаться byte-identical; итоговая проверка перечислена в MASTER_PROJECT_PLAN.

## Проверка созданного planning package

Фактически выполнено 30.10.2026:

- Все 14 требуемых Markdown-документов существуют, непустые и читаются как UTF-8.
- Локальные Markdown-ссылки ведут на существующие документы; code fences сбалансированы.
- В roadmap ровно 49 уникальных task cards, каждая содержит все 20 обязательных полей; dependency graph не содержит неизвестных prerequisite IDs или cycles.
- В protocol ровно 96 уникальных matrix rows: 2 regimes × 3 seeds × 16 conditions. Для каждого run присутствуют clean0 и все 15 corruption conditions.
- Все 29 исходных tracked файлов совпадают с HEAD после штатных Git clean filters; `git diff --exit-code` и `git diff --check` проходят. У LICENSE checkout использует CRLF при LF в Git blob из-за `core.autocrlf=true`; это не изменение файла этой задачей.
- `git status --short` показывает только `?? docs/`. Commits, push, branch changes и remote mutations не выполнялись.

Это проверка документации и неизменности исходников. ML unit/integration tests, imports/ABI/CUDA checks, data download, training и evaluation по-прежнему **не выполнялись**.
