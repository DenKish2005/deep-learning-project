# Промпт следующего этапа — DATA-001

Скопируйте текст ниже в новый чат этого проекта.

```text
Продолжи проект CSCI 494 — Robustness to Noise.

Поставь цель: выполнить DATA-001 — получение CIFAR-10, проверку целостности и
offline loading. Выполни работу до проверяемого результата, а не составляй общий план.

Сначала прочитай AGENTS.md, если он существует, README.md и документы:
- docs/MASTER_PROJECT_PLAN.md
- docs/START_HERE.md
- docs/CURRENT_STATUS.md
- docs/REQUIREMENTS.md
- docs/TECHNICAL_ARCHITECTURE.md
- docs/EXPERIMENT_PROTOCOL.md
- docs/IMPLEMENTATION_ROADMAP.md — особенно DATA-001 и его зависимости
- docs/REPRODUCIBILITY.md
- docs/TESTING_PLAN.md
- docs/SETUP_DECISIONS.md
- report/course_materials/README.md
- report/ai_usage.md
- report/reproduction.md

Проверь текущую ветку, git status, последние commits и состояние origin/main.
Учитывай реальные изменения после foundation, не полагайся только на этот промпт.
Foundation уже содержит отдельную .venv, exact dependency lock, CPU/CUDA smoke,
два полных YAML и strict config validation. На момент подготовки этого промпта
текущая suite имела 116 passing tests; data pipeline ещё не реализован.

ОБЪЁМ

1. Используй существующую .venv и проверенные dependencies. Не меняй глобальное
   окружение и не обновляй versions без обоснованной необходимости. Проверь pip
   check и текущие tests перед изменениями. Повторный CUDA smoke нужен при
   изменении environment/backend, а не для каждого изменения loader.

2. Реализуй DATA-001 в src/data.py, scripts/download_data.py и
   tests/test_data_pipeline.py. Сохрани согласованный контракт:
   load_cifar10(root: Path, train: bool, download=False).
   По умолчанию loader работает offline: не делает скрытого download, а при
   отсутствии/повреждении данных выдаёт понятную ошибку с действием для исправления.

3. Разрешаю скачать официальный CIFAR-10 в ignored data/raw/. Перед скачиванием
   проверь, нет ли уже корректных данных. Проверь official source, archive/file
   checksums и соответствие установленной torchvision по официальным источникам.
   Не используй непроверенные зеркала. Не перезаписывай и не удаляй повреждённые
   или чужие artifacts молча; покажи точную причину и безопасный способ восстановления.

4. Реализуй работающий CLI из root:
   .\.venv\Scripts\python.exe -m scripts.download_data --root data/raw
   .\.venv\Scripts\python.exe -m scripts.download_data --root data/raw --verify-only
   Первый режим явно получает отсутствующие данные и проверяет их; verify-only
   не скачивает и не создаёт/изменяет dataset artifacts. Invalid arguments/errors
   должны давать ненулевой exit code. Не объявляй будущий --prepare-split работающим.

5. Проверь actual dataset: original train 50000, test 10000, RGB 32x32, 10 classes,
   labels и class mapping, counts 5000/class train и 1000/class test, checksums.
   Integrity inspection test partition разрешена; не используй test images или
   labels для model/hyperparameter/preprocessing/augmentation decisions.
   Сохрани компактный versioned integrity manifest с source/checksum/counts и
   фактическим provenance в ignored data directory. Повторная успешная проверка
   должна быть идемпотентной и не выдавать ранее не проверенные данные за verified.

6. Добавь содержательные offline tests: missing/corrupt data, integrity failures,
   оба partition, class/count/shape contract, download=False без network,
   явный download mode, повторная проверка и CLI success/failure.
   Default pytest не должен скачивать CIFAR-10 или зависеть от его наличия.
   Actual local CIFAR checks запускай отдельно с explicit opt-in и зафиксируй
   результат; отсутствие данных/skip не выдавай за PASS.

7. Обнови README, CURRENT_STATUS, roadmap, TESTING_PLAN, REPRODUCIBILITY и
   reproduction/AI usage records по фактам. Отдели working commands от будущих
   интерфейсов. Не приписывай собственную проверку команде. Полный task DoD оставь
   PARTIAL, если остаётся обязательный independent human review, и перечисли остаток.

ГРАНИЦЫ

Не реализуй split, normalization, transforms/loaders для training, corruptions,
CNN, training, evaluation, metrics или plotting. Не запускай обучение и не меняй
scientific configs. Split/stats hashes остаются null до DATA-002/003.
Из REPRO-001 добавляй только минимальные необходимые helpers для integrity
manifest и их tests; не закрывай весь REPRO-001 по наличию одного helper.
Не добавляй frameworks, сервисы и экспериментальные результаты.
Source/policy/team uncertainties остаются явно pending и не блокируют независимую
техническую работу. Не объявляй course compliance или submitted proposal verified.

GIT — НОВОЕ ЯВНОЕ ПРАВИЛО ПОЛЬЗОВАТЕЛЯ

Работай, коммить и пушь только в main. Ветки codex/* и другие feature branches
не создавай и не используй. Это правило заменяет старые branch/PR рекомендации
planning docs. Разрешаю небольшие локальные commits и обычный push в origin main.

- Перед изменениями проверь status и remote history; не захватывай чужую работу.
- После каждого небольшого завершённого логического блока делай отдельный
  работающий commit. Реализацию и относящиеся к ней tests коммить вместе.
- Перед каждым commit проверь diff, конкретные staged files, git diff --check
  и relevant checks. Не используй git add . или git add -A.
- После каждого прошедшего проверки commit пушь main в origin/main.
- Не используй force push, amend, reset или переписывание истории. Не создавай
  заведомо неработающие промежуточные commits и не откладывай всё в один commit.
- Если remote изменился, сначала проверь чужие commits. Безопасный fast-forward
  допустим; при несовместимом расхождении не перезаписывай remote и не применяй
  конфликтный merge автоматически: сохрани работу и сообщи точную блокировку.
- Не коммить .venv, CIFAR archives/images, caches, credentials, локальные
  абсолютные пути и временные outputs. Integrity manifest с raw data оставь ignored;
  в Git сохраняй код, tests и компактное документальное evidence без самих данных.

Возможные границы commits: offline loader/integrity checks + tests; download/CLI
и manifest + tests; проверенные инструкции и статусы. Это примеры, не квота:
выбирай границы по завершённости и работоспособности каждого блока.

КРИТЕРИИ ГОТОВНОСТИ

Offline loader и оба CLI режима работают; official CIFAR-10 реально получен и
проверен; default tests offline и проходят; actual integrity check зафиксирован;
повторный запуск не портит данные; foundation tests сохранены. Каждый завершённый
блок закоммичен и запушен в main. Рабочее дерево clean, либо точно перечислены
чужие/неразрешённые изменения. Не имитируй успех при network/source failures.

В конце по-русски сообщи: что реализовано; какие проверки PASS/FAIL/SKIP;
что реально проверено на CIFAR-10; commits и результат push; pending сведения
и следующий конкретный этап DATA-002. Продолжай самостоятельно до завершения
этого объёма; вопросы задавай только при действительно блокирующей неопределённости.
```
