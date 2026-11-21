# Следующий bounded этап — EVAL-001: оценщик одной модели

Состояние на 21.11.2026: METRIC-001/002 технически реализованы в `src/metrics.py`;
62 синтетических теста `tests/test_metrics.py` проходят. Полное принятие метрик
ждёт формального review PROTO-001/SETUP-002 и реальных результатов EVAL-004.
Оценщик, общий trainer, checkpoints и все реальные model evaluations пока не
реализованы. В Git нет CIFAR-10 в `data/raw/` и нет обученных checkpoints.

## Цель

Выполнить EVAL-001 по [roadmap](IMPLEMENTATION_ROADMAP.md): реализовать
single-condition inference и синтетические integration tests, не объявляя real
CIFAR evaluation завершённой. Перед загрузкой checkpoint согласовать точный
payload и provenance с Person 3/4: `TRAIN-002` ещё не создал checkpoint schema.
Для реального прогона нужны проверенные test data от Person 2 и завершённый gate
`TEST-002` от Person 1.

## Перед изменениями

Прочитать `AGENTS.md` если он есть, затем `README.md`, `docs/CURRENT_STATUS.md`,
`docs/EXPERIMENT_PROTOCOL.md`, `docs/TECHNICAL_ARCHITECTURE.md`,
`docs/IMPLEMENTATION_ROADMAP.md` (EVAL-001/002), `docs/TESTING_PLAN.md`,
`docs/REPRODUCIBILITY.md`, configs и соответствующие `src/data.py`,
`src/corruptions.py`, `src/models.py`, `src/metrics.py`.

Уточнить интерфейс TRAIN-002 и используемые loaders до написания checkpoint
loader. Нельзя подставлять случайно инициализированную модель при ошибке загрузки.
Сохранить порядок test original indices 0..9999. Corruption применяется в pixel
space до normalization; одинаковая пара (index, condition) обязана давать
одинаковое изображение для всех моделей и seed. Не использовать test results для
настройки policy или выбора checkpoint.

## Файлы и ожидаемый API

- `src/evaluate.py`: `build_conditions()` возвращает clean-0 и 15 повреждённых
  условий в протокольном порядке; `evaluate_condition(...)` выполняет
  `eval()`/`inference_mode()`, проверяет checkpoint/data/config identity и считает
  классификационные метрики; `save_predictions(...)` сохраняет индексы, targets
  и logits; CLI вызывает эти функции без копирования логики.
- `tests/test_evaluation.py`: TEST-E01 known-logit fixture, reconstruction
  метрик из сохранённых logits и корректная сериализация; TEST-E03 повторный
  запуск, изменение batch size и перестановка входных примеров с восстановлением
  результатов по original index. TEST-E02 matrix checks остаётся на следующем
  orchestration этапе.
- Один JSON summary и NPZ predictions на условие по контракту
  `docs/EXPERIMENT_PROTOCOL.md`. JSON хранит scalar metrics, run/config/checkpoint/
  dataset/condition hashes и output paths; NPZ — int64 indices/targets и float32
  logits. Не складывать массивы confusion/per-class внутрь итогового CSV.

## Проверки и границы

Начать с synthetic fixture и checkpoint test double; сначала пройти новые тесты,
затем совместимые offline model/corruption/metrics tests. Отмечать skips и
ограничения версии окружения. Реальный CIFAR smoke, все 16 условий и core test
evaluations запускаются только после готовности данных/checkpoints и прохождения
научного gate; пропущенные строки нельзя выдавать за результаты.

После реализации согласовать EVAL-001 с Person 3/4 и Person 1. Отдельно обновить
статус и точную границу выполненного DoD. Технический synthetic PASS не означает
real inference, 96 evaluations или полного acceptance.

## Работа с Git

Продолжать только в существующей `main`, небольшими проверяемыми блоками. Не
создавать ветки и PR. Не делать force push, amend/reset или перезапись чужих
изменений. Перед публикацией проверить diff и связанные тесты; обновлять GitHub
только с точными содержимым файлов и после проверки SHA. Не включать raw data,
checkpoints, окружение, caches и временные outputs в Git.
