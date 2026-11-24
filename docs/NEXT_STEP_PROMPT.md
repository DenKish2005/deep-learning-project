# Следующий шаг — EVAL-001: запуск одного core checkpoint

## Текущее состояние

В `src/evaluate.py` реализован production CLI для одного validation-selected
`best.pt`: загрузка чекпойнта, проверка его core/config/dataset identity и
оценка на clean и 15 corruption conditions. На каждое условие сохраняются
JSON summary и NPZ с original IDs, targets и logits.

Реальную оценку пока не запускали. Для неё нужны доступные локально проверенные
CIFAR-10, frozen split/normalization, готовый core checkpoint и пройденный
`TEST-002` scientific gate. Нельзя использовать test results для выбора
чекпойнта или настройки policy.

## Команда

Из корня репозитория:

```bash
python -m src.evaluate --checkpoint outputs/runs/<run_id>/best.pt --config configs/clean_cnn.yaml --output-dir outputs --device auto
```

Для Robust использовать `configs/robust_cnn.yaml` и соответствующий Robust
checkpoint. Допустимо заменить `auto` на `cpu` или `cuda`. Выходы появятся в
`outputs/metrics/<run_id>/` и `outputs/predictions/<run_id>/`; существующие
файлы не перезаписываются, поэтому для повторного прогона нужен новый run ID или
output directory.

## После одиночного прогона

1. Person 3/4 передают шесть завершённых core checkpoints и их run IDs.
2. Person 5 добавляет `scripts/run_all_evaluations.py`, который запускает 16
   условий для каждой пары model/seed (всего 96), обнаруживает пропуски и
   конфликты и собирает проверенный results CSV.
3. Пропустить CSV через `python -m src.metrics --input <results.csv> --output-dir <new-dir>`
   и сохранить итоговые таблицы для отчёта.

METRIC-001/002 уже реализуют расчёт и проверку агрегированных метрик. До появления
всех 96 валидных core строк научные итоги и сравнительные выводы не публиковать.
