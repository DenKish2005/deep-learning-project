# Следующий шаг — анализ ошибок и результаты для отчёта

Все шесть core runs и 96 test evaluations завершены. Начните с
[результатов](../report/core_results.md), [протокола](EXPERIMENT_PROTOCOL.md)
и [полной таблицы](../report/core_v1/raw_results.csv).

1. RQ1–RQ2. Сравните severity curves Clean CNN, затем измерьте падение
   recall по каждому классу и разберите матрицы ошибок. Используйте
   заранее выбранный seed 42 для основных confusion plots.
2. RQ3–RQ4. Покажите все три paired deltas mCA и clean accuracy,
   средние и sample SD. Сформулируйте выводы в пределах трёх
   corruption families и фиксированного split.
3. Сохраните подписи, исходные таблицы и ссылки на чекпойнты. Для
   примеров ошибок используйте правило выбора исходных IDs из
   [списка deliverables](FINAL_DELIVERABLES.md#план-figures).
4. Попросите участника команды сверить числа с артефактами, пройти
   [independent replay](IMPLEMENTATION_ROADMAP.md#repro-002--independent-fresh-machine-reproduction)
   и записать статус формальной проверки.

[ANALYSIS-001](IMPLEMENTATION_ROADMAP.md#analysis-001--failure-analysis-и-rq-evidence-map)
