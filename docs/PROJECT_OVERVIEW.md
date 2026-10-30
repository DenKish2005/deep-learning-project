# Описание проекта

Дата планирования: **30 октября 2026**, Asia/Qyzylorda. Статус: план до реализации, экспериментальных результатов нет.

CSCI 494 — Deep Learning, Nazarbayev University, Fall 2026. Тема: **Robustness of Convolutional Neural Networks to Common Image Corruptions**. Команда: шесть студентов. Основной документ: [MASTER_PROJECT_PLAN.md](MASTER_PROJECT_PLAN.md).

## Цель и границы

Провести контролируемое воспроизводимое сравнение компактной CNN, обученной на clean CIFAR-10, и той же CNN с corruption-aware augmentation. Измерить, как Gaussian noise, Gaussian blur и progressive darkening влияют на классификацию, и помогает ли augmentation при сохранении clean accuracy.

Вклад — качество экспериментального сравнения и анализа. Новая архитектура, SOTA и положительное улучшение не являются условиями успеха. Отрицательный результат допустим при корректной методике.

На текущем этапе требуется только аудит и полный план. Реализация, установка окружения, скачивание данных, обучение и оценка ещё не выполняются. Существующий код не заменяется; Git commits и push не выполняются.

## Исследовательские вопросы

Исходные RQ1–RQ4 из задания сохраняются по смыслу. Ниже формулировки уточняют измерения и границы выводов:

| ID | Уточнённый вопрос | Измерение |
|---|---|---|
| RQ1 | Как меняется test accuracy Clean CNN на каждом из пяти заранее заданных уровней noise, blur и darkening? | Accuracy по 15 условиям, drop относительно clean, mean ± sample SD по трём seeds |
| RQ2 | Какое из трёх семейств при выбранных шкалах имеет наибольший средний accuracy drop и какие классы наиболее чувствительны? | Среднее по пяти severity, per-class drop, confusion matrices |
| RQ3 | Насколько corruption-aware augmentation меняет mCA при фиксированных архитектуре, split, optimizer и числе updates? | Парные разницы Robust − Clean для seeds 42, 123, 2026 |
| RQ4 | Как связаны изменение clean accuracy и изменение mCA между двумя режимами обучения? | Δclean, ΔmCA и robustness gap; заранее выбранный ориентир clean loss ≤2 percentage points |

Порог 2 pp — проектный ориентир для интерпретации «без существенной потери», не официальный норматив и не статистический тест. Нужно показать фактические разницы для всех seeds. RQ2 относится только к выбранным шкалам: severity 3 noise и severity 3 blur не означают равную perceptual difficulty. Для RQ4 двух точек недостаточно, чтобы установить универсальную Pareto frontier.

## Гипотезы

- H1: accuracy в целом убывает с severity; строгая монотонность каждой точки не гарантирована.
- H2: noise и blur могут сильнее ухудшать accuracy, чем умеренное darkening.
- H3: augmentation может повысить mCA и уменьшить degradation.
- H4: прирост robustness может сопровождаться потерей clean accuracy.

Это ожидаемые направления, а не результаты. В отчёте отдельно фиксируются наблюдения, интерпретации и неразрешённая неопределённость.

## Минимальный полный эксперимент

1. Оригинальный CIFAR-10: 50 000 train и 10 000 test. Из исходного train сохранить один stratified split 45 000/5 000, общий для всех запусков.
2. Одна CNN с шестью Conv, BatchNorm, ReLU, тремя MaxPool, global average pooling и Linear 128→10.
3. Clean CNN: crop + flip. Robust CNN: те же преобразования плюс одна corruption per image с общей вероятностью 0.5.
4. По три training runs для каждой конфигурации: seeds 42, 123, 2026. Ровно 40 epochs без early stopping; одинаковый бюджет updates.
5. Каждый validation-selected checkpoint оценить на clean и 15 corrupted conditions: **6 training runs, 96 model-condition evaluations, 960 000 image predictions**.
6. Сохранить raw predictions, metrics, manifests, checkpoints и сформировать figures/tables программно.
7. Подготовить report, slides, rehearsal и воспроизводимый release.

Числа выше — планируемые объёмы. Ни один запуск ещё не завершён. Smoke/pilot runs отдельно маркируются и в core matrix не включаются.

## Источники и их статус

- Репозиторий на commit `5ce6b1a` — проверенный источник текущего состояния.
- Пользовательский MASTER PROMPT, приложенный к этому чату, — источник проектного scope и предварительных сведений о курсе. Он не заменяет course guidelines или подписанный proposal.
- `report/proposal.md` пуст; оригинальных guidelines, rubric, LMS calendar и подтверждённой AI policy в репозитории нет.
- Предварительные даты: proposal 20.09.2026 уже прошёл; presentation Week 14 без календарной даты; report 12.12.2026 без времени и timezone. Проверка назначена SETUP-001.

Факты о размере и составе CIFAR-10 проверены по [странице авторов](https://cave.cs.toronto.edu/kriz/cifar.html). Методика и технические источники приведены в [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) и [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Что будет считаться готовым проектом

Шесть проверяемых checkpoints, 96 полных оценок по 10 000 images, анализ uncertainty и всех RQ, восемь типов figures, шесть типов tables, report, slides и независимый reproduction smoke run. Все claims связаны с raw artifacts. Подробные pass/fail критерии: [FINAL_DELIVERABLES.md](FINAL_DELIVERABLES.md).

За пределами core scope: frontend, production deployment, adversarial training, архитектурный поиск, новые datasets, облачная инфраструктура и обязательное превосходство над опубликованными моделями. CIFAR-10-C и calibration — только OPTIONAL после core milestones.
