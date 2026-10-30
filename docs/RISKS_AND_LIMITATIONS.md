# Риски, ограничения и assumptions

Вероятности качественные: Low/Medium/High; impact — Low/Medium/High/Critical. Это экспертные planning estimates, не измеренная частота событий. Owners Person 1..6 из TEAM_TASKS. Все даты предварительно сверяются с S1.

## Risk register

| ID / риск | Likelihood / impact | Early warning | Prevention | Mitigation | Owner | Fallback |
|---|---|---|---|---|---|---|
| R01 Недостаточно GPU | High / High | До 08.11 нет подтверждённого GPU slot/VRAM | SETUP-002 inventory, pilot runtime, заранее определить machine и доступ | Один GPU sequential, persistent checkpoints, использовать проверенный Colab/университетский GPU при доступности | 6,3 | CPU только после замера; симметрично сокращённый budget до test с новой protocol version |
| R02 Training медленнее графика | Medium / High | Pilot estimate +30% превышает окно 16–22.11 | Измерить t_clean/t_robust и CPU augmentation/I/O | Исправить bottleneck без изменения научного treatment; zero workers reference→tested workers при необходимости | 3,4 | Убрать OPTIONAL, сдвинуть writing параллельно; раскрыть partial delivery при невозможности core |
| R03 PyTorch/CUDA mismatch | Medium / High | Import torchvision ops fail, cuda unavailable, driver error | Согласованная version pair, fresh env, CPU/GPU smoke, exact recipe | Исправить install/driver совместимость, не менять package во время core runs | 6 | Verified CPU backend; per-run environment limits disclosed |
| R04 Download/integrity CIFAR | Medium / Medium | Timeout, missing archive, checksum mismatch | Official source, verified archive, download до implementation gate | Retry/resume через поддерживаемый loader, проверенная локальная копия | 2 | Shared archive с тем же checksum; не другой dataset |
| R05 Разные локальные окружения | High / High | Разные package versions, outputs расхождены | One lock/install recipe, same backend для paired runs | Повторить smoke, сравнить manifests, reinstall clean env | 6 | Единая reference machine, остальные делают review/analysis |
| R06 Невоспроизводимый RNG | Medium / Critical | Same index corruption differs, resume mismatch | Stable per-image streams, saved split, deterministic backend, unit tests | Изолировать input RNG от inference nondeterminism; исправить и пересчитать затронутые результаты | 2,6 | CPU reference, no bitwise cross-platform claims |
| R07 Неверные corruptions | Medium / Critical | Brightening вместо darkening, неверные kernels, extreme artifacts | Exact mapping, numeric identity/range tests, visual grid | Block evaluation, fix versioned implementation, rerun all affected conditions/runs | 2 | Core scope тот же; нельзя заменить severity post hoc без disclosure |
| R08 Data/test leakage | Medium / Critical | Test loader в train, stats включают val/test, policy меняется после test | Train-only stats, code review, clean-val selection, freeze audit | Остановить claims; восстановить честный history и определить contaminated decisions | 1,5 | Если test уже повлиял на tuning, original comparison считать exploratory; консультироваться о новом holdout, не объявлять чистым rerun того же test |
| R09 Robust underfitting | Medium / High | Low train/val, plateau, policy frequency wrong | Overfit64, pilot, probabilities tests, same budget | Проверить preprocessing/LR/labels и treatment; диагностировать без test tuning | 4,3 | Сохранить отрицательный результат; exploratory study отдельно |
| R10 Малое/отрицательное улучшение | Medium / Medium | Все paired ΔmCA около 0 или <0 | Не обещать H3, primary estimand заранее | Публиковать all3 seeds, clean trade-off, family/class results и uncertainty | 4,5 | Полноценный report с negative finding; не менять p ради выигрыша |
| R11 Merge conflicts/config drift | Medium / High | Две копии trainer, divergent YAML settings | Module owners, small PRs, common-field validator | Owner resolves conflicts, reruns relevant tests, pair config review | 1 | Sequential merges, main stable, frozen source for runs |
| R12 Неравная нагрузка/участник недоступен | Medium / High | Tasks без updates,6 перегружен plotting+QA | Explicit owners, weekly availability, chapter coauthors, cross-review | Передать writing/verification Persons3/5, pair review для bus factor | 1 | Minimum scope, backup owner на каждом critical module |
| R13 Потеря checkpoints | Medium / Critical | Только ephemeral Colab/local copy, hash list отсутствует | Atomic best/last, external backup + retrieval test после run | Restore verified copy; epoch resume по last | 6,3/4 | Retrain только если budget позволяет; missing run явно disclosed |
| R14 Plotting/aggregation bug | Medium / High | Поменяны axes, SD=population, missing rows silently averaged | Hand metrics fixtures, n/cardinality asserts, figure provenance | Rebuild tables/figures from frozen raw logits; сверить sample numbers | 5,6 | Простые checked tables вместо сломанной декоративной визуализации; required plots восстановить до full DoD |
| R15 Report writing поздно | High / High | До 15.11 нет Methods draft, до 29.11 нет full draft | REPORT-001 starts early, chapter owners, no dependence on positive result | Writing sprint по реальным artifacts; cut OPTIONAL | 1,4 | Complete core report с ясными limitations, без extra experiments |
| R16 Presentation preparation fail | Medium / High | Неизвестна дата/slot, нет notes, speakers не знают code | SETUP-001, two full rehearsals, all-six pipeline review | Early draft, backup speakers, offline PDF, timed Q&A | 4,1 | PDF talk на другом device; recorded/precomputed inference demo если допускается |
| R17 Course/AI policy conflict | Unknown / High | Новая rubric требует другой scope/ограничивает AI assistance | Получить S1/S2, отмечать provisional facts, log assistance | Adapt subsequent implementation/writing, explain contributions, use required disclosure | 1 | Самостоятельная реализация командой с разрешённой помощью; planning не выдавать за автоматически разрешённый code |
| R18 Deadline/timezone/format error | Medium / Critical | «Week 14» без даты, deadline без cutoff, oversized upload | Verify official calendar/format, internal09.12 deadline | Early upload, second-person reopen, keep receipt | 1,6 | Reserve10–12.12 for исправление; без предположения о late submission permission |
| R19 Optional benchmark съедает бюджет | Medium / Medium | Core report ещё pending, скачивается 2.9GB benchmark | OPTIONAL gate после core+repro, separate owner/time cap | Остановить extension, сохранить core artifacts | 4 | SKIPPED, честно описанный future work |

## Если Robust CNN не превосходит Clean CNN

1. Проверить implementation, severity mapping, actual policy frequencies, paired init/order/split/stats/budget, checkpoint selection и common corrupted input hashes. Это QA по корректности, а не поиск выгодного seed.
2. Рассчитать и показать все три paired ΔmCA, Δclean, family-level changes, sample SD и full severity curves. Возможно часть conditions выигрывает, часть проигрывает; это требует полного изображения, а не selection.
3. Проверить train/val curves: Robust могла underfit в фиксированном budget. Наблюдение не доказывает, что метод augmentation в принципе бесполезен.
4. Сформулировать вывод ограниченно: «при данной архитектуре, policy, budget, split и выбранных шкалах улучшение не обнаружено / observed mean отрицательный». Не утверждать отсутствие эффекта для всех CNN.
5. Если остаётся время, отдельный **exploratory** validation-only protocol может исследовать policy/budget, но исходный core result не удаляется и не заменяется красивым outcome. После просмотра final test нельзя объявить последующий выбор p независимым confirmatory experiment.
6. Продолжить report/rehearsals по расписанию. Успех проекта — проверяемый эксперимент, не подтверждение H1–H4.

## Неустранимые этим core design ограничения

- CIFAR-10 32×32 ограничивает детализацию и перенос на high-resolution tasks.
- Synthetic noise/blur/darkening не представляют все real-world degradations и sensors.
- Только 3families и одна architecture; severity scales не calibrated по perceived difficulty.
- Robust model видит те же corruption families при training, levels1–3; levels4–5 проверяют новую intensity, не unseen family.
- Fixed split и fixed noise realization контролируют comparison, но не оценивают variability от split или random test-noise draws.
- Три seeds дают ограниченную оценку training variability; normality/CI/significance claims ненадёжны без дополнительных предпосылок.
- Clean-val checkpoint selection задаёт конкретный trade-off и может выбирать не максимальную robustness; это deliberate shared policy.
- Equal updates не равно equal wall time; augmentation computational overhead надо отдельно измерить.
- Complete label preservation математически следует из неизменённых targets, но severe corruption может сделать visual evidence недостаточным для человека.
- Bitwise reproducibility разных GPU/versions не гарантируется; reference environment и numerical tolerance ограничивают claims.

## Assumptions register

| ID | Предположение | Безопасность / действие |
|---|---|---|
| AS01 | Project scope S3 отражает намерение команды | Безопасно для planning, не подтверждает submitted proposal; SETUP-001 |
| AS02 | Report deadline12.12.2026, presentation Week 14 | Временное; до сверки не считать authoritative, нужны cutoff/timezone |
| AS03 | Python 3.12 и согласованная torch/vision pair доступны команде | Candidate для env validation, не pass; SETUP-002 |
| AS04 | Шесть участников доступны 5–8active h/week | Требует TEAM-001; график пересчитать по actual availability |
| AS05 | GPU/CPU budget позволяет 6×40 epochs до 22.11 | Неизвестно до measured pilot; compute gate15.11 |
| AS06 | Shared clean validation selection, stats45k, split 494 допустимы | Разумные scientific choices; freeze PROTO-001, не official mandates |
| AS07 | Clean loss≤2pp — полезный practical threshold | Recommendation для интерпретации; не guarantee и не significance |
| AS08 | 12–14min talk,14 slides | Только provisional layout; verify slot/language/demo requirements |
| AS09 | Dataset и artifact storage доступны всем | Verify download, storage capacity, permissions и second-copy retrieval |

Открытые вопросы не блокируют создание этого planning package. Они назначены owners и должны закрываться до соответствующего implementation/submission gate, а не скрываться как факты.
