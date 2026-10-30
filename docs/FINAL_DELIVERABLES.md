# Итоговые deliverables и acceptance

Это описание будущей готовности, не существующих результатов. Сейчас code, checkpoints, metrics, report и slides отсутствуют. См. [roadmap](IMPLEMENTATION_ROADMAP.md) и [protocol](EXPERIMENT_PROTOCOL.md).

## Source, models и research outputs

- Working modules `src/{data,corruptions,models,train,evaluate,metrics,utils}.py`, CLI scripts согласно architecture, unit/integration/scientific tests и actual CIFAR smoke.
- Шесть `best.pt` и шесть recovery `last.pt` для 2 regimes × 3 seeds, 6 resolved configs, 6 environment/run manifests, 6 histories по 40 epochs, hashes и внешняя backup-копия.
- **96 per-condition JSON + 96 predictions NPZ**, каждый на 10000 indices; results.csv на 96 rows, per_class.csv на 960 rows и confusion_counts.csv на 9600 cells (96×100).
- Clean/corrupted accuracy, accuracy drop, relative robustness, mCA, robustness gap, macro-F1, per-class accuracy, confusion matrices, mean ± sample SD и paired deltas.
- Report source/PDF, slides source/PDF, два rehearsal logs, AI disclosure по policy, independent reproduction log и submission receipt.

SOTA, высокая заранее обещанная accuracy и положительный ΔmCA не являются условиями готовности.

## Канонические tables

Каталог `outputs/metrics/aggregate/`. Numeric artifacts генерируются из raw results, а не редактируются под narrative.

| ID / файл | Source fields и вычисление | Rows / keys | Проверка |
|---|---|---|---|
| TAB 01 architecture_hyperparameters.csv | Architecture/config manifests: layers, shapes, params, optimizer, batch, epochs, policy | Layer table и 2 config profiles | 288746 params, common fields equal |
| TAB 02 results.csv | 96 summaries: counts, accuracy, loss, F1, paths; derived drop/RR через clean row | (model, seed, dataset, corruption, severity), 96 rows | Exact matrix, total 960000 predictions, hashes |
| TAB 03 condition_summary.csv | Group TAB 02 by model/corruption/severity; mean и sample SD для accuracy/F1/drop/RR | 2×16=32 groups, n=3 | ddof=1, unit columns explicit |
| TAB 04 robustness_summary.csv | Per-run clean, mCA, family mCA, gap; model mean/SD отдельно | 6 run rows и 2 model aggregates в отдельных sections/files | mCA: 15 equal weights, clean excluded |
| TAB 05 relative_robustness.csv | a_corrupt/a_clean сначала per seed, затем condition mean/SD | 90 corrupted seed rows и 30 aggregate rows | Zero denominator null + flag; не ratio of means |
| TAB 06 paired_comparison.csv | B−A join by seed: clean Δ, mCA Δ, gap Δ, family Δ | 3 seed rows и 1 summary | Все deltas показаны, pp units |
| AUX per_class.csv, confusion_counts.csv, failures.csv | Indexed predictions → class counts/CM; sample selection ниже | 960 class rows, 9600 CM cells, выбранные failure IDs | Supports, count sums, unique IDs |

## План figures

Output каждого графика: `outputs/figures/<name>.png` с 300 dpi и `.pdf` с vector elements, где возможно. Размер шрифта ≥10 pt при конечном масштабе. Цвета Clean/Robust одинаковы на всех figures. `figure_manifest.json` фиксирует input paths/hashes, script SHA, aggregation, selection criteria и output hashes. Synthetic fixtures допустимы только для tests.

| ID / имя | Input data / поля | Calculation и seeds | Вид, оси и legend | Интерпретация и validation |
|---|---|---|---|---|
| FIG 01 corruption_examples | Raw train IDs: первый index каждого class; corruption table и manifest | 10 images × (clean+15 conditions); fixed ε с eval seed 100494 и train namespace | Grid: columns condition/severity, rows class/index; подписи σ/factor | Sanity corruption; no auto-contrast, правильные ranges, source IDs сохранены |
| FIG 02 loss_curves | history.csv: epoch, train_loss, val_loss, model_name, seed | Все 40 epochs каждого run; panels по seed, без averaging | Lines, x=epoch, y=CE loss; legend train/val + model | Over/underfit; weighted loss, нет missing epochs |
| FIG 03 accuracy_curves | history.csv: epoch, train_accuracy, val_accuracy, best_epoch | Те же panels; accuracy×100; mark selected best | Lines, x=epoch, y=accuracy %; train/val + model | Train metric измерена на augmented inputs; best marker совпадает с manifest |
| FIG 04 baseline_severity | condition_summary: Clean model, corruption, severity, accuracy_mean/sd/n | Mean ± sample SD по 3 seeds; clean point как severity 0 внутри каждого panel | 3 panels, x=severity 0..5, y=accuracy %, family labels, SD ribbon | RQ1; 16 исходных conditions, SD не CI, no forced smoothing |
| FIG 05 clean_vs_robust | TAB 03 для обоих models | По каждой family 2 lines, mean ± sample SD; clean 0 + levels1..5 | 3 panels, x=severity, y=accuracy %, legend Clean/Robust + SD | RQ3; шкалы разных families не эквивалентны; points match table |
| FIG 06 clean_vs_mca | Per-run summary: clean_accuracy, mca, model, seed | 6 seed points, 2 mean points; optional SD error bars по обеим осям | Scatter, x=clean accuracy %, y=mCA %, color=model, marker=seed | RQ4; показывать отрицательные deltas; оси %, разницы pp |
| FIG 07 per_class_drop | per_class.csv: class, model, seed, clean_recall, corrupted_recall, condition | Per seed: clean recall − mean 15 corrupted recalls, ×100; затем mean ± SD 3 | Grouped bars, x=10 classes, y=mean recall drop pp, color=model | RQ2; support1000/class, canonical order, negative drop допустим |
| FIG 08 confusion_comparison | confusion_counts.csv: model, seed 42, condition, true_class, pred_class, count | Предвыбранный seed 42; clean/noise5/blur5/brightness5 ×2 models; row normalization | 2×4 heatmaps, x=predicted, y=true; shared colorbar0..1 | CM sum10000, row support 1000, correct orientation; raw counts доступны |

Дополнительная failure grid: для каждой модели и family level 5 взять первые три original indices, где clean prediction correct, corrupted wrong. Если таких случаев меньше трёх, показать все и подписать count. Отдельно сравнить Clean wrong/Robust correct и наоборот по первым indices. Не выбирать вручную только выигрышные примеры. FIG 08 seed 42 выбран до результатов; matrices остальных seeds остаются в raw artifacts.

## Final report: chapter-by-chapter plan

Official page count, template и citation style неизвестны. Ниже content plan, а не университетский format. Source: `report/final_report/report.md` либо согласованный TeX/Word, output PDF. Person 1 интегрирует главы; каждую проверяет другой участник.

| Chapter / owner | Purpose / RQ | Technical content и sources | Figures/tables / evidence | Ошибки и completion criterion |
|---|---|---|---|---|
| 1. Introduction /1 | Motivation, scope, RQ1–4 | Common corruption robustness, вклад контролируемого сравнения, RQs и hypotheses из PROJECT_OVERVIEW | Overview experiment, verified background citations | Не обещать gain/SOTA; точная задача понятна |
| 2. Related Background /1, review5 | Position study, RQ1/3 | CNN, distribution shift, corruption benchmarks, augmentation; primary papers Hendrycks & Dietterich2019 и AugMix2020 | Verified references.bib, comparison of methods | Наша policy не AugMix; full text прочитан перед substantive claims |
| 3. Dataset /2 | Inputs и reproducibility, все RQ | Source/checksum, class mapping, split 494, train-only stats, transforms | Counts table, FIG 01, split/stats manifests | Нет test statistics; counts45k/5k/10k подтверждены |
| 4. Method /3 с2/4 | Architecture и treatment, RQ3/4 | Six Conv, shapes, params, init, BN/dropout; corruption formulas/kernel/policy | TAB 01, architecture table, FIG 01 | Conv bias/dropout placement однозначны; одна architecture |
| 5. Experimental Setup /3 с 6 | Reproduction/validity, все RQ | Seeds, budget, AdamW/cosine, clean-val selection/ties, RNG, software/hardware, metric formulas | Config table, complete matrix, manifests и QA | Другой участник повторяет без выбора missing defaults; no test tuning |
| 6. Results /5 | Actual measurements, RQ1–4 | Все seeds, accuracy/mCA/F1, drops/gaps, per-class/CM, paired deltas ± SD | TAB 02..06, FIG 02..08, raw artifact IDs | Только реальные числа; все три seeds; values reconstructable |
| 7. Discussion /4 с 5 | Interpretation, RQ1–4 | Family differences, trade-offs, diagnostics, nonmonotonic/negative findings | RQ evidence map, failure grid, paired deltas | Наблюдение отделено от conjecture; no unsupported causal claims |
| 8. Limitations /4 | Границы выводов | 32×32, synthetic vs real, 3 families, 1 architecture, training family overlap, scales, compute, n 3, fixed split/noise | Protocol facts и observed issues | Generalization и uncertainty limits описаны явно |
| 9. Conclusion /1 | Краткие ответы на RQs | Actual direction/magnitude и clean trade-off, negative outcomes допустимы | TAB04/06 и уже показанные results | Нет новых claims или чисел вне Results |
| 10. Appendix /6 со всеми | Audit/reproduction | Commands/configs, hashes, full tables, tests/recovery, contributions и AI disclosure | Manifest, lock, reproduction log, extra matrices | Links доступны; core seeds не пропущены |

Methods пишется до experiments. Results содержит measurements; Discussion — интерпретации. В protocol bibliographic metadata проверены по primary sources; REPORT-001 требует full text review и проверку citation format до содержательного literature review.

Report review: Person 5 сверяет минимум 10 чисел с raw predictions; Person 2 — data/corruptions; Person 3 — training/checkpoints; Person 6 — commands/artifacts; Person 4 — limitations; Person 1 — rubric/citations. PDF открыть на другом устройстве, проверить captions, axes, page breaks и references. PDF и source должны соответствовать одной версии.

## Presentation plan

Предположение для layout: **12–14 минут + Q&A**, не official requirement. SETUP-001 уточняет slot, slide limit, language и обязательность demo. При коротком slot объединить slides2/3,5/6,10/11. Полные tables/configs оставить в backup. Все шесть speakers имеют содержательный блок.

| Slide | Content / visual | Evidence | Speaker / target |
|---|---|---|---|
| 1. Problem/title | Team, motivation, одна corruption illustration | Scope и verified background, без real-world performance claims | Person 1,30s |
| 2. RQs/hypotheses | Четыре RQ, primary outcome, clean trade-off | Frozen protocol, не results | Person 1,45s |
| 3. Dataset/split | CIFAR samples, class balance,45/5/10k | Dataset/split manifest | Person 2,45s |
| 4. Corruption ladder | FIG 01, σ и darkening factors | Exact table, sample IDs | Person 2,60s |
| 5. Custom CNN | Layers/shapes/288746 params | Model tests, TAB 01 | Person 3,50s |
| 6. Controlled training | Same budget, paired seeds, selection | Config/history,40epochs/14080updates | Person 3,60s |
| 7. Robust policy | Per-image probabilities, severity 1–3 | Frequency/RNG isolation tests | Person 4,50s |
| 8. Matrix/metrics | 6runs/96eval, mCA/drop/SD | Matrix, formulas, n 3 | Person 5,50s |
| 9. Main results | Clean/mCA/paired delta table | TAB04/06 с frozen provenance | Person 5,70s |
| 10. Severity curves | FIG 05, все families | TAB 03, SD ribbon не CI | Person 5,60s |
| 11. Trade-off/reproduction | FIG 06 и краткая provenance chain | Per-seed summaries, actual reproduction log | Person 6,60s |
| 12. Failures/classes | FIG07/08 + fixed examples | Indexed predictions/CM | Person 4,60s |
| 13. Limitations |32×32,1architecture, n 3, family overlap | Actual scope и observed issues | Person 6,45s |
| 14. Conclusions | Four RQ answers, actual gain или negative finding | Уже показанные results | Person 1,35s |

Transitions:1→2 «какие данные позволяют измерить эффект»;2→3 «сравниваем одну architecture»;3→4 «меняем train policy»;4→5 «измеряем разницу»;5→6 «clean trade-off и reproducibility»;6→4 «failure cases»; затем 1 — conclusions. Notes должны помогать объяснять доказательства, не заучивать неподтверждённые claims.

Две **полные** rehearsals всех шести: первая — timed run, Q&A, issues/owners log; вторая — после исправлений, повторный timing и offline PDF на другом device. Даты привязать к actual presentation date по PROJECT_SCHEDULE. Каждый готов заменить соседнего speaker.

Backup: slides PDF, local figures, full tables, config/hyperparameters, all seed results и precomputed inference examples. Если course требует demo, использовать frozen checkpoint на few images с offline backup. Live training в presentation не планируется.

### Сложные вопросы на защите

| Вопрос | Правильный ответ должен покрыть |
|---|---|
| Почему CIFAR-10 и custom CNN? | Accessible balanced data, compute budget, controlled comparison;32×32 и one-architecture limits |
| В чём вклад без новой architecture? | Paired reproducible protocol, quantitative analysis, inspectable artifacts; не SOTA |
| Как выбирали best и почему clean validation? | Predeclared common policy, ties, no test; clean-oriented criterion — limitation |
| Почему 40 epochs и 3 seeds? | Shared budget, measured feasibility; training variability и small-n limits, без обещаний accuracy |
| Как доказать одинаковую noise? | Original index + eval seed 100494 + stable hash, CPU generator, model seed excluded, input hashes |
| Почему corruption до Normalize? | σ/factors определены в pixel units; normalization меняет масштаб; regression test |
| Чем mCA отличается от mCE? | Mean15 accuracies против benchmark normalized errors; напрямую не приравниваются |
| Drop20% или 20pp? | Разность fractions ×100 даёт pp; relative decrease — другая формула |
| Почему нет corrupted validation selection? | Общий простой clean criterion; robustness не оптимизируется напрямую; alternative требует отдельного protocol |
| Severity3 разных families эквивалентны? | Perceptual equivalence не calibrated; ranking зависит от chosen scales |
| Robust не выиграла — проект провален? | Negative finding valid при QA и all-seed evidence; hypotheses не guarantees |
| Это unseen-corruption generalization? | Same families видны 1–3;4–5 — новая intensity, не новая family |
| Что означает SD? | Sample SD по 3trainingseeds, не по images/conditions; uncertainty test population не измерена |
| Почему nonmonotonic curve? | Сначала проверить mapping/RNG/counts; finite-sample/model variability возможна, smoothing запрещено |
| Как использовался AI? | Actual disclosure, verified policy, human review и понимание code; permissions не выдумываются |
| Как повторить после Colab reset? | Code/lock/data/split/config и persistent artifact backup, проверка hashes |

## Final Acceptance Checklist

Каждый item получает PASS/FAIL или OPTIONAL SKIPPED, evidence path и reviewer. **Сейчас implementation items NOT RUN.**

- [ ] **AC01 Sources:** original rubric, dates и AI policy verified; unresolved exceptions отмечены, compliance не объявлена confirmed без evidence.
- [ ] **AC02 Data:** checksum,50k/10k, persisted disjoint/exhaustive stratified45k/5k split, train-only stats provenance.
- [ ] **AC03 Corruptions:**15 условий, pixel-space order, range/shape/determinism tests и visual review.
- [ ] **AC04 Model:** custom from-scratch architecture,288746 params, N×10 logits, strict load и BN/dropout tests.
- [ ] **AC05 Control:** same split/stats/init/geometry/order/commonconfig и14080updates/run; только train corruption policy различается.
- [ ] **AC06 Runs:**6 core combinations complete,40historyepochs/run, best по clean validation; no test selection.
- [ ] **AC07 Models:**6 best и 6 last load, metadata и hashes сохранены; backup retrieval verified.
- [ ] **AC08 Evaluations:**96unique keys,10000indices/result, labels aligned, shared condition hashes;960000 predictions.
- [ ] **AC09 Metrics:** reconstruction из logits совпадает; mCA15weights, pp units, RR zero policy, sampleSD ddof 1.
- [ ] **AC10 Seeds:** все 3 paired deltas, n3/group; no dropped seeds, smoke/pilot excluded.
- [ ] **AC11 Tables/figures:**TAB 01..06 и FIG 01..08 из frozen artifacts, readablePNG/PDF, source manifest, no invented/interpolated data.
- [ ] **AC12 Tests:** required unit/integration/scientific suite и actual smoke passed; used backend tested, skips disclosed.
- [ ] **AC13 Reproduction:** другой участник сделал fresh-env install/smoke и frozen clean/noise5 comparison; actual log сохранён.
- [ ] **AC14 Report:**10 chapters, все RQ, honest limitations/negative findings, verified citations, PDF соответствует rubric.
- [ ] **AC15 Presentation:** date/slot verified, all six understand pipeline, two complete rehearsals logged, offline backup opens.
- [ ] **AC16 Release:** README commands work, tree current, artifacts retrievable by team, stable code revision и complete manifest.
- [ ] **AC17 Submission:** correct filenames/formats/size/deadline/timezone, receipt и second-person reopen.
- [ ] **AC18 Optional:** CIFAR-10-C/ECE отдельно verified либо SKIPPED; core не зависит от них.

Неподтверждённая hypothesis, low accuracy или negative ΔmCA сами по себе не FAIL. Missing seed, leakage, разные corrupted inputs или неподтверждаемые числа — FAIL scientific core. Emergency partial scope нельзя считать полным PASS.
