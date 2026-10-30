# План тестирования и QA

Статус 30.10.2026 после foundation: TEST-U01 реализован в tests/test_utils.py;
полная текущая suite **116 passed**, 0failed/0skipped. Проверены schema/types/ranges,
required/unknown/duplicate keys, probabilities, exact severity maps, controlled
settings, invalid CLI и config loading без dataset artifacts. Environment smoke
CPU/CUDA/Pillow/torchvision проходит отдельно: [evidence](../report/reproduction.md).
Остальные test cases ниже — будущие проверки ML implementation; пустые
test_data_pipeline/test_corruptions не считаются покрытием. ID TEST-* — cases,
а TEST-001..003 в roadmap — tasks, которые их интегрируют.

## Организация fixtures

`tests/conftest.py`: seeded float32 RGB ramps, checkerboard, constant .5 image, impulses, tiny labeled dataset с известными original indices, hand metrics fixtures, temporary run directory, tiny valid config. Unit tests offline, downloads запрещены по умолчанию; actual CIFAR integration требует explicit opt-in `RUN_CIFAR_INTEGRATION=1` и verified local dataset. CUDA tests skip с причиной при отсутствии CUDA; skips не выдавать за GPU pass. Freeze versions и reference CPU перед comparisons.

## Unit cases: data

| ID / функция теста | Файл и target | Действия / oracle | Acceptance |
|---|---|---|---|
| TEST-D01 `test_load_integrity_and_missing_data` | test_data_pipeline.py → load_cifar10 | Monkeypatch missing file/checksum; actual dataset separate opt-in | Ошибка actionable; real counts 50k/10k, targets 0..9 |
| TEST-D02 `test_stratified_persisted_split` | create_split/load_split | Balanced synthetic 50k labels; generate/load; duplicate/tamper manifest | Train 45k, val5k, 4500/500 per class, disjoint union, saved hashes equal, tamper rejected |
| TEST-D03 `test_stats_train_only` | compute_normalization | Val/test outlier values, known train mean/std; change val values | Stats unchanged; numeric oracle tolerance1e-7, positive finite std |
| TEST-D04 `test_transform_order_and_eval_determinism` | datasets | Ramp image, known darkening; compare normalize(f*x) vs f*normalize(x); repeat eval | First formula matches; second differs; val/test repeat equal, no geometry |
| TEST-D05 `test_loader_indices_and_seeds` | make_loaders | Epoch 0/1, seed match, batch sizes, workers0/2 where supported | Paired sample order/geometry equal, epochs differ, labels int64, N×3×32×32, no drops |

Train/test overlap check использует dataset partition namespace + original indices. Numerical index equality между разными official partitions не leakage. Optional duplicate content audit документируется отдельно.

## Unit cases: corruptions

| ID / функции | Target / fixture | Acceptance |
|---|---|---|
| TEST-C01 `test_contract_all_conditions` | All 3×5, zero/one/ramp images | same shape/device/float32; [0,1], finite, input unchanged; wrapper targets/indices unchanged |
| TEST-C02 `test_parameter_mapping_and_errors` | severity_params, known table; severity0/6, unknown kind, wrong channels/dtype/range | Exact σ/factor/kernel mapping; invalid input explicit error; clean separate |
| TEST-C03 `test_noise_fixed_by_original_index` | Same image/seed across model seeds, batches and workers | Bitwise equal CPU, common ε across severity pre-clamp; changed index produces changed ε |
| TEST-C04 `test_noise_statistics_and_clipping` | Large constant .5 batch with low σ, preclamp ε | ε mean within .01, std within .02 of1; boundary clip yields correct range; no false demand output std=σ at boundary |
| TEST-C05 `test_blur_kernel_and_identity` | kernel weights, constant, impulse and checkerboard | odd kernels [3,7,9,11,13], sum≈1 atol1e-6, symmetric, constant unchanged atol1e-6, high frequency reduced |
| TEST-C06 `test_brightness_exact` | Known RGB values + factors .9.. .4; helper factor1/0 | allclose factor*x atol1e-7, factor 1 identity, factor 0 black; severity order exact |
| TEST-C07 `test_policy_sampling_and_isolation` | 100000 keyed policy draws, paired geometry/global RNG snapshot | Event frequencies within .01 of p, conditional severity within .02 of1/3; no severity4/5 or multiple events; global model RNG unchanged |
| TEST-C08 `test_device_and_preprocess_order` | CPU; optional CUDA own generator | Correct device/dtype; wrong generator device clear error; corruption before normalization regression fixture |

Visual QA дополнительно: Person 2 экспортирует 10 train images (по одному/class) ×clean+15 conditions, Person 5 проверяет darkening, blur padding/artifacts, clipping и узнаваемость объектов. Visual review не заменяет numeric tests. Ни accuracy, ни every-pixel monotonic noise не обязательны.

## Unit cases: model, config, checkpoint, metrics

| ID / функция | Файл / target | Acceptance |
|---|---|---|
| TEST-M01 `test_model_shapes_and_parameter_count` | test_models.py / CompactCNN | Batch1/4/128 → N×10 logits; stage shapes как architecture; ровно 288746 trainable params |
| TEST-M02 `test_backward_and_modes` | CompactCNN | finite loss/gradients; train BN updates, eval freezes BN/dropout; повтор eval одинаковый |
| TEST-M03 `test_init_and_strict_checkpoint` | CompactCNN/load_checkpoint | Seed повторяет state; paired initial hashes равны; mismatched classes/architecture rejected |
| TEST-U01 `test_config_schema_and_controlled_fields` | test_utils.py | Empty YAML/unknown key/bad p/lr/epochs rejected; common fields equal, allowed policy differs |
| TEST-U02 `test_stable_seeds_and_identity_hash` | utils | Stable across subprocess and hash randomization; resolved scientific change changes config hash, timestamp/path alone не меняет |
| TEST-U03 `test_atomic_checkpoint_roundtrip` | utils | Save/load state and optimizer/RNG; interrupted temp write не уничтожает previous complete checkpoint; map_location CPU |
| TEST-U04 `test_manifest_and_output_collision` | utils | Missing metadata/tampered hash fails; new run refuses overwrite; environment IDs linked |
| TEST-K01 `test_known_classification_metrics` | test_metrics.py | Hand fixture y=[0,0,1,1], pred=[0,1,1,1]; 2-class CM[[1,1],[0,2]], acc=.75, macroF1=(2/3+.8)/2; отдельный 10-class zero-support fixture |
| TEST-K02 `test_robustness_units_and_denominators` | metrics | Artificial a_clean=.8, a_corrupt=.6 →20pp, RR=.75; 15 rows .6 →mCA=.6, gap20pp; clean 0→null RR |
| TEST-K03 `test_seed_aggregation_and_pairing` | metrics | Artificial values [.6,.7,.8] mean.7 sampleSD.1; join by seed; shuffled order same; missing seed rejected |
| TEST-K04 `test_confusion_and_f1_edge_cases` | metrics | Class support 0 handled explicitly, F1 zero_division0; rows sum to supports; sumCM=N; no micro/macro confusion |

Numbers in metric fixtures are synthetic test oracles, **не experimental results**.

## Integration и end-to-end

| ID / тест | Chain | Действия и gate |
|---|---|---|
| TEST-T01 `test_epoch_logging_and_validation` | loader→CNN→optimizer→history | Tiny synthetic dataset,2 epochs; params change, loss/acc finite, correct weighting nonfull last batch; validation no gradients/BN mutation |
| TEST-T02 `test_resume_matches_uninterrupted_cpu` | train→last→resume | 2 epochs uninterrupted vs1+resume; same weights/optimizer/scheduler/history (кроме timing); LR/epoch count correct |
| TEST-T03 `test_training_orchestration` | configs→6 run specs | dry-run 6, seed/config IDs unique; incomplete manifest not treated complete; profile guard |
| TEST-E01 `test_checkpoint_to_evaluation` | checkpoint→loader→metrics→JSON/NPZ | Known tiny logits classifier; n_correct, loss, F1 exact; predictions reload reproduces metrics |
| TEST-E02 `test_complete_matrix` | six manifests→96 conditions | All keys exactly once; missing/duplicate row, mismatched dataset/condition hash or stale checkpoint → failure |
| TEST-E03 `test_eval_idempotent_and_order_independent` | fixed conditions→inference | Repeat checkpoint gives same predictions within declared backend tolerance; batch size change same images; reorder samples restores index-keyed metrics |
| TEST-P01 `test_plot_exports_and_provenance` | synthetic complete metrics→figures | Files PNG/PDF exist, nonempty, correct axes/legends/source IDs; missing row/column/NaN fails, no interpolation of missing result |
| TEST-S01 `test_controlled_comparison` | paired manifests | Same architecture/split/stats/init/order/budget/common config; only policy differs; all test conditions same hashes |
| TEST-S02 `test_no_test_loader_in_training` | train config/module review | Data access trace fixtures show train/val only; code review no test-based selection/tuning |
| TEST-S03 `test_all_seeds_and_profiles` | manifests→aggregate | Exactly core seeds; failed/smoke/pilot excluded with explicit counts; no selecting best seed |
| TEST-END01 `test_smoke_pipeline` | actual CIFAR subset→train→checkpoint→16eval→aggregate→plots | Explicit profile smoke, 128train/64val/100test,1 epoch, one seed for each model,32 condition results each n 100; core validator refuses smoke |

Smoke uses persisted full split to select tiny subsets and frozen train-only normalization; never overwrites core directories. It tests mechanics, not model quality. Synthetic offline integration always available; real smoke opt-in when CIFAR ready. For full training separate debug overfit64 no aug, target ≥95% within300 updates; failure blocks long runs until diagnostic evidence reviewed. Threshold is health gate, not promised CIFAR test accuracy.

## Scientific QA gates

1. Before full runs: SETUP-002/DATA/MODEL/CORR/TRAIN tests pass; official policy reviewed; protocol frozen; init/order geometry identity verified for paired seed; seed 42 pilot both regimes fits budget.
2. Before final test: select best by clean val logs only, all core configs frozen, test params unchanged, complete checkpoint metadata and no test tuning in history.
3. Before plots/report: 96 conditions ×10000 samples, 6 runs×40 epochs, every artifact hash verified; aggregate includes all seeds and correct ddof/units; reconstruction from logits matches accuracy/F1/CM.
4. Before release: second-person installation/smoke, external artifact backup, no scientific validity failures. Optional CUDA skip acceptable only with declared CPU-only execution; used GPU path must have actual smoke evidence.

## Failure analysis decision table

| Симптом | Проверить / evidence | Корректное действие / предел вывода |
|---|---|---|
| Low train и val | Loss, labels, preprocessing range, gradients,64-image overfit, LR | Сначала bugs; capacity/budget change только shared versioned protocol |
| Train high, val deteriorates | Curves per seed, best epoch, dropout/BN modes | Использовать predeclared best; test не выбирать epoch |
| Nonmonotonic severity | Parameter map, fixed noise hashes, count correctness, SD, visual grid | Проверить bug; если всё верно — публиковать, не сглаживать monotonic forcing |
| Robust теряет >2pp clean | Seed-paired clean deltas, policy frequencies, train loss | Сообщить trade-off; не скрывать seed и не менять p по test |
| ΔmCA≤0 | Все seeds/conditions, noise identity, budget, underfit, val | Валидный отрицательный результат; exploratory retrain отдельно, core не заменять |
| Class-specific drop | Supports=1000, per-class confusion, matched index failures | Qualitative sampling по заранее заданному правилу; никаких claims про causal semantics |
| Один seed unstable | NaN/Inf, LR logs, interruption, env/init hashes | Сохранить attempt, исправить cause; unrecorded cherry-pick запрещён |
| Implausibly good corrupted result | Normalization/order/label alignment, test leakage, severity mapping | Блокировать publication до независимой проверки |
| CPU/GPU mismatch | Versions/device algorithms, same CPU corrupted inputs, tolerance | Разделить deterministic input и numeric inference; bitwise cross-platform не обещать |
| Custom/CIFAR-10-C disagree | Different severity, darkening vs benchmark brightness, blur type | Описать разные distributions; не объявлять benchmark ошибочным по одному расхождению |

Artifacts QA: `report/reproduction.md` должен содержать date, code/config hashes, commands, pass/fail/skip, machine, logs path, reviewer. Тестовый отчёт создаётся при реализации, сейчас его нет.
