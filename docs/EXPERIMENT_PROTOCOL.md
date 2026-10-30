# Экспериментальный протокол v1

Дата планирования 30.10.2026. **Proposed freeze:** PROTO-001 до core training. Никаких measured results пока нет. Изменения после freeze записывать с причиной, датой и protocol_version; test set не используется для выбора решений. Конфигурация: [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md).

## 1. Estimand и controlled comparison

Основной outcome: парная разница mean corrupted accuracy Robust − Clean по трём run seeds. Дополнительные: clean accuracy difference, family-wise robustness, per-class failures, clean/mCA trade-off. Model B отличается только corruption policy при training; architecture, initialization per paired seed, split, normalization, geometry, sample order, optimizer, LR schedule, batch size, epochs, validation и selection совпадают.

Budget: 45000 samples, batch 128, drop_last=False → 352 optimizer steps/epoch (последний batch 72), 40 epochs → 14080 steps/run; 6 runs → 84480 steps. Не одинаковое wall-clock time: robust augmentation добавляет overhead. Default AMP выключен. Три training seeds не означают три независимых dataset splits; оценивается training variability на фиксированном split и фиксированном corrupted test.

Primary selection: **максимум clean validation accuracy**, tie lower validation loss, затем earliest epoch. Это единый прозрачный критерий, который не оптимизирует напрямую robustness; ограничение обсуждается в report. Corrupted validation по умолчанию не используется ни для model selection, ни для настройки policy. При будущей validation-only tuning campaign нужен отдельный protocol, одинаковый tuning budget, freeze до любых test runs и честное описание history. Test нельзя превращать обратно в validation после просмотра результатов.

## 2. Dataset и защита от leakage

Использовать torchvision CIFAR10, original Python archive. Источник подтверждает 60k RGB images 32×32, 10 classes, исходные 50k train/10k test; archive MD5 `c58f30108f718f92721af3b95e74349a`. [CIFAR authors](https://cave.cs.toronto.edu/kriz/cifar.html).

1. DATA-001 скачивает с официального source, проверяет archive/file checksums и class mapping. При недоступности network показать ошибку и поддержать заранее скачанный проверенный archive; не заменять dataset без записи.
2. Original training indices 0..49999 разделить **stratified**. Для каждого class k: sorted candidate indices → local `np.random.default_rng(494).permutation` последовательно в class order 0..9; первые 500 — validation, остальные 4500 — train; final indices каждого split сортировать. Сохранённый NPZ, а не повторное исполнение алгоритма, является canonical source после freeze.
3. `split_manifest.json`: schema_version, dataset/source/archive checksum, algorithm/version, seed 494, arrays SHA-256, counts, per-class counts, creation code SHA. Train/val disjoint; union 50000; unique indices. Test — отдельный official partition, не числовое сравнение test index 5 с train index 5.
4. На raw pixel/255 только 45000 train images вычислить channel mean/std float64 (population std), сохранить full precision и train_indices hash в normalization.json. Val/test не участвуют в statistics. Проверить std>0, finite.
5. EDA: counts, labels и примеры train/validation. Не выбирать hard cases/test conditions для настройки pipeline по final test. Image-content duplicates между official partitions, если обнаружатся, документировать; не менять официальный test произвольно.
6. Download/split/stats idempotent: существующие manifests verify, mismatch → stop; нельзя молча создать новый split для очередного seed.

### Преобразования и loaders

Train: raw RGB uint8 → zero padding 4 + RandomCrop(32) + RandomHorizontalFlip(p=.5) → float32 CHW /255 → (только Robust policy) → Normalize `(x−mean)/std`. Geometry можно выполнить через функциональные transforms с локальным generator; это эквивалент заявленного pipeline crop/flip/ToTensor, но RNG явно управляем.

Val/test clean: raw → float32/255 → Normalize. Corrupted eval: raw → float32/255 → corruption → clamp where needed → Normalize. **Никакой pixel-scale noise после normalization.** Не применять crop/flip при eval. Визуализации denormalize и показывают исходные pixel values без auto-contrast.

Training loader shuffle=True через epoch-specific generator, val/test shuffle=False, all drop_last=False. Reference workers=0, persistent_workers=False. Если включить workers, top-level worker_init_fn seeds Python/NumPy from torch.initial_seed, main guards обязательны; worker count сохранить в metadata. Local per-image RNG сохраняет transforms независимо от batching/workers. pin_memory=True только CUDA; transfer non_blocking при pinned memory. CPU fallback полноценный, но runtime сначала измерить.

## 3. Corruptions в pixel space

Input tensor RGB CHW float32 `[0,1]`, размеры 3×32×32; output такой же, labels неизменны. Не мутировать input. Main eval corruption на CPU; model inference может быть CUDA. No JPEG/uint8 round trip после corruption.

| Severity | Gaussian noise σ | Gaussian blur σ | Kernel | Brightness factor |
|---|---:|---:|---:|---:|
| 1 | 0.02 | 0.3 | 3 | 0.90 |
| 2 | 0.05 | 0.7 | 7 | 0.80 |
| 3 | 0.08 | 1.1 | 9 | 0.70 |
| 4 | 0.12 | 1.5 | 11 | 0.55 |
| 5 | 0.18 | 2.0 | 13 | 0.40 |

- Noise: `clip(x + σ*ε,0,1)`, ε~N(0,1) independently per channel/pixel. Empirical σ проверять на pre-clamp noise; clipping меняет observed variance. Для paired severity используем один ε на original_index, масштабируем σ; обеспечивает контролируемую лестницу, но не гарантирует монотонность accuracy.
- Blur: normalized separable/discrete Gaussian, `k=2*ceil(3σ)+1`, reflect padding, одинаковый σ по двум spatial axes, no channel mixing. Все kernels odd и radius≤6<32. На 32×32 σ=2 уже заметно удаляет детали; severity limits фиксируются заранее. Constant image сохраняется в пределах numerical tolerance, normalized kernel sum≈1, impulse symmetric. На CUDA результат может отличаться численно; main corruption CPU фиксирует реализацию.
- Brightness: `x_dark=clip(factor*x,0,1)` — **только progressive darkening**; не brightening и не additive shift. Для factor≤1 clamp обычно ничего не меняет. Семантический target не меняется, но очень повреждённые изображения могут стать трудноразличимыми человеком; это limitation.
- Clean condition имеет severity=0; main three-family API принимает только 1..5. Boundary helpers sigma=0/factor=1 могут поддерживать identity для unit tests; negative sigma, invalid dtype/range, channels или severity → explicit errors.

## 4. RNG и идентичные изображения

Использовать SHA-256 stable seeds, никогда built-in hash. Namespace и поля сохранять в protocol manifest.

| Stream | Seed definition | Что одинаково |
|---|---|---|
| Split | 494; saved NPZ | Все runs |
| Model init и dropout global RNG | run seed 42/123/2026 | Парные Clean/Robust при равном порядке операций |
| Epoch shuffle | stable_seed(run_seed, epoch, 'order') | Парные Clean/Robust |
| Crop/flip | stable_seed(run_seed, epoch, original_index, 'geometry') | Парные модели; разные epochs |
| Robust event/severity/noise | stable_seed(run_seed, epoch, original_index, 'train_corruption') | Не влияет на geometry/global model RNG |
| Test noise ε | stable_seed(100494, 'cifar10-test', original_index, 'gaussian_noise', 'v1') | Все 6 models и severity; seed не содержит model/run seed |

Для test noise level не включён в seed намеренно: общий ε масштабируется разными σ. Не тянуть RNG в порядке inference, иначе batch/worker count меняют images. Nonstochastic blur/darkening не требуют RNG. Hash condition включает dataset identity, original index order, implementation/version, severity params, eval seed, normalization hash; контрольная выборка corrupted tensors сохраняется с SHA для regression.

## 5. Training policy и процедура

Policy — **одно событие на image**, не на batch, не композиция трёх corruptions:

| Event | Probability | Conditional severity |
|---|---:|---|
| Identity (геометрия остаётся) | .50 | 0 |
| Noise | .17 | Uniform {1,2,3} |
| Blur | .17 | Uniform {1,2,3} |
| Darkening | .16 | Uniform {1,2,3} |

Один uniform u выбирает intervals [0,.50), [.50,.67), [.67,.84), [.84,1); severity sample отдельно uniform int 1..3. В frequency test на 100000 selections требовать |observed−p|<.01 и близкую uniform conditional severity. Rare mismatches investigate, не менять вероятности по test accuracy. Не делать exact quota по batch.

1. Проверить locked env, hardware, deterministic settings, config schema, split/stats hashes и отсутствие output collision.
2. Init seed; создать модель; сохранить initial state hash. Сначала weights init, затем optimizer AdamW lr=.001, betas(.9,.999), eps1e-8, weight_decay=.0001 на все params. CE loss. CosineAnnealingLR T_max=40 eta_min=0.
3. Выполнить finite forward/backward sanity. Отдельный debug overfit: 64 train images, no geometry/corruption/dropout, до 300 updates; criterion train accuracy≥.95 и loss сокращается. Это diagnostic с отдельным profile, не core result.
4. Pilot обеих policies на train subset для runtime/memory/shape checks и validation curves. Pilot не даёт доказательств конечной accuracy и не читает test.
5. Full 40 epochs, batch 128, fixed steps. На каждом epoch set_epoch, deterministic sampler, model.train, optimizer steps; weighted train metrics. Затем clean validation model.eval/inference_mode. Scheduler.step после epoch. Сохранять last и при улучшении best атомарно.
6. Early stopping не используем: variable epoch count нарушил бы простой budget control. Best epoch может различаться, training budget остаётся одинаковым.
7. После success проверить 40 history rows и metadata. Best selected только по validation. Final test запускается после freeze и sanity gates, не в каждом epoch.
8. Повторить 6 core runs без cherry-picking. Failed seed чинить по diagnostics, original attempt сохранять; protocol-changing fix потребует пересчёта всех затронутых runs.

Underfit: low train and val, loss plateau; проверить labels/range/LR/gradients до изменения capacity. Overfit: train растёт, val падает; сохранить заранее выбранное best, не подбирать по test. Настройка после freeze — separate exploratory experiment, не подмена core.

## 6. Logging и serialization

### Per epoch: `outputs/runs/<run_id>/history.csv`

Одна строка на complete epoch: `schema_version, experiment_id(run_id), model_name, seed, epoch, train_loss, train_accuracy, val_loss, val_accuracy, learning_rate, epoch_seconds, cumulative_seconds, best_epoch, checkpoint_path, config_hash, git_sha, device, environment_id, num_train, num_val, optimizer_steps`. Loss weighted by actual batch size; accuracy fractions 0..1. Git dirty flag и code snapshot hash хранятся в manifest, чтобы git_sha не скрывал uncommitted changes.

### Per run: manifest/config/environment/checkpoint

`run_manifest.json`: protocol/schema versions, profile, timestamps UTC, seeds, split/stats/dataset hashes, init hash, config hash, code identity, status, total steps, best epoch, artifacts with SHA-256. `environment.json`: Python, packages, torch/vision/CUDA/cuDNN, OS, GPU model/VRAM, determinism flags, worker count, device. Epoch checkpoint: model/optimizer/scheduler state, epoch, best_score/loss/epoch, CPU/CUDA/Python/NumPy RNG states, resolved config and hashes; architecture ID, class mapping, normalization and split IDs.

Resume: только epoch boundary из last.pt, verify architecture/config/data/code policy. Restore RNG + optimizer/scheduler, next epoch correct LR, history append без duplicate epochs. Mid-epoch restart возвращается к предыдущему complete epoch и не считается дополнительной core эпохой. Более новая implementation требует явного compatibility check, не silent reuse.

### Per condition

JSON: `schema_version, run_id, model_name, seed, profile, dataset, corruption, severity, parameter_value, n_samples, n_correct, accuracy, loss, macro_f1, per_class_accuracy, confusion_matrix, checkpoint_sha256, config_hash, condition_hash, eval_seed, split_hash, normalization_hash, git_sha, environment_id, duration_seconds, predictions_path, predictions_sha256, status`.

NPZ: sorted original test indices 0..9999, targets int64, logits float32 [10000,10]. Derived accuracy drop/RR добавляет aggregation после наличия clean row. Missing condition — error, не nanmean. CSV `results.csv` содержит 96 rows с scalar fields + artifact paths. Confusion/per-class arrays остаются JSON/NPZ и export отдельными long tables.

## 7. Метрики и averaging

Пусть `C_ij` — count true=i, pred=j, `N=ΣC`, `a=trace(C)/N` (fraction). Все accuracy в JSON/CSV — fraction; графики могут показывать %. Явные `*_pp` — percentage points.

| Метрика | Формула / program contract |
|---|---|
| Clean accuracy | a_clean на 10000 original test |
| Corrupted accuracy | a_c, s на тех же indices |
| Accuracy drop | `100*(a_clean-a_c,s)` pp; negative допустим |
| Relative robustness | `a_c,s/a_clean`; unitless; clean=0 → null + reason, не infinity |
| mCA | `(1/15)*Σ_cΣ_s a_c,s`, только corrupted, равные веса 3×5 |
| Family mCA | `(1/5)*Σ_s a_c,s` |
| Robustness gap | `100*(a_clean-mCA)` pp |
| Robust training improvement | `100*(mCA_B-mCA_A)` pp для matched seed |
| Clean change | `100*(a_clean_B-a_clean_A)` pp; acceptable loss ориентир ≥−2 pp |
| Per-class accuracy | `C_kk/Σ_j C_kj`, фактически recall; test support=1000/class |
| Macro-F1 | mean по всем 10 classes `2TP/(2TP+FP+FN)`; zero denominator →0; labels фиксированы 0..9 |
| Confusion matrix | 10×10 int counts; row-normalized copy только visualization, не усреднение logits |

F1 per condition рассчитывается из всех predictions данного run. Если нужен средний corrupted macro-F1 — среднее 15 значений, отдельное имя, не путать с F1 pooled images. Aggregation: сначала per seed mCA/gap/Δ, затем mean и **sample SD ddof=1** по трём seeds. Для каждого of 32 model-condition groups n=3. mCA summary имеет 2 model rows. Не считать 15 conditions или 10000 images независимыми training repeats. SD по seeds не является test-population uncertainty.

Paired Δ: для каждого seed B−A, публиковать все три Δ и mean±SD. При n=3 не заявлять statistical significance по наличию положительного mean; уверенные обобщения ограничены. Optional 95% t interval `mean ±4.303*SD/sqrt(3)` явно exploratory, normality assumptions не проверить надёжно; bootstrap над тремя seeds не устраняет малый sample. Testing многих metrics без correction не использовать для confirmatory claims.

Optional ECE: 15 fixed equal-width confidence bins, `(lower,upper]`, первый включает 0; `Σ(n_b/N)*|acc_b-conf_b|`, empty bins contribution 0. Calibration/tuning temperature по test запрещены; этот outcome не нужен для core acceptance.

## 8. Optional CIFAR-10-C

Отдельный benchmark после core: `gaussian_noise`, **defocus_blur** (стандартная blur category), `brightness`, severity 1..5. Defocus blur не Gaussian blur; standardized brightness не наша darkening шкала. Модель та же, training на CIFAR-10-C запрещён. Никакого смешивания custom и benchmark в одной mCA.

Zenodo artifact v1: CIFAR-10-C.tar ~2.9GB, MD5 `56bf5dcef84df0e2308c6dcbcbbd8499`. В каждом category `.npy` severity blocks по 10000; block `[(s−1)*10000:s*10000]`. Проверить shape `[50000,32,32,3]`, dtype, labels.npy length/consistency; при labels length 50000 брать соответствующий block, при 10000 verify mapping и reuse. [Оригинальный dataset record](https://zenodo.org/records/2535967).

Шесть checkpoints ×15 corrupted =90 **additional** evaluations; clean исходного CIFAR-10 уже есть. Отдельные IDs dataset=cifar10c-v1, artifact directory/prefix и summary; optional scope не меняет core 96. Критерий — load integrity, no train access, complete mapping, transparent differences, а не совпадение accuracy с custom.

## 9. Источники и ограничения выводов

- Hendrycks & Dietterich, **Benchmarking Neural Network Robustness to Common Corruptions and Perturbations**, ICLR 2019: проверенная библиографическая запись и основа benchmark context. [Paper](https://arxiv.org/abs/1903.12261). Наш mCA не benchmark mCE и не сравнивается напрямую с published ImageNet-C scores.
- Hendrycks et al., **AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty**, 2020: проверенная запись для related work; текущая простая policy не реализация AugMix. [Paper](https://arxiv.org/abs/1912.02781). Для содержательного literature review Person 1 читает full text до claims о методе/числах.
- Dataset citation: Krizhevsky, **Learning Multiple Layers of Features from Tiny Images**, 2009, technical report linked на [странице dataset](https://cave.cs.toronto.edu/kriz/cifar.html); полные BibTeX metadata и формат citation проверить REPORT-001.

Нельзя заключить: устойчивость ко всем corruptions, real-world deployment, adversarial robustness, превосходство других architectures или unseen corruption families. Training использует те же семьи и severity 1..3; severity 4..5 — held-out intensity, не unseen corruption type. Labels предполагаются preserved, perceptual evidence ограничено 32×32. No guaranteed monotonic curves или positive improvement.

## 10. Полная core experiment matrix

Каждая строка ниже — **одна оценка уже обученного checkpoint**, не отдельное обучение. Все entries сейчас `TODO`. Шесть runs перечисляются явно; один Clean и один Robust на каждый seed.

Обозначения, применяемые к каждой строке без исключений:

- `R` в колонке run — короткое имя; фактический run_id `core-v1-<model>-s<seed>-<config_hash8>`, связывается через run_manifest.
- `M=outputs/metrics`, `P=outputs/predictions`; output `M/R/C.json + P/R/C.npz` однозначно заменяет R на фактический run_id, C на condition ID.
- `K` = полный per-condition JSON schema из §6, включая accuracy/loss/macro_f1/CM/per-class/counts и provenance. Drops/RR вычисляются после clean row.
- `V` = verified best checkpoint; all 10000 unique indices; finite logits; labels alignment; common condition hash; recomputed metrics equal saved; schema/status complete. Эти validation требования относятся к каждой строке.

| # | Model | Seed | Run R | Dataset | Corruption | Severity | Expected files | Metrics | Status | Validation |
|---:|---|---:|---|---|---|---:|---|---|---|---|
| 01 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | clean | 0 | M/clean_cnn-s42/clean-0.json + P/clean_cnn-s42/clean-0.npz | K | TODO | V |
| 02 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | gaussian_noise | 1 | M/clean_cnn-s42/gaussian_noise-1.json + P/clean_cnn-s42/gaussian_noise-1.npz | K | TODO | V |
| 03 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | gaussian_noise | 2 | M/clean_cnn-s42/gaussian_noise-2.json + P/clean_cnn-s42/gaussian_noise-2.npz | K | TODO | V |
| 04 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | gaussian_noise | 3 | M/clean_cnn-s42/gaussian_noise-3.json + P/clean_cnn-s42/gaussian_noise-3.npz | K | TODO | V |
| 05 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | gaussian_noise | 4 | M/clean_cnn-s42/gaussian_noise-4.json + P/clean_cnn-s42/gaussian_noise-4.npz | K | TODO | V |
| 06 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | gaussian_noise | 5 | M/clean_cnn-s42/gaussian_noise-5.json + P/clean_cnn-s42/gaussian_noise-5.npz | K | TODO | V |
| 07 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | gaussian_blur | 1 | M/clean_cnn-s42/gaussian_blur-1.json + P/clean_cnn-s42/gaussian_blur-1.npz | K | TODO | V |
| 08 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | gaussian_blur | 2 | M/clean_cnn-s42/gaussian_blur-2.json + P/clean_cnn-s42/gaussian_blur-2.npz | K | TODO | V |
| 09 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | gaussian_blur | 3 | M/clean_cnn-s42/gaussian_blur-3.json + P/clean_cnn-s42/gaussian_blur-3.npz | K | TODO | V |
| 10 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | gaussian_blur | 4 | M/clean_cnn-s42/gaussian_blur-4.json + P/clean_cnn-s42/gaussian_blur-4.npz | K | TODO | V |
| 11 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | gaussian_blur | 5 | M/clean_cnn-s42/gaussian_blur-5.json + P/clean_cnn-s42/gaussian_blur-5.npz | K | TODO | V |
| 12 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | brightness | 1 | M/clean_cnn-s42/brightness-1.json + P/clean_cnn-s42/brightness-1.npz | K | TODO | V |
| 13 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | brightness | 2 | M/clean_cnn-s42/brightness-2.json + P/clean_cnn-s42/brightness-2.npz | K | TODO | V |
| 14 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | brightness | 3 | M/clean_cnn-s42/brightness-3.json + P/clean_cnn-s42/brightness-3.npz | K | TODO | V |
| 15 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | brightness | 4 | M/clean_cnn-s42/brightness-4.json + P/clean_cnn-s42/brightness-4.npz | K | TODO | V |
| 16 | clean_cnn | 42 | clean_cnn-s42 | CIFAR-10 test | brightness | 5 | M/clean_cnn-s42/brightness-5.json + P/clean_cnn-s42/brightness-5.npz | K | TODO | V |
| 17 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | clean | 0 | M/clean_cnn-s123/clean-0.json + P/clean_cnn-s123/clean-0.npz | K | TODO | V |
| 18 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | gaussian_noise | 1 | M/clean_cnn-s123/gaussian_noise-1.json + P/clean_cnn-s123/gaussian_noise-1.npz | K | TODO | V |
| 19 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | gaussian_noise | 2 | M/clean_cnn-s123/gaussian_noise-2.json + P/clean_cnn-s123/gaussian_noise-2.npz | K | TODO | V |
| 20 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | gaussian_noise | 3 | M/clean_cnn-s123/gaussian_noise-3.json + P/clean_cnn-s123/gaussian_noise-3.npz | K | TODO | V |
| 21 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | gaussian_noise | 4 | M/clean_cnn-s123/gaussian_noise-4.json + P/clean_cnn-s123/gaussian_noise-4.npz | K | TODO | V |
| 22 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | gaussian_noise | 5 | M/clean_cnn-s123/gaussian_noise-5.json + P/clean_cnn-s123/gaussian_noise-5.npz | K | TODO | V |
| 23 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | gaussian_blur | 1 | M/clean_cnn-s123/gaussian_blur-1.json + P/clean_cnn-s123/gaussian_blur-1.npz | K | TODO | V |
| 24 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | gaussian_blur | 2 | M/clean_cnn-s123/gaussian_blur-2.json + P/clean_cnn-s123/gaussian_blur-2.npz | K | TODO | V |
| 25 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | gaussian_blur | 3 | M/clean_cnn-s123/gaussian_blur-3.json + P/clean_cnn-s123/gaussian_blur-3.npz | K | TODO | V |
| 26 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | gaussian_blur | 4 | M/clean_cnn-s123/gaussian_blur-4.json + P/clean_cnn-s123/gaussian_blur-4.npz | K | TODO | V |
| 27 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | gaussian_blur | 5 | M/clean_cnn-s123/gaussian_blur-5.json + P/clean_cnn-s123/gaussian_blur-5.npz | K | TODO | V |
| 28 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | brightness | 1 | M/clean_cnn-s123/brightness-1.json + P/clean_cnn-s123/brightness-1.npz | K | TODO | V |
| 29 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | brightness | 2 | M/clean_cnn-s123/brightness-2.json + P/clean_cnn-s123/brightness-2.npz | K | TODO | V |
| 30 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | brightness | 3 | M/clean_cnn-s123/brightness-3.json + P/clean_cnn-s123/brightness-3.npz | K | TODO | V |
| 31 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | brightness | 4 | M/clean_cnn-s123/brightness-4.json + P/clean_cnn-s123/brightness-4.npz | K | TODO | V |
| 32 | clean_cnn | 123 | clean_cnn-s123 | CIFAR-10 test | brightness | 5 | M/clean_cnn-s123/brightness-5.json + P/clean_cnn-s123/brightness-5.npz | K | TODO | V |
| 33 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | clean | 0 | M/clean_cnn-s2026/clean-0.json + P/clean_cnn-s2026/clean-0.npz | K | TODO | V |
| 34 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | gaussian_noise | 1 | M/clean_cnn-s2026/gaussian_noise-1.json + P/clean_cnn-s2026/gaussian_noise-1.npz | K | TODO | V |
| 35 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | gaussian_noise | 2 | M/clean_cnn-s2026/gaussian_noise-2.json + P/clean_cnn-s2026/gaussian_noise-2.npz | K | TODO | V |
| 36 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | gaussian_noise | 3 | M/clean_cnn-s2026/gaussian_noise-3.json + P/clean_cnn-s2026/gaussian_noise-3.npz | K | TODO | V |
| 37 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | gaussian_noise | 4 | M/clean_cnn-s2026/gaussian_noise-4.json + P/clean_cnn-s2026/gaussian_noise-4.npz | K | TODO | V |
| 38 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | gaussian_noise | 5 | M/clean_cnn-s2026/gaussian_noise-5.json + P/clean_cnn-s2026/gaussian_noise-5.npz | K | TODO | V |
| 39 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | gaussian_blur | 1 | M/clean_cnn-s2026/gaussian_blur-1.json + P/clean_cnn-s2026/gaussian_blur-1.npz | K | TODO | V |
| 40 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | gaussian_blur | 2 | M/clean_cnn-s2026/gaussian_blur-2.json + P/clean_cnn-s2026/gaussian_blur-2.npz | K | TODO | V |
| 41 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | gaussian_blur | 3 | M/clean_cnn-s2026/gaussian_blur-3.json + P/clean_cnn-s2026/gaussian_blur-3.npz | K | TODO | V |
| 42 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | gaussian_blur | 4 | M/clean_cnn-s2026/gaussian_blur-4.json + P/clean_cnn-s2026/gaussian_blur-4.npz | K | TODO | V |
| 43 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | gaussian_blur | 5 | M/clean_cnn-s2026/gaussian_blur-5.json + P/clean_cnn-s2026/gaussian_blur-5.npz | K | TODO | V |
| 44 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | brightness | 1 | M/clean_cnn-s2026/brightness-1.json + P/clean_cnn-s2026/brightness-1.npz | K | TODO | V |
| 45 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | brightness | 2 | M/clean_cnn-s2026/brightness-2.json + P/clean_cnn-s2026/brightness-2.npz | K | TODO | V |
| 46 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | brightness | 3 | M/clean_cnn-s2026/brightness-3.json + P/clean_cnn-s2026/brightness-3.npz | K | TODO | V |
| 47 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | brightness | 4 | M/clean_cnn-s2026/brightness-4.json + P/clean_cnn-s2026/brightness-4.npz | K | TODO | V |
| 48 | clean_cnn | 2026 | clean_cnn-s2026 | CIFAR-10 test | brightness | 5 | M/clean_cnn-s2026/brightness-5.json + P/clean_cnn-s2026/brightness-5.npz | K | TODO | V |
| 49 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | clean | 0 | M/robust_cnn-s42/clean-0.json + P/robust_cnn-s42/clean-0.npz | K | TODO | V |
| 50 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | gaussian_noise | 1 | M/robust_cnn-s42/gaussian_noise-1.json + P/robust_cnn-s42/gaussian_noise-1.npz | K | TODO | V |
| 51 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | gaussian_noise | 2 | M/robust_cnn-s42/gaussian_noise-2.json + P/robust_cnn-s42/gaussian_noise-2.npz | K | TODO | V |
| 52 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | gaussian_noise | 3 | M/robust_cnn-s42/gaussian_noise-3.json + P/robust_cnn-s42/gaussian_noise-3.npz | K | TODO | V |
| 53 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | gaussian_noise | 4 | M/robust_cnn-s42/gaussian_noise-4.json + P/robust_cnn-s42/gaussian_noise-4.npz | K | TODO | V |
| 54 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | gaussian_noise | 5 | M/robust_cnn-s42/gaussian_noise-5.json + P/robust_cnn-s42/gaussian_noise-5.npz | K | TODO | V |
| 55 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | gaussian_blur | 1 | M/robust_cnn-s42/gaussian_blur-1.json + P/robust_cnn-s42/gaussian_blur-1.npz | K | TODO | V |
| 56 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | gaussian_blur | 2 | M/robust_cnn-s42/gaussian_blur-2.json + P/robust_cnn-s42/gaussian_blur-2.npz | K | TODO | V |
| 57 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | gaussian_blur | 3 | M/robust_cnn-s42/gaussian_blur-3.json + P/robust_cnn-s42/gaussian_blur-3.npz | K | TODO | V |
| 58 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | gaussian_blur | 4 | M/robust_cnn-s42/gaussian_blur-4.json + P/robust_cnn-s42/gaussian_blur-4.npz | K | TODO | V |
| 59 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | gaussian_blur | 5 | M/robust_cnn-s42/gaussian_blur-5.json + P/robust_cnn-s42/gaussian_blur-5.npz | K | TODO | V |
| 60 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | brightness | 1 | M/robust_cnn-s42/brightness-1.json + P/robust_cnn-s42/brightness-1.npz | K | TODO | V |
| 61 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | brightness | 2 | M/robust_cnn-s42/brightness-2.json + P/robust_cnn-s42/brightness-2.npz | K | TODO | V |
| 62 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | brightness | 3 | M/robust_cnn-s42/brightness-3.json + P/robust_cnn-s42/brightness-3.npz | K | TODO | V |
| 63 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | brightness | 4 | M/robust_cnn-s42/brightness-4.json + P/robust_cnn-s42/brightness-4.npz | K | TODO | V |
| 64 | robust_cnn | 42 | robust_cnn-s42 | CIFAR-10 test | brightness | 5 | M/robust_cnn-s42/brightness-5.json + P/robust_cnn-s42/brightness-5.npz | K | TODO | V |
| 65 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | clean | 0 | M/robust_cnn-s123/clean-0.json + P/robust_cnn-s123/clean-0.npz | K | TODO | V |
| 66 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | gaussian_noise | 1 | M/robust_cnn-s123/gaussian_noise-1.json + P/robust_cnn-s123/gaussian_noise-1.npz | K | TODO | V |
| 67 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | gaussian_noise | 2 | M/robust_cnn-s123/gaussian_noise-2.json + P/robust_cnn-s123/gaussian_noise-2.npz | K | TODO | V |
| 68 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | gaussian_noise | 3 | M/robust_cnn-s123/gaussian_noise-3.json + P/robust_cnn-s123/gaussian_noise-3.npz | K | TODO | V |
| 69 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | gaussian_noise | 4 | M/robust_cnn-s123/gaussian_noise-4.json + P/robust_cnn-s123/gaussian_noise-4.npz | K | TODO | V |
| 70 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | gaussian_noise | 5 | M/robust_cnn-s123/gaussian_noise-5.json + P/robust_cnn-s123/gaussian_noise-5.npz | K | TODO | V |
| 71 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | gaussian_blur | 1 | M/robust_cnn-s123/gaussian_blur-1.json + P/robust_cnn-s123/gaussian_blur-1.npz | K | TODO | V |
| 72 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | gaussian_blur | 2 | M/robust_cnn-s123/gaussian_blur-2.json + P/robust_cnn-s123/gaussian_blur-2.npz | K | TODO | V |
| 73 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | gaussian_blur | 3 | M/robust_cnn-s123/gaussian_blur-3.json + P/robust_cnn-s123/gaussian_blur-3.npz | K | TODO | V |
| 74 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | gaussian_blur | 4 | M/robust_cnn-s123/gaussian_blur-4.json + P/robust_cnn-s123/gaussian_blur-4.npz | K | TODO | V |
| 75 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | gaussian_blur | 5 | M/robust_cnn-s123/gaussian_blur-5.json + P/robust_cnn-s123/gaussian_blur-5.npz | K | TODO | V |
| 76 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | brightness | 1 | M/robust_cnn-s123/brightness-1.json + P/robust_cnn-s123/brightness-1.npz | K | TODO | V |
| 77 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | brightness | 2 | M/robust_cnn-s123/brightness-2.json + P/robust_cnn-s123/brightness-2.npz | K | TODO | V |
| 78 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | brightness | 3 | M/robust_cnn-s123/brightness-3.json + P/robust_cnn-s123/brightness-3.npz | K | TODO | V |
| 79 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | brightness | 4 | M/robust_cnn-s123/brightness-4.json + P/robust_cnn-s123/brightness-4.npz | K | TODO | V |
| 80 | robust_cnn | 123 | robust_cnn-s123 | CIFAR-10 test | brightness | 5 | M/robust_cnn-s123/brightness-5.json + P/robust_cnn-s123/brightness-5.npz | K | TODO | V |
| 81 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | clean | 0 | M/robust_cnn-s2026/clean-0.json + P/robust_cnn-s2026/clean-0.npz | K | TODO | V |
| 82 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | gaussian_noise | 1 | M/robust_cnn-s2026/gaussian_noise-1.json + P/robust_cnn-s2026/gaussian_noise-1.npz | K | TODO | V |
| 83 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | gaussian_noise | 2 | M/robust_cnn-s2026/gaussian_noise-2.json + P/robust_cnn-s2026/gaussian_noise-2.npz | K | TODO | V |
| 84 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | gaussian_noise | 3 | M/robust_cnn-s2026/gaussian_noise-3.json + P/robust_cnn-s2026/gaussian_noise-3.npz | K | TODO | V |
| 85 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | gaussian_noise | 4 | M/robust_cnn-s2026/gaussian_noise-4.json + P/robust_cnn-s2026/gaussian_noise-4.npz | K | TODO | V |
| 86 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | gaussian_noise | 5 | M/robust_cnn-s2026/gaussian_noise-5.json + P/robust_cnn-s2026/gaussian_noise-5.npz | K | TODO | V |
| 87 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | gaussian_blur | 1 | M/robust_cnn-s2026/gaussian_blur-1.json + P/robust_cnn-s2026/gaussian_blur-1.npz | K | TODO | V |
| 88 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | gaussian_blur | 2 | M/robust_cnn-s2026/gaussian_blur-2.json + P/robust_cnn-s2026/gaussian_blur-2.npz | K | TODO | V |
| 89 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | gaussian_blur | 3 | M/robust_cnn-s2026/gaussian_blur-3.json + P/robust_cnn-s2026/gaussian_blur-3.npz | K | TODO | V |
| 90 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | gaussian_blur | 4 | M/robust_cnn-s2026/gaussian_blur-4.json + P/robust_cnn-s2026/gaussian_blur-4.npz | K | TODO | V |
| 91 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | gaussian_blur | 5 | M/robust_cnn-s2026/gaussian_blur-5.json + P/robust_cnn-s2026/gaussian_blur-5.npz | K | TODO | V |
| 92 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | brightness | 1 | M/robust_cnn-s2026/brightness-1.json + P/robust_cnn-s2026/brightness-1.npz | K | TODO | V |
| 93 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | brightness | 2 | M/robust_cnn-s2026/brightness-2.json + P/robust_cnn-s2026/brightness-2.npz | K | TODO | V |
| 94 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | brightness | 3 | M/robust_cnn-s2026/brightness-3.json + P/robust_cnn-s2026/brightness-3.npz | K | TODO | V |
| 95 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | brightness | 4 | M/robust_cnn-s2026/brightness-4.json + P/robust_cnn-s2026/brightness-4.npz | K | TODO | V |
| 96 | robust_cnn | 2026 | robust_cnn-s2026 | CIFAR-10 test | brightness | 5 | M/robust_cnn-s2026/brightness-5.json + P/robust_cnn-s2026/brightness-5.npz | K | TODO | V |
