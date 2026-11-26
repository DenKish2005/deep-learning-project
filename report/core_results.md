# Core CIFAR-10 results

On 26 November 2026, Robust CNN achieved **77.55% mean corrupted accuracy**
versus **55.64%** for Clean CNN, a gain of **21.91 percentage points**. On
clean test images, accuracy went from **87.17%** to **86.60%**. The gain on
corrupted images held across all three matched seeds.

## Experiment

Both training regimes used the same 288,746-parameter CompactCNN, persisted
45,000/5,000 train/validation split, train-only normalization, and paired
initialization. Each run covered 40 epochs and 14,080 optimizer updates. Both
regimes used crops and flips. The Robust regime also applied noise, blur, or
darkening to about half of its training images at severities 1–3.

We selected each checkpoint on clean validation accuracy, breaking ties by
lower validation loss and then earlier epoch. Each selected checkpoint was
evaluated on all 10,000 original test images under 16 conditions. The matrix
contains 96 evaluations and 960,000 predictions across repeated views of
the same test set. [Experiment protocol](../docs/EXPERIMENT_PROTOCOL.md)

## Results across seeds

mCA is the unweighted mean accuracy across all 15 corrupted conditions,
calculated for each seed before averaging. Each reported standard deviation is
the sample SD of the three seed results. The robustness gap is clean accuracy
minus mCA. Changes are Robust minus Clean.

| Metric | Clean CNN | Robust CNN | Mean change |
|---|---:|---:|---:|
| Clean test accuracy | 87.17% ± 0.33 pp | 86.60% ± 0.13 pp | −0.57 pp |
| Mean corrupted accuracy (mCA) | 55.64% ± 1.01 pp | 77.55% ± 0.08 pp | +21.91 pp |
| Robustness gap | 31.53 pp | 9.05 pp | −22.49 pp |
| Gaussian noise, mean accuracy | 39.77% | 73.97% | +34.20 pp |
| Gaussian blur, mean accuracy | 44.48% | 73.76% | +29.28 pp |
| Darkening, mean accuracy | 82.66% | 84.93% | +2.27 pp |

| Seed | Clean CNN selected epoch | Robust CNN selected epoch | Change in clean test accuracy | Change in mCA |
|---:|---:|---:|---:|---:|
| 42 | 35 | 40 | −0.43 pp | +21.46 pp |
| 123 | 38 | 40 | −1.08 pp | +21.24 pp |
| 2026 | 34 | 39 | −0.21 pp | +23.04 pp |

Noise caused the largest average drop for the Clean CNN under the chosen
severity scales. The Robust CNN improved on 12 of the 15 corrupted conditions
on average. The three declines were blur severity 1 and darkening severities
1 and 2. The experiment measures a single architecture, three seeds, a fixed
split, and fixed corruption scales. Test severities 4 and 5 extend beyond the
training intensities within the same corruption families.

## Training and test figures

**Figure 02. Training and validation loss.** Each panel follows one seed for
all 40 epochs. Solid lines show training cross-entropy and dashed lines show
clean-validation cross-entropy. [PNG](../outputs/figures/loss_curves.png) · [PDF](../outputs/figures/loss_curves.pdf)

**Figure 03. Training and validation accuracy.** Diamonds mark each manifest’s
selected checkpoint on its clean-validation curve. Robust CNN training
accuracy is measured on augmented inputs. [PNG](../outputs/figures/accuracy_curves.png) · [PDF](../outputs/figures/accuracy_curves.pdf)

**Figure 04. Clean CNN accuracy by corruption severity.** Each panel starts at
the same clean-test reference. Points show the three-seed mean and shading
spans one sample SD on either side. Mean accuracy rises slightly from severity
0 to 1 for both blur and darkening. [PNG](../outputs/figures/baseline_severity.png) · [PDF](../outputs/figures/baseline_severity.pdf)

**Figure 05. Clean and Robust CNN accuracy by corruption severity.** Each model
starts at its own clean-test mean. Shading shows ±1 sample SD across seeds at
each level. Severity levels use a separate intensity scale for each family.
[PNG](../outputs/figures/clean_vs_robust.png) · [PDF](../outputs/figures/clean_vs_robust.pdf)

**Figure 06. Clean-test accuracy versus mean corrupted accuracy.** Each
seed marker represents one checkpoint. Its mCA gives equal weight to all 15
corrupted conditions. Stars mark the three-seed means on both axes. All three
paired seeds gained mCA and lost clean-test accuracy under robust training.
[PNG](../outputs/figures/clean_vs_mca.png) · [PDF](../outputs/figures/clean_vs_mca.pdf)

Regenerate Figures 02–06 from the repository root with
`python -m scripts.plot_core_figures`. The [figure manifest](../outputs/figures/figure_manifest.json)
records source and output hashes.

## Evidence and review

- [Per-class findings, source tables, and figures](core_v1/per_class_analysis/findings.md)
  compare every class across seeds 42, 123, and 2026. Robust CNN improves mean
  corrupted recall in all 30 class–seed pairs, with the largest mean gain for
  automobile (+34.15 pp) and the smallest for bird (+2.73 pp). The report also
  records severity-specific regressions and recurring misclassifications.
  Figure 07 shows recall drops with sample SD; Figure 08 compares seed-42
  clean/noise-5/blur-5/darkening-5 matrices for both models.
- [Six checkpoints and training records](../outputs/checkpoints/core-v1/README.md)
  include the original histories and SHA-256 manifests.
- [Full 96-row matrix](core_v1/raw_results.csv) includes per-class recall,
  confusion counts, checkpoint hashes, and prediction hashes. Prediction paths
  are relative to the repository root and refer to the original local exports.
  The original NPZ files are not included in this checkout; the per-class
  analysis reconstructs metrics from the saved confusion counts.
- [Condition summary](core_v1/tables/condition_summary.csv),
  [model summary](core_v1/tables/model_summary.csv),
  [paired deltas](core_v1/tables/paired_deltas.csv), and
  [per-class summary](core_v1/tables/per_class_summary.csv) support the reported
  comparisons. The full [table set](core_v1/tables/) can be regenerated with
  `python -m src.metrics --input report/core_v1/raw_results.csv --output-dir outputs/metrics/aggregate/rebuilt`.
- [Validation curves](core_v1/figures/core-training-validation.png) and
  [test accuracy by severity](core_v1/figures/core-test-results.png) are
  preliminary summaries. The per-class and confusion-matrix figures are now
  available in the linked analysis; independent report review remains open.

All six runs used Linux, Python 3.14.4, PyTorch 2.9.0+cu126, and an RTX 4060 Ti.
Training took about 38 minutes. All 96 prediction files were checked against
their recorded SHA-256 values and the classification metrics were rebuilt from
the saved logits. Independent reproduction and team review remain open. The
repository has no recorded signed pre-test `TEST-002` approval for this run.
[Verification record](reproduction.md)
