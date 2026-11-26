"""Rebuild the core per-class analysis from saved counts; no inference or training.

Run from the repository root: python -m scripts.analyze_per_class
Only named generated exports in --output-dir are replaced on rerun.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from src.metrics import aggregate_seeds, validate_matrix


ROOT = Path(__file__).resolve().parents[1]
MODELS = ('clean_cnn', 'robust_cnn')
SEEDS = (42, 123, 2026)
FAMILIES = ('gaussian_noise', 'gaussian_blur', 'brightness')
LABELS = {'clean_cnn': 'Clean CNN', 'robust_cnn': 'Robust CNN',
          'clean': 'Clean', 'gaussian_noise': 'Gaussian noise',
          'gaussian_blur': 'Gaussian blur', 'brightness': 'Darkening'}
DISPLAY_CONDITIONS = (('clean', 0), *((f, 5) for f in FAMILIES))
KEY = ['model_name', 'seed', 'corruption', 'severity']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_results(path):
    frame = pd.read_csv(path)
    for column in ('confusion_matrix', 'per_class_accuracy'):
        frame[column] = frame[column].map(json.loads)
    return frame


def verify_counts(frame, support):
    """Check the full matrix and independently supplied saved test supports."""
    support = np.asarray(support)
    if support.shape != (10,) or support.dtype.kind not in 'iu' or np.any(support <= 0):
        raise ValueError('saved test class counts must be ten positive integers')
    frame = validate_matrix(frame)
    checks = []
    for row in frame.itertuples(index=False):
        counts = np.asarray(row.confusion_matrix)
        if not np.array_equal(counts.sum(axis=1), support):
            raise ValueError('confusion row sums disagree with saved test class counts')
        recall = counts.diagonal() / support
        np.testing.assert_allclose(recall, row.per_class_accuracy, rtol=0, atol=1e-12)
        np.testing.assert_allclose(counts.trace() / support.sum(), row.accuracy,
                                   rtol=0, atol=1e-12)
        checks.append({**{k: getattr(row, k) for k in KEY},
                       'n_samples': int(counts.sum()), 'n_correct': int(counts.trace()),
                       'min_class_support': int(counts.sum(axis=1).min()),
                       'max_class_support': int(counts.sum(axis=1).max()),
                       'recall_max_abs_error': float(np.max(np.abs(recall - row.per_class_accuracy)))})
    return frame, pd.DataFrame(checks)


def summarize(frame, keys, values):
    grouped = frame.groupby(keys, sort=False)[values]
    means = grouped.mean().add_suffix('_mean')
    sds = grouped.std(ddof=1).add_suffix('_sample_sd')
    return means.join(sds).join(grouped.size().rename('n')).reset_index()


def build_tables(frame, classes):
    class_rows, cells = [], []
    clean = {(r.model_name, r.seed): np.diag(r.confusion_matrix) / np.sum(r.confusion_matrix, axis=1)
             for r in frame[frame.corruption == 'clean'].itertuples(index=False)}
    for row in frame.itertuples(index=False):
        key = {k: getattr(row, k) for k in KEY}
        counts = np.asarray(row.confusion_matrix)
        for i, name in enumerate(classes):
            support = int(counts[i].sum())
            recall = counts[i, i] / support
            reference = clean[row.model_name, row.seed][i]
            class_rows.append({**key, 'class_id': i, 'class_name': name,
                               'support': support, 'correct': int(counts[i, i]),
                               'recall': recall, 'clean_recall': reference,
                               'recall_drop_pp': 100 * (reference - recall)})
            for j, predicted in enumerate(classes):
                cells.append({**key, 'true_class_id': i, 'true_class': name,
                              'pred_class_id': j, 'pred_class': predicted,
                              'count': int(counts[i, j]), 'true_support': support,
                              'rate': counts[i, j] / support})
    per_class = pd.DataFrame(class_rows)
    confusion = pd.DataFrame(cells)
    condition_summary = summarize(per_class,
        ['model_name', 'corruption', 'severity', 'class_id', 'class_name'],
        ['recall', 'recall_drop_pp'])
    keys = ['model_name', 'seed', 'class_id', 'class_name']
    corrupted = per_class[per_class.corruption != 'clean']
    # Average within each training seed before computing uncertainty across seeds.
    run = corrupted.groupby(keys, sort=False)[['clean_recall', 'recall', 'recall_drop_pp']].mean().reset_index()
    run = run.rename(columns={'recall': 'mean_corrupted_recall'})
    summary = summarize(run, ['model_name', 'class_id', 'class_name'],
                        ['clean_recall', 'mean_corrupted_recall', 'recall_drop_pp'])
    family_run = corrupted.groupby([*keys, 'corruption'], sort=False)[
        ['clean_recall', 'recall', 'recall_drop_pp']].mean().reset_index()
    family_summary = summarize(family_run, ['model_name', 'class_id', 'class_name', 'corruption'],
                               ['clean_recall', 'recall', 'recall_drop_pp'])
    paired = run[run.model_name == MODELS[0]].merge(
        run[run.model_name == MODELS[1]], on=['seed', 'class_id', 'class_name'],
        suffixes=('_clean', '_robust'), validate='one_to_one')
    deltas = paired[['seed', 'class_id', 'class_name']].copy()
    for metric in ('clean_recall', 'mean_corrupted_recall'):
        deltas[metric + '_change_pp'] = 100 * (paired[metric + '_robust'] - paired[metric + '_clean'])
    delta_summary = summarize(deltas, ['class_id', 'class_name'],
                              ['clean_recall_change_pp', 'mean_corrupted_recall_change_pp'])
    off = confusion[confusion.true_class_id != confusion.pred_class_id].copy()
    # All off-diagonal counts remain available, including ties and zero counts.
    rank_keys = [*KEY, 'true_class_id']
    top = off.sort_values([*rank_keys, 'count', 'pred_class_id'],
                         ascending=[True] * len(rank_keys) + [False, True])
    top['rank'] = top.groupby(rank_keys).cumcount() + 1
    top = top[top['rank'] <= 3]
    # Mean condition errors preserve each seed, rather than treating views as replicates.
    error_run = off[off.corruption != 'clean'].groupby(
        ['model_name', 'seed', 'true_class_id', 'true_class', 'pred_class_id', 'pred_class'],
        sort=False)[['count', 'rate']].mean().reset_index()
    error_summary = summarize(error_run,
        ['model_name', 'true_class_id', 'true_class', 'pred_class_id', 'pred_class'], ['count', 'rate'])
    return {'per_class': per_class, 'confusion_counts': confusion,
            'class_condition_summary': condition_summary,
            'per_class_run_summary': run, 'per_class_summary': summary,
            'family_class_run_summary': family_run, 'family_class_summary': family_summary,
            'per_class_paired_deltas': deltas, 'per_class_paired_summary': delta_summary,
            'common_misclassifications': top, 'corrupted_confusion_run_summary': error_run,
            'corrupted_confusion_summary': error_summary}


def export_figures(tables, classes, output):
    output.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10}):
        fig = Figure(figsize=(12, 5.8), layout='constrained')
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        for offset, model, color in zip((-.19, .19), MODELS, ('#3569a6', '#cf6d27')):
            rows = tables['per_class_summary'].query('model_name == @model').sort_values('class_id')
            ax.bar(np.arange(10) + offset, rows.recall_drop_pp_mean, width=.36,
                   yerr=rows.recall_drop_pp_sample_sd, capsize=3, color=color, label=LABELS[model])
        ax.set_xticks(np.arange(10), classes, rotation=30, ha='right')
        ax.set_ylabel('Clean recall − mean corrupted recall (pp)')
        ax.set_title('Per-class recall drop across all 15 corrupted conditions\n'
                     'Mean ± sample SD across seeds 42, 123, 2026')
        ax.axhline(0, color='#555555', linewidth=.8)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend()
        for extension in ('png', 'pdf'):
            metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None
            fig.savefig(output / f'per_class_drop.{extension}', dpi=180, metadata=metadata)

        fig = Figure(figsize=(20, 12), layout='constrained')
        fig.get_layout_engine().set(h_pad=.15, w_pad=.06, hspace=.08)
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 4)
        for row, model in enumerate(MODELS):
            for col, (family, severity) in enumerate(DISPLAY_CONDITIONS):
                selected = tables['confusion_counts'].query(
                    'model_name == @model and seed == 42 and corruption == @family and severity == @severity')
                values = selected.pivot(index='true_class_id', columns='pred_class_id', values='rate').to_numpy()
                ax = axes[row, col]
                im = ax.imshow(values, vmin=0, vmax=1, cmap='Blues', interpolation='nearest')
                for i in range(10):
                    for j in range(10):
                        ax.text(j, i, f'{100 * values[i, j]:.1f}', ha='center', va='center',
                                fontsize=6.5, color='white' if values[i, j] > .5 else '#172033')
                condition = LABELS[family] + (f' s{severity}' if severity else '')
                ax.set_title(f'{LABELS[model]} · {condition}', fontsize=12)
                ax.set_xticks(range(10), classes, rotation=55, ha='right', fontsize=8)
                ax.set_yticks(range(10), classes, fontsize=8)
                if row == 1:
                    ax.set_xlabel('Predicted class')
                ax.set_ylabel('True class')
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.8, pad=.015, label='Fraction of true class (0–1)')
        fig.suptitle('Seed 42 confusion matrices · 1,000 images per true class\n'
                     'Cells show percentages; each row sums to 100% before rounding', fontsize=16)
        for extension in ('png', 'pdf'):
            metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None
            fig.savefig(output / f'confusion_seed42.{extension}', dpi=180, metadata=metadata)


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def write_findings(tables, aggregate, classes, output):
    summary = tables['per_class_summary'].set_index(['model_name', 'class_id'])
    deltas = tables['per_class_paired_summary'].set_index('class_id')
    rows = []
    for i, name in enumerate(classes):
        a, b, d = summary.loc[(MODELS[0], i)], summary.loc[(MODELS[1], i)], deltas.loc[i]
        rows.append([name, f'{100*a.clean_recall_mean:.2f} / {100*b.clean_recall_mean:.2f}',
                     f'{100*a.mean_corrupted_recall_mean:.2f} / {100*b.mean_corrupted_recall_mean:.2f}',
                     f'{a.recall_drop_pp_mean:.2f} ± {a.recall_drop_pp_sample_sd:.2f}',
                     f'{b.recall_drop_pp_mean:.2f} ± {b.recall_drop_pp_sample_sd:.2f}',
                     f'{d.mean_corrupted_recall_change_pp_mean:+.2f} ± {d.mean_corrupted_recall_change_pp_sample_sd:.2f}'])
    ordered = deltas.sort_values('mean_corrupted_recall_change_pp_mean', ascending=False)
    improved = int((tables['per_class_paired_deltas'].mean_corrupted_recall_change_pp > 0).sum())
    text = ['# Per-class analysis of the saved core results', '',
            'Generated with `python -m scripts.analyze_per_class` from the repository root. '
            'This analysis uses the 96 saved confusion matrices; it does not retrain or run inference.', '',
            f'Robust CNN improves mean corrupted recall in **{improved}/30 class–seed pairs**. '
            f'The largest average gain is {classes[ordered.index[0]]} '
            f'(+{ordered.iloc[0].mean_corrupted_recall_change_pp_mean:.2f} pp), followed by '
            f'{classes[ordered.index[1]]} (+{ordered.iloc[1].mean_corrupted_recall_change_pp_mean:.2f} pp). '
            f'{classes[ordered.index[-1]].capitalize()} gains least '
            f'(+{ordered.iloc[-1].mean_corrupted_recall_change_pp_mean:.2f} pp). '
            'Individual severe conditions still contain class-level regressions.', '',
            '## Recall across all three seeds', '',
            'Recall is diagonal count / true-class support. The mean corrupted recall averages all '
            '15 conditions within each seed first. Drops are clean minus corrupted recall; paired '
            'changes are Robust minus Clean within the same seed. All ± values are sample SD '
            '(ddof=1) across seeds 42, 123, 2026, not confidence intervals.', '',
            markdown_table(['Class', 'Clean recall % (Clean / Robust)',
                            'Corrupted recall % (Clean / Robust)', 'Clean drop pp',
                            'Robust drop pp', 'Paired corrupted gain pp'], rows), '',
            '![Per-class recall drops](figures/per_class_drop.png)', '',
            '**Figure 07.** Each bar is the seed mean of clean recall minus mean recall over '
            '15 equally weighted corrupted conditions. Error bars show sample SD over three '
            'seeds. Class order follows the saved dataset manifest. '
            '[PDF](figures/per_class_drop.pdf); [per-seed source](tables/per_class_run_summary.csv).', '',
            'All 960 class/condition/seed rows, including clean references and recall drops, '
            'are in [per_class.csv](tables/per_class.csv). '
            '[Family summaries](tables/family_class_summary.csv) average the five severities '
            'within each seed; [paired class changes](tables/per_class_paired_deltas.csv) '
            'retain all three seeds. '
            '[Class-by-condition means and SDs](tables/class_condition_summary.csv) '
            'retain severity-specific effects.', '',
            'A small recall drop does not imply high recall: Robust cat recall starts lower '
            'on clean images and remains among its lowest corrupted recalls. Likewise, '
            'high recall for a class attracting many false positives can mask poor discrimination; '
            'the off-diagonal counts below expose that behavior.', '',
            '## Common misclassifications', '',
            'The following are the three largest directed off-diagonal cells for each model '
            'and displayed condition, ranked by their mean count across all three seeds. '
            'Counts are out of 1,000 true-class images in each seed; they are not percentages '
            'of errors. Ties use ascending true-class ID, then predicted-class ID.', '']
    confusion = tables['confusion_counts']
    error_rows = []
    for model in MODELS:
        for family, severity in DISPLAY_CONDITIONS:
            selected = confusion.query('model_name == @model and corruption == @family and severity == @severity and true_class_id != pred_class_id')
            means = selected.groupby(['true_class_id', 'pred_class_id'])['count'].mean().reset_index()
            leaders = means.sort_values(['count', 'true_class_id', 'pred_class_id'], ascending=[False, True, True]).head(3)
            for leader in leaders.itertuples(index=False):
                counts = selected.query('true_class_id == @leader.true_class_id and pred_class_id == @leader.pred_class_id').set_index('seed')['count']
                error_rows.append([LABELS[model], LABELS[family] + f' s{severity}',
                                   f'{classes[leader.true_class_id]} → {classes[leader.pred_class_id]}',
                                   *[int(counts[s]) for s in SEEDS], f'{leader.count:.2f}'])
    text += [markdown_table(['Model', 'Condition', 'True → predicted', '42', '123', '2026', 'Mean count'], error_rows), '',
             'Clean-image errors prominently confuse cats and dogs in both models. '
             'At noise s5, Clean CNN sends many other classes to frog; at blur s5, '
             'many errors go to bird. These directions recur across all three seeds, '
             'although their counts vary substantially. Robust CNN reduces these dominant '
             'patterns but retains substantial errors, including ship → airplane under '
             'noise and dog → cat under blur and darkening. This describes observed '
             'predictions; it does not identify a causal feature mechanism.', '',
             'For every true class, model, seed, and condition, '
             '[common_misclassifications.csv](tables/common_misclassifications.csv) retains the '
             'top three destinations (ties by predicted-class ID). '
             '[All 9,600 cells](tables/confusion_counts.csv) preserve the full matrices. '
             '[Corrupted confusion summaries](tables/corrupted_confusion_summary.csv) average '
             '15 conditions within each seed, then report mean and sample SD over seeds.', '',
             '## Seed-42 confusion matrices', '',
             '![Seed-42 confusion matrices](figures/confusion_seed42.png)', '',
             '**Figure 08.** The preselected seed 42, with Clean CNN on the top row and '
             'Robust CNN below; columns are clean, Gaussian noise s5, Gaussian blur s5, '
             'and darkening s5. Rows are true labels and columns predicted labels. '
             'All panels share the 0–1 color scale; annotations are percentages. '
             'Every row has 1,000 images and every panel has 10,000. '
             '[PDF](figures/confusion_seed42.pdf).', '',
             '## Paired overall comparison', '']
    paired = aggregate['paired_deltas']
    text += [markdown_table(['Seed', 'Clean accuracy change pp', 'mCA change pp'],
                            [[int(r.seed), f'{r.clean_accuracy_change_pp:+.2f}', f'{r.mca_improvement_pp:+.2f}']
                             for r in paired.itertuples(index=False)]), '']
    p = aggregate['paired_summary'].iloc[0]
    text += [f'Mean paired clean change: **{p.clean_accuracy_change_pp_mean:+.2f} ± '
             f'{p.clean_accuracy_change_pp_sample_sd:.2f} pp**. Mean paired mCA change: '
             f'**{p.mca_improvement_pp_mean:+.2f} ± {p.mca_improvement_pp_sample_sd:.2f} pp**.', '',
             '## Severity-specific exceptions and RQ evidence', '']
    values = tables['per_class'].pivot(index=['corruption', 'severity', 'class_id', 'seed'],
                                       columns='model_name', values='recall')
    values['delta_pp'] = 100 * (values.robust_cnn - values.clean_cnn)
    means = values.groupby(['corruption', 'severity', 'class_id']).mean()
    exceptions = means[(means.index.get_level_values('severity') == 5) & (means.delta_pp < 0)]
    exception_rows = []
    for (family, severity, class_id), row in exceptions.iterrows():
        seed_values = values.loc[(family, severity, class_id), 'delta_pp']
        exception_rows.append([LABELS[family] + ' s5', classes[class_id],
            f'{100*row.clean_cnn:.2f}', f'{100*row.robust_cnn:.2f}',
            *[f'{seed_values.loc[s]:+.2f}' for s in SEEDS], f'{row.delta_pp:+.2f}'])
    text += ['All severity-5 class regressions in the three-seed mean are shown below. '
             'The per-seed signs demonstrate why seed 42 alone is insufficient for inference.', '',
             markdown_table(['Condition', 'Class', 'Clean recall %', 'Robust recall %',
                             'Δ42 pp', 'Δ123 pp', 'Δ2026 pp', 'Mean Δ pp'], exception_rows), '']
    severity_rows = []
    condition = aggregate['condition_summary']
    for model in MODELS:
        for family in FAMILIES:
            rows = condition.query('model_name == @model and corruption == @family').sort_values('severity')
            severity_rows.append([LABELS[model], LABELS[family], *[f'{100*x:.2f}' for x in rows['mean']]])
    text += [markdown_table(['Model', 'Family', 's1 %', 's2 %', 's3 %', 's4 %', 's5 %'], severity_rows), '',
             '- **RQ1:** Clean CNN accuracy falls sharply with noise and blur severity; '
             'darkening is less damaging on these specified scales. Its very mild blur and '
             'darkening means are slightly above clean accuracy; these measured departures '
             'are retained rather than smoothed away.',
             '- **RQ2:** Recall drops vary widely by class. Figure 07 and the full class '
             'tables quantify them; Figure 08 and directed error counts show the recurring '
             'frog, bird, and cat prediction concentrations under severe corruptions.',
             '- **RQ3:** Every seed has positive overall mCA improvement, and every '
             'class–seed pair has positive mean corrupted recall improvement. This does '
             'not imply improvement in every class/condition; the exception table shows '
             'the severity-5 reversals.',
             '- **RQ4:** Every seed has lower overall clean accuracy with Robust CNN. '
             'The mean loss and its paired SD are reported above, alongside the much '
             'larger mCA gain. Individual clean class recalls can improve or decline.', '',
             '## Verification and provenance', '',
             'All 96 matrices pass the existing core-matrix validator and an additional '
             'comparison of every row sum against `report/data003_evidence.json`, '
             '`observations.full_loaders.test.class_counts`: 1,000 for each of the ten classes. '
             'Each matrix totals 10,000 and its trace matches saved `n_correct`; reconstructed '
             'recalls, accuracy and macro-F1 match the saved results. Class names agree with '
             '`data/splits/split_manifest.json`. All six existing aggregate summaries '
             '(run, model, condition, paired deltas, paired summary, per-class) are rebuilt '
             'and compared numerically to their saved CSVs.', '',
             '[verification.json](verification.json) records input and output hashes, runtime '
             'versions, dimensions, checkpoint identities and prediction-file availability. '
             '[Per-condition checks](tables/count_verification.csv) expose the numeric audit. '
             'The six [selected checkpoints](../../../outputs/checkpoints/core-v1/README.md) '
             'are hashed and compared to the result rows. The original prediction NPZ files '
             'are absent from this checkout, so this run verifies saved counts, not logits, '
             'per-image labels, or individual failure IDs. The earlier logits check is recorded '
             'in [reproduction.md](../../reproduction.md).', '',
             'Conclusions describe training variability on one fixed split/test set, one '
             'architecture and three specified corruption families. Severity scales are not '
             'perceptually matched. Severities 4–5 extend training intensities 1–3 within known '
             'families; these results do not establish robustness to unseen corruption families. '
             'Repeated views and seeds are not independent samples of new test populations. '
             'Independent human review, fresh-machine replay, and formal pre-test sign-off '
             'remain open; this automated count check does not satisfy them.', '']
    (output / 'findings.md').write_text('\n'.join(text), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'report/core_v1/raw_results.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'report/core_v1/per_class_analysis')
    args = parser.parse_args(argv)
    if args.output_dir.resolve() == args.input.parent.resolve():
        parser.error('--output-dir must differ from the source results directory')
    evidence_path = ROOT / 'report/data003_evidence.json'
    split_path = ROOT / 'data/splits/split_manifest.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    split = json.loads(split_path.read_text(encoding='utf-8'))
    support = evidence['observations']['full_loaders']['test']['class_counts']
    classes = split['dataset']['classes']
    if classes != evidence['normalization']['record']['provenance']['dataset']['classes']:
        raise ValueError('saved class orders disagree')
    frame, checks = verify_counts(read_results(args.input), support)
    aggregate = aggregate_seeds(frame)
    inputs = [args.input, evidence_path, split_path, Path(__file__), ROOT / 'src/metrics.py']
    verified_summaries = []
    for name, rebuilt in aggregate.items():
        if name == 'results':
            continue
        path = args.input.parent / 'tables' / f'{name}.csv'
        saved = pd.read_csv(path)
        pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True), saved,
                                      check_dtype=False, rtol=0, atol=1e-12)
        inputs.append(path)
        verified_summaries.append(name)
    checkpoints = []
    for (model, seed), group in frame.groupby(['model_name', 'seed']):
        run = group.iloc[0]
        path = ROOT / 'outputs/checkpoints/core-v1' / run.run_id / 'best.pt'
        digest = sha256(path)
        if digest != run.checkpoint_sha256:
            raise ValueError(f'checkpoint hash mismatch: {path}')
        checkpoints.append({'model_name': model, 'seed': int(seed),
                            'path': path.relative_to(ROOT).as_posix(), 'sha256': digest})
    tables = build_tables(frame, classes)
    tables['count_verification'] = checks
    output = args.output_dir
    (output / 'tables').mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / 'tables' / f'{name}.csv', index=False, float_format='%.12g')
    export_figures(tables, classes, output / 'figures')
    write_findings(tables, aggregate, classes, output)
    def relative(path):
        try:
            return Path(path).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return str(Path(path).resolve())
    record = {'schema_version': 1, 'status': 'PASS: saved-count analysis',
              'inputs_sha256': {relative(p): sha256(p) for p in inputs},
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                          'pandas': pd.__version__, 'matplotlib': matplotlib.__version__},
              'classes': classes, 'saved_test_class_counts': support,
              'evaluations': len(frame), 'class_rows': len(tables['per_class']),
              'confusion_cells': len(tables['confusion_counts']),
              'total_repeated_views': int(frame.n_samples.sum()),
              'aggregate_tables_verified': verified_summaries,
              'max_recall_abs_error': float(checks.recall_max_abs_error.max()),
              'checkpoints': checkpoints,
              'prediction_files_present': sum((ROOT / p).is_file() for p in frame.predictions_path),
              'prediction_files_expected': len(frame),
              'verification_scope': 'saved confusion counts, class supports, summaries and checkpoint bytes; no logits replay',
              'outputs_sha256': {p.relative_to(output).as_posix(): sha256(p)
                                 for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'verification.json'}}
    (output / 'verification.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {len(frame)} matrices, 960 recalls, 9600 cells; six saved summaries and checkpoints verified.')
    print(f'Analysis written to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
