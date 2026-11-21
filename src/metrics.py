"""Classification and per-run robustness metrics for the CIFAR-10 protocol."""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


_CORRUPTED_FAMILIES = ('gaussian_noise', 'gaussian_blur', 'brightness')
_CONDITIONS = {('clean', 0)} | {
    (family, severity) for family in _CORRUPTED_FAMILIES for severity in range(1, 6)
}
_RESULT_COLUMNS = ('corruption', 'severity', 'n_samples', 'n_correct', 'accuracy')
_MODELS = ('clean_cnn', 'robust_cnn')
_SEEDS = (42, 123, 2026)
_IDENTITY_COLUMNS = ('dataset', 'split_hash', 'normalization_hash', 'eval_seed')
_SHA256_COLUMNS = ('checkpoint_sha256', 'config_hash', 'condition_hash',
                   'split_hash', 'normalization_hash')


def _is_integer(value) -> bool:
    if isinstance(value, np.integer):
        return value.dtype.kind in 'iu'
    return isinstance(value, int) and not isinstance(value, bool)


def _label_ids(labels, name: str, num_classes: int) -> np.ndarray:
    if isinstance(labels, np.ndarray):
        if labels.ndim != 1 or labels.dtype.kind not in 'iu':
            raise ValueError(f'{name} must be a one-dimensional integer array')
        in_range = np.all((labels >= 0) & (labels < num_classes))
    elif isinstance(labels, (list, tuple)):
        if not all(_is_integer(label) for label in labels):
            raise ValueError(f'{name} must contain only integer class IDs')
        in_range = all(0 <= int(label) < num_classes for label in labels)
    else:
        raise ValueError(f'{name} must be a one-dimensional integer array, list or tuple')
    if len(labels) == 0:
        raise ValueError(f'{name} must be nonempty')
    if not in_range:
        raise ValueError(f'{name} class IDs must be in [0, {num_classes - 1}]')
    return np.asarray(labels, dtype=np.int64)


def confusion_counts(y_true, y_pred, num_classes: int = 10) -> np.ndarray:
    """Count integer class IDs in a fixed int64 matrix with true classes as rows."""
    if not _is_integer(num_classes) or num_classes < 1:
        raise ValueError('num_classes must be a positive integer')
    num_classes = int(num_classes)
    true = _label_ids(y_true, 'y_true', num_classes)
    predicted = _label_ids(y_pred, 'y_pred', num_classes)
    if len(true) != len(predicted):
        raise ValueError('y_true and y_pred must have the same length')

    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(counts, (true, predicted), 1)
    return counts


def classification_metrics(y_true, y_pred, num_classes: int = 10) -> dict:
    """Return counts, accuracy, per-class recall and macro-F1 for one condition.

    Accuracy, per_class_accuracy (recall), and macro_f1 are fractions. Recall
    is zero for classes without true examples. Macro-F1 averages over every
    configured class, with undefined class F1 contributing zero.
    """
    counts = confusion_counts(y_true, y_pred, num_classes)
    true_positives = counts.diagonal()
    support = counts.sum(axis=1)
    f1_denominator = support + counts.sum(axis=0)
    per_class_accuracy = np.divide(true_positives, support,
                                   out=np.zeros(len(counts)), where=support > 0)
    per_class_f1 = np.divide(2 * true_positives, f1_denominator,
                             out=np.zeros(len(counts)), where=f1_denominator > 0)
    n_samples = int(support.sum())
    n_correct = int(true_positives.sum())
    return {
        'confusion_matrix': counts,
        'n_samples': n_samples,
        'n_correct': n_correct,
        'accuracy': n_correct / n_samples,
        'per_class_accuracy': per_class_accuracy,
        'macro_f1': float(per_class_f1.mean()),
    }


def _validated_condition_scores(results: pd.DataFrame) -> dict[tuple[str, int], float]:
    if not isinstance(results, pd.DataFrame):
        raise ValueError('results must be a pandas DataFrame')
    if not results.columns.is_unique:
        raise ValueError('results must not contain duplicate columns')
    missing_columns = set(_RESULT_COLUMNS).difference(results.columns)
    if missing_columns:
        raise ValueError(f'results is missing required columns: {sorted(missing_columns)}')
    if 'run_id' in results:
        run_ids = results['run_id']
        if not all(isinstance(run_id, str) and run_id.strip() for run_id in run_ids):
            raise ValueError('run_id must contain nonempty strings')
        if run_ids.nunique() > 1:
            raise ValueError('results must contain a single run_id')

    scores = {}
    sample_totals = set()
    rows = results[list(_RESULT_COLUMNS)].itertuples(index=False, name=None)
    for row_number, (corruption, severity, n_samples, n_correct, accuracy) in enumerate(rows):
        row = f'results row {row_number}'
        if not isinstance(corruption, str) or corruption not in ('clean', *_CORRUPTED_FAMILIES):
            raise ValueError(f'{row}: unknown corruption {corruption!r}')
        if not _is_integer(severity) or (corruption, int(severity)) not in _CONDITIONS:
            raise ValueError(f'{row}: invalid severity {severity!r} for {corruption}')
        condition = (corruption, int(severity))
        if condition in scores:
            raise ValueError(f'{row}: duplicate condition {condition!r}')
        if not _is_integer(n_samples) or int(n_samples) <= 0:
            raise ValueError(f'{row}: n_samples must be a positive integer')
        if not _is_integer(n_correct) or not 0 <= int(n_correct) <= int(n_samples):
            raise ValueError(f'{row}: n_correct must be an integer in [0, n_samples]')
        if (not (_is_integer(accuracy) or isinstance(accuracy, (float, np.floating)))
                or not 0 <= accuracy <= 1):
            raise ValueError(f'{row}: accuracy must be a finite fraction in [0, 1]')
        accuracy = float(accuracy)
        # Accept float32 rounding while requiring a zero count to yield exact zero.
        if not math.isclose(accuracy, int(n_correct) / int(n_samples), rel_tol=1e-7, abs_tol=0.0):
            raise ValueError(f'{row}: accuracy must agree with n_correct / n_samples')
        scores[condition] = accuracy
        sample_totals.add(int(n_samples))

    missing = _CONDITIONS.difference(scores)
    if missing:
        raise ValueError(f'results is missing conditions: {sorted(missing)}')
    if len(sample_totals) != 1:
        raise ValueError('all conditions must have the same n_samples')
    return scores


def robustness_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Annotate the 16 condition rows for one run without changing the input.

    Per-condition and family metrics are missing on the clean row. mca and
    robustness_gap_pp are run-level values repeated on every row. Accuracy and
    mCA are fractions, relative robustness is a ratio, and *_pp fields are
    percentage points. Relative robustness is missing with a reason when the
    clean accuracy is zero.
    """
    scores = _validated_condition_scores(results)
    clean_accuracy = scores['clean', 0]

    scored = results.copy()
    families = scored['corruption'].to_numpy()
    accuracies = scored['accuracy'].to_numpy(dtype=np.float64)
    corrupted = families != 'clean'
    corrupted_accuracies = accuracies[corrupted]

    drops = pd.array([None] * len(scored), dtype='Float64')
    drops[corrupted] = 100 * (clean_accuracy - corrupted_accuracies)
    relative = pd.array([None] * len(scored), dtype='Float64')
    relative_reason = pd.array([None] * len(scored), dtype='string')
    if clean_accuracy > 0:
        relative[corrupted] = corrupted_accuracies / clean_accuracy
    else:
        relative_reason[corrupted] = 'zero_clean_accuracy'

    family_mca = pd.array([None] * len(scored), dtype='Float64')
    for family in _CORRUPTED_FAMILIES:
        in_family = families == family
        family_mca[in_family] = accuracies[in_family].mean()

    mca = float(corrupted_accuracies.mean())
    scored['accuracy_drop_pp'] = drops
    scored['relative_robustness'] = relative
    scored['relative_robustness_reason'] = relative_reason
    scored['family_mca'] = family_mca
    scored['mca'] = mca
    scored['robustness_gap_pp'] = 100 * (clean_accuracy - mca)
    return scored


def validate_matrix(results: pd.DataFrame) -> pd.DataFrame:
    """Validate the complete 2-model × 3-seed × 16-condition core matrix.

    Each input row is a completed model/seed/condition evaluation summary. The
    returned copy is sorted into protocol order. ``condition_hash`` is required
    so every model is proven to have received the same indexed test inputs for
    each condition. Optional dataset/split/normalization/eval identity columns,
    when present, must also be constant throughout the matrix.
    """
    required = {
        'run_id', 'model_name', 'seed', 'profile', 'status', 'dataset',
        'checkpoint_sha256', 'config_hash', 'condition_hash', 'eval_seed',
        'split_hash', 'normalization_hash', 'macro_f1', 'per_class_accuracy',
        'confusion_matrix', *_RESULT_COLUMNS,
    }
    if not isinstance(results, pd.DataFrame):
        raise ValueError('results must be a pandas DataFrame')
    if not results.columns.is_unique:
        raise ValueError('results must not contain duplicate columns')
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f'results is missing required columns: {sorted(missing)}')
    if len(results) != len(_MODELS) * len(_SEEDS) * len(_CONDITIONS):
        raise ValueError('core matrix must contain exactly 96 rows')

    frame = results.copy()
    if frame['model_name'].isna().any() or not frame['model_name'].isin(_MODELS).all():
        raise ValueError(f'model_name values must be exactly {_MODELS}')
    if frame['seed'].isna().any() or not frame['seed'].map(_is_integer).all():
        raise ValueError('seed values must be integer protocol seeds')
    frame['seed'] = frame['seed'].astype('int64')
    if set(frame['seed']) != set(_SEEDS):
        raise ValueError(f'seed values must be exactly {_SEEDS}')
    if frame['condition_hash'].map(lambda x: not isinstance(x, str) or not x.strip()).any():
        raise ValueError('condition_hash must contain nonempty strings')
    if not (frame['status'] == 'complete').all():
        raise ValueError('all core evaluation rows must have status=complete')
    if not (frame['profile'] == 'core').all():
        raise ValueError('only profile=core rows may enter the core matrix')
    for column in ('run_id', 'dataset'):
        if frame[column].map(lambda x: not isinstance(x, str) or not x.strip()).any():
            raise ValueError(f'{column} must contain nonempty strings')
    for column in _SHA256_COLUMNS:
        if frame[column].map(lambda x: not isinstance(x, str)
                             or len(x) != 64
                             or any(c not in '0123456789abcdef' for c in x)).any():
            raise ValueError(f'{column} must contain lowercase SHA-256 hex digests')
    if not frame['eval_seed'].map(_is_integer).all():
        raise ValueError('eval_seed must contain integer values')
    if not (frame['macro_f1'].map(
            lambda x: (_is_integer(x) or isinstance(x, (float, np.floating)))
            and math.isfinite(float(x)) and 0 <= float(x) <= 1)).all():
        raise ValueError('macro_f1 must contain finite fractions in [0, 1]')

    key_columns = ['model_name', 'seed', 'corruption', 'severity']
    if frame.duplicated(key_columns).any():
        raise ValueError('core matrix contains duplicate model/seed/condition keys')
    expected_keys = {
        (model, seed, corruption, severity)
        for model in _MODELS for seed in _SEEDS
        for corruption, severity in _CONDITIONS
    }
    actual_keys = set(frame[key_columns].itertuples(index=False, name=None))
    if actual_keys != expected_keys:
        raise ValueError('core matrix is missing or has unexpected model/seed/condition keys')

    if not (frame['n_samples'] == 10_000).all():
        raise ValueError('every core condition must contain exactly 10000 samples')
    run_ids = []
    for (_, _), group in frame.groupby(['model_name', 'seed'], sort=False):
        _validated_condition_scores(group)
        if group['run_id'].nunique() != 1 or group['checkpoint_sha256'].nunique() != 1 \
                or group['config_hash'].nunique() != 1:
            raise ValueError('run_id, checkpoint_sha256 and config_hash must be stable within a run')
        run_ids.append(group['run_id'].iloc[0])
    if len(set(run_ids)) != 6:
        raise ValueError('each model/seed pair must have a unique run_id')

    for row in frame.itertuples(index=False):
        matrix = np.asarray(row.confusion_matrix)
        if (matrix.shape != (10, 10) or matrix.dtype.kind not in 'iu'
                or np.any(matrix < 0) or int(matrix.sum()) != int(row.n_samples)
                or not np.array_equal(matrix.sum(axis=1), np.full(10, 1000, dtype=matrix.dtype))
                or int(np.trace(matrix)) != int(row.n_correct)):
            raise ValueError(
                'confusion_matrix must be nonnegative int 10x10 with 1000 test labels per class '
                'and counts matching the row summary')
        support = matrix.sum(axis=1)
        recalls = np.divide(np.diag(matrix), support, out=np.zeros(10, dtype=np.float64),
                            where=support > 0)
        f1_denominator = support + matrix.sum(axis=0)
        f1 = np.divide(2 * np.diag(matrix), f1_denominator,
                       out=np.zeros(10, dtype=np.float64), where=f1_denominator > 0)
        stored_recalls = np.asarray(row.per_class_accuracy, dtype=np.float64)
        if (stored_recalls.shape != (10,) or not np.isfinite(stored_recalls).all()
                or np.any((stored_recalls < 0) | (stored_recalls > 1))
                or not np.allclose(stored_recalls, recalls, rtol=1e-7, atol=1e-12)):
            raise ValueError('per_class_accuracy does not agree with confusion_matrix')
        if not math.isclose(float(row.macro_f1), float(f1.mean()), rel_tol=1e-7, abs_tol=1e-12):
            raise ValueError('macro_f1 does not agree with confusion_matrix')

    hash_counts = frame.groupby(['corruption', 'severity'], sort=False)['condition_hash'].nunique()
    if not (hash_counts == 1).all():
        raise ValueError('condition_hash differs across models or seeds for a condition')
    hashes_by_condition = frame.groupby(['corruption', 'severity'], sort=False)[
        'condition_hash'].first()
    if hashes_by_condition.nunique() != len(_CONDITIONS):
        raise ValueError('each of the 16 conditions must have a unique condition_hash')
    for column in _IDENTITY_COLUMNS:
        if column in frame and frame[column].nunique(dropna=False) != 1:
            raise ValueError(f'{column} must be identical across the core matrix')

    model_order = {value: i for i, value in enumerate(_MODELS)}
    condition_order = {condition: i for i, condition in enumerate(
        [('clean', 0)] + [(family, severity) for family in _CORRUPTED_FAMILIES
                         for severity in range(1, 6)])}
    frame['_model_order'] = frame['model_name'].map(model_order)
    frame['_condition_order'] = [condition_order[(c, int(s))]
                                 for c, s in zip(frame['corruption'], frame['severity'])]
    frame = frame.sort_values(['_model_order', 'seed', '_condition_order'])
    return frame.drop(columns=['_model_order', '_condition_order']).reset_index(drop=True)


def aggregate_seeds(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Validate core results and produce seed-aware result tables.

    Standard deviations use sample SD (ddof=1) across the three training
    seeds. Paired treatment effects are computed within seed and reported in
    percentage points; corrupted conditions and images are never treated as
    independent training replicates.
    """
    matrix = validate_matrix(results)
    per_run = []
    scored_runs = []
    for (model, seed), group in matrix.groupby(['model_name', 'seed'], sort=False):
        scored = robustness_metrics(group)
        scored_runs.append(scored)
        clean = scored.loc[scored['corruption'] == 'clean'].iloc[0]
        corrupted = scored.loc[scored['corruption'] != 'clean']
        row = {
            'model_name': model, 'seed': int(seed),
            'clean_accuracy': float(clean['accuracy']),
            'mca': float(scored['mca'].iloc[0]),
            'robustness_gap_pp': float(scored['robustness_gap_pp'].iloc[0]),
        }
        if 'macro_f1' in scored:
            row['mean_corrupted_macro_f1'] = float(corrupted['macro_f1'].mean())
        per_run.append(row)
    run_summary = pd.DataFrame(per_run).sort_values(['model_name', 'seed']).reset_index(drop=True)
    annotated_results = pd.concat(scored_runs, ignore_index=True)

    condition_group = matrix.groupby(['model_name', 'corruption', 'severity'], sort=False)
    condition_summary = condition_group['accuracy'].agg(
        mean='mean', sample_sd=lambda values: values.std(ddof=1), n='count').reset_index()
    if 'macro_f1' in matrix:
        f1_summary = condition_group['macro_f1'].agg(
            macro_f1_mean='mean', macro_f1_sample_sd=lambda values: values.std(ddof=1)).reset_index()
        condition_summary = condition_summary.merge(
            f1_summary, on=['model_name', 'corruption', 'severity'], validate='one_to_one')

    model_aggregations = {
        'n': ('seed', 'count'),
        'clean_accuracy_mean': ('clean_accuracy', 'mean'),
        'clean_accuracy_sample_sd': ('clean_accuracy', lambda x: x.std(ddof=1)),
        'mca_mean': ('mca', 'mean'),
        'mca_sample_sd': ('mca', lambda x: x.std(ddof=1)),
        'robustness_gap_pp_mean': ('robustness_gap_pp', 'mean'),
        'robustness_gap_pp_sample_sd': ('robustness_gap_pp', lambda x: x.std(ddof=1)),
    }
    if 'mean_corrupted_macro_f1' in run_summary:
        model_aggregations.update({
            'mean_corrupted_macro_f1_mean': ('mean_corrupted_macro_f1', 'mean'),
            'mean_corrupted_macro_f1_sample_sd': ('mean_corrupted_macro_f1', lambda x: x.std(ddof=1)),
        })
    model_summary = run_summary.groupby('model_name', sort=False).agg(
        **model_aggregations,
    ).reset_index()

    paired = run_summary.pivot(index='seed', columns='model_name',
                               values=['clean_accuracy', 'mca', 'robustness_gap_pp'])
    paired_deltas = pd.DataFrame({
        'seed': paired.index.to_numpy(dtype=np.int64),
        'clean_accuracy_change_pp': 100 * (
            paired['clean_accuracy']['robust_cnn'] - paired['clean_accuracy']['clean_cnn']),
        'mca_improvement_pp': 100 * (
            paired['mca']['robust_cnn'] - paired['mca']['clean_cnn']),
        'robustness_gap_change_pp': (
            paired['robustness_gap_pp']['robust_cnn'] - paired['robustness_gap_pp']['clean_cnn']),
    }).reset_index(drop=True)
    paired_summary = pd.DataFrame([{
        'n': len(paired_deltas),
        **{f'{column}_mean': float(paired_deltas[column].mean())
           for column in paired_deltas.columns if column.endswith('_pp')},
        **{f'{column}_sample_sd': float(paired_deltas[column].std(ddof=1))
           for column in paired_deltas.columns if column.endswith('_pp')},
    }])

    class_rows = []
    for row in matrix.itertuples(index=False):
        for class_id, recall in enumerate(row.per_class_accuracy):
            class_rows.append({
                'model_name': row.model_name, 'corruption': row.corruption,
                'severity': int(row.severity), 'class_id': class_id,
                'accuracy': float(recall),
            })
    per_class = pd.DataFrame(class_rows).groupby(
        ['model_name', 'corruption', 'severity', 'class_id'], sort=False)['accuracy'].agg(
            mean='mean', sample_sd=lambda x: x.std(ddof=1), n='count').reset_index()

    return {
        'results': annotated_results,
        'run_summary': run_summary,
        'condition_summary': condition_summary,
        'model_summary': model_summary,
        'paired_deltas': paired_deltas,
        'paired_summary': paired_summary,
        'per_class_summary': per_class,
    }


def _parse_array_columns(results: pd.DataFrame) -> pd.DataFrame:
    """Decode the JSON array cells used by the portable results CSV format."""
    decoded = results.copy()
    for column in ('per_class_accuracy', 'confusion_matrix'):
        if column not in decoded:
            continue
        def parse(value):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{column} CSV cell is not valid JSON') from exc
            return value
        decoded[column] = decoded[column].map(parse)
    return decoded


def _csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for column in ('per_class_accuracy', 'confusion_matrix'):
        if column in safe:
            safe[column] = safe[column].map(
                lambda value: json.dumps(np.asarray(value).tolist(), separators=(',', ':')))
    return safe


def write_aggregate_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    """Publish the complete set of aggregate CSVs as one new directory.

    Existing output is never replaced. Files are assembled in a temporary
    sibling directory and the directory rename publishes them together.
    """
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'aggregate output already exists: {output}')
    expected = {
        'results', 'run_summary', 'condition_summary', 'model_summary',
        'paired_deltas', 'paired_summary', 'per_class_summary',
    }
    if set(tables) != expected:
        raise ValueError(f'tables must contain exactly {sorted(expected)}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.', dir=output.parent) as temporary:
        staged = Path(temporary)
        for name in sorted(expected):
            table = tables[name]
            if name == 'results':
                # The protocol keeps large per-condition arrays in JSON/NPZ and
                # exports only scalar metrics and artifact paths to results.csv.
                table = table.drop(columns=['confusion_matrix', 'per_class_accuracy'],
                                   errors='ignore')
            _csv_safe(table).to_csv(staged / f'{name}.csv', index=False)
        os.replace(staged, output)
    return output


def main(argv=None) -> int:
    """Validate a complete core results CSV and publish aggregate tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path,
                        help='CSV containing the 96 completed core evaluation summaries')
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='new directory for the seven aggregate CSV tables')
    args = parser.parse_args(argv)
    if not args.input.is_file() or args.input.is_symlink():
        parser.error(f'input must be an existing regular file: {args.input}')
    results = _parse_array_columns(pd.read_csv(args.input))
    tables = aggregate_seeds(results)
    output = write_aggregate_tables(tables, args.output_dir)
    print(f'Validated {len(tables["results"])} core rows; wrote {len(tables)} tables to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
