"""Classification and per-run robustness metrics for the CIFAR-10 protocol."""

import math

import numpy as np
import pandas as pd


_CORRUPTED_FAMILIES = ('gaussian_noise', 'gaussian_blur', 'brightness')
_CONDITIONS = {('clean', 0)} | {
    (family, severity) for family in _CORRUPTED_FAMILIES for severity in range(1, 6)
}
_RESULT_COLUMNS = ('corruption', 'severity', 'n_samples', 'n_correct', 'accuracy')


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
