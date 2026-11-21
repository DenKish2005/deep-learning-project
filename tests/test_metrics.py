import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    aggregate_seeds, classification_metrics, confusion_counts, robustness_metrics,
    main as metrics_main, validate_matrix,
)


def test_known_classification_metrics():
    true = [0, 0, 1, 1]
    predicted = (0, 1, 1, 1)
    expected = np.array([[1, 1], [0, 2]], dtype=np.int64)

    counts = confusion_counts(true, predicted, num_classes=2)
    metrics = classification_metrics(true, predicted, num_classes=2)

    np.testing.assert_array_equal(counts, expected, strict=True)
    np.testing.assert_array_equal(metrics['confusion_matrix'], expected, strict=True)
    assert metrics['n_samples'] == 4
    assert metrics['n_correct'] == 3
    assert metrics['accuracy'] == 0.75
    np.testing.assert_array_equal(metrics['per_class_accuracy'], [0.5, 1.0])
    assert metrics['macro_f1'] == pytest.approx((2 / 3 + 0.8) / 2)


def test_fixed_ten_classes_with_missing_support():
    true = np.array([0, 0, 3], dtype=np.uint8)
    predicted = np.array([0, 2, 9], dtype=np.int16)
    true.setflags(write=False)
    predicted.setflags(write=False)

    metrics = classification_metrics(true, predicted)

    counts = metrics['confusion_matrix']
    assert counts.shape == (10, 10)
    assert counts.sum() == metrics['n_samples'] == 3
    np.testing.assert_array_equal(counts.sum(axis=1), [2, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    assert counts[0, 2] == counts[3, 9] == 1
    np.testing.assert_array_equal(metrics['per_class_accuracy'], [0.5] + [0.0] * 9)
    assert metrics['macro_f1'] == pytest.approx((2 / 3) / 10)
    np.testing.assert_array_equal(true, [0, 0, 3])
    np.testing.assert_array_equal(predicted, [0, 2, 9])


@pytest.mark.parametrize(('predicted', 'score'), [([0, 1, 2], 1.0), ([1, 2, 0], 0.0)])
def test_all_correct_or_all_wrong(predicted, score):
    metrics = classification_metrics([0, 1, 2], predicted, num_classes=3)

    assert metrics['accuracy'] == metrics['macro_f1'] == score
    np.testing.assert_array_equal(metrics['per_class_accuracy'], [score] * 3)


@pytest.mark.parametrize(('true', 'predicted', 'diagnostic'), [
    ([], [], 'nonempty'),
    ([0, 1], [0], 'same length'),
    (np.array([[0, 1]]), [0, 1], 'one-dimensional'),
    (np.array([0.0, 1.0]), [0, 1], 'integer'),
    (np.array([0, 1], dtype=object), [0, 1], 'integer'),
    ([0, True], [0, 1], 'integer'),
    ([0, np.bool_(True)], [0, 1], 'integer'),
    ([0, np.timedelta64(1, 'D')], [0, 1], 'integer'),
    ([0, 1], [0, '1'], 'y_pred'),
    ([0, 1], [0, np.nan], 'y_pred'),
    ([0, 1], [0, np.inf], 'y_pred'),
    ([0, 1], [0, -1], 'y_pred'),
    ([0, 2], [0, 1], 'y_true'),
    ([0, 10**100], [0, 1], 'y_true'),
    (np.array([0, 2**64 - 1], dtype=np.uint64), [0, 1], 'y_true'),
])
def test_invalid_label_inputs(true, predicted, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        classification_metrics(true, predicted, num_classes=2)


@pytest.mark.parametrize('num_classes', [0, -1, True, np.bool_(True), 2.0, '2'])
def test_invalid_number_of_classes(num_classes):
    with pytest.raises(ValueError, match='num_classes'):
        confusion_counts([0], [0], num_classes=num_classes)


@pytest.fixture
def condition_results():
    conditions = [('clean', 0, 80)] + [
        (family, severity, 60)
        for family in ('gaussian_noise', 'gaussian_blur', 'brightness')
        for severity in range(1, 6)
    ]
    return pd.DataFrame([
        {'run_id': 'synthetic-run', 'corruption': family, 'severity': severity,
         'n_samples': 100, 'n_correct': correct, 'accuracy': correct / 100}
        for family, severity, correct in conditions
    ])


def test_robustness_units_and_denominators(condition_results):
    results = condition_results.assign(extra_metadata='fixture').sample(frac=1, random_state=5)
    # Evaluation records may be concatenated without resetting the index.
    results.index = pd.Index([19] * len(results))
    original = results.copy(deep=True)

    scored = robustness_metrics(results)

    pd.testing.assert_frame_equal(results, original)
    pd.testing.assert_frame_equal(scored[results.columns], original)
    corrupted = scored['corruption'] != 'clean'
    np.testing.assert_allclose(scored.loc[corrupted, 'accuracy_drop_pp'], 20.0)
    np.testing.assert_allclose(scored.loc[corrupted, 'relative_robustness'], 0.75)
    np.testing.assert_allclose(scored.loc[corrupted, 'family_mca'], 0.60)
    np.testing.assert_allclose(scored['mca'], 0.60)
    np.testing.assert_allclose(scored['robustness_gap_pp'], 20.0)
    assert scored.loc[~corrupted, ['accuracy_drop_pp', 'relative_robustness',
                                   'family_mca']].isna().all().all()
    assert scored['relative_robustness_reason'].isna().all()


def test_family_means_clean_exclusion_and_negative_drop(condition_results):
    scores = [80, 20, 30, 40, 50, 60, 60, 70, 80, 90, 100, 50, 50, 50, 50, 50]
    condition_results['n_correct'] = scores
    condition_results['accuracy'] = np.array(scores) / 100

    scored = robustness_metrics(condition_results)

    for family, mean in [('gaussian_noise', 0.4), ('gaussian_blur', 0.8), ('brightness', 0.5)]:
        np.testing.assert_allclose(scored.loc[scored['corruption'] == family, 'family_mca'], mean)
    np.testing.assert_allclose(scored['mca'], 17 / 30)
    np.testing.assert_allclose(scored['robustness_gap_pp'], 70 / 3)
    assert scored.at[10, 'accuracy_drop_pp'] == pytest.approx(-20)
    assert scored.at[10, 'relative_robustness'] == pytest.approx(1.25)


def test_zero_clean_accuracy_produces_missing_relative_robustness(condition_results):
    condition_results.loc[0, ['n_correct', 'accuracy']] = [0, 0.0]

    scored = robustness_metrics(condition_results)

    assert scored['relative_robustness'].isna().all()
    assert pd.isna(scored.at[0, 'relative_robustness_reason'])
    assert (scored.loc[1:, 'relative_robustness_reason'] == 'zero_clean_accuracy').all()
    np.testing.assert_allclose(scored.loc[1:, 'accuracy_drop_pp'], -60.0)
    np.testing.assert_allclose(scored['robustness_gap_pp'], -60.0)


def test_float32_accuracy_and_rows_without_run_id(condition_results):
    results = condition_results.drop(columns='run_id')
    results['accuracy'] = results['accuracy'].astype(np.float32)

    scored = robustness_metrics(results)

    np.testing.assert_allclose(scored['mca'], 0.60)


@pytest.mark.parametrize(('row', 'column', 'value', 'diagnostic'), [
    (1, 'corruption', 'darkening', 'corruption'),
    (1, 'corruption', None, 'corruption'),
    (1, 'severity', 0, 'severity'),
    (1, 'severity', 6, 'severity'),
    (0, 'severity', 1, 'severity'),
    (1, 'severity', True, 'severity'),
    (1, 'severity', 1.0, 'severity'),
    (1, 'n_samples', 0, 'n_samples'),
    (1, 'n_samples', True, 'n_samples'),
    (1, 'n_samples', 100.0, 'n_samples'),
    (1, 'n_correct', -1, 'n_correct'),
    (1, 'n_correct', 101, 'n_correct'),
    (1, 'n_correct', 60.0, 'n_correct'),
    (1, 'accuracy', -0.1, 'accuracy'),
    (1, 'accuracy', 1.1, 'accuracy'),
    (1, 'accuracy', np.nan, 'accuracy'),
    (1, 'accuracy', np.inf, 'accuracy'),
    (1, 'accuracy', '0.6', 'accuracy'),
    (1, 'accuracy', False, 'accuracy'),
    (1, 'accuracy', 0.5, 'n_correct / n_samples'),
    (1, 'run_id', 'other-run', 'run_id'),
    (1, 'run_id', None, 'run_id'),
])
def test_invalid_condition_value(condition_results, row, column, value, diagnostic):
    condition_results[column] = condition_results[column].astype(object)
    condition_results.at[row, column] = value

    with pytest.raises(ValueError, match=diagnostic):
        robustness_metrics(condition_results)


@pytest.mark.parametrize('missing_row', [0, 1])
def test_missing_condition(condition_results, missing_row):
    with pytest.raises(ValueError, match='missing conditions'):
        robustness_metrics(condition_results.drop(index=missing_row))


@pytest.mark.parametrize('duplicate_row', [0, 1])
def test_duplicate_condition(condition_results, duplicate_row):
    duplicate = pd.concat([condition_results, condition_results.iloc[[duplicate_row]]])

    with pytest.raises(ValueError, match='duplicate condition'):
        robustness_metrics(duplicate)


def test_mismatched_sample_totals(condition_results):
    condition_results.loc[1, ['n_samples', 'n_correct']] = [50, 30]

    with pytest.raises(ValueError, match='same n_samples'):
        robustness_metrics(condition_results)


def test_invalid_robustness_input_structure(condition_results):
    with pytest.raises(ValueError, match='DataFrame'):
        robustness_metrics(condition_results.to_dict('records'))
    with pytest.raises(ValueError, match='required columns.*accuracy'):
        robustness_metrics(condition_results.drop(columns='accuracy'))
    with pytest.raises(ValueError, match='duplicate columns'):
        robustness_metrics(pd.concat([condition_results, condition_results[['accuracy']]], axis=1))
    with pytest.raises(ValueError, match='missing conditions'):
        robustness_metrics(condition_results.iloc[:0])


@pytest.fixture
def core_matrix():
    import hashlib

    from src.metrics import classification_metrics

    conditions = [('clean', 0)] + [
        (family, severity)
        for family in ('gaussian_noise', 'gaussian_blur', 'brightness')
        for severity in range(1, 6)
    ]
    rows = []
    for model_index, model in enumerate(('clean_cnn', 'robust_cnn')):
        for seed_index, seed in enumerate((42, 123, 2026)):
            run_id = f'{model}-s{seed}'
            for corruption, severity in conditions:
                baseline = 600 if corruption == 'clean' else 300
                correct_per_class = baseline + seed_index * 100 + model_index * 100
                confusion = np.zeros((10, 10), dtype=np.int64)
                for class_id in range(10):
                    confusion[class_id, class_id] = correct_per_class
                    confusion[class_id, (class_id + 1) % 10] = 1000 - correct_per_class
                stats = classification_metrics(
                    np.repeat(np.arange(10), 1000),
                    np.array([prediction for truth in range(10)
                              for prediction in ([truth] * correct_per_class
                                                 + [(truth + 1) % 10] * (1000 - correct_per_class))]),
                )
                condition_hash = hashlib.sha256(
                    f'{corruption}:{severity}:protocol-v1'.encode()).hexdigest()
                rows.append({
                    'run_id': run_id, 'model_name': model, 'seed': seed,
                    'profile': 'core', 'status': 'complete', 'dataset': 'cifar10-test-v1',
                    'corruption': corruption, 'severity': severity,
                    'n_samples': 10_000, 'n_correct': int(np.trace(confusion)),
                    'accuracy': float(np.trace(confusion) / 10_000),
                    'macro_f1': stats['macro_f1'],
                    'per_class_accuracy': stats['per_class_accuracy'].tolist(),
                    'confusion_matrix': confusion.tolist(),
                    'checkpoint_sha256': hashlib.sha256(run_id.encode()).hexdigest(),
                    'config_hash': hashlib.sha256(model.encode()).hexdigest(),
                    'condition_hash': condition_hash, 'eval_seed': 100494,
                    'split_hash': 'a' * 64, 'normalization_hash': 'b' * 64,
                })
    assert len(rows) == 96
    return pd.DataFrame(rows)


def test_validate_complete_core_matrix_and_identity(core_matrix):
    validated = validate_matrix(core_matrix)

    assert len(validated) == 96
    assert validated.groupby(['model_name', 'seed']).size().eq(16).all()
    assert validated.groupby(['corruption', 'severity'])['condition_hash'].nunique().eq(1).all()
    assert validated.iloc[0]['corruption'] == 'clean'
    assert validated.iloc[1]['corruption'] == 'gaussian_noise'
    pd.testing.assert_frame_equal(core_matrix, core_matrix.copy())


def test_validate_matrix_rejects_missing_duplicate_hash_mismatch_and_smoke(core_matrix):
    with pytest.raises(ValueError, match='exactly 96'):
        validate_matrix(core_matrix.iloc[:-1])
    duplicate = pd.concat([core_matrix.iloc[:-1], core_matrix.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match='duplicate'):
        validate_matrix(duplicate)
    changed_hash = core_matrix.copy()
    changed_hash.at[1, 'condition_hash'] = 'c' * 64
    with pytest.raises(ValueError, match='condition_hash differs'):
        validate_matrix(changed_hash)
    reused_hash = core_matrix.copy()
    reused_hash['condition_hash'] = 'c' * 64
    with pytest.raises(ValueError, match='16 conditions.*unique'):
        validate_matrix(reused_hash)
    smoke = core_matrix.copy()
    smoke.at[0, 'profile'] = 'smoke'
    with pytest.raises(ValueError, match='profile=core'):
        validate_matrix(smoke)
    undersized = core_matrix.copy()
    undersized.at[0, 'n_samples'] = 100
    with pytest.raises(ValueError, match='exactly 10000'):
        validate_matrix(undersized)


def test_validate_matrix_rejects_inconsistent_confusion_summary(core_matrix):
    invalid = core_matrix.copy()
    invalid.at[0, 'n_correct'] -= 1
    invalid.at[0, 'accuracy'] = invalid.at[0, 'n_correct'] / invalid.at[0, 'n_samples']
    with pytest.raises(ValueError, match='confusion_matrix'):
        validate_matrix(invalid)


def test_aggregate_seeds_uses_paired_seeds_and_sample_sd(core_matrix):
    tables = aggregate_seeds(core_matrix)

    assert set(tables) == {
        'results', 'run_summary', 'condition_summary', 'model_summary',
        'paired_deltas', 'paired_summary', 'per_class_summary',
    }
    assert len(tables['run_summary']) == 6
    assert len(tables['condition_summary']) == 32
    assert len(tables['model_summary']) == 2
    assert tables['model_summary']['n'].tolist() == [3, 3]
    assert len(tables['paired_deltas']) == 3
    np.testing.assert_allclose(tables['paired_deltas']['mca_improvement_pp'], 10.0)
    np.testing.assert_allclose(tables['paired_summary']['mca_improvement_pp_sample_sd'], 0.0,
                               atol=1e-12)
    clean_rows = tables['condition_summary'].query(
        "model_name == 'clean_cnn' and corruption == 'clean'")
    np.testing.assert_allclose(clean_rows['sample_sd'], 0.1)
    assert len(tables['per_class_summary']) == 2 * 16 * 10
    assert tables['per_class_summary']['n'].eq(3).all()


def test_metrics_cli_publishes_complete_tables_and_refuses_overwrite(core_matrix, tmp_path):
    import json

    from src.metrics import _csv_safe

    source = tmp_path / 'results.csv'
    output = tmp_path / 'aggregate'
    _csv_safe(core_matrix).to_csv(source, index=False)

    assert metrics_main(['--input', str(source), '--output-dir', str(output)]) == 0
    assert {path.name for path in output.iterdir()} == {
        'results.csv', 'run_summary.csv', 'condition_summary.csv', 'model_summary.csv',
        'paired_deltas.csv', 'paired_summary.csv', 'per_class_summary.csv',
    }
    saved = pd.read_csv(output / 'results.csv')
    assert len(saved) == 96
    assert len(pd.read_csv(output / 'condition_summary.csv')) == 32
    assert len(pd.read_csv(output / 'paired_deltas.csv')) == 3
    assert len(json.loads(saved.iloc[0]['confusion_matrix'])) == 10
    with pytest.raises(FileExistsError, match='already exists'):
        metrics_main([
            '--input', str(source), '--output-dir', str(output),
        ])
