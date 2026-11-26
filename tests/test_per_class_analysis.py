"""Saved-count analysis checks; no dataset download, torch, or inference needed."""

import copy
import json
import unittest

import numpy as np

from scripts.analyze_per_class import ROOT, build_tables, read_results, verify_counts


class PerClassAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = read_results(ROOT / 'report/core_v1/raw_results.csv')
        evidence = json.loads((ROOT / 'report/data003_evidence.json').read_text(encoding='utf-8'))
        cls.support = evidence['observations']['full_loaders']['test']['class_counts']
        manifest = json.loads((ROOT / 'data/splits/split_manifest.json').read_text(encoding='utf-8'))
        cls.classes = manifest['dataset']['classes']
        cls.frame, cls.checks = verify_counts(cls.raw, cls.support)
        cls.tables = build_tables(cls.frame, cls.classes)

    def test_all_counts_and_known_orientation(self):
        self.assertEqual(len(self.checks), 96)
        self.assertEqual(self.checks.n_samples.sum(), 960000)
        self.assertLessEqual(self.checks.recall_max_abs_error.max(), 1e-12)
        cells = self.tables['confusion_counts']
        self.assertEqual(len(cells), 9600)
        selected = cells.query("model_name == 'clean_cnn' and seed == 42 and corruption == 'clean'")
        cat_dog = selected.query("true_class == 'cat' and pred_class == 'dog'").iloc[0]
        dog_cat = selected.query("true_class == 'dog' and pred_class == 'cat'").iloc[0]
        self.assertEqual(cat_dog['count'], 112)
        self.assertEqual(dog_cat['count'], 87)
        self.assertAlmostEqual(cat_dog.rate, .112)

    def test_rejects_transposed_confusion(self):
        broken = self.raw.copy(deep=True)
        broken.at[0, 'confusion_matrix'] = np.asarray(broken.at[0, 'confusion_matrix']).T.tolist()
        with self.assertRaisesRegex(ValueError, 'confusion_matrix'):
            verify_counts(broken, self.support)

    def test_rejects_inconsistent_external_support(self):
        support = self.support.copy()
        support[0] += 1
        support[1] -= 1  # Same total must not hide different class counts.
        with self.assertRaisesRegex(ValueError, 'saved test class counts'):
            verify_counts(self.raw, support)

    def test_rejects_changed_saved_recall(self):
        broken = self.raw.copy(deep=True)
        recalls = copy.copy(broken.at[0, 'per_class_accuracy'])
        recalls[0] += .001
        broken.at[0, 'per_class_accuracy'] = recalls
        with self.assertRaisesRegex(ValueError, 'per_class_accuracy'):
            verify_counts(broken, self.support)

    def test_requires_every_seed_and_condition(self):
        with self.assertRaisesRegex(ValueError, '96 rows'):
            verify_counts(self.raw.iloc[:-1], self.support)

    def test_paired_averaging_and_sample_sd(self):
        # Independently sum diagonal integer counts over the 15 conditions.
        gains = []
        for seed in (42, 123, 2026):
            scores = []
            for model in ('clean_cnn', 'robust_cnn'):
                rows = self.frame.query('model_name == @model and seed == @seed and corruption != "clean"')
                scores.append(sum(r[1][1] for r in rows.confusion_matrix) / 15000)
            gains.append(100 * (scores[1] - scores[0]))
        summary = self.tables['per_class_paired_summary'].query('class_id == 1').iloc[0]
        self.assertAlmostEqual(summary.mean_corrupted_recall_change_pp_mean, np.mean(gains))
        self.assertAlmostEqual(summary.mean_corrupted_recall_change_pp_sample_sd, np.std(gains, ddof=1))
        self.assertEqual(summary.n, 3)

    def test_rank_excludes_diagonal_and_breaks_ties_by_class_id(self):
        selected = self.tables['common_misclassifications']
        self.assertEqual(len(selected), 96 * 10 * 3)
        self.assertTrue((selected.true_class_id != selected.pred_class_id).all())
        for _, rows in selected.groupby(['model_name', 'seed', 'corruption', 'severity', 'true_class_id']):
            actual = rows.sort_values('rank').pred_class_id.tolist()
            expected = rows.sort_values(['count', 'pred_class_id'], ascending=[False, True]).pred_class_id.tolist()
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
