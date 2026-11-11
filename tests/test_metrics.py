import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from embeddings import load_embedding_cache, save_embedding_cache
from metrics import (
    angular_error_degrees,
    evaluate_classification_map,
    evaluate_continuous_doa,
    evaluate_records,
    parse_binary,
    parse_doa,
)


class ParsingTests(unittest.TestCase):
    def test_doa_natural_language(self):
        self.assertEqual(
            parse_doa("right, front, below; about 4.5 meters away"),
            {"lr": "right", "fb": "front", "ud": "below", "dist": 4.5},
        )

    def test_binary_requires_standalone_answer(self):
        self.assertIs(parse_binary("Final answer: Yes"), True)
        self.assertIs(parse_binary("No."), False)
        self.assertIsNone(parse_binary("I think yes because the layout matches"))


class OfficialMetricTests(unittest.TestCase):
    def test_unparseable_predictions_stay_in_denominator(self):
        records = [
            {"qt": "DOA", "gt": "left, front, above; 2m", "pred": "left, front, above; 2.5m"},
            {"qt": "DOA", "gt": "right, behind, below; 3m", "pred": "unparseable"},
        ]
        result = evaluate_records(records)
        doa = result["ssqa_official"]["doa"]["DOA"]
        self.assertEqual(doa["doa_accuracy"], 0.5)
        self.assertEqual(doa["direction_valid_rate"], 0.5)
        self.assertEqual(doa["distance_der"], 0.5)
        self.assertEqual(doa["distance_valid_rate"], 0.5)

    def test_der_boundary_is_strictly_greater_than_half_meter(self):
        records = [
            {"qt": "DOA", "gt": "left, front, above; 2m", "pred": "left, front, above; 2.5m"},
            {"qt": "DOA", "gt": "left, front, above; 2m", "pred": "left, front, above; 2.51m"},
        ]
        der = evaluate_records(records)["ssqa_official"]["doa"]["DOA"]["distance_der"]
        self.assertEqual(der, 0.5)

    def test_raw_task_is_recovered_from_normalized_aura_key(self):
        records = [{
            "key": "MIXUP_SINGLE_DOA-42",
            "qt": "DOA",
            "gt": "left, front, above; 2m",
            "pred": "left, front, above; 2m",
        }]
        result = evaluate_records(records)
        self.assertIn("MIXUP_SINGLE_DOA", result["ssqa_official"]["doa"])
        self.assertEqual(result["question_type_counts"], {"MIXUP_SINGLE_DOA": 1})

    def test_average_ba_is_unweighted_task_mean(self):
        records = [
            {"qt": "MIXUP_DIRECTION", "gt": "Yes", "pred": "Yes"},
            {"qt": "MIXUP_DIRECTION", "gt": "No", "pred": "No"},
            {"qt": "MIXUP_DISTANCE_BOTH", "gt": "Yes", "pred": "No"},
        ]
        result = evaluate_records(records)["ssqa_official"]
        self.assertEqual(result["direction_binary_accuracy"]["accuracy"], 1.0)
        self.assertEqual(result["distance_binary_accuracy"]["accuracy"], 0.0)
        self.assertEqual(result["average_binary_accuracy"], 0.5)


class ClassificationMapTests(unittest.TestCase):
    def test_unquoted_comma_in_subset_label_is_preserved(self):
        import tempfile
        from metrics import load_labels

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            path.write_text(
                "index,mid,display_name\n"
                "0,/m/a,Speech\n"
                "1,/m/b,Domestic animals, pets\n",
                encoding="utf-8",
            )
            self.assertEqual(load_labels(path), ["Speech", "Domestic animals, pets"])

    def test_perfect_ranking_has_perfect_map(self):
        records = [
            {"gt": "speech", "pred": "speech"},
            {"gt": "music", "pred": "music"},
        ]
        labels = ["Speech", "Music"]
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        result = evaluate_classification_map(records, labels, vectors, vectors)
        self.assertEqual(result["map"], 1.0)
        self.assertEqual(result["num_classes_evaluated"], 2)


class ContinuousDirectionTests(unittest.TestCase):
    def test_wraparound_uses_spherical_angle(self):
        error = angular_error_degrees(179, 0, -179, 0)
        self.assertAlmostEqual(error, 2.0, places=6)

    def test_er20_is_strict(self):
        result = evaluate_continuous_doa([20.0, 20.1, 10.0])
        self.assertAlmostEqual(result["er20"], 1 / 3)
        self.assertAlmostEqual(result["angular_mae_deg"], 50.1 / 3)


class EmbeddingCacheTests(unittest.TestCase):
    def test_model_mismatch_is_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            save_embedding_cache(path, "text-embedding-3-small", ["x"], np.ones((1, 2)))
            with self.assertRaisesRegex(ValueError, "model mismatch"):
                load_embedding_cache(path, "text-embedding-ada-002")


if __name__ == "__main__":
    unittest.main()
