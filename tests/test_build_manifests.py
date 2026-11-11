import math
import sys
import unittest
from pathlib import Path


PART2 = Path(__file__).resolve().parents[1] / "data_preparation"
sys.path.insert(0, str(PART2))

from build_manifests import fmt_tenth, geometry_row, round_tenth  # noqa: E402


class FakeGeometry:
    POSITIONS = {
        "scene/a.npy": {
            "x": 1.141710,
            "y": -0.356227,
            "z": 1.479843,
        },
        "scene/b.npy": {
            "x": 0.129974,
            "y": -0.064216,
            "z": -0.258742,
        },
    }

    def listener_position(self, reverb_id):
        result = dict(self.POSITIONS[reverb_id])
        result["distance_m"] = math.sqrt(sum(v * v for v in result.values()))
        result["listener"] = [0.0, 0.0, 0.0]
        result["source"] = [result["x"], result["y"], result["z"]]
        return result


LABELS = {"audio/a": ["alpha"], "audio/b": ["beta"]}


def row(qtype, question, answer):
    return {
        "audio_id": "audio/a",
        "reverb_id": "scene/a.npy",
        "audio_id2": "audio/b",
        "reverb_id2": "scene/b.npy",
        "question_id": 7,
        "question_type": qtype,
        "question": question,
        "answer": answer,
    }


class Stage3GeometryManifestTest(unittest.TestCase):
    def enrich(self, qtype, question, answer):
        return geometry_row(row(qtype, question, answer), LABELS, FakeGeometry(), "eval")

    def test_half_up_tenth_and_signed_zero(self):
        self.assertEqual(round_tenth(1.25), 1.3)
        self.assertEqual(round_tenth(-0.04), 0.0)
        self.assertEqual(fmt_tenth(-0.04, signed=True), "-0.0")
        self.assertEqual(fmt_tenth(0.04, signed=True), "+0.0")

    def test_binary_direction(self):
        item = self.enrich(
            "MIXUP_DIRECTION",
            "Can you hear alpha from the right and beta from the left?",
            "No",
        )
        self.assertIn("not consistent", item["geometric_analysis"])

    def test_binary_listener_distance(self):
        item = self.enrich(
            "MIXUP_DISTANCE_BOTH",
            "Is alpha further from you than beta?",
            "Yes",
        )
        self.assertIn("1.9 m", item["geometric_analysis"])
        self.assertIn("0.3 m", item["geometric_analysis"])

    def test_nonbinary_relative_direction(self):
        item = self.enrich(
            "MIXUP_NONBINARY_DIRECTION",
            "Is alpha above or below beta?",
            "below",
        )
        self.assertEqual(item["pair_geometry"]["a_relative_to_b"]["ud"], "below")

    def test_nonbinary_inter_source_distance(self):
        item = self.enrich(
            "MIXUP_NONBINARY_DISTANCE",
            "What is the distance between alpha and beta?",
            "2m",
        )
        self.assertEqual(item["pair_geometry"]["inter_source_distance_m_0p1"], 2.0)
        self.assertIn("nearest 0.5 m", item["geometric_analysis"])

    def test_nonbinary_source_identity(self):
        item = self.enrich(
            "MIXUP_NONBINARY_SOURCE",
            "What can be heard behind beta?",
            "alpha",
        )
        self.assertIn("contains alpha", item["geometric_analysis"])

    def test_original_answer_must_match_geometry(self):
        with self.assertRaisesRegex(ValueError, "geometry does not reproduce the answer"):
            self.enrich(
                "MIXUP_DISTANCE_BOTH",
                "Is alpha further from you than beta?",
                "No",
            )

    def test_source_order_is_observable_and_query_mapping_is_retained(self):
        item = self.enrich(
            "MIXUP_NONBINARY_DIRECTION",
            "Is alpha left or right of beta?",
            "right",
        )
        self.assertEqual(item["scene_sources"][0]["labels"], ["beta"])
        self.assertEqual(item["pair_geometry"]["source_a_index"], 2)
        self.assertEqual(item["pair_geometry"]["source_b_index"], 1)


if __name__ == "__main__":
    unittest.main()
