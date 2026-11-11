import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from scene_attribution import evaluate, parse_scene


class SceneParsingTests(unittest.TestCase):
    def test_parse_two_source_scene(self):
        parsed = parse_scene(
            "Scene description:\n"
            "- Source at left, front, and above, approximately 2 meters away: dog; bark.\n"
            "- Source at right, behind, and below, approximately 4.5 meters away: music.\n"
            "Final answer: dog; bark"
        )
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["labels"], {"dog", "bark"})
        self.assertEqual(parsed[1]["location"]["distance_m"], 4.5)


class SceneMetricTests(unittest.TestCase):
    def test_source_order_is_irrelevant(self):
        manifest = [{
            "key": "MIXUP_SINGLE_DOA-1",
            "scene_sources": [
                {"labels": ["dog"], "location": {"lr": "left", "fb": "front", "ud": "above", "distance_m": 2.0}, "is_target": True},
                {"labels": ["music"], "location": {"lr": "right", "fb": "behind", "ud": "below", "distance_m": 4.0}, "is_target": False},
            ],
        }]
        predictions = [{
            "subset": "held-out",
            "key": "MIXUP_SINGLE_DOA-1",
            "raw": (
                "Scene description:\n"
                "- Source at right, behind, and below, approximately 4 meters away: music.\n"
                "- Source at left, front, and above, approximately 2 meters away: dog.\n"
                "Final answer: left, front, above; 2m"
            ),
        }]
        result = evaluate(manifest, predictions)
        self.assertEqual(result["format"]["two_source_parse_rate"], 1.0)
        self.assertEqual(result["scene"]["full_scene_tuple_exact_rate"], 1.0)
        self.assertEqual(result["query"]["MIXUP_SINGLE_DOA"]["final_answer_full_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
