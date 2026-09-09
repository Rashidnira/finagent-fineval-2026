import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module

class Subtask1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load("s1_rules_test", ROOT / "subtask1/scripts/s1_rules.py")

    def test_numeric_entities_are_source_spans(self):
        text = "Στις 15/03/2026 η αξία ήταν 12,5 εκατ. ευρώ και αυξήθηκε κατά 7%."
        entities = self.rules.extract(text)
        self.assertTrue(entities)
        for surface, entity_type in entities:
            self.assertIn(surface, text)
            self.assertTrue(entity_type)

class Subtask2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load("s2_rules_test", ROOT / "subtask2/scripts/s2_rules.py")
        cls.cal = load("s2_calibrate_test", ROOT / "subtask2/scripts/s2_calibrate.py")

    def test_canonical_examples(self):
        cases = {
            "+2": "本日の取締役会で決議しました。来月1日に実施します。",
            "+1": "許認可が得られれば、来年度の着工を目指したいと考えています。",
            "0": "両者の影響を精査しており、現在は分析している段階です。",
            "-1": "現時点では難しいと考えています。ただし環境が変われば再検討します。",
            "-2": "この計画を実施する予定はありません。将来も行いません。",
        }
        for expected, response in cases.items():
            scores = self.rules.score_row("", response)
            self.assertEqual(expected, self.rules.LABELS[int(np.argmax(scores))])

    def test_sinkhorn_is_finite_and_shaped(self):
        result = self.cal.sinkhorn(np.full((50, 5), 0.2), np.full(5, 0.2))
        self.assertEqual((50, 5), result.shape)
        self.assertTrue(np.isfinite(result).all())

if __name__ == "__main__":
    unittest.main()
