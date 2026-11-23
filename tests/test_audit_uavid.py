"""Synthetic fixtures only; these tests do not validate the UAVid download."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.audit_uavid import COLORS, audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "seq1" / "Images"
        self.labels = self.root / "seq1" / "Labels"
        self.images.mkdir(parents=True)
        self.labels.mkdir()

    def pair(self, label_color=COLORS["human"], label_size=(2, 2)):
        Image.new("RGB", (2, 2)).save(self.images / "000000.png")
        Image.new("RGB", label_size, label_color).save(self.labels / "000000.png")

    def test_valid_pair_counts_pixels(self):
        self.pair()
        result = audit(self.root)
        self.assertTrue(result["ok"])
        self.assertEqual(result["class_pixels"]["human"], 4)
        self.assertEqual(result["sequences"][0]["checked_pairs"], 1)

    def test_unknown_color_is_not_clutter(self):
        self.pair(label_color=(1, 2, 3))
        result = audit(self.root)
        self.assertFalse(result["ok"])
        self.assertEqual(result["unknown_pixels"], 4)
        self.assertEqual(result["class_pixels"]["clutter"], 0)

    def test_dimension_mismatch(self):
        self.pair(label_size=(1, 2))
        result = audit(self.root)
        self.assertFalse(result["ok"])
        self.assertEqual(result["sequences"][0]["checked_pairs"], 0)

    def test_missing_label(self):
        Image.new("RGB", (2, 2)).save(self.images / "unpaired.png")
        result = audit(self.root)
        self.assertFalse(result["ok"])
        self.assertEqual(result["sequences"][0]["missing_labels"], ["unpaired.png"])

    def test_empty_root_rejected(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertFalse(audit(empty)["ok"])

    def test_unreadable_image(self):
        self.pair()
        (self.images / "000000.png").write_bytes(b"broken")
        self.assertFalse(audit(self.root)["ok"])


if __name__ == "__main__":
    unittest.main()
