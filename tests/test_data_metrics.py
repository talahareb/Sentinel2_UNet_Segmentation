"""No-TensorFlow unit tests for data handling and binary segmentation metrics."""
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel_unet.data import load_dataset
from sentinel_unet.metrics import evaluate_masks


class TestDataset(unittest.TestCase):
    def test_load_three_way_and_original_split(self):
        with tempfile.TemporaryDirectory() as t:
            file = Path(t) / "sentinel2.npz"
            x = np.ones((50, 10, 10, 12), dtype=np.float32)
            x[0] = 2
            y = np.zeros((50, 10, 10, 1), dtype=np.float32)
            y[:, :4, :4] = 1
            np.savez(file, X=x, Y=y)
            s = load_dataset(file, mode="three-way", pad=1)
            self.assertEqual((len(s.x_train), len(s.x_val), len(s.x_test)), (35, 5, 10))
            self.assertEqual(s.x_train.shape, (35, 12, 12, 12))
            self.assertAlmostEqual(float(s.x_train.max()), 1)
            orig = load_dataset(file, mode="original", pad=1)
            self.assertEqual((len(orig.x_train), len(orig.x_val), len(orig.x_test)), (40, 10, 10))
            np.testing.assert_array_equal(orig.x_val, orig.x_test)

    def test_invalid_masks(self):
        with tempfile.TemporaryDirectory() as t:
            file = Path(t) / "sentinel2.npz"
            np.savez(file, X=np.ones((50, 8, 8, 12)), Y=np.ones((50, 8, 8))*2)
            with self.assertRaises(ValueError):
                load_dataset(file, pad=0)


class TestMetrics(unittest.TestCase):
    def test_counts_and_iou_dice(self):
        true = np.array([1, 1, 0, 0], dtype=np.float32).reshape(1, 2, 2, 1)
        pred = np.array([.9, .2, .8, .1], dtype=np.float32).reshape(1, 2, 2, 1)
        m = evaluate_masks(true, pred)
        self.assertEqual((m['true_positive'], m['false_positive'], m['false_negative'], m['true_negative']), (1, 1, 1, 1))
        self.assertEqual(m['foreground_iou'], 1/3)
        self.assertEqual(m['foreground_dice'], .5)
        self.assertEqual(m['pixel_accuracy'], .5)

    def test_crop_excludes_padding(self):
        true = np.zeros((1, 4, 4, 1))
        pred = np.ones_like(true)
        pred[:, 1:-1, 1:-1] = 0
        self.assertEqual(evaluate_masks(true, pred, pad=1)["pixel_accuracy"], 1.0)
        self.assertLess(evaluate_masks(true, pred, pad=0)["pixel_accuracy"], 1.0)

if __name__ == '__main__':
    unittest.main()
