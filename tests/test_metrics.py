import unittest

import numpy as np

from sam2hot.metrics import evaluate_one_sequence_CNT


class MetricsTest(unittest.TestCase):
    def test_identical_boxes(self):
        boxes = np.array([[0, 0, 10, 20], [100, 200, 12, 24]], dtype=float)
        success, precision = evaluate_one_sequence_CNT(boxes, boxes)
        np.testing.assert_array_equal(success, np.ones(21))
        np.testing.assert_array_equal(precision, np.ones(51))

    def test_precision_is_center_not_box_overlap(self):
        gt = np.array([[50, 50, 10, 10]], dtype=float)
        center_preserving = np.array([[30, 30, 50, 50]], dtype=float)
        _, precision = evaluate_one_sequence_CNT(gt, center_preserving)
        self.assertEqual(precision[0], 1.0)
        shifted = center_preserving + [21, 0, 0, 0]
        _, precision = evaluate_one_sequence_CNT(gt, shifted)
        self.assertEqual(precision[20], 0.0)

    def test_incomplete_results_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_one_sequence_CNT(np.ones((2, 4)), np.ones((1, 4)))


if __name__ == "__main__":
    unittest.main()
