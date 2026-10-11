import unittest

import numpy as np

from run_hot20 import mask_to_box, quantize, summarize


class RunnerTests(unittest.TestCase):
    def test_mask_box_inclusive_pixel_extent(self):
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[2:5, 4:9] = 1
        np.testing.assert_array_equal(mask_to_box(mask), [4, 2, 5, 3])
        np.testing.assert_array_equal(mask_to_box(np.zeros_like(mask)), [0, 0, 0, 0])

    def test_preserves_saved_observation_precision(self):
        self.assertEqual(quantize(0.999962451234, 8), 0.99996245)
        self.assertEqual(quantize(4.12345678, 6), 4.123457)
        with self.assertRaises(ValueError):
            quantize(float("nan"), 8)

    def test_partial_mean_is_not_complete_benchmark(self):
        records = [{"raw": {"auc": 0.6, "dp20": 0.8},
                    "final": {"auc": 0.7, "dp20": 0.9}}]
        self.assertEqual(summarize(records, [], 35)["status"], "incomplete")
        self.assertEqual(summarize(records, [], 1)["status"], "completed")


if __name__ == "__main__":
    unittest.main()
