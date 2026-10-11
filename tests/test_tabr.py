"""CPU tests for the fixed HOT20 input path; no checkpoint or GPU required."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from sam2hot.data import Sequence, mosaic_to_cube
from sam2hot.tabr import TABRConfig, aggregate, build_groups, first_frame_sam_mask, fit_tabr


class TABRTests(unittest.TestCase):
    def test_vis_mosaic_order_and_normalization(self):
        cube = np.arange(3 * 5 * 16, dtype=np.uint16).reshape(3, 5, 16)
        raw = np.empty((12, 20), dtype=np.uint16)
        for band in range(16):
            raw[band // 4::4, band % 4::4] = cube[:, :, band]
        low, high = cube.min((0, 1), keepdims=True), cube.max((0, 1), keepdims=True)
        expected = ((cube - low) / (high - low + 1e-6) * 255).astype(np.uint8)
        np.testing.assert_array_equal(mosaic_to_cube(raw), expected)
        with self.assertRaises(ValueError):
            mosaic_to_cube(np.zeros((3, 4), dtype=np.uint16))

    def test_groupwise_rounding_and_reversed_channels(self):
        scores = np.arange(16, dtype=np.float32)
        groups, ranked = build_groups(scores, scores)
        # 16 bands split 6/5/5: Python round(.9*5)=4, not ceil(4.5).
        self.assertEqual([group[0] for group in groups], [[12, 13, 14, 15], [7, 8, 9, 10], [1, 2, 3, 4, 5]])
        self.assertEqual(len(ranked), 13)
        cube = np.broadcast_to(np.arange(16, dtype=np.uint8), (3, 4, 16))
        projected = aggregate(cube, groups)
        self.assertEqual(projected.shape, (3, 4, 3))
        self.assertTrue(np.all(projected[..., 0] > projected[..., 1]))
        self.assertTrue(np.all(projected[..., 1] > projected[..., 2]))

    def test_exact_target_mask_and_full_background(self):
        rng = np.random.default_rng(16)
        cube = rng.integers(0, 256, (12, 15, 16), dtype=np.uint8)
        mask = np.zeros((12, 15), dtype=bool)
        mask[2:5, 3:8] = True
        fitted = fit_tabr(cube, mask)
        self.assertEqual(fitted["foreground_pixels"], 15)
        self.assertEqual(fitted["background_pixels"], 165)
        self.assertEqual(fitted["retained_bands"], 13)
        self.assertEqual(fitted["config"]["temperature"], 1.0)
        for invalid in (np.zeros_like(mask), np.ones_like(mask), np.eye(12, 15) == 7):
            with self.assertRaises(ValueError):
                fit_tabr(cube, invalid)
        one = np.zeros_like(mask)
        one[0, 0] = True
        with self.assertRaises(ValueError):
            fit_tabr(cube, one)

    def test_mapping_profile(self):
        self.assertEqual(TABRConfig.from_mapping({"lambda": 0.6, "temperature": 1.0}), TABRConfig())
        with self.assertRaises(ValueError):
            TABRConfig.from_mapping({"other_modality": "nir"})

    def test_first_prompt_is_original_fci_with_rounded_xyxy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original" / "0001.jpg"
            original.parent.mkdir()
            Image.fromarray(np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)).save(original)
            seq = Sequence("ball", original.parent, root / "HSI", root / "gt.txt", [original], [],
                           np.array([[1.4, 2.6, 4.2, 3.1]]))
            class Predictor:
                def init_state(self, folder, **kwargs):
                    self.kwargs = kwargs
                    self.files = list(Path(folder).glob("*.jpg"))
                    return {}
                def add_new_points_or_box(self, state, **kwargs):
                    self.prompt = kwargs
                    return 0, [0], np.array([[[[-1., 2.], [3., -4.]]]], dtype=np.float32)
            predictor = Predictor()
            mask = first_frame_sam_mask(seq, predictor, root / "output", (8, 12), "cpu")
            self.assertEqual(predictor.prompt, {"box": (1, 3, 6, 6), "frame_idx": 0, "obj_id": 0})
            self.assertEqual(predictor.kwargs, {"offload_video_to_cpu": True, "offload_state_to_cpu": True,
                                               "async_loading_frames": True})
            self.assertEqual(len(predictor.files), 1)
            self.assertEqual(original.read_bytes(), predictor.files[0].read_bytes())
            expected = cv2.resize(np.array([[-1., 2.], [3., -4.]], dtype=np.float32),
                                  (12, 8), interpolation=cv2.INTER_LINEAR) > 0
            np.testing.assert_array_equal(mask, expected)


if __name__ == "__main__":
    unittest.main()
