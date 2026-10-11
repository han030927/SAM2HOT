"""CPU-only tests, including optional exact comparison to the original project.

Run: python -m unittest discover -s tests -p test_sdmr.py -v

For all 35 archived HOT20 observation sequences, additionally set:
  SAM2HOT_REFERENCE_PROJECT: original SENTRY directory
  SAM2HOT_REFERENCE_RUNTIME: original offline_test_center_point_mlp_v1.py
  SAM2HOT_REFERENCE_CORE: original offline_hot23_multimodal_motion_ablation_v2_3.py
No absolute machine-specific dependency is required for the regular tests.
"""
from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from sam2hot.motion import LoadedMotion, MotionGateMLP, load_motion
from sam2hot.sdmr import SDMR, clip_box


torch.set_num_threads(1)


def make_loaded():
    torch.manual_seed(123)
    return LoadedMotion(MotionGateMLP().eval().requires_grad_(False), torch.device("cpu"))


class SDMRTests(unittest.TestCase):
    def test_model_size_and_image_relative_encoding(self):
        model = make_loaded().model
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 20184)
        history = torch.tensor([[[.30 + i * .01, .25 + i * .005, .1, .1]
                                 for i in range(5)]])
        altered = history.clone()
        altered[..., 2:] *= 2.0
        with torch.no_grad():
            first, second = model(history), model(altered)
        for key in ("pred_center", "center_delta", "motion_coefficients", "temporal_weights_dynamic"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_empty_quality_stays_finite(self):
        model = make_loaded().model
        boxes = torch.zeros((2, 5, 4))
        boxes[..., :2], boxes[..., 2:] = .5, .1
        with torch.no_grad():
            result = model(boxes, torch.zeros((2, 5, 1)), torch.zeros((2, 5, 1)))
        self.assertTrue(torch.isfinite(result["pred_center"]).all())
        for order, length in enumerate((4, 3, 2)):
            weights = result["temporal_weights_dynamic"][:, order, :, -length:]
            torch.testing.assert_close(weights.sum(-1), torch.ones((2, 2)))

    def test_history_warmup_and_empty_forward_fill(self):
        runtime = SDMR(make_loaded(), [10, 10, 20, 20], 200, 100)
        for index in range(1, 5):
            raw = np.asarray([10 + index, 10, 20, 20], dtype=float)
            final, debug = runtime.step(raw, 0.2, 0.9, False)
            np.testing.assert_array_equal(final, raw)
            self.assertFalse(debug["ready"])
            self.assertEqual(runtime.valid[-1], 1.0)  # Low probability is not invalid.
        prior = runtime.last_valid.copy()
        final, debug = runtime.step([0, 0, 0, 0], 0.0, 0.1, True)
        self.assertTrue(debug["ready"])
        self.assertTrue(debug["applied"])
        self.assertTrue(np.all(np.isfinite(final)))
        np.testing.assert_array_equal(runtime.history[-1], prior)
        self.assertEqual(runtime.valid[-1], 0.0)
        self.assertEqual(runtime.scores[-1], 0.1)

    def test_gate_boundary_size_update_and_raw_history(self):
        runtime = SDMR(make_loaded(), [10, 10, 20, 20], 200, 100)
        # Supply an exact prior to isolate gate and scale semantics from weights.
        runtime._predict = lambda: {"center_px": np.asarray([100., 50.])}
        raw = np.asarray([15., 10., 24., 24.])
        final, debug = runtime.step(raw, .9, .75, False)
        np.testing.assert_array_equal(final, raw)
        self.assertFalse(debug["scheduled"])
        self.assertTrue(debug["size_updated"])
        expected = np.exp(.8 * np.log(20.) + .2 * np.log(24.))
        np.testing.assert_array_equal(debug["stable_size"], [expected, expected])
        raw = np.asarray([16., 10., 24., 24.])
        final, debug = runtime.step(raw, .89999999, .75, False)
        self.assertTrue(debug["applied"])
        np.testing.assert_array_equal(runtime.history[-1], runtime._normalise(raw))
        self.assertNotEqual(float(final[0]), float(raw[0]))
        previous = runtime.log_size.copy()
        runtime.step([16., 10., 100., 100.], .99, .99, False)
        np.testing.assert_array_equal(runtime.log_size, previous)  # Ratio rejection.

    def test_clipping_and_invalid_prior_fall_back_to_raw(self):
        np.testing.assert_array_equal(clip_box([-5, -4, 10, 8], 100, 50), [0, 0, 5, 4])
        np.testing.assert_array_equal(clip_box([101, 0, 10, 10], 100, 50), np.zeros(4))
        runtime = SDMR(make_loaded(), [10, 10, 20, 20], 100, 50)
        runtime._predict = lambda: {"center_px": np.asarray([1000., 1000.])}
        raw = np.asarray([4., 5., 6., 7.])
        final, debug = runtime.step(raw, .2, .1, False)
        np.testing.assert_array_equal(final, raw)
        self.assertTrue(debug["scheduled"])
        self.assertFalse(debug["applied"])

    def test_checkpoint_strict_loading_and_reject_wrong_variant(self):
        model = make_loaded().model
        payload = {"motion_gate": model.state_dict(), "temporal_window": 5,
                   "motion_order": 3, "args": {"hidden_dim": 64, "num_layers": 3}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pth"
            torch.save(payload, path)
            loaded = load_motion(path, "cpu")
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.model.state_dict()[key], rtol=0, atol=0)
            self.assertFalse(loaded.model.training)
            self.assertFalse(any(parameter.requires_grad for parameter in loaded.model.parameters()))
            payload["temporal_window"] = 4
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_motion(path, "cpu")


def load_reference(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


_REFERENCE_VARIABLES = ("SAM2HOT_REFERENCE_PROJECT", "SAM2HOT_REFERENCE_RUNTIME", "SAM2HOT_REFERENCE_CORE")


@unittest.skipUnless(all(os.environ.get(name) for name in _REFERENCE_VARIABLES),
                     "Optional original-project references not configured")
class OriginalParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project = Path(os.environ["SAM2HOT_REFERENCE_PROJECT"])
        cls.runtime = load_reference(Path(os.environ["SAM2HOT_REFERENCE_RUNTIME"]), "_reference_sdmr_runtime")
        cls.core = load_reference(Path(os.environ["SAM2HOT_REFERENCE_CORE"]), "_reference_sdmr_core")
        model_path = project / "baseline/SAMHOTD/sam2/samhsi_kalman/motion_gate_model_center_point_image_norm_v2.py"
        original_model = load_reference(model_path, "_reference_motion_v2")
        checkpoint = project / "checkpoints/motion_center_point_image_norm_hot23_all_l5_o3_v2_0/best.pth"
        cls.original = cls.runtime.LoadedCenterPointModel(original_model, checkpoint, "cpu")
        cls.clean = load_motion(checkpoint, "cpu")
        cls.observations = project / ("outputs/hot/all_methods_eta0p7_best_full_with_observations/"
                                     "DAM4SAM/HOT2023/VIS/topk0.9_lambda0.6_eta0.7_applyboth/observations")

    def test_model_outputs_exactly_match_original(self):
        torch.manual_seed(42)
        boxes = torch.rand((16, 5, 4))
        scores, valid = torch.rand((16, 5, 1)), torch.randint(0, 2, (16, 5, 1))
        valid[0] = 0
        with torch.no_grad():
            original = self.original.model(boxes, history_scores=scores, history_valid=valid)
            clean = self.clean.model(boxes, history_scores=scores, history_valid=valid)
        for key, value in clean.items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0, msg=key)

    def test_all_35_observation_sequences_exactly_match_original(self):
        total_frames = 0
        for name in self.runtime.HOT20_SEQUENCES:
            with self.subTest(sequence=name):
                with (self.observations / f"vis-{name}_observations.csv").open(encoding="utf-8-sig", newline="") as stream:
                    csv_rows = list(csv.DictReader(stream))
                raw = np.asarray([[float(row[f"raw_{axis}"]) for axis in "xywh"] for row in csv_rows])
                rows = [{"mask_empty": int(row["mask_empty"]), "object_prob": float(row["object_prob"]),
                         "sam_pred_iou": float(row["sam_pred_iou"])} for row in csv_rows]
                width, height = float(csv_rows[0]["image_width"]), float(csv_rows[0]["image_height"])
                sequence = SimpleNamespace(raw=raw, gt=raw, rows=rows, image_width=width, image_height=height)
                centers, ready, _ = self.runtime.replay_sequence(sequence, self.original, self.core)
                sizes, size_updated = self.core.stable_sizes(sequence, .2)
                expected, applied = self.core.final_predictions(sequence, centers, ready, sizes, .9, False)
                fresh = SDMR(self.clean, raw[0], width, height)
                output = [fresh.initial_bbox]
                for index in range(1, len(raw)):
                    row = rows[index]
                    final, debug = fresh.step(raw[index], row["object_prob"], row["sam_pred_iou"], row["mask_empty"])
                    output.append(final)
                    self.assertEqual(debug["applied"], bool(applied[index]))
                    self.assertEqual(debug["ready"], bool(ready[index]))
                    self.assertEqual(debug["size_updated"], bool(size_updated[index]))
                    np.testing.assert_array_equal(debug["stable_size"], sizes[index])
                    if ready[index]:
                        np.testing.assert_array_equal(debug["center_px"], centers[index])
                np.testing.assert_array_equal(np.asarray(output), expected)
                total_frames += len(raw)
        self.assertEqual(total_frames, 16574)


if __name__ == "__main__":
    unittest.main()
