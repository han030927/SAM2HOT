"""CPU-only training smoke tests; no private data or pretrained weights required."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from sam2hot.config import ROOT, load_config
from sam2hot.motion import MotionGateMLP, load_motion
from sam2hot.sdmr import SDMR
from train_motion import center_objective, main, seed_training


torch.set_num_threads(1)


def make_config(root):
    data, scores = root / "data", root / "scores"
    scores.mkdir()
    for modality in ("VIS", "NIR", "RedNIR"):
        for index in range(2):
            name = f"synthetic{index}"
            folder = data / f"HSI-{modality}" / name
            folder.mkdir(parents=True)
            rows = [f"{10 + frame * .5} {15 + frame * .15} 12 8" for frame in range(12)]
            (folder / "groundtruth_rect.txt").write_text("\n".join(rows), encoding="utf-8")
            path = scores / f"{modality.lower()}-score-{name}.txt"
            path.write_text("\n".join(row + " 0.9" for row in rows), encoding="utf-8")
            path.with_suffix(".meta.json").write_text(
                json.dumps({"frame_width": 100, "frame_height": 80}), encoding="utf-8")
    config = yaml.safe_load((ROOT / "configs/hot20.yaml").read_text(encoding="utf-8"))
    config["device"] = "cpu"
    config["training"].update(data_root=str(data), score_root=str(scores),
                              output_dir=str(root / "run"), epochs=2,
                              batch_size=4, samples_per_epoch=16, val_samples=8)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class TrainingTests(unittest.TestCase):
    def test_center_loss_and_gradients(self):
        prediction = torch.tensor([[.6, .6]], requires_grad=True)
        target = torch.tensor([[.5, .5, .1, .2]])
        loss, error = center_objective(prediction, target)
        self.assertAlmostEqual(float(loss), .3125, places=5)
        self.assertAlmostEqual(float(error), 1.0, places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0)

    def test_inference_config_still_accepts_no_training_section(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = yaml.safe_load((ROOT / "configs/hot20.yaml").read_text(encoding="utf-8"))
            config.pop("training")
            path = Path(temporary) / "inference.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            self.assertNotIn("training", load_config(path))

    def test_check_is_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_config(root)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--config", str(config), "--check"])
            self.assertFalse((root / "run").exists())

    def test_train_save_reload_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_config(root)
            seed_training(123)
            initial = {key: value.clone() for key, value in MotionGateMLP().state_dict().items()}
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--config", str(config)])
            best = root / "run/best.pth"
            payload = torch.load(best, map_location="cpu", weights_only=False)
            self.assertEqual(payload["loss_name"], "smooth_l1_center")
            self.assertTrue(any(not torch.equal(initial[key], value)
                                for key, value in payload["motion_gate"].items()))
            metadata = {key: value for key, value in payload.items() if key != "motion_gate"}
            self.assertNotIn("direction", json.dumps(metadata).lower())
            loaded = load_motion(best, "cpu")
            for key, value in loaded.model.state_dict().items():
                torch.testing.assert_close(value, payload["motion_gate"][key], atol=0, rtol=0)
            runtime = SDMR(loaded, [10., 15., 12., 8.], 100, 80)
            for index in range(7):
                box, _ = runtime.step([10. + index * .5, 15., 12., 8.], .2, .9, False)
                self.assertTrue(np.isfinite(box).all())
            lines = (root / "run/history.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue((root / "run/last.pth").is_file())
            split = json.loads((root / "run/split.json").read_text(encoding="utf-8"))
            train = {(item["modality"], item["name"]) for item in split["train"]}
            validation = {(item["modality"], item["name"]) for item in split["validation"]}
            self.assertFalse(train & validation)
            before = best.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                main(["--config", str(config)])
            self.assertEqual(before, best.read_bytes())
            # Fresh deterministic training should select the same model on CPU.
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--config", str(config), "--output-dir", str(root / "repeat")])
            repeated = torch.load(root / "repeat/best.pth", map_location="cpu", weights_only=False)
            self.assertEqual(repeated["epoch"], payload["epoch"])
            for key, value in repeated["motion_gate"].items():
                torch.testing.assert_close(value, payload["motion_gate"][key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
