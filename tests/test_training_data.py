import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from sam2hot.training_data import MotionDataset, MotionRecord, load_records, split_records


def make_record(name="example", modality="vis", count=18):
    pred = np.tile(np.asarray([10, 20, 20, 10], dtype=np.float32), (count, 1))
    pred[:, 0] += np.arange(count, dtype=np.float32)
    return MotionRecord(
        source_name="HOTC2023", modality=modality, name=name, sequence_id=name,
        score_path="", gt_path="", width=100, height=50,
        pred_xywh=pred, pred_score=np.full(count, 0.5, dtype=np.float32),
        gt_xywh=pred + np.asarray([2, 1, 0, 0], dtype=np.float32),
    )


class TrainingDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.scores = self.root / "scores"
        self.data.mkdir()
        self.scores.mkdir()

    def export(self, name="example", modality="vis", *, metadata=True):
        record = make_record(name=name, modality=modality)
        hsi = {"vis": "HSI-VIS", "nir": "HSI-NIR", "rednir": "HSI-RedNIR"}[modality]
        folder = self.data / hsi / name
        folder.mkdir(parents=True)
        gt_path = folder / "groundtruth_rect.txt"
        score_path = self.scores / f"{modality}-score-{name}.txt"
        np.savetxt(gt_path, record.gt_xywh, delimiter=",")
        np.savetxt(score_path, np.column_stack((record.pred_xywh, record.pred_score)))
        if metadata:
            score_path.with_suffix(".meta.json").write_text(json.dumps({
                "frame_width": 100, "frame_height": 50,
            }), encoding="utf-8")
        return record, gt_path, score_path

    def test_load_modalities_exclusion_and_image_dimensions(self):
        self.export("ball")
        for modality in ("vis", "nir", "rednir"):
            self.export("example", modality)
        records = load_records(self.data, self.scores)
        self.assertEqual([record.modality for record in records], ["nir", "rednir", "vis"])
        self.assertEqual({record.name for record in records}, {"example"})
        self.assertEqual((records[0].width, records[0].height), (100, 50))

    def test_relative_metadata_paths_and_identity_exclusion(self):
        record, gt_path, score_path = self.export()
        alternate = self.data / "metadata-target.txt"
        gt_path.rename(alternate)
        meta = {"gt_path": "Z:/missing/old-machine.txt", "gt_path_relative": alternate.name,
                "frame_width": 100, "frame_height": 50, "sequence_id": "renamed-identity"}
        score_path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
        loaded = load_records(self.data, self.scores)[0]
        self.assertEqual(loaded.sequence_id, "renamed-identity")
        np.testing.assert_array_equal(loaded.gt_xywh, record.gt_xywh)
        meta["sequence_id"] = "BALL"
        score_path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "HOTC2020 exclusion"):
            load_records(self.data, self.scores)

    def test_falsecolor_image_size_and_no_box_size_guess(self):
        self.export(metadata=False)
        with self.assertRaisesRegex(FileNotFoundError, "tracking image dimensions"):
            load_records(self.data, self.scores)
        frames = self.data / "HSI-VIS-FalseColor" / "example"
        frames.mkdir(parents=True)
        Image.new("RGB", (80, 40)).save(frames / "0001.png")
        record = load_records(self.data, self.scores)[0]
        self.assertEqual((record.width, record.height), (80, 40))

    def test_prediction_scores_required_and_duplicates_rejected(self):
        record, _, score_path = self.export()
        np.savetxt(score_path, record.pred_xywh)
        with self.assertRaisesRegex(ValueError, "5 columns"):
            load_records(self.data, self.scores)
        np.savetxt(score_path, np.column_stack((record.pred_xywh, record.pred_score)))
        duplicate_dir = self.scores / "duplicate"
        duplicate_dir.mkdir()
        (duplicate_dir / score_path.name).write_text(score_path.read_text(), encoding="utf-8")
        (duplicate_dir / score_path.with_suffix(".meta.json").name).write_text(
            score_path.with_suffix(".meta.json").read_text(), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_records(self.data, self.scores)

    def test_car62_end_alignment_and_other_start_alignment(self):
        for name in ("car62", "example"):
            record, _, score_path = self.export(name)
            shortened = np.column_stack((record.pred_xywh[:12], record.pred_score[:12]))
            np.savetxt(score_path, shortened)
        with self.assertWarns(UserWarning):
            records = load_records(self.data, self.scores)
        by_name = {record.name: record for record in records}
        np.testing.assert_array_equal(by_name["car62"].gt_xywh, make_record().gt_xywh[-12:])
        np.testing.assert_array_equal(by_name["example"].gt_xywh, make_record().gt_xywh[:12])

    def test_normalized_windows_and_next_frame_gt(self):
        record = make_record()
        sample = MotionDataset([record], training=False)[0]
        self.assertEqual(sample["history_boxes"].shape, (5, 4))
        self.assertEqual(sample["history_scores"].shape, (5, 1))
        self.assertEqual(sample["history_valid"].shape, (5, 1))
        self.assertEqual(sample["target_box"].shape, (4,))
        np.testing.assert_allclose(sample["history_boxes"][0], [0.2, 0.5, 0.2, 0.2])
        np.testing.assert_allclose(sample["target_box"], [0.27, 0.52, 0.2, 0.2])
        for tensor in sample.values():
            self.assertEqual(tensor.dtype, torch.float32)

    def test_all_five_history_frames_must_be_valid_but_low_scores_survive(self):
        record = make_record()
        record.pred_score[0] = -2.0
        dataset = MotionDataset([record], training=False)
        self.assertEqual(dataset[0]["history_scores"][0].item(), -2.0)
        record.pred_xywh[2, 2] = 1.0
        dataset = MotionDataset([record], training=False)
        self.assertEqual(dataset.samples[0], (0, 8))
        record.pred_score[8] = np.nan
        record.gt_xywh[14, 2] = 0.0
        dataset = MotionDataset([record], training=False)
        self.assertTrue(all(target not in {9, 10, 11, 12, 13, 14}
                            for _, target in dataset.samples))

    def test_fixed_validation_subsample_matches_reference_seed(self):
        records = [make_record()]
        all_samples = MotionDataset(records, training=False).samples
        expected = np.sort(np.random.default_rng(123 + 104729).choice(
            len(all_samples), size=4, replace=False))
        dataset = MotionDataset(records, sample_count=4, training=False, seed=123)
        self.assertEqual(dataset.samples, [all_samples[index] for index in expected])
        self.assertEqual(len(dataset), 4)
        torch.testing.assert_close(dataset[0]["history_boxes"], dataset[0]["history_boxes"])
        self.assertEqual(len(MotionDataset(records, sample_count=200, training=False)), 13)

    def test_training_samples_with_replacement_and_seeded_rng(self):
        dataset = MotionDataset([make_record()], sample_count=50, training=True)
        self.assertEqual(len(dataset), 50)
        random.seed(123)
        first = [dataset[index]["history_boxes"] for index in range(10)]
        random.seed(123)
        second = [dataset[index]["history_boxes"] for index in range(10)]
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right)

    def test_stratified_split_matches_reference(self):
        records = [make_record(f"sequence{index:02d}", modality)
                   for modality in ("vis", "nir", "rednir") for index in range(10)]
        train, validation = split_records(records)
        self.assertEqual((len(train), len(validation)), (27, 3))
        expected = []
        for group_index, modality in enumerate(("nir", "rednir", "vis")):
            indices = list(range(10))
            random.Random(123 + group_index * 1009).shuffle(indices)
            expected.append((modality, f"sequence{indices[0]:02d}"))
        self.assertEqual([(record.modality, record.name) for record in validation], expected)
        self.assertFalse({(record.modality, record.name) for record in train} & set(expected))

    def test_tiny_split_fails_without_train_validation_overlap(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            split_records([make_record(modality="vis"), make_record(modality="nir")])
        with self.assertRaisesRegex(ValueError, "val_ratio"):
            split_records([make_record()], val_ratio=0)

    def test_invalid_record_or_no_windows_is_rejected(self):
        record = make_record(count=5)
        with self.assertRaisesRegex(ValueError, "No valid"):
            MotionDataset([record])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            MotionDataset([replace(record, width=0)])
        with self.assertRaisesRegex(ValueError, "Unaligned"):
            MotionDataset([replace(record, pred_score=np.zeros(3))])


if __name__ == "__main__":
    unittest.main()
