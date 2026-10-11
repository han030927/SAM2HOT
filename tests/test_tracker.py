"""CPU checks for streaming state and native DAM4SAM DRM selection."""
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from sam2hot.tracker import SAMTracker


class FakePredictor:
    """Supplies native-format SAM outputs without loading SAM weights."""

    def __init__(self):
        self.drm_frames = []
        self.prepared_images = []

    def reset_state(self, state):
        pass

    def _get_image_feature(self, state, frame_idx, batch_size):
        return None, None, [torch.arange(16).reshape(4, 1, 4).float()], None, [(2, 2)]

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        self.prepared_images.append(inference_state["images"][0].clone())
        return 0, [0], torch.tensor(mask[None, None], dtype=torch.float32) * 2 - 1

    def propagate_in_video(self, state, start_frame_idx, **kwargs):
        index = start_frame_idx
        self.prepared_images.append(state["images"][index].clone())
        mask = np.zeros((36, 48), dtype=np.uint8)
        if index != 17:
            x = 4 + index % 3
            mask[8:14, x:x + (2 if index == 23 else 6)] = 1
        alternative = mask.copy()
        alternative[20:26, 32:38] = 1
        distractor = np.zeros_like(mask)
        distractor[20:26, 32:38] = 1
        logits = torch.tensor(mask[None, None], dtype=torch.float32) * 2 - 1
        alternatives = [torch.tensor(value[None, None], dtype=torch.float32) * 2 - 1
                        for value in (alternative, mask, distractor)]
        scores = np.asarray([0.2, 0.6 if index == 20 else 0.9, 0.3])
        state["output_dict"]["non_cond_frame_outputs"][index] = {
            "object_score_logits": torch.tensor([[index * 0.1 - 1.0]])
        }
        yield index, [0], logits, (alternatives, scores)

    def add_to_drm(self, inference_state, frame_idx, obj_id):
        self.drm_frames.append(frame_idx)


def sample_image():
    values = np.arange(36 * 48 * 3, dtype=np.uint32).reshape(36, 48, 3)
    return Image.fromarray((values % 256).astype(np.uint8))


def initial_mask():
    mask = np.zeros((36, 48), dtype=np.uint8)
    mask[8:14, 4:10] = 1
    return mask


class TrackerTests(unittest.TestCase):
    @patch("torch.cuda.is_available", return_value=False)
    def test_streaming_score_and_drm(self, _):
        predictor = FakePredictor()
        tracker = SAMTracker("unused-in-test.pt", device="cpu", predictor=predictor)
        tracker.input_image_size = 32
        output = tracker.initialize(sample_image(), init_mask=initial_mask())
        self.assertEqual(output["observation_debug"]["object_prob"], 1.0)
        self.assertEqual(tracker.inference_state["images"], {})
        for index in range(1, 34):
            output = tracker.track(sample_image())
            observation = output["observation_debug"]
            self.assertEqual(output["frame_idx"], index)
            self.assertEqual(observation["selected_idx"], 1)
            self.assertEqual(observation["mask_empty"], index == 17)
            self.assertAlmostEqual(observation["object_logit"], index * 0.1 - 1.0, places=6)
            self.assertEqual(tracker.inference_state["images"], {})
        self.assertEqual(predictor.drm_frames, [2, 8, 14, 21, 27, 33])
        self.assertEqual(predictor.prepared_images[0].shape, (3, 32, 32))
        self.assertEqual(predictor.prepared_images[1].shape, (1, 3, 32, 32))

    @patch("torch.cuda.is_available", return_value=False)
    def test_missing_object_score_does_not_silently_substitute_iou(self, _):
        tracker = SAMTracker("unused-in-test.pt", device="cpu", predictor=FakePredictor())
        tracker.input_image_size = 32
        tracker.initialize(sample_image(), init_mask=initial_mask())
        tracker.frame_index = 1
        with self.assertRaisesRegex(RuntimeError, "object confidence is absent"):
            tracker._object_score()


if __name__ == "__main__":
    unittest.main()
