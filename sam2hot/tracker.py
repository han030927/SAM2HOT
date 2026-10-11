"""Streaming DAM4SAM, retaining native RAM/DRM and read-only SAM2 scores.

Adapted from the user's SENTRY integration of DAM4SAM. The native SAM2/DAM4SAM
model is not modified. Experimental reranking and visualization branches are
removed; recovered SDMR states are never written into the visual tracker.
See third_party/licenses/SENTRY-NOTICE.txt for upstream licensing restrictions.
"""

from collections import OrderedDict
from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

from sam2hot._vendor import activate

activate()

from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.transforms import SAM2Transforms


def _load_vot_region_helpers():
    """Retain the original adapter's VOT-compatible DRM box operations."""
    try:
        from vot.region import RegionType
        from vot.region.raster import calculate_overlaps
        from vot.region.shapes import Mask
    except ImportError:
        class RegionType:
            RECTANGLE = "rectangle"

        class Mask:
            def __init__(self, data):
                self.data = np.asarray(data, dtype=np.uint8)

            def convert(self, region_type):
                if region_type != RegionType.RECTANGLE:
                    raise ValueError(f"Unsupported region type: {region_type}")
                rows, cols = np.nonzero(self.data)
                if rows.size == 0:
                    return [0.0, 0.0, 0.0, 0.0]
                x1, x2 = float(cols.min()), float(cols.max() + 1)
                y1, y2 = float(rows.min()), float(rows.max() + 1)
                return [x1, y1, x2 - x1, y2 - y1]

        def calculate_overlaps(first, second):
            overlaps = []
            for left, right in zip(first, second):
                lx, ly, lw, lh = (float(value) for value in left)
                rx, ry, rw, rh = (float(value) for value in right)
                width = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
                height = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
                intersection = width * height
                union = lw * lh + rw * rh - intersection
                overlaps.append(intersection / union if union > 0 else 0.0)
            return np.asarray(overlaps, dtype=np.float32)
    return calculate_overlaps, Mask, RegionType


def _keep_largest_component(mask):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    result = np.zeros_like(mask)
    result[labels == largest] = 1
    return result


class SAMTracker:
    """The single supported tracker: DAM4SAM with SAM2.1 Hiera Large."""

    def __init__(self, checkpoint, model_cfg="sam21pp_hiera_l.yaml",
                 device="cuda:0", predictor=None, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if model_cfg != "sam21pp_hiera_l.yaml":
            raise ValueError("This repository supports only DAM4SAM SAM2.1 Hiera Large.")
        if predictor is None and not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        self.checkpoint = str(checkpoint)
        self.model_cfg = model_cfg
        self.device = device
        self.input_image_size = 1024
        self.img_mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.img_std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        self.predictor = predictor or build_sam2_video_predictor(
            model_cfg, self.checkpoint, device=device
        )

    def _prepare_image(self, image):
        value = torch.from_numpy(np.array(image)).to(self.inference_state["device"])
        value = value.permute(2, 0, 1).float() / 255.0
        value = F.resize(value, (self.input_image_size, self.input_image_size))
        return (value - self.img_mean) / self.img_std

    def init_state_tw(self):
        device = torch.device(self.device)
        self.img_mean = self.img_mean.to(device)
        self.img_std = self.img_std.to(device)
        return {
            "images": None,
            "num_frames": 0,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": None,
            "video_width": None,
            "device": device,
            "storage_device": device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "adds_in_drm_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {
                "cond_frame_outputs": set(), "non_cond_frame_outputs": set()
            },
            "tracking_has_started": False,
            "frames_already_tracked": {},
            "frames_tracked_per_obj": {},
        }

    @torch.inference_mode()
    def initialize(self, image, init_mask=None, bbox=None):
        if isinstance(init_mask, list):
            init_mask = init_mask[0]
        if init_mask is None and bbox is None:
            raise ValueError("Initialization requires a bounding box or mask.")
        self.frame_index = 0
        self.object_sizes = []
        self.last_added = -1
        self.img_width, self.img_height = image.size
        self.inference_state = self.init_state_tw()
        self.inference_state["video_height"] = self.img_height
        self.inference_state["video_width"] = self.img_width
        self.inference_state["images"] = {0: self._prepare_image(image)}
        self.inference_state["num_frames"] = 1
        self.predictor.reset_state(self.inference_state)
        self.predictor._get_image_feature(self.inference_state, frame_idx=0, batch_size=1)
        if init_mask is None:
            init_mask = self.estimate_mask_from_box(bbox)
        _, _, logits = self.predictor.add_new_mask(
            inference_state=self.inference_state, frame_idx=0, obj_id=0, mask=init_mask
        )
        mask = (logits[0, 0] > 0).float().cpu().numpy().astype(np.uint8)
        self.inference_state["images"].pop(0)
        return {
            "pred_mask": mask,
            "frame_idx": 0,
            "observation_debug": {
                "object_prob": 1.0,
                "object_logit": None,
                "sam_iou_score": 1.0,
                "selected_idx": None,
                "sam_ious": None,
                "mask_area": int(np.count_nonzero(mask)),
                "mask_empty": bool(not np.any(mask)),
                "score_type": "selected_sam_pred_iou",
            },
        }

    def _candidate_mask_to_original(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().float()
        else:
            value = torch.as_tensor(np.asarray(value), dtype=torch.float32)
        while value.ndim > 2:
            value = value[0]
        if value.ndim != 2:
            return None
        binary = (value > 0.0).float()[None, None]
        if binary.shape[-2:] != (self.img_height, self.img_width):
            binary = torch.nn.functional.interpolate(
                binary, size=(self.img_height, self.img_width), mode="nearest"
            )
        return binary[0, 0].to(torch.uint8).cpu().numpy()

    def _maybe_add_to_drm(self, frame_idx, mask, alternatives, selected_idx, score):
        # Native DAM4SAM stability, area, stride, and distractor tests.
        pixels = (mask == 1).sum()
        self.object_sizes.append(pixels)
        if len(self.object_sizes) > 1 and pixels >= 1:
            ratio = pixels / np.median([
                size for size in self.object_sizes[-300:] if size >= 1
            ][-10:])
        else:
            ratio = -1
        if not (score > 0.8 and 0.8 <= ratio <= 1.2 and pixels >= 1
                and (self.frame_index - self.last_added > 5 or self.last_added == -1)):
            return
        alternatives = [self._candidate_mask_to_original(value)
                        for index, value in enumerate(alternatives) if index != selected_idx]
        alternatives = [value for value in alternatives if value is not None]
        calculate_overlaps, Mask, RegionType = _load_vot_region_helpers()
        chosen = mask.copy()
        chosen_box = Mask(chosen).convert(RegionType.RECTANGLE)
        alternatives = [np.logical_and(value, np.logical_not(chosen)).astype(np.uint8)
                        for value in alternatives]
        alternatives = [_keep_largest_component(value) for value in alternatives
                        if np.sum(value) >= 1]
        if not alternatives:
            return
        alternatives = [np.logical_or(value, chosen).astype(np.uint8)
                        for value in alternatives]
        boxes = [Mask(value).convert(RegionType.RECTANGLE) for value in alternatives]
        overlaps = [calculate_overlaps([chosen_box], [box])[0] for box in boxes]
        if np.min(np.array(overlaps)) <= 0.7:
            self.last_added = self.frame_index
            self.predictor.add_to_drm(
                inference_state=self.inference_state, frame_idx=frame_idx, obj_id=0
            )

    def _object_score(self):
        state = self.inference_state
        stores = [state.get("output_dict", {})]
        stores.extend(state.get("output_dict_per_obj", {}).values())
        stores.extend(state.get("temp_output_dict_per_obj", {}).values())
        for store in stores:
            for key in ("non_cond_frame_outputs", "cond_frame_outputs"):
                output = store.get(key, {}).get(self.frame_index, {})
                value = output.get("object_score_logits")
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        value = value.detach().float().cpu().numpy()
                    values = np.asarray(value, dtype=np.float64).reshape(-1)
                    values = values[np.isfinite(values)]
                    if values.size:
                        logit = float(values[0])
                        return logit, float(1.0 / (1.0 + np.exp(-logit)))
        raise RuntimeError(f"SAM2 object confidence is absent at frame {self.frame_index}.")

    @torch.inference_mode()
    def track(self, image):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.frame_index += 1
        self.inference_state["num_frames"] += 1
        self.inference_state["images"][self.frame_index] = self._prepare_image(image).unsqueeze(0)
        # Preserve frame-0 cached features until propagation preflight has encoded memory.
        for output in self.predictor.propagate_in_video(
            self.inference_state, start_frame_idx=self.frame_index,
            max_frame_num_to_track=0, return_all_masks=True
        ):
            if len(output) != 4:
                raise RuntimeError("Expected the native DAM4SAM candidate-mask output.")
            frame_idx, _, logits, (alternatives, scores) = output
            mask = (logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)
            scores = np.asarray(scores)
            selected_idx = int(np.argmax(scores))
            selected_score = float(scores.reshape(-1)[selected_idx])
            self._maybe_add_to_drm(frame_idx, mask, alternatives, selected_idx, selected_score)
            object_logit, object_prob = self._object_score()
            self.inference_state["images"].pop(self.frame_index)
            return {
                "pred_mask": mask,
                "frame_idx": self.frame_index,
                "observation_debug": {
                    "object_logit": object_logit,
                    "object_prob": object_prob,
                    "sam_iou_score": selected_score,
                    "selected_idx": selected_idx,
                    "sam_ious": scores.reshape(-1).astype(float).tolist(),
                    "mask_area": int(np.count_nonzero(mask)),
                    "mask_empty": bool(not np.any(mask)),
                    "score_type": "selected_sam_pred_iou",
                },
            }
        raise RuntimeError(f"DAM4SAM returned no output at frame {self.frame_index}.")

    def estimate_mask_from_box(self, bbox):
        _, _, features, _, sizes = self.predictor._get_image_feature(self.inference_state, 0, 1)
        box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])[None, :]
        box = torch.as_tensor(box, dtype=torch.float, device=features[0].device)
        transforms = SAM2Transforms(
            resolution=self.predictor.image_size, mask_threshold=0.0,
            max_hole_area=0.0, max_sprinkle_area=0.0
        )
        box = transforms.transform_boxes(box, normalize=True, orig_hw=(self.img_height, self.img_width))
        coords = box.reshape(-1, 2, 2)
        labels = torch.tensor([[2, 3]], dtype=torch.int, device=box.device).repeat(box.size(0), 1)
        sparse, dense = self.predictor.sam_prompt_encoder(
            points=(coords, labels), boxes=None, masks=None
        )
        high_res = []
        for index in range(2):
            _, batch, channels = features[index].shape
            high_res.append(features[index].permute(1, 2, 0).view(batch, channels, *sizes[index]))
        embedding = features[2]
        if self.predictor.directly_add_no_mem_embed:
            embedding = embedding + self.predictor.no_mem_embed
        _, batch, channels = features[2].shape
        embedding = embedding.permute(1, 2, 0).view(batch, channels, *sizes[2])
        low_res, _, _, _ = self.predictor.sam_mask_decoder(
            image_embeddings=embedding, image_pe=self.predictor.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=coords.shape[0] > 1,
            high_res_features=high_res,
        )
        masks = transforms.postprocess_masks(low_res, (self.img_height, self.img_width)) > 0
        return masks.squeeze(0).float().detach().cpu().numpy()[0]


DAM4SAMTracker = SAMTracker
