"""Streaming SDMR with exact parity to the selected V2 offline recovery.

Initialize once from frame-zero GT. Call ``step`` only for frames 1 onward.
The caller supplies raw boxes rounded to six decimals and probabilities/SAM
IoUs rounded to eight decimals, matching the original observation CSV boundary.
Recovered boxes never enter the tracker's memory or this module's history.
"""
from __future__ import annotations

import numpy as np
import torch

from .motion import LoadedMotion


def valid_box(box) -> bool:
    value = np.asarray(box, dtype=np.float64).reshape(-1)
    return bool(value.size >= 4 and np.all(np.isfinite(value[:4]))
                and value[2] > 0.0 and value[3] > 0.0)


def clip_box(box, width: float, height: float) -> np.ndarray:
    if not valid_box(box):
        return np.zeros(4, dtype=np.float64)
    x, y, w, h = [float(value) for value in box[:4]]
    x1 = float(np.clip(x, 0.0, max(width, 0.0)))
    y1 = float(np.clip(y, 0.0, max(height, 0.0)))
    x2 = float(np.clip(x + w, 0.0, max(width, 0.0)))
    y2 = float(np.clip(y + h, 0.0, max(height, 0.0)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(4, dtype=np.float64)
    return np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)


class SDMR:
    """Output-only center recovery plus reliability-gated log-size EMA.

    ``initial_bbox`` is frame-zero XYWH in original-image pixels; it is also
    the unchanged first output. ``step`` returns ``(final_xywh, debug)``.
    ``object_score`` means sigmoid object probability, never the raw logit.
    """

    def __init__(self, loaded: LoadedMotion, initial_bbox, image_width: float,
                 image_height: float, *, object_threshold: float = 0.9,
                 ema_alpha: float = 0.2) -> None:
        if not valid_box(initial_bbox):
            raise ValueError("Frame-zero initialization must be a valid XYWH box")
        if not np.isfinite(image_width) or not np.isfinite(image_height) or min(image_width, image_height) <= 0:
            raise ValueError("Image dimensions must be finite and positive")
        if not 0.0 <= object_threshold <= 1.0 or not 0.0 < ema_alpha <= 1.0:
            raise ValueError("Invalid object threshold or EMA alpha")
        self.loaded = loaded
        self.initial_bbox = np.asarray(initial_bbox, dtype=np.float64).reshape(-1)[:4].copy()
        self.image_width, self.image_height = float(image_width), float(image_height)
        self.object_threshold, self.ema_alpha = float(object_threshold), float(ema_alpha)
        self.history: list[np.ndarray] = []
        self.scores: list[float] = []
        self.valid: list[float] = []
        self.last_valid: np.ndarray | None = None
        self.frame_index = 0
        self.log_size = np.log(np.maximum(np.asarray(self.initial_bbox[2:4], dtype=float), 1e-6))
        self._append(self.initial_bbox, 1.0, True)

    def _normalise(self, box) -> np.ndarray | None:
        if not valid_box(box):
            return None
        x, y, width, height = [float(item) for item in box[:4]]
        return np.asarray([(x + 0.5 * width) / max(self.image_width, 1.0),
                           (y + 0.5 * height) / max(self.image_height, 1.0),
                           width / max(self.image_width, 1.0),
                           height / max(self.image_height, 1.0)], dtype=np.float32)

    def _append(self, box, score: float, valid: bool) -> None:
        normalised = self._normalise(box)
        structurally_valid = bool(valid and normalised is not None)
        if structurally_valid:
            self.last_valid = normalised.copy()
            stored = normalised
        else:
            stored = self.last_valid.copy()
        self.history.append(stored)
        self.scores.append(float(np.clip(score, 0.0, 1.0)))
        self.valid.append(1.0 if structurally_valid else 0.0)
        self.history, self.scores, self.valid = self.history[-32:], self.scores[-32:], self.valid[-32:]

    def _predict(self) -> dict | None:
        if len(self.history) < self.loaded.history_len:
            return None
        length, device = self.loaded.history_len, self.loaded.device
        boxes = np.stack(self.history[-length:], axis=0)
        scores = np.asarray(self.scores[-length:], dtype=np.float32)[:, None]
        valid = np.asarray(self.valid[-length:], dtype=np.float32)[:, None]
        # The original SDMR runtime runs float32, independently of tracker AMP.
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            output = self.loaded.model(
                torch.as_tensor(boxes, device=device).unsqueeze(0),
                history_scores=torch.as_tensor(scores, device=device).unsqueeze(0),
                history_valid=torch.as_tensor(valid, device=device).unsqueeze(0),
            )
        def array(name):
            return output[name][0].detach().float().cpu().numpy().astype(np.float64)
        center, delta = array("pred_center"), array("center_delta")
        expected = np.clip(boxes[-1, :2].astype(np.float64) + delta, 0.0, 1.0)
        if not np.allclose(center, expected, rtol=2e-4, atol=2e-6):
            raise RuntimeError("Motion predicted-center/delta mismatch")
        scale = np.asarray([self.image_width, self.image_height], dtype=np.float64)
        return {"center_px": center * scale, "delta_px": delta * scale,
                "coefficients": array("motion_coefficients"),
                "temporal_weights": array("temporal_weights_dynamic"),
                "contributions_px": array("order_contributions") * scale[None, :]}

    def step(self, raw_bbox, object_score: float, sam_iou: float,
             mask_empty: bool) -> tuple[np.ndarray, dict]:
        """Consume one quantized raw observation (frame 1 onward)."""
        raw = np.asarray(raw_bbox, dtype=np.float64).reshape(-1)
        if raw.size != 4 or not np.all(np.isfinite(raw)):
            raise ValueError("Raw bbox must contain four finite XYWH values")
        if not np.isfinite(object_score) or not np.isfinite(sam_iou):
            raise ValueError("Observation scores must be finite")
        self.frame_index += 1
        prediction = self._predict()  # Never include frame t in its own prediction.
        current = np.exp(self.log_size)
        ratios = np.asarray([np.nan, np.nan])
        if valid_box(raw):
            ratios = np.asarray(raw[2:4], dtype=float) / np.maximum(current, 1e-6)
        score_gate = bool(valid_box(raw) and not mask_empty and object_score >= 0.70 and sam_iou >= 0.75)
        size_gate = bool(np.all((ratios >= 0.75) & (ratios <= 1.33)))
        size_updated = score_gate and size_gate
        if size_updated:
            self.log_size = ((1.0 - self.ema_alpha) * self.log_size
                             + self.ema_alpha * np.log(np.maximum(raw[2:4], 1e-6)))
        stable_size = np.exp(self.log_size)
        scheduled = bool(prediction is not None and (mask_empty or object_score < self.object_threshold))
        final, applied = raw.copy(), False
        if scheduled and np.all(np.isfinite(prediction["center_px"])):
            cx, cy = [float(value) for value in prediction["center_px"]]
            width, height = [max(float(value), 1e-6) for value in stable_size]
            proposed = np.asarray([cx - 0.5 * width, cy - 0.5 * height, width, height], dtype=np.float64)
            clipped = clip_box(proposed, self.image_width, self.image_height)
            if valid_box(clipped):
                final, applied = clipped, True
        self._append(raw, sam_iou, bool(not mask_empty and valid_box(raw)))
        debug = {"frame_index": self.frame_index, "ready": prediction is not None,
                 "scheduled": scheduled, "applied": applied,
                 "size_updated": size_updated, "stable_size": stable_size.tolist()}
        if prediction is not None:
            debug.update({key: value.tolist() for key, value in prediction.items()})
        return final, debug
