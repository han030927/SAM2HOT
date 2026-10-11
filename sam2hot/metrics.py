"""The user's CNT metric, without plotting or multi-benchmark configuration.

Matches auc_pre_plot_2023.py: 21 inclusive IoU thresholds, DP@20 from
unclipped centers, and sequence-level macro averaging. The 640x480 IoU
clip dimensions deliberately retain the reference evaluator's defaults.
"""
from pathlib import Path

import numpy as np


def read_boxes(path):
    boxes = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            values = line.strip().replace(",", " ").split()
            if len(values) >= 4:
                boxes.append([float(value) for value in values[:4]])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def clip_bbox_xywh(box, img_w=640, img_h=480):
    x = np.clip(box[:, 0], 0, img_w - 1)
    y = np.clip(box[:, 1], 0, img_h - 1)
    w = np.clip(box[:, 2], 0, img_w - x)
    h = np.clip(box[:, 3], 0, img_h - y)
    return np.stack([x, y, w, h], axis=1)


def evaluate_one_sequence_CNT(gt, pred, img_w=640, img_h=480):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    # Unlike the old plotting script, fail on incomplete prediction files;
    # valid, full-length files have identical metric arithmetic.
    if gt.shape != pred.shape or gt.ndim != 2 or gt.shape[1] != 4 or not len(gt):
        raise ValueError("GT and predictions must have matching nonempty (N,4) shapes")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("Non-finite boxes cannot be evaluated")
    valid = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    gt, pred = gt[valid], pred[valid]
    if not len(gt):
        raise ValueError("Sequence has no valid ground truth")
    pc, gc = clip_bbox_xywh(pred, img_w, img_h), clip_bbox_xywh(gt, img_w, img_h)
    x_a, y_a = np.maximum(pc[:, 0], gc[:, 0]), np.maximum(pc[:, 1], gc[:, 1])
    x_b = np.minimum(pc[:, 0] + pc[:, 2], gc[:, 0] + gc[:, 2])
    y_b = np.minimum(pc[:, 1] + pc[:, 3], gc[:, 1] + gc[:, 3])
    intersection = np.maximum(0, x_b - x_a) * np.maximum(0, y_b - y_a)
    union = pc[:, 2] * pc[:, 3] + gc[:, 2] * gc[:, 3] - intersection
    ious = intersection / np.clip(union, 1e-6, None)
    error_x = (pred[:, 0] + pred[:, 2] / 2) - (gt[:, 0] + gt[:, 2] / 2)
    error_y = (pred[:, 1] + pred[:, 3] / 2) - (gt[:, 1] + gt[:, 3] / 2)
    distances = np.sqrt(error_x**2 + error_y**2)
    success = np.array([np.mean(ious >= t) for t in np.arange(0, 1.05, 0.05)])
    precision = np.array([np.mean(distances <= t) for t in np.arange(0, 51)])
    return success, precision


def evaluate(gt, pred):
    success, precision = evaluate_one_sequence_CNT(gt, pred)
    return {"auc": float(success.mean()), "dp20": float(precision[20]),
            "success_curve": success.tolist(), "precision_curve": precision.tolist()}
