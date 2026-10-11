"""HOTC2023 SAM-trajectory data for the selected five-frame motion predictor.

Only the released recipe is supported: SAM history, predicted-IoU scores,
image-normalized boxes, and fully structurally valid L=5 training windows.
"""
from __future__ import annotations

import json
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .motion import HISTORY_LEN


HOT20_SEQUENCE_NAMES = frozenset({
    "ball", "basketball", "board", "book", "bus", "bus2", "campus", "car",
    "car2", "car3", "card", "coin", "coke", "drive", "excavator", "face",
    "face2", "forest", "forest2", "fruit", "hand", "kangaroo", "paper",
    "pedestrain", "pedestrian2", "player", "playground", "rider1", "rider2",
    "rubik", "student", "toy1", "toy2", "trucker", "worker",
})
_MODALITIES = {"vis": "VIS", "nir": "NIR", "rednir": "RedNIR"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MotionRecord:
    source_name: str
    modality: str
    name: str
    sequence_id: str
    score_path: str
    gt_path: str
    width: int
    height: int
    pred_xywh: np.ndarray
    pred_score: np.ndarray
    gt_xywh: np.ndarray


def _read_rows(path: Path, columns: int) -> np.ndarray:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = re.split(r"[,\s]+", line.strip())
        if len(fields) < columns:
            raise ValueError(f"{path}:{line_number}: expected at least {columns} columns")
        try:
            rows.append([float(value) for value in fields[:columns]])
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid numeric row") from error
    if not rows:
        raise ValueError(f"No trajectory rows in {path}")
    return np.asarray(rows, dtype=np.float32)


def _meta_path(meta: dict, key: str, data_root: Path, *, directory: bool) -> Path | None:
    candidates = []
    if meta.get(key):
        candidates.append(Path(str(meta[key])).expanduser())
    if meta.get(key + "_relative"):
        candidates.append(data_root / str(meta[key + "_relative"]))
    for path in candidates:
        if path.is_dir() if directory else path.is_file():
            return path
    return None


def _image_size(frame_dirs: Sequence[Path]) -> tuple[int, int]:
    for folder in frame_dirs:
        if not folder.is_dir():
            continue
        images = [path for path in folder.iterdir()
                  if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS]

        def frame_key(path: Path):
            numbers = re.findall(r"\d+", path.stem)
            return (0, int(numbers[-1])) if numbers else (1, path.stem.lower())

        for path in sorted(images, key=frame_key):
            try:
                with Image.open(path) as image:
                    return image.size
            except (OSError, ValueError):
                continue
    raise FileNotFoundError(
        "No readable tracking image found. Provide false-color frames or "
        "frame_width/frame_height in the score .meta.json; HSI mosaics cannot "
        "supply the tracking image dimensions."
    )


def load_records(data_root: str | Path, score_root: str | Path) -> list[MotionRecord]:
    """Read HOTC2023 GT and ``{modality}-score-{sequence}.txt`` exports.

    Optional sibling ``.meta.json`` files use the original exporter keys:
    gt_path[_relative], frame_dir[_relative], frame_width, frame_height,
    and sequence_id. HOTC2020 identities are excluded before splitting.
    Ordinary unequal-length trajectories align at their starts; car62 uses
    end alignment, matching the reference loader. Length changes are reported.
    """
    data_root, score_root = Path(data_root).expanduser(), Path(score_root).expanduser()
    for root in (data_root, score_root):
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}")
    discovered = []
    for path in score_root.rglob("*.txt"):
        match = re.fullmatch(r"(vis|nir|rednir)-score-(.+)\.txt", path.name, re.IGNORECASE)
        if match:
            discovered.append((match[1].lower(), match[2], path))
    records, seen = [], set()
    for modality, name, score_path in sorted(discovered, key=lambda row: (row[0], row[1], str(row[2]).lower())):
        meta_path = score_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig")) if meta_path.is_file() else {}
        if not isinstance(meta, dict):
            raise ValueError(f"Expected a JSON object in {meta_path}")
        sequence_id = str(meta.get("sequence_id") or name)
        if name.strip().lower() in HOT20_SEQUENCE_NAMES or sequence_id.strip().lower() in HOT20_SEQUENCE_NAMES:
            continue
        key = (modality, name.lower())
        if key in seen:
            raise ValueError(f"Duplicate {modality} trajectory for {name}")
        seen.add(key)
        hsi = "HSI-" + _MODALITIES[modality]
        gt_candidates = [
            data_root / hsi / name / "groundtruth_rect.txt",
            data_root / hsi / name / "HSI" / "groundtruth_rect.txt",
            data_root / name / "groundtruth_rect.txt",
            data_root / name / "HSI" / "groundtruth_rect.txt",
        ]
        gt_path = _meta_path(meta, "gt_path", data_root, directory=False)
        if gt_path is None:
            gt_path = next((path for path in gt_candidates if path.is_file()), None)
        if gt_path is None:
            raise FileNotFoundError(f"GT not found for {modality}/{name} under {data_root}")
        prediction = _read_rows(score_path, 5)
        target = _read_rows(gt_path, 4)
        count = min(len(prediction), len(target))
        if len(prediction) != len(target):
            alignment = "end" if "car62" in name.lower() else "start"
            warnings.warn(f"{score_path.name}: aligning {len(prediction)} prediction rows "
                          f"and {len(target)} GT rows at the {alignment}", stacklevel=2)
            selection = slice(-count, None) if alignment == "end" else slice(0, count)
            prediction, target = prediction[selection], target[selection]
        width, height = int(meta.get("frame_width") or 0), int(meta.get("frame_height") or 0)
        if width <= 0 or height <= 0:
            explicit_frames = _meta_path(meta, "frame_dir", data_root, directory=True)
            frame_dirs = ([explicit_frames] if explicit_frames is not None else []) + [
                data_root / (hsi + "-FalseColor") / name,
                data_root / hsi / name / "HSI-FalseColor",
                data_root / name / "HSI-FalseColor",
            ]
            width, height = _image_size(frame_dirs)
        records.append(MotionRecord(
            source_name="HOTC2023", modality=modality, name=name,
            sequence_id=sequence_id, score_path=str(score_path.resolve()),
            gt_path=str(gt_path.resolve()), width=width, height=height,
            pred_xywh=prediction[:, :4].copy(), pred_score=prediction[:, 4].copy(),
            gt_xywh=target.copy(),
        ))
    if not records:
        raise ValueError("No training trajectories remain after HOTC2020 exclusion")
    return records


def split_records(records: Sequence[MotionRecord], val_ratio: float = 0.1,
                  seed: int = 123) -> tuple[list[MotionRecord], list[MotionRecord]]:
    """Reproduce the source/modality-stratified reference split without overlap."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    groups: dict[tuple[str, str], list[MotionRecord]] = {}
    for record in records:
        groups.setdefault((record.source_name, record.modality), []).append(record)
    train, validation = [], []
    for group_index, key in enumerate(sorted(groups)):
        group = sorted(groups[key], key=lambda record: record.name.lower())
        indices = list(range(len(group)))
        random.Random(int(seed) + group_index * 1009).shuffle(indices)
        count = min(len(group) - 1, max(1, round(len(group) * val_ratio))) if len(group) > 1 else 0
        selected = set(indices[:count])
        for index, record in enumerate(group):
            (validation if index in selected else train).append(record)
    if not train or not validation:
        raise ValueError("Need at least two sequences in a source/modality group for held-out validation")
    ordering = lambda record: (record.modality, record.source_name, record.name)
    return sorted(train, key=ordering), sorted(validation, key=ordering)


def _normalized_boxes(xywh: np.ndarray, width: int, height: int) -> np.ndarray:
    output = np.asarray(xywh, dtype=np.float32).copy()
    output[..., :2] += 0.5 * output[..., 2:]
    output /= np.asarray([width, height, width, height], dtype=np.float32)
    output[..., :2] = np.clip(output[..., :2], 0.0, 1.0)
    output[..., 2:] = np.clip(output[..., 2:], 1e-6, 1.0)
    return output


class MotionDataset(Dataset):
    """Five valid SAM observations followed by the next GT box.

    Training resamples with replacement using Python's worker-seeded RNG, as
    in the original trainer. Seed Python's RNG before constructing loaders.
    Validation uses a fixed sorted subset drawn with ``seed + 104729``.
    A non-positive sample_count uses every eligible window once.
    """

    def __init__(self, records: Sequence[MotionRecord], sample_count: int = 0,
                 seed: int = 123, training: bool = True) -> None:
        self.records = list(records)
        self.sample_count = int(sample_count)
        self.training = bool(training)
        self.samples: list[tuple[int, int]] = []
        for sequence_index, record in enumerate(self.records):
            pred, gt, scores = record.pred_xywh, record.gt_xywh, record.pred_score
            if record.width <= 0 or record.height <= 0:
                raise ValueError(f"Invalid tracking image dimensions for {record.name}")
            if len(pred) != len(gt) or len(scores) != len(pred):
                raise ValueError(f"Unaligned trajectory arrays for {record.name}")
            structural = np.isfinite(pred).all(axis=1) & np.isfinite(scores)
            structural &= (pred[:, 2:] > 1.0).all(axis=1)
            structural &= np.abs(pred).sum(axis=1) > 1e-6
            target_valid = np.isfinite(gt).all(axis=1) & (gt[:, 2:] > 1.0).all(axis=1)
            for target_index in range(HISTORY_LEN, len(gt)):
                if target_valid[target_index] and structural[target_index - HISTORY_LEN:target_index].all():
                    self.samples.append((sequence_index, target_index))
        self.available_samples = len(self.samples)
        if not self.samples:
            raise ValueError("No valid five-frame motion windows found")
        if not self.training and 0 < self.sample_count < len(self.samples):
            generator = np.random.default_rng(int(seed) + 104729)
            indices = np.sort(generator.choice(len(self.samples), size=self.sample_count, replace=False))
            self.samples = [self.samples[int(index)] for index in indices]

    def __len__(self) -> int:
        if self.training and self.sample_count > 0:
            return self.sample_count
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.training and self.sample_count > 0:
            index = random.randrange(len(self.samples))
        sequence_index, target_index = self.samples[index]
        record = self.records[sequence_index]
        history = slice(target_index - HISTORY_LEN, target_index)
        return {
            "history_boxes": torch.from_numpy(_normalized_boxes(
                record.pred_xywh[history], record.width, record.height)),
            "history_scores": torch.from_numpy(record.pred_score[history, None].copy()),
            "history_valid": torch.ones((HISTORY_LEN, 1), dtype=torch.float32),
            "target_box": torch.from_numpy(_normalized_boxes(
                record.gt_xywh[target_index], record.width, record.height)),
        }
