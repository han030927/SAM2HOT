"""Strict, read-only HOTC2020 VIS data access (35 fixed sequences)."""
from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image


HOT20_NAMES = (
    "ball", "basketball", "board", "book", "bus", "bus2", "campus", "car",
    "car2", "car3", "card", "coin", "coke", "drive", "excavator", "face",
    "face2", "forest", "forest2", "fruit", "hand", "kangaroo", "paper",
    "pedestrain", "pedestrian2", "player", "playground", "rider1", "rider2",
    "rubik", "student", "toy1", "toy2", "trucker", "worker",
)


@dataclass
class Sequence:
    name: str
    frame_dir: Path
    hsi_dir: Path
    gt_path: Path
    frames: list[Path]
    hsi_frames: list[Path]
    gt: np.ndarray

    @property
    def initial_box_xyxy(self):
        x, y, width, height = self.gt[0]
        return tuple(int(round(v)) for v in (x, y, x + width, y + height))


def read_groundtruth(path):
    boxes = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = re.split(r"[,\s]+", line.strip())
        if len(values) < 4:
            raise ValueError(f"Invalid ground truth row {number}: {path}")
        boxes.append([float(v) for v in values[:4]])
    result = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(f"Empty/nonfinite ground truth: {path}")
    if result[0, 2] <= 0 or result[0, 3] <= 0:
        raise ValueError(f"Invalid initialization box: {path}")
    return result


def _frame_key(path):
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else path.stem


def discover_sequences(root, sequence=None):
    """Discover only named parent directories, never nested image directories."""
    root = Path(root).expanduser().resolve(strict=True)
    names = HOT20_NAMES if sequence is None else (sequence,)
    invalid = set(names).difference(HOT20_NAMES)
    if invalid:
        raise ValueError(f"Not a HOTC2020 sequence: {sorted(invalid)}")
    result = []
    for name in names:
        folder = root / name
        frame_dir, hsi_dir = folder / "HSI-FalseColor", folder / "HSI"
        gt_path = hsi_dir / "groundtruth_rect.txt"
        frames = sorted(frame_dir.glob("*.jpg"), key=_frame_key)
        hsi_frames = [hsi_dir / (frame.stem + ".png") for frame in frames]
        gt = read_groundtruth(gt_path)
        if not frames or len(frames) != len(gt):
            raise ValueError(f"{name}: {len(frames)} false-color frames != {len(gt)} annotations")
        missing = [str(path) for path in hsi_frames if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing paired HSI frame: {missing[0]}")
        if len(list(hsi_dir.glob("*.png"))) != len(frames):
            raise ValueError(f"{name}: HSI/false-color frame counts differ")
        result.append(Sequence(name, frame_dir, hsi_dir, gt_path, frames, hsi_frames, gt))
    return result


def mosaic_to_cube(raw):
    """Exact VIS X2Cube: row-major 4x4 bands, per-frame/band uint8 min-max."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[0] % 4 or raw.shape[1] % 4:
        raise ValueError(f"Expected a two-dimensional 4x4 VIS mosaic, got {raw.shape}")
    height, width = raw.shape
    # Keep the original indexing and arithmetic dtype: uint16 subtraction occurs
    # before floating-point division in the reference implementation.
    start = np.arange(4)[:, None] * width + np.arange(4)
    offsets = np.arange(height - 3)[:, None] * width + np.arange(width - 3)
    pixels = np.take(raw, start.ravel()[:, None] + offsets[::4, ::4].ravel())
    cube = pixels.T.reshape(height // 4, width // 4, 16)
    minimum = np.min(cube, axis=(0, 1), keepdims=True)
    maximum = np.max(cube, axis=(0, 1), keepdims=True)
    return ((cube - minimum) / ((maximum - minimum) + 1e-6) * 255).astype(np.uint8)


def read_hsi(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot read HSI mosaic: {path}")
    return mosaic_to_cube(raw)


def read_first_fci(sequence):
    with Image.open(sequence.frames[0]) as image:
        return image.convert("RGB")


def iter_hsi(sequence):
    for path in sequence.hsi_frames:
        yield path, read_hsi(path)
