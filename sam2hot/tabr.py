"""Exact fixed-profile TABR for HOTC2020 VIS and DAM4SAM.

The first-frame SAM mask defines the foreground and its full-image complement
defines background. Spectral statistics, pruning and weights are fixed for the
sequence; each frame retains the reference per-band uint8 normalization.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import hashlib
import json
import shutil

import cv2
import numpy as np

from .data import read_hsi


@dataclass(frozen=True)
class TABRConfig:
    topk: float = 0.9
    target_lambda: float = 0.6
    eta: float = 0.7
    temperature: float = 1.0
    wavelength_sigma: float = 3.0

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, cls):
            return value
        value = dict(value)
        allowed = {"topk", "lambda", "eta", "temperature", "wavelength_sigma"}
        if set(value).difference(allowed):
            raise ValueError(f"Unsupported TABR keys: {sorted(set(value).difference(allowed))}")
        if "lambda" in value:
            value["target_lambda"] = value.pop("lambda")
        return cls(**value)

    def __post_init__(self):
        if not 0 < self.topk <= 1 or not 0 <= self.target_lambda <= 1 or not 0 <= self.eta <= 1:
            raise ValueError("Invalid TABR ratio, lambda or eta")
        if self.temperature <= 0 or self.wavelength_sigma <= 0:
            raise ValueError("TABR temperature and wavelength sigma must be positive")


def minmax01(value, eps=1e-6):
    value = np.asarray(value, dtype=np.float32)
    if not value.size:
        return value.astype(np.float32)
    low, high = float(value.min()), float(value.max())
    if high - low < float(eps):
        return np.zeros_like(value, dtype=np.float32)
    return ((value - low) / (high - low + float(eps))).astype(np.float32)


def _softmax(scores, temperature):
    scores = np.asarray(scores, dtype=np.float32)
    scaled = scores / max(float(temperature), 1e-6)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    return exp / (exp.sum() + 1e-6)


def build_groups(selection_scores, weight_scores, config=TABRConfig()):
    """Preserve three spectral groups, prune locally, then reverse RGB order."""
    config = TABRConfig.from_mapping(config)
    selection = np.asarray(selection_scores, dtype=np.float32).reshape(-1)
    weights = np.asarray(weight_scores, dtype=np.float32).reshape(-1)
    if selection.shape != (16,) or weights.shape != (16,):
        raise ValueError("HOTC2020 VIS requires exactly 16 band scores")
    groups, selected = [], []
    for base in np.array_split(np.arange(16, dtype=np.int64), 3):
        keep_n = max(1, min(int(round(float(base.size) * config.topk)), int(base.size)))
        kept = np.sort(base[np.argsort(-selection[base])][:keep_n].astype(np.int64))
        selected.extend(kept.tolist())
        soft_weights = _softmax(weights[kept], config.temperature).astype(np.float32)
        groups.append((kept.tolist(), soft_weights.tolist()))
    selected = np.asarray(selected, dtype=np.int64)
    ranked = selected[np.argsort(-selection[selected])].tolist()
    return groups[::-1], ranked


def fit_tabr(first_cube, first_mask, config=TABRConfig()):
    """Compute full-precision fixed groups and provenance without model inference."""
    config = TABRConfig.from_mapping(config)
    cube = np.asarray(first_cube, dtype=np.float32)
    if cube.ndim != 3 or cube.shape[2] != 16:
        raise ValueError(f"Expected HxWx16 VIS cube, got {cube.shape}")
    height, width, bands = cube.shape
    mask = np.squeeze(np.asarray(first_mask))
    if mask.ndim != 2:
        raise ValueError(f"Expected a two-dimensional SAM mask, got {mask.shape}")
    if mask.shape != (height, width):
        mask = cv2.resize((mask > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    mask = mask > 0
    if int(mask.sum()) < 2 or int((~mask).sum()) == 0:
        raise ValueError("TABR needs at least two SAM-target pixels and nonempty background")

    fg, bg = cube[mask], cube[~mask]
    mu_fg, mu_bg = fg.mean(axis=0), bg.mean(axis=0)
    std_fg, std_bg = fg.std(axis=0), bg.std(axis=0)
    fisher = np.abs(mu_fg - mu_bg) / (std_fg + std_bg + 1e-6)
    fisher = np.nan_to_num(fisher, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    target = minmax01(fisher)

    samples = cube[mask, :]
    if samples.shape[0] > 50000:
        samples = samples[np.linspace(0, samples.shape[0] - 1, num=50000).astype(np.int64)]
    samples = samples - samples.mean(axis=0, keepdims=True)
    samples = samples / (samples.std(axis=0, keepdims=True) + 1e-6)
    correlation = np.matmul(samples.T, samples) / max(float(samples.shape[0] - 1), 1.0)
    spectral = 0.5 * (np.clip(correlation, -1.0, 1.0) + 1.0)
    np.fill_diagonal(spectral, 1.0)
    spectral = spectral.astype(np.float32)
    indices = np.arange(bands, dtype=np.float32)
    distances = np.abs(indices[:, None] - indices[None, :])
    wavelength = np.exp(-distances / max(float(config.wavelength_sigma), 1e-6))
    np.fill_diagonal(wavelength, 1.0)
    wavelength = wavelength.astype(np.float32)
    affinity = np.clip(config.eta * spectral + (1.0 - config.eta) * wavelength, 0.0, 1.0).astype(np.float32)
    all_indices = np.arange(bands, dtype=np.int64)
    intra, inter = np.zeros(bands, dtype=np.float32), np.zeros(bands, dtype=np.float32)
    for group in np.array_split(all_indices, 3):
        other = np.setdiff1d(all_indices, group)
        for band in group:
            same = group[group != band]
            intra[band] = float(np.mean(affinity[band, same])) if same.size else 1.0
            inter[band] = 1.0 - float(np.mean(affinity[band, other])) if other.size else 1.0
    intra, inter = minmax01(intra), minmax01(inter)
    prior = minmax01((0.5 * intra + 0.5 * inter) / 1.0)
    hybrid = minmax01(config.target_lambda * minmax01(target) + (1.0 - config.target_lambda) * minmax01(prior))
    groups, ranked = build_groups(hybrid, hybrid, config)
    return {
        "groups": groups,
        "ranked_bands": ranked,
        "retained_bands": len(ranked),
        "target_raw": fisher.tolist(),
        "target_normalized": target.tolist(),
        "spectral_prior": prior.tolist(),
        "hybrid_scores": hybrid.tolist(),
        "intra": intra.tolist(), "inter": inter.tolist(),
        "foreground_pixels": int(mask.sum()), "background_pixels": int((~mask).sum()),
        "config": asdict(config),
        "foreground": "exact_first_frame_sam_mask",
        "background": "full_image_complement",
        "normalization": "per_frame_per_band_uint8",
    }


def aggregate(cube, groups):
    """Apply the sequence's fixed band weights with reference uint8 rounding."""
    cube = np.asarray(cube).astype(np.float32)
    channels = []
    for indices, weights in groups:
        indices = np.asarray(indices, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-6)
        channels.append(np.sum(cube[:, :, indices] * weights.reshape(1, 1, -1), axis=2))
    return np.clip(np.stack(channels, axis=2), 0.0, 255.0).astype(np.uint8)


def _state_output(state):
    stores = [state.get("output_dict", {})]
    stores.extend(state.get("output_dict_per_obj", {}).values())
    for store in stores:
        for key in ("non_cond_frame_outputs", "cond_frame_outputs"):
            if 0 in store.get(key, {}):
                return store[key][0]
    return {}


def first_frame_sam_mask(sequence, predictor, output_dir, shape, device="cuda:0"):
    """Prompt original FCI with rounded XYXY; never initialize on TABR imagery."""
    import torch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first = sequence.frames[0]
    destination = output_dir / first.name
    if destination.resolve() == first.resolve():
        raise ValueError("SAM prompt copy must not be the dataset original")
    extra_images = [p for p in output_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p != destination]
    if extra_images:
        raise ValueError("First-mask prompt directory must contain exactly one frame")
    shutil.copy2(first, destination)
    state = None
    try:
        state = predictor.init_state(str(output_dir), offload_video_to_cpu=True,
                                     offload_state_to_cpu=True, async_loading_frames=True)
        result = predictor.add_new_points_or_box(state, box=sequence.initial_box_xyxy,
                                                frame_idx=0, obj_id=0)
        masks = result[2] if isinstance(result, tuple) and len(result) >= 3 else None
        if masks is None:
            output = _state_output(state)
            masks = output.get("pred_masks")
            if masks is None:
                masks = output.get("pred_masks_high_res")
        if masks is None:
            raise RuntimeError("SAM did not return the first-frame box-prompted mask")
        if torch.is_tensor(masks):
            mask = masks.detach().float().cpu()
            while mask.ndim > 2:
                mask = mask[0]
            mask = mask.numpy()
        else:
            mask = np.asarray(masks)
            while mask.ndim > 2:
                mask = mask[0]
        mask = mask.astype(np.float32)
        height, width = shape
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        return mask > 0.0
    finally:
        if state is not None:
            del state
        gc.collect()
        if str(device).startswith("cuda"):
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()


def _fingerprint(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def prepare_tabr(sequence, predictor, output_dir, config=TABRConfig(), device="cuda:0"):
    """Generate quality-100 JPEGs only under output_dir; return ordered paths.

    A cache is reusable only with a recorded checkpoint identity, matching
    configuration/data fingerprint and every generated frame present.
    """
    config = TABRConfig.from_mapping(config)
    output_dir = Path(output_dir).expanduser().resolve()
    dataset_sequence = sequence.hsi_dir.parent.resolve()
    if output_dir == dataset_sequence or dataset_sequence in output_dir.parents:
        raise ValueError("TABR output must be outside the original dataset sequence")
    outputs = [output_dir / (p.stem + ".jpg") for p in sequence.frames]
    checkpoint = getattr(predictor, "_sam2hot_checkpoint", None)
    checkpoint_id = _fingerprint(checkpoint) if checkpoint and Path(checkpoint).is_file() else None
    provenance = {
        "version": "sam2hot-tabr-hot20-v1", "sequence": sequence.name,
        "config": asdict(config), "checkpoint": checkpoint_id,
        "gt_sha256": hashlib.sha256(sequence.gt_path.read_bytes()).hexdigest(),
        "original_fci": _fingerprint(sequence.frames[0]),
        "hsi_frames": [_fingerprint(p) for p in sequence.hsi_frames],
        "jpeg_quality": 100,
    }
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    manifest = output_dir / "tabr.json"
    if checkpoint_id and manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("signature") == signature and all(p.is_file() for p in outputs):
            return outputs
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Existing TABR cache is incomplete or incompatible; choose a fresh output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cube = read_hsi(sequence.hsi_frames[0])
    mask = first_frame_sam_mask(sequence, predictor, output_dir / "first_frame_prompt", cube.shape[:2], device)
    fitted = fit_tabr(cube, mask, config)
    if not cv2.imwrite(str(output_dir / "first_sam_mask.png"), mask.astype(np.uint8) * 255):
        raise RuntimeError("Could not save initial SAM mask")
    for source, destination in zip(sequence.hsi_frames, outputs):
        rgb = aggregate(read_hsi(source), fitted["groups"])
        ok = cv2.imwrite(str(destination), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
        if not ok:
            raise RuntimeError(f"Could not save TABR frame: {destination}")
    record = {"signature": signature, "provenance": provenance, **fitted}
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return outputs
