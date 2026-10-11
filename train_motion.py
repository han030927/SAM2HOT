#!/usr/bin/env python3
"""Train the released image-relative L5/O3 MLP with Smooth L1 center loss only."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sam2hot.config import ROOT, load_config
from sam2hot.motion import HISTORY_LEN, MODEL_NAME, MOTION_ORDER, MotionGateMLP
from sam2hot.training_data import MotionDataset, load_records, split_records


def center_objective(predicted_center, target_box):
    """The original center term, with GT-size normalization and beta=1."""
    predicted_center, target_box = predicted_center.float(), target_box.float()
    target_center = target_box[:, :2]
    target_size = target_box[:, 2:].clamp_min(1e-4)
    residual = (predicted_center - target_center) / target_size
    loss = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1.0)
    scale = torch.sqrt(target_size.prod(dim=-1).clamp_min(1e-6))
    error = torch.linalg.vector_norm(predicted_center - target_center, dim=-1) / scale
    return loss, error.mean()


def seed_training(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def make_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs, min_lr_ratio):
    total = epochs * steps_per_epoch
    warmup = min(max(round(warmup_epochs * steps_per_epoch), 0), total - 1)

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def run_epoch(model, loader, device, *, optimizer=None, scaler=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, error_sum, count = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                output = model(batch["history_boxes"], batch["history_scores"], batch["history_valid"])
                loss, error = center_objective(output["pred_center"], batch["target_box"])
            if not torch.isfinite(loss) or not torch.isfinite(error):
                raise FloatingPointError("Non-finite training/validation objective")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # An overflow skips the optimizer update; do not advance the LR.
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
            batch_size = len(batch["target_box"])
            loss_sum += float(loss.detach()) * batch_size
            error_sum += float(error.detach()) * batch_size
            count += batch_size
    if not count:
        raise ValueError("Empty training/validation loader")
    return {"loss": loss_sum / count, "center_norm_error": error_sum / count}


def save_checkpoint(path, model, epoch, settings, metrics):
    payload = {
        "motion_gate": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_name": MODEL_NAME, "epoch": epoch,
        "temporal_window": HISTORY_LEN, "motion_order": MOTION_ORDER,
        "training_history": "sam", "use_sam_score": True,
        "temporal_aggregation": "weighted", "center_encoding": model.center_encoding,
        "box_format": "cxcywh_norm", "score_type": "selected_sam_pred_iou",
        "loss_name": "smooth_l1_center", "selection_metric": "val_center_norm_error",
        "args": {"hidden_dim": 64, "num_layers": 3, "dropout": 0.0, "eta": 1.0,
                 "model_score_floor": settings["score_floor"]},
        "training_config": settings, "metrics": metrics,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def get_settings(args):
    profile = load_config(args.config)
    settings = dict(profile.get("training", {}))
    required = {"data_root", "score_root", "output_dir", "epochs", "batch_size",
                "learning_rate", "weight_decay", "samples_per_epoch", "val_samples",
                "val_ratio", "seed", "score_floor", "warmup_epochs", "min_lr_ratio"}
    if set(settings) != required:
        raise ValueError(f"The training section must contain: {sorted(required)}")
    for key in ("data_root", "score_root", "output_dir", "epochs"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    for key in ("data_root", "score_root", "output_dir"):
        path = Path(settings[key]).expanduser()
        settings[key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    for key in ("epochs", "batch_size", "samples_per_epoch", "val_samples"):
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    if not isinstance(settings["seed"], int) or not 0 <= settings["seed"] < 2 ** 32:
        raise ValueError("training.seed must be an integer in [0, 2**32)")
    if not 0 < settings["val_ratio"] < 1:
        raise ValueError("training.val_ratio must be between 0 and 1")
    for key in ("score_floor", "min_lr_ratio"):
        if not 0 <= settings[key] <= 1:
            raise ValueError(f"training.{key} must be in [0, 1]")
    for key in ("learning_rate", "weight_decay", "warmup_epochs"):
        if not math.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f"Invalid training.{key}")
    if settings["learning_rate"] == 0:
        raise ValueError("training.learning_rate must be positive")
    if settings["samples_per_epoch"] < settings["batch_size"]:
        raise ValueError("training.samples_per_epoch must be at least batch_size")
    device = args.device or profile["device"]
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    settings["device"] = device
    return settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/hot20.yaml"))
    parser.add_argument("--data-root", help="HOTC2023 training partition containing HSI-VIS/NIR/RedNIR")
    parser.add_argument("--score-root", help="Directory of raw DAM4SAM *-score-*.txt exports")
    parser.add_argument("--output-dir", help="New directory for best.pth and last.pth")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--check", action="store_true", help="Validate data and split without training or writing files")
    args = parser.parse_args(argv)
    settings = get_settings(args)
    records = load_records(settings["data_root"], settings["score_root"])
    train_records, val_records = split_records(records, settings["val_ratio"], settings["seed"])
    train_data = MotionDataset(train_records, settings["samples_per_epoch"], settings["seed"], training=True)
    val_data = MotionDataset(val_records, settings["val_samples"], settings["seed"], training=False)
    print(f"Sequences: {len(train_records)} train / {len(val_records)} validation; "
          f"windows: {train_data.available_samples} train / {len(val_data)} fixed validation", flush=True)
    # --check remains read-only and never initializes the GPU.
    if args.check:
        return
    device = torch.device(settings["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use a CPU or CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu or install a CUDA-enabled PyTorch")
    output_dir = Path(settings["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Training output already exists; choose a new --output-dir: {output_dir}")
    seed_training(settings["seed"])
    model = MotionGateMLP(score_floor=settings["score_floor"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    train_loader = DataLoader(train_data, batch_size=settings["batch_size"], num_workers=0,
                              shuffle=False, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=settings["batch_size"], num_workers=0, shuffle=False)
    scheduler = make_scheduler(optimizer, settings["epochs"], len(train_loader),
                               settings["warmup_epochs"], settings["min_lr_ratio"])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    def describe(record):
        return {key: getattr(record, key) for key in
                ("source_name", "modality", "name", "sequence_id", "score_path", "gt_path")}
    split = {"train": [describe(record) for record in train_records],
             "validation": [describe(record) for record in val_records],
             "validation_windows": [[val_records[index].modality, val_records[index].name, target]
                                    for index, target in val_data.samples]}
    (output_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    best_error, best_epoch = math.inf, 0
    with (output_dir / "history.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, settings["epochs"] + 1):
            train = run_epoch(model, train_loader, device, optimizer=optimizer, scaler=scaler, scheduler=scheduler)
            validation = run_epoch(model, val_loader, device)
            metrics = {"epoch": epoch, "train": train, "validation": validation,
                       "learning_rate": optimizer.param_groups[0]["lr"]}
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            if validation["center_norm_error"] < best_error:
                best_error, best_epoch = validation["center_norm_error"], epoch
                save_checkpoint(output_dir / "best.pth", model, epoch, settings, metrics)
            save_checkpoint(output_dir / "last.pth", model, epoch, settings, metrics)
            print(f"Epoch {epoch:03d}: train={train['loss']:.6f} val={validation['loss']:.6f} "
                  f"center_norm_error={validation['center_norm_error']:.6f} best_epoch={best_epoch}", flush=True)
    print(f"Best checkpoint: {output_dir / 'best.pth'}", flush=True)


if __name__ == "__main__":
    main()
