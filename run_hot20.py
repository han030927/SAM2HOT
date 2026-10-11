"""Single entry point for fresh HOTC2020 TABR + DAM4SAM + SDMR inference."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from sam2hot.config import ROOT, load_config
from sam2hot.metrics import evaluate, read_boxes


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def mask_to_box(mask):
    points = np.argwhere(np.asarray(mask) > 0)
    if not len(points):
        return np.zeros(4, dtype=np.float64)
    y1, x1 = points.min(axis=0)
    y2, x2 = points.max(axis=0)
    return np.asarray([x1, y1, x2 - x1 + 1, y2 - y1 + 1], dtype=np.float64)


def quantize(value, places):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Non-finite tracker observation; refusing a proxy score")
    return float(f"{value:.{places}f}")


def run_sequence(sequence, config, destination):
    import torch
    from sam2hot.tracker import SAMTracker
    from sam2hot.tabr import prepare_tabr
    from sam2hot.motion import load_motion
    from sam2hot.sdmr import SDMR

    destination.mkdir(parents=True, exist_ok=False)
    tracker = SAMTracker(config["backbone_checkpoint"], config["backbone_config"],
                         config["device"], seed=config["seed"])
    tracker.predictor._sam2hot_checkpoint = config["backbone_checkpoint"]
    print(f"[{sequence.name}] Generating fresh TABR images", flush=True)
    frames = prepare_tabr(sequence, tracker.predictor, destination / "tabr",
                          config["tabr"], device=config["device"])
    if len(frames) != len(sequence.gt):
        raise ValueError("TABR frame count differs from ground truth")
    observations = []
    raw_boxes = []
    width = height = None
    print(f"[{sequence.name}] DAM4SAM: {len(frames)} frames", flush=True)
    for index, path in enumerate(frames):
        with Image.open(path) as source:
            image = source.convert("RGB")
        if width is None:
            width, height = image.size
        elif image.size != (width, height):
            raise ValueError("Frame dimensions changed within a sequence")
        if index == 0:
            tracker.initialize(image, bbox=sequence.gt[0].tolist())
            raw = sequence.gt[0]
            object_prob, sam_iou, empty, logit = 1.0, 1.0, False, None
        else:
            result = tracker.track(image)
            raw = mask_to_box(result["pred_mask"])
            scores = result["observation_debug"]
            object_prob, sam_iou = scores["object_prob"], scores["sam_iou_score"]
            empty = bool(scores["mask_empty"] or raw[2] <= 0 or raw[3] <= 0)
            logit = scores["object_logit"]
        image.close()
        # Retain the old saved-observation boundary exactly. SDMR previously
        # consumed 6-decimal boxes and 8-decimal confidence values from CSV.
        raw = [quantize(value, 6) for value in raw]
        observation = {
            "frame_index": index, "frame_number": index + 1, "frame_name": path.name,
            "image_width": width, "image_height": height,
            "raw_x": raw[0], "raw_y": raw[1], "raw_w": raw[2], "raw_h": raw[3],
            "mask_empty": int(empty), "object_logit": "" if logit is None else quantize(logit, 8),
            "object_prob": quantize(object_prob, 8), "sam_pred_iou": quantize(sam_iou, 8),
        }
        raw_boxes.append(raw)
        observations.append(observation)
        if (index + 1) % 50 == 0 or index == len(frames) - 1:
            print(f"[{sequence.name}] {index + 1}/{len(frames)}", flush=True)
    np.savetxt(destination / "raw.txt", raw_boxes, fmt="%.6f", delimiter=",")
    with (destination / "observations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    # SDMR is output-only: never feed recovered boxes into DAM4SAM or its memory.
    del tracker
    gc.collect()
    if str(config["device"]).startswith("cuda"):
        torch.cuda.empty_cache()
    loaded = load_motion(config["motion_checkpoint"], device=config["device"])
    recovery = SDMR(loaded, sequence.gt[0], width, height, **config["sdmr"])
    final_boxes = [sequence.gt[0].copy()]
    recovery_frames = 0
    for raw, observation in zip(raw_boxes[1:], observations[1:]):
        final, debug = recovery.step(raw, observation["object_prob"], observation["sam_pred_iou"],
                                     bool(observation["mask_empty"]))
        final_boxes.append(final)
        recovery_frames += int(not np.array_equal(np.asarray(raw), final))
    np.savetxt(destination / "predictions.txt", final_boxes, fmt="%.6f", delimiter=",")
    result = {"sequence": sequence.name, "frames": len(frames), "finished_at": stamp(),
              "changed_output_frames": recovery_frames,
              "raw": evaluate(sequence.gt, read_boxes(destination / "raw.txt")),
              "final": evaluate(sequence.gt, read_boxes(destination / "predictions.txt"))}
    save_json(destination / "metrics.json", result)
    print(json.dumps({"sequence": sequence.name, "auc": result["final"]["auc"],
                      "dp20": result["final"]["dp20"]}), flush=True)


def summarize(results, failures, expected):
    summary = {"status": "completed" if len(results) == expected and not failures else "incomplete",
               "sequence_count": len(results), "expected_sequences": expected,
               "failed": failures, "metric": "CNT reference: inclusive IoU; 640x480 IoU clipping; unclipped-center DP20; sequence macro mean"}
    for kind in ("raw", "final"):
        summary[kind] = {key: float(np.mean([item[kind][key] for item in results]))
                         for key in ("auc", "dp20")} if results else None
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hot20.yaml")
    parser.add_argument("--sequence", help="One HOTC2020 sequence; omit to run all 35")
    parser.add_argument("--data-root", type=Path, help="Override the local HOTC2020 path")
    parser.add_argument("--output-root", type=Path, help="New output directory; existing runs are never overwritten")
    parser.add_argument("--device", help="Override cuda:0 or cpu")
    parser.add_argument("--check", action="store_true", help="Check data, imports and checkpoint without GPU inference")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("data_root", "output_root", "device"):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value.resolve()) if isinstance(value, Path) else value
    for key in ("backbone_checkpoint", "motion_checkpoint"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"{key}: {config[key]}")
    from sam2hot.data import discover_sequences
    sequences = discover_sequences(config["data_root"], sequence=args.sequence)
    if not sequences:
        raise ValueError("No HOTC2020 sequences found")
    if args.check:
        from sam2hot.tracker import SAMTracker
        from sam2hot.motion import load_motion
        loaded = load_motion(config["motion_checkpoint"], device="cpu")
        print(json.dumps({"status": "preflight_passed", "sequences": len(sequences),
                          "frames": sum(len(item.gt) for item in sequences),
                          "motion_parameters": sum(p.numel() for p in loaded.model.parameters()),
                          "data_root": config["data_root"], "device": config["device"]}, indent=2))
        return 0
    output = Path(config["output_root"])
    if args.worker:
        if len(sequences) != 1:
            raise ValueError("Each worker must handle one sequence")
        run_sequence(sequences[0], config, output / "sequences" / sequences[0].name)
        return 0
    if output.exists():
        raise FileExistsError(f"Use a new --output-root; refusing to overwrite: {output}")
    output.mkdir(parents=True)
    fingerprint = {key: sha256(config[key]) for key in ("backbone_checkpoint", "motion_checkpoint")}
    save_json(output / "run.json", {"started_at": stamp(), "config": config, "python": sys.executable,
                                   "checkpoint_sha256": fingerprint, "sequences": [item.name for item in sequences]})
    results, failures = [], []
    for index, sequence in enumerate(sequences, 1):
        save_json(output / "status.json", {"state": "running", "current": sequence.name,
                                           "completed": len(results), "total": len(sequences), "failed": failures})
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--config", str(args.config.resolve()),
                   "--sequence", sequence.name, "--data-root", config["data_root"],
                   "--output-root", str(output), "--device", config["device"]]
        print(f"[{index}/{len(sequences)}] {sequence.name} (see {sequence.name}.log)", flush=True)
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        with (output / f"{sequence.name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        folder = output / "sequences" / sequence.name
        if result.returncode:
            failures.append({"sequence": sequence.name, "exit_code": result.returncode})
            print(f"FAILED: {sequence.name}; retained partial files and log", flush=True)
            continue
        record = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        results.append(record)
        for kind, filename in (("raw", "raw.txt"), ("predictions", "predictions.txt")):
            target = output / kind / f"vis-{sequence.name}.txt"
            target.parent.mkdir(exist_ok=True)
            target.write_bytes((folder / filename).read_bytes())
        print(f"AUC={record['final']['auc']:.6f} DP={record['final']['dp20']:.6f}", flush=True)
    summary = summarize(results, failures, len(sequences))
    save_json(output / ("summary.json" if not failures else "partial_summary.json"), summary)
    save_json(output / "per_sequence.json", results)
    save_json(output / "status.json", {"state": summary["status"], "completed": len(results),
                                       "total": len(sequences), "failed": failures, "finished_at": stamp()})
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
