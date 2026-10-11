"""Image-normalized L=5/O=3 motion model for training and inference.

Layer names and tensor operations match the selected V2 checkpoint. No legacy
target-size normalization, training losses, or alternate motion models are kept.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


HISTORY_LEN = 5
MOTION_ORDER = 3
MODEL_NAME = "MotionCenterPointMLP_ImageRelative_V2"


def _take_last(value: torch.Tensor, length: int, name: str) -> torch.Tensor:
    if value.ndim < 2 or value.shape[1] < length:
        raise ValueError(f"{name} needs at least {length} time steps")
    return value[:, -length:]


def _axis_masked_softmax(logits: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    quality = quality.unsqueeze(1).expand_as(logits).clamp_min(0.0)
    mask = quality > 0.0
    no_valid = ~mask.any(dim=-1, keepdim=True)
    safe_mask = torch.where(no_valid, torch.ones_like(mask), mask)
    safe_quality = torch.where(no_valid, torch.ones_like(quality), quality)
    adjusted = logits + torch.log(safe_quality.clamp_min(1e-4))
    return torch.softmax(adjusted.masked_fill(~safe_mask, -1e4), dim=-1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.block(value))


class MotionGateMLP(nn.Module):
    """The selected weighted, score-aware image-relative motion network only."""

    history_len = HISTORY_LEN
    motion_order = MOTION_ORDER
    order_lengths = (4, 3, 2)
    center_encoding = "image_normalized_relative"

    def __init__(self, *, hidden_dim: int = 64, num_layers: int = 3,
                 dropout: float = 0.0, eta: float = 1.0,
                 score_floor: float = 0.10) -> None:
        super().__init__()
        if hidden_dim != 64 or num_layers != 3:
            raise ValueError("This package supports the selected 64-wide, 3-layer model only")
        self.eta = float(eta)
        self.score_floor = float(score_floor)
        self.input_projection = nn.Sequential(
            nn.Linear(HISTORY_LEN * 4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.trunk = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_layers - 1)]
        )
        self.velocity_weight_head = nn.Linear(hidden_dim, 8)
        self.acceleration_weight_head = nn.Linear(hidden_dim, 6)
        self.jerk_weight_head = nn.Linear(hidden_dim, 4)
        self.coefficient_head = nn.Linear(hidden_dim, 6)
        # Preserve the reference training bias; checkpoint loading replaces it.
        with torch.no_grad():
            self.coefficient_head.bias.copy_(
                torch.logit(torch.tensor([0.20, 0.10, 0.05] * 2))
            )

    def forward(self, history_boxes: torch.Tensor,
                history_scores: torch.Tensor | None = None,
                history_valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        history = _take_last(history_boxes.float(), HISTORY_LEN, "history_boxes")
        cx, cy, width, height = history.unbind(dim=-1)
        history = torch.stack([cx.clamp(0.0, 1.0), cy.clamp(0.0, 1.0),
                               width.clamp(1e-4, 1.0), height.clamp(1e-4, 1.0)], dim=-1)
        batch = history.shape[0]
        if history_scores is None:
            scores = history.new_ones((batch, HISTORY_LEN, 1))
        else:
            scores = history_scores.to(device=history.device, dtype=history.dtype)
            if scores.ndim == 2:
                scores = scores.unsqueeze(-1)
            scores = _take_last(scores, HISTORY_LEN, "history_scores")
            scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=1.0, neginf=0.0)
            in_range = (scores >= 0.0) & (scores <= 1.0)
            scores = torch.where(in_range, scores, torch.sigmoid(scores)).clamp(0.0, 1.0)
            scores = scores.to(dtype=history.dtype)
        if history_valid is None:
            valid = history.new_ones((batch, HISTORY_LEN, 1))
        else:
            valid = history_valid.to(device=history.device, dtype=history.dtype)
            if valid.ndim == 2:
                valid = valid.unsqueeze(-1)
            valid = (_take_last(valid, HISTORY_LEN, "history_valid") > 0.5).to(history.dtype)
        floor = min(max(self.score_floor, 0.0), 1.0)
        quality = valid * (floor + (1.0 - floor) * scores)
        centers = history[..., :2]
        last_box = history[:, -1]
        # Retain the exact V2 operation: centers are already divided by image W/H.
        relative_centers = (centers - centers[:, -1:, :]) / last_box.new_ones((batch, 1, 1))
        model_input = torch.cat([relative_centers.reshape(batch, -1),
                                 scores.reshape(batch, -1), valid.reshape(batch, -1)], dim=-1)
        encoded = self.trunk(self.input_projection(model_input))
        heads = (self.velocity_weight_head, self.acceleration_weight_head, self.jerk_weight_head)
        aggregated_orders, weight_orders = [], []
        for order, head in enumerate(heads, start=1):
            sequence = torch.diff(centers, n=order, dim=1)
            logits = head(encoded).view(-1, 2, sequence.shape[1])
            reliability = quality.squeeze(-1).unfold(1, order + 1, 1).amin(dim=-1)
            weights = _axis_masked_softmax(logits, reliability)
            weight_orders.append(weights)
            aggregated_orders.append((sequence.transpose(1, 2) * weights).sum(dim=-1))
        raw_coefficients = self.coefficient_head(encoded).view(-1, 2, MOTION_ORDER)
        coefficients = torch.sigmoid(raw_coefficients)
        contributions = torch.stack([
            coefficients[..., order - 1] * (self.eta ** order) * aggregated_orders[order - 1]
            for order in range(1, MOTION_ORDER + 1)
        ], dim=1)
        delta = contributions.sum(dim=1)
        center = (last_box[:, :2] + delta).clamp(0.0, 1.0)
        temporal_weights = history.new_zeros((batch, MOTION_ORDER, 2, 4))
        for index, weights in enumerate(weight_orders):
            temporal_weights[:, index, :, -weights.shape[-1]:] = weights
        return {
            "pred_center": center, "center_delta": delta,
            "motion_coefficients": coefficients, "raw_motion_coefficients": raw_coefficients,
            "temporal_weights_dynamic": temporal_weights, "order_contributions": contributions,
            "order_lengths": torch.tensor(self.order_lengths, device=history.device, dtype=torch.long),
            "history_boxes": history, "history_scores": scores, "history_valid": valid,
            "relative_center_input": relative_centers,
        }


@dataclass(frozen=True)
class LoadedMotion:
    model: MotionGateMLP
    device: torch.device
    checkpoint_path: Path | None = None
    history_len: int = HISTORY_LEN
    motion_order: int = MOTION_ORDER


def load_motion(checkpoint: str | Path, device: str | torch.device = "cpu") -> LoadedMotion:
    """Load the trusted selected checkpoint, strictly checking every state key.

    Like the validated original loader, this accepts its training metadata via
    ``weights_only=False``. Only load checkpoints from a trusted source.
    """
    path = Path(checkpoint).expanduser().resolve(strict=True)
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("args", {}), dict):
        raise TypeError("Expected a checkpoint dictionary with dictionary args")
    args = payload.get("args", {})
    length = int(payload.get("temporal_window", payload.get("history_len", 4)))
    order = int(payload.get("motion_order", 3))
    use_score = bool(payload.get("use_sam_score", args.get("use_sam_score", True)))
    history = str(payload.get("training_history", args.get("training_history", "sam"))).strip().lower()
    aggregation = str(payload.get("temporal_aggregation", args.get("temporal_aggregation", "weighted")))
    if (length, order, use_score, history, aggregation) != (5, 3, True, "sam", "weighted"):
        raise ValueError("Expected the selected SAM-history, score-aware, weighted L5/O3 checkpoint")
    state = payload.get("motion_gate", payload.get("model"))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint has no motion_gate/model state dictionary")
    model = MotionGateMLP(
        hidden_dim=int(args.get("hidden_dim", 64)), num_layers=int(args.get("num_layers", 3)),
        dropout=float(args.get("dropout", 0.0)), eta=float(args.get("eta", 1.0)),
        score_floor=float(args.get("model_score_floor", args.get("score_floor", 0.10))),
    )
    model.load_state_dict(state, strict=True)
    if str(device) == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device).eval().requires_grad_(False)
    return LoadedMotion(model=model, device=device, checkpoint_path=path)
