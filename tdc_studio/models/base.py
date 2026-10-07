import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseTherapeuticsModel(nn.Module, ABC):
    """Abstract base model for molecular property and interaction prediction."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.task_type = config.get("task_type", "regression")

        # Dynamic Loss Strategy
        loss_type = str(
            config.get("loss_type", "smooth_l1" if self.task_type == "regression" else "default")
        ).lower()

        if self.task_type == "regression":
            if loss_type in ("smooth_l1", "smoothl1", "huber"):
                beta = float(config.get("huber_beta", 1.0))
                self.criterion: nn.Module = nn.SmoothL1Loss(beta=beta)
            elif loss_type in ("l1", "mae"):
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()
        elif self.task_type == "binary_classification":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.task_type == "multiclass_classification":
            self.criterion = nn.CrossEntropyLoss()
        self.pearson_weight = float(config.get("pearson_weight", 0.0))
        self.r2_weight = float(config.get("r2_weight", 0.0))

    @abstractmethod
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass taking a collated batch dictionary."""
        pass

    def compute_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute task-appropriate loss between predictions and targets with optional masking."""
        preds_flat = preds.squeeze(-1) if preds.ndim > 1 else preds
        targets_flat = targets.squeeze(-1) if targets.ndim > 1 else targets

        if mask is not None:
            mask_flat = mask.squeeze(-1) if mask.ndim > 1 else mask
            valid = mask_flat.bool() & ~torch.isnan(targets_flat)
            if not valid.any():
                return torch.tensor(0.0, device=preds.device, requires_grad=True)
            preds_flat = preds_flat[valid]
            targets_flat = targets_flat[valid]
        else:
            valid = ~torch.isnan(targets_flat)
            if not valid.all():
                preds_flat = preds_flat[valid]
                targets_flat = targets_flat[valid]

        if preds_flat.numel() == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        base_loss = self.criterion(preds_flat, targets_flat)

        if self.task_type == "regression" and preds_flat.numel() >= 8 and self.pearson_weight > 0:
            var_x = torch.var(preds_flat, unbiased=False)
            var_y = torch.var(targets_flat, unbiased=False)

            if var_x >= 1e-3 and var_y >= 1e-3:
                vx = preds_flat - torch.mean(preds_flat)
                vy = targets_flat - torch.mean(targets_flat)
                std_x = torch.sqrt(var_x * preds_flat.numel() + 1e-4)
                std_y = torch.sqrt(var_y * targets_flat.numel() + 1e-4)
                r = torch.sum(vx * vy) / (std_x * std_y)
                p_loss = 1.0 - torch.clamp(r, -0.999, 0.999)
                if torch.isfinite(p_loss) and float(p_loss.item()) > 0.0:
                    base_loss = base_loss + self.pearson_weight * p_loss

        return base_loss

    def save(self, path: str) -> None:
        """Save model state dict ensuring parent directory exists."""
        parent_dir = os.path.dirname(os.path.abspath(path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str = "cpu") -> None:
        """Load model state dict with safe weights_only=True."""
        try:
            state_dict = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
