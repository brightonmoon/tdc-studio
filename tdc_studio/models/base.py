"""Base model architecture with dynamic loss calculation."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BaseTherapeuticsModel(nn.Module, ABC):
    """Abstract base model for molecular property and interaction prediction."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.task_type = config.get("task_type", "regression")

        # Dynamic Loss Strategy
        if self.task_type == "regression":
            self.criterion: nn.Module = nn.MSELoss()
        elif self.task_type == "binary_classification":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.task_type == "multiclass_classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

    @abstractmethod
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass taking a collated batch dictionary."""
        pass

    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute task-appropriate loss between predictions and targets."""
        preds_flat = preds.squeeze(-1) if preds.ndim > 1 else preds
        targets_flat = targets.squeeze(-1) if targets.ndim > 1 else targets
        return self.criterion(preds_flat, targets_flat)
