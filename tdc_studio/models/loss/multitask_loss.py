"""Masked Multi-Task Loss with Kendall et al. Homoscedastic Uncertainty Weighting."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMultiTaskLoss(nn.Module):
    """Multi-task loss supporting missing labels (masking) and mixed regression/classification tasks."""

    def __init__(
        self,
        task_names: List[str],
        task_types: List[str],
        use_uncertainty: bool = True,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.task_names = task_names
        self.task_types = [t.lower() for t in task_types]
        self.num_tasks = len(task_names)
        self.use_uncertainty = use_uncertainty
        self.task_weights = task_weights or {}

        if len(self.task_types) != self.num_tasks:
            raise ValueError("task_names and task_types must have identical length.")

        # Trainable log variance parameters (log(sigma^2)) initialized to 0 (sigma=1)
        if self.use_uncertainty:
            self.log_vars = nn.Parameter(torch.zeros(self.num_tasks, dtype=torch.float32))
        else:
            self.register_parameter("log_vars", None)

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute masked multi-task loss.

        Args:
            preds: Tensor of shape [B, num_tasks]
            targets: Tensor of shape [B, num_tasks]
            mask: Optional boolean or binary Tensor of shape [B, num_tasks] (1: valid, 0: missing)

        Returns:
            total_loss: Scalar Tensor for backpropagation
            task_losses_dict: Dictionary of unweighted loss values per task for logging
        """
        device = preds.device
        if mask is None:
            # If mask not provided, consider non-NaN targets as valid
            mask = ~torch.isnan(targets)
        else:
            mask = mask.bool() & ~torch.isnan(targets)

        task_losses: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_task_count = 0

        for i in range(self.num_tasks):
            valid_idx = mask[:, i]
            n_valid = int(valid_idx.sum().item())
            if n_valid == 0:
                continue

            t_preds = preds[valid_idx, i]
            t_targets = targets[valid_idx, i]
            t_type = self.task_types[i]

            if t_type == "regression":
                t_loss = F.smooth_l1_loss(t_preds, t_targets, beta=1.0)
            elif t_type in ("binary_classification", "classification"):
                t_preds_clamped = torch.clamp(t_preds, min=-15.0, max=15.0)
                t_loss = F.binary_cross_entropy_with_logits(t_preds_clamped, t_targets)
            else:
                t_loss = F.smooth_l1_loss(t_preds, t_targets, beta=1.0)

            task_losses[self.task_names[i]] = float(t_loss.item())
            valid_task_count += 1

            w = self.task_weights.get(self.task_names[i], 1.0)
            if self.use_uncertainty and self.log_vars is not None:
                log_var = torch.clamp(self.log_vars[i], min=-4.0, max=4.0)
                precision = torch.exp(-log_var)
                if t_type == "regression":
                    weighted_loss = 0.5 * precision * t_loss + 0.5 * log_var
                else:
                    weighted_loss = precision * t_loss + 0.5 * log_var
                total_loss = total_loss + w * weighted_loss
            else:
                total_loss = total_loss + w * t_loss

        if valid_task_count > 0 and not self.use_uncertainty:
            denom = sum(
                self.task_weights.get(self.task_names[i], 1.0)
                for i in range(self.num_tasks)
                if mask[:, i].any()
            ) if self.task_weights else float(valid_task_count)
            total_loss = total_loss / max(1e-6, denom)

        return total_loss, task_losses
