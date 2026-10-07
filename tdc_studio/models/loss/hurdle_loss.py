"""Two-Stage Hurdle Multi-Task Loss Formulation for extreme binding distributions."""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class HurdleMultiTaskLoss(nn.Module):
    """End-to-End Two-Stage Hurdle Multi-Task Loss.

    Simultaneously optimizes:
    1. Binary Gating Cross-Entropy: separates extreme high-binding (>= threshold) from low/mid binding.
    2. High-binding Specialist Loss: MSE on extreme high-binding compounds.
    3. Low/Mid-binding Specialist Loss: Weighted MSE heavily prioritizing rare low-binding (< 70%) compounds.
    4. End-to-End Mixture Loss: MSE of the gated mixture prediction p_gate * y_high + (1 - p_gate) * y_low.
    """

    def __init__(
        self,
        high_threshold: float = 90.0,
        high_subthreshold: float = 85.0,
        low_threshold: float = 70.0,
        weight_gate: float = 1.0,
        weight_high: float = 0.5,
        weight_low: float = 0.8,
        weight_mixture: float = 1.0,
        low_sample_weight: float = 3.5,
        is_logit_target: bool = False,
    ):
        super().__init__()
        self.high_threshold = high_threshold
        self.high_subthreshold = high_subthreshold
        self.low_threshold = low_threshold
        self.weight_gate = weight_gate
        self.weight_high = weight_high
        self.weight_low = weight_low
        self.weight_mixture = weight_mixture
        self.low_sample_weight = low_sample_weight
        self.is_logit_target = is_logit_target

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        preds: Union[Dict[str, torch.Tensor], torch.Tensor],
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined Hurdle loss.

        Args:
            preds: Dict containing 'gate_logits', 'pred_high', 'pred_low', 'pred_mixture',
                   or a Tensor of shape (B, 4) with [gate_logits, pred_high, pred_low, pred_mixture].
            targets: Continuous target binding percentage (0 to 100) or logit tensor of shape (B,).
            mask: Optional boolean mask of shape (B,).

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        if isinstance(preds, torch.Tensor):
            if preds.dim() == 2 and preds.size(1) >= 4:
                gate_logits = preds[:, 0]
                pred_high = preds[:, 1]
                pred_low = preds[:, 2]
                pred_mixture = preds[:, 3]
            elif preds.dim() == 2 and preds.size(1) == 1:
                # Fallback if single prediction passed
                pred_mixture = preds.squeeze(-1)
                loss = F.mse_loss(pred_mixture, targets.squeeze(-1))
                return loss, {"total_loss": loss.item()}
            else:
                gate_logits = preds[..., 0]
                pred_high = preds[..., 1]
                pred_low = preds[..., 2]
                pred_mixture = preds[..., 3]
        else:
            gate_logits = preds["gate_logits"].squeeze(-1)
            pred_high = preds["pred_high"].squeeze(-1)
            pred_low = preds["pred_low"].squeeze(-1)
            pred_mixture = preds.get(
                "pred_mixture",
                torch.sigmoid(gate_logits) * pred_high + (1.0 - torch.sigmoid(gate_logits)) * pred_low,
            ).squeeze(-1)

        y = targets.squeeze(-1).float()

        if mask is not None:
            m = mask.squeeze(-1).bool()
            gate_logits = gate_logits[m]
            pred_high = pred_high[m]
            pred_low = pred_low[m]
            pred_mixture = pred_mixture[m]
            y = y[m]

        if y.numel() == 0:
            zero_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
            return zero_loss, {"total_loss": 0.0}

        # 1. Gating Binary Cross Entropy
        # Define high-binding ground truth
        y_is_high = (y >= self.high_threshold).float()
        loss_gate = self.bce_loss(gate_logits, y_is_high)

        # 2. High-binding Specialist Loss
        high_mask = y >= self.high_subthreshold
        if high_mask.sum() > 0:
            loss_high = F.mse_loss(pred_high[high_mask], y[high_mask])
        else:
            loss_high = F.mse_loss(pred_high, y) * 0.1

        # 3. Low-binding Specialist Loss (weighted by low binding priority)
        sample_weights = torch.ones_like(y)
        sample_weights[y < self.low_threshold] = self.low_sample_weight
        sample_weights[(y >= self.low_threshold) & (y < self.high_threshold)] = 2.0
        sample_weights[y >= self.high_threshold] = 0.5
        loss_low = torch.mean(sample_weights * (pred_low - y) ** 2)

        # 4. Mixture Loss
        loss_mixture = F.mse_loss(pred_mixture, y)

        total_loss = (
            self.weight_gate * loss_gate
            + self.weight_high * loss_high
            + self.weight_low * loss_low
            + self.weight_mixture * loss_mixture
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_gate": float(loss_gate.item()),
            "loss_high": float(loss_high.item()),
            "loss_low": float(loss_low.item()),
            "loss_mixture": float(loss_mixture.item()),
        }

        return total_loss, metrics
