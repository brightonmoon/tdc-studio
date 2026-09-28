"""Numerically Stable Binary Focal Loss with Logits for Imbalanced Classification."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLossWithLogits(nn.Module):
    """Binary Focal Loss with Logits for hard-example focusing and class imbalance mitigation.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where:
        p_t = sigmoid(z) if y=1 else 1 - sigmoid(z)
        alpha_t = alpha if y=1 else 1 - alpha

    Numerically stable formulation using standard BCEWithLogits:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = exp(-bce)
        loss = alpha_t * ((1 - p_t) ** gamma) * bce
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.70,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha) if alpha is not None else None
        self.reduction = reduction.lower()
        self.register_buffer("pos_weight", pos_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute binary focal loss.

        Args:
            logits: Unnormalized logits of shape [N] or [N, 1].
            targets: Binary ground truth labels (0.0 or 1.0) of matching shape.
            mask: Optional valid element mask of boolean type.

        Returns:
            Computed scalar or unreduced loss tensor.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        if mask is not None:
            valid = mask.view(-1).bool() & ~torch.isnan(targets)
            if not valid.any():
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            logits = logits[valid]
            targets = targets[valid]
        else:
            valid = ~torch.isnan(targets)
            if not valid.all():
                logits = logits[valid]
                targets = targets[valid]

        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # 1. Base binary cross entropy per element
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )

        # 2. Compute p_t = exp(-bce)
        p_t = torch.exp(-bce)

        # 3. Modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        # 4. Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_loss = alpha_t * modulating_factor * bce
        else:
            focal_loss = modulating_factor * bce

        # 5. Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
