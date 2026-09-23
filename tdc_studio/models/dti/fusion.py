"""BilinearAttentionFusion — Drug × Target interaction head for DTA prediction.

Design rationale:
- Simple Concat + MLP ignores the *interaction structure* between drug and target.
  A drug atom may specifically contact a target residue, not the whole protein.
- nn.Bilinear(h_drug, h_target) learns a weight matrix W such that the output
  captures pairwise relationships: score = h_drug^T W h_target + b.
- This is the core fusion strategy from DrugBAN (Nature Comp. Sci. 2022),
  which showed +0.05 CI over Concat on cold drug splits.
- Registered as "bilinear" in MODELS registry for YAML-driven config.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from tdc_studio.core.registry import MODELS


@MODELS.register("bilinear_fusion")
class BilinearAttentionFusion(nn.Module):
    """Bilinear interaction head combining drug and target representations.

    Computes: interaction = Bilinear(h_drug, h_target)
              affinity    = MLP(interaction)

    This explicitly models pairwise drug-atom × target-residue interactions
    rather than naively concatenating the two representations.

    Args:
        config: Dict with keys:
            drug_dim   : Drug encoder output dimension (default 256).
            target_dim : Target encoder output dimension (default 256).
            hidden_dim : Bilinear output / MLP hidden dimension (default 512).
            dropout    : Dropout rate in MLP (default 0.2).
            out_dim    : Output dimension — 1 for affinity regression (default 1).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        drug_dim   = config.get("drug_dim",   256)
        target_dim = config.get("target_dim", 256)
        hidden_dim = config.get("hidden_dim", 512)
        dropout    = config.get("dropout",    0.2)
        out_dim    = config.get("out_dim",    1)

        # Bilinear layer: h_drug^T W h_target → hidden_dim
        # Unlike Concat + Linear, this allows cross-modal feature interaction
        self.bilinear = nn.Bilinear(drug_dim, target_dim, hidden_dim)

        # MLP head for final affinity prediction
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(
        self,
        h_drug: torch.Tensor,
        h_target: torch.Tensor,
    ) -> torch.Tensor:
        """Predict binding affinity from drug and target representations.

        Args:
            h_drug   : FloatTensor [B, drug_dim]   from drug encoder.
            h_target : FloatTensor [B, target_dim] from target encoder.

        Returns:
            FloatTensor [B, 1] — predicted affinity (normalised Kd/Ki/etc.).
        """
        interaction = self.bilinear(h_drug, h_target)  # [B, hidden_dim]
        return self.head(interaction)                   # [B, out_dim]
