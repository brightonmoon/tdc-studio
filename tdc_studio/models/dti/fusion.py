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

from typing import Any, Dict, Optional

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
        drug_dim = config.get("drug_dim", 256)
        target_dim = config.get("target_dim", 256)
        hidden_dim = config.get("hidden_dim", 512)
        dropout = config.get("dropout", 0.2)
        out_dim = config.get("out_dim", 1)

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
        return self.head(interaction)  # [B, out_dim]


@MODELS.register("cross_attention_fusion")
@MODELS.register("cross_attention")
class CrossAttentionFusion(nn.Module):
    """Bidirectional Multi-Head Cross-Attention interaction head for DTA prediction.

    Replaces simple Bilinear/MLP pooling with bidirectional cross-attention:
    - Drug queries Target (which protein residues does the drug interact with?)
    - Target queries Drug (which molecular pharmacophores does the protein recognize?)

    Key advantages:
    - Superior expressive power modeling pairwise atom-residue contacts
    - Native Explainable AI (XAI): extracts per-residue attention weights for binding pocket analysis
    - Dimension-flexible: accepts either pooled vectors [B, D] or full token sequences [B, L, D]

    Args:
        config: Dict with keys:
            drug_dim   : Drug encoder output dimension (default 256).
            target_dim : Target encoder output dimension (default 256).
            hidden_dim : Attention embedding dimension (default 256).
            num_heads  : Number of attention heads (default 4).
            dropout    : Dropout probability (default 0.1).
            out_dim    : Final prediction output dimension (default 1).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        drug_dim = config.get("drug_dim", 256)
        target_dim = config.get("target_dim", 256)
        hidden_dim = config.get("hidden_dim", 256)
        num_heads = config.get("num_heads", 4)
        dropout = config.get("dropout", 0.1)
        out_dim = config.get("out_dim", 1)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Linear projections into shared attention dimension
        self.proj_drug = nn.Linear(drug_dim, hidden_dim)
        self.proj_target = nn.Linear(target_dim, hidden_dim)

        self.norm_drug = nn.LayerNorm(hidden_dim)
        self.norm_target = nn.LayerNorm(hidden_dim)

        # Cross-Attention modules: Drug -> Target & Target -> Drug
        self.cross_d2t = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_t2d = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Residual LayerNorms
        self.post_norm_d = nn.LayerNorm(hidden_dim)
        self.post_norm_t = nn.LayerNorm(hidden_dim)

        # Feed-Forward Network per branch
        self.ffn_d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_t = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.final_norm_d = nn.LayerNorm(hidden_dim)
        self.final_norm_t = nn.LayerNorm(hidden_dim)

        # Regression MLP Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
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
        return_attention: bool = False,
        drug_padding_mask: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """Predict binding affinity with bidirectional cross-attention.

        Args:
            h_drug              : FloatTensor [B, drug_dim] or [B, L_drug, drug_dim]
            h_target            : FloatTensor [B, target_dim] or [B, L_target, target_dim]
            return_attention    : If True, returns (affinity, attention_dict)
            drug_padding_mask   : Optional Byte/Bool tensor [B, L_drug] (True for padding)
            target_padding_mask : Optional Byte/Bool tensor [B, L_target] (True for padding)

        Returns:
            affinity: FloatTensor [B, out_dim]
            (optional) attention_dict: {
                "attn_d2t": Tensor,
                "attn_t2d": Tensor,
                "contact_map": Tensor [B, L_drug, L_target]
            }
        """
        # Ensure 3D sequence tensors [B, L, D]
        if h_drug.dim() == 2:
            h_drug = h_drug.unsqueeze(1)  # [B, 1, D_drug]
        if h_target.dim() == 2:
            h_target = h_target.unsqueeze(1)  # [B, 1, D_target]

        # Auto-detect padding masks if not explicitly provided
        if drug_padding_mask is None and h_drug.size(1) > 1:
            drug_padding_mask = h_drug.abs().sum(dim=-1) < 1e-5
        if target_padding_mask is None and h_target.size(1) > 1:
            target_padding_mask = h_target.abs().sum(dim=-1) < 1e-5

        d_proj = self.norm_drug(self.proj_drug(h_drug))
        t_proj = self.norm_target(self.proj_target(h_target))

        # 1. Drug queries Target (Q=drug, K=target, V=target)
        d_cross, attn_d2t = self.cross_d2t(
            query=d_proj,
            key=t_proj,
            value=t_proj,
            key_padding_mask=target_padding_mask
            if (target_padding_mask is not None and target_padding_mask.any())
            else None,
            need_weights=return_attention,
            average_attn_weights=False if return_attention else True,
        )
        d_inter = self.post_norm_d(d_proj + d_cross)
        d_out = self.final_norm_d(d_inter + self.ffn_d(d_inter))

        # 2. Target queries Drug (Q=target, K=drug, V=drug)
        t_cross, attn_t2d = self.cross_t2d(
            query=t_proj,
            key=d_proj,
            value=d_proj,
            key_padding_mask=drug_padding_mask
            if (drug_padding_mask is not None and drug_padding_mask.any())
            else None,
            need_weights=return_attention,
            average_attn_weights=False if return_attention else True,
        )
        t_inter = self.post_norm_t(t_proj + t_cross)
        t_out = self.final_norm_t(t_inter + self.ffn_t(t_inter))

        # Masked pooling across token dimension (strictly exclude padding noise)
        if drug_padding_mask is not None and drug_padding_mask.any():
            d_mask = (~drug_padding_mask).unsqueeze(-1).float()
            d_pooled = (d_out * d_mask).sum(dim=1) / d_mask.sum(dim=1).clamp(min=1.0)
        else:
            d_pooled = d_out.mean(dim=1)  # [B, hidden_dim]

        if target_padding_mask is not None and target_padding_mask.any():
            t_mask = (~target_padding_mask).unsqueeze(-1).float()
            t_pooled = (t_out * t_mask).sum(dim=1) / t_mask.sum(dim=1).clamp(min=1.0)
        else:
            t_pooled = t_out.mean(dim=1)  # [B, hidden_dim]

        fused = torch.cat([d_pooled, t_pooled], dim=-1)  # [B, hidden_dim * 2]
        affinity = self.head(fused)  # [B, out_dim]

        if return_attention:
            contact_map = (
                attn_d2t.mean(dim=1) if (attn_d2t is not None and attn_d2t.dim() == 4) else attn_d2t
            )
            if contact_map is not None:
                if drug_padding_mask is not None and drug_padding_mask.any():
                    contact_map = contact_map * (~drug_padding_mask).unsqueeze(-1).float()
                if target_padding_mask is not None and target_padding_mask.any():
                    contact_map = contact_map * (~target_padding_mask).unsqueeze(1).float()
            return affinity, {
                "attn_d2t": attn_d2t,
                "attn_t2d": attn_t2d,
                "contact_map": contact_map,
            }
        return affinity
