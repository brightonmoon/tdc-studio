"""GraphDTAModel — assembled Drug-Target Affinity model for Phase A.

Architecture:
    Drug SMILES → GINEConv (drug encoder, reused from ADMET)
              → extract_features() → h_drug [B, hidden_dim*2]
    AA Sequence → ProteinCNNEncoder (target encoder, new)
              → encode_sequence() → h_target [B, out_dim]
    (h_drug, h_target) → BilinearAttentionFusion → affinity [B, 1]

Key design decisions:
1. Drug encoder is MODELS.build(drug_enc_cfg) — dynamic from config.
   Phase A: type="gine"  →  Phase B: type="chembert_encoder"
   Only YAML changes needed, no model code modification.

2. Target encoder is similarly dynamic.
   Phase A: type="protein_cnn"  →  Phase B: type="esm2_encoder"

3. forward(batch) reads both batch["drug_graph"] and batch["target_seq"],
   so the DTADataModule must supply both (which it does).

4. extract_features(batch) returns (h_drug, h_target) for interpretability
   and domain adaptation (Phase C domain discriminator needs feature vectors).

5. Registered as "graph_dta" for YAML: model: {type: graph_dta}.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from tdc_studio.core.registry import MODELS
from tdc_studio.models.base import BaseTherapeuticsModel
from tdc_studio.models.dti.protein_encoder import ProteinCNNEncoder


def _build_drug_encoder(cfg: Dict[str, Any]) -> nn.Module:
    """Build drug encoder from config. Falls back to GINEModel if type not set."""
    encoder_type = cfg.get("type", "gine")
    # GINEModel is registered under "gine" — reused from ADMET, no changes needed
    return MODELS.build({**cfg, "type": encoder_type})


def _build_target_encoder(cfg: Dict[str, Any]) -> nn.Module:
    """Build target encoder from config. Falls back to ProteinCNNEncoder."""
    encoder_type = cfg.get("type", "protein_cnn")
    if encoder_type == "protein_cnn":
        return ProteinCNNEncoder(cfg)
    # Phase B: esm2_encoder or chembert_encoder will be registered here
    return MODELS.build({**cfg, "type": encoder_type})


def _build_fusion(cfg: Dict[str, Any]) -> nn.Module:
    """Build fusion head from config. Falls back to BilinearAttentionFusion."""
    fusion_type = cfg.get("type", "bilinear_fusion")
    return MODELS.build({**cfg, "type": fusion_type})


@MODELS.register("graph_dta")
class GraphDTAModel(BaseTherapeuticsModel):
    """Drug-Target Affinity model with GNN/LM drug encoder + CNN/LM target encoder.

    Config keys (all nested under top-level config dict):
        drug_encoder  : dict — passed to _build_drug_encoder()
            type        : "gine" (Phase A) | "chembert_encoder" (Phase B)
            hidden_dim  : 256
            num_layers  : 5
        target_encoder: dict — passed to _build_target_encoder()
            type        : "protein_cnn" (Phase A) | "esm2_encoder" (Phase B)
            out_dim     : 256
        fusion        : dict — passed to _build_fusion()
            type        : "bilinear_fusion" (Phase A/B) | "cross_attention" (Phase C)
            hidden_dim  : 256 or 512
            num_heads   : 4 (for cross_attention)
        use_domain_adaptation: bool — False (Phase A/B), True (Phase C)
    """

    def __init__(self, config: Dict[str, Any]):
        # task_type="dta" triggers CI+MSE metrics in BaseTherapeuticsModel
        super().__init__({**config, "task_type": "dta"})

        # ── Drug encoder (GINEModel or ChemBERTaEncoder) ──
        drug_enc_cfg = config.get(
            "drug_encoder", {"type": "gine", "hidden_dim": 256, "num_layers": 5}
        )
        self.drug_encoder: nn.Module = _build_drug_encoder(drug_enc_cfg)
        drug_out_dim = getattr(
            self.drug_encoder,
            "out_dim",
            drug_enc_cfg.get("out_dim", drug_enc_cfg.get("hidden_dim", 256) * 2),
        )

        # ── Target encoder (ProteinCNNEncoder or ESM2Encoder) ──
        target_enc_cfg = config.get("target_encoder", {"type": "protein_cnn", "out_dim": 256})
        self.target_encoder: nn.Module = _build_target_encoder(target_enc_cfg)
        target_out_dim = getattr(
            self.target_encoder,
            "out_dim",
            target_enc_cfg.get("out_dim", 256),
        )

        # ── Fusion head (BilinearAttentionFusion or CrossAttentionFusion) ──
        fusion_cfg = config.get("fusion", {"hidden_dim": 512})
        out_dim = config.get("out_dim", fusion_cfg.get("out_dim", 1))
        fusion_full_cfg = {
            **fusion_cfg,
            "drug_dim": drug_out_dim,
            "target_dim": target_out_dim,
            "out_dim": out_dim,
        }
        self.fusion: nn.Module = _build_fusion(fusion_full_cfg)
        self.out_dim = out_dim

        # ── Phase C placeholder: domain adversarial head ──
        self.use_domain_adaptation: bool = config.get("use_domain_adaptation", False)
        self.domain_head: Optional[nn.Module] = None  # set externally in Phase C

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def extract_features(
        self, batch: Dict[str, Any], return_sequence: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (h_drug, h_target) before fusion.

        Used by:
        - Phase C: DomainAdversarialHead receives h_drug for distribution alignment.
        - Interpretability: visualise drug/target representation spaces.

        Returns:
            h_drug   : [B, drug_out_dim] or [B, L_drug, drug_out_dim]
            h_target : [B, target_out_dim] or [B, L_target, target_out_dim]
        """
        # Drug encoding — GINEModel reads batch["drug_graph"], ChemBERTa reads batch["drug_smiles_str"]
        if hasattr(self.drug_encoder, "extract_features"):
            try:
                h_drug = self.drug_encoder.extract_features(batch, return_sequence=return_sequence)
            except TypeError:
                h_drug = self.drug_encoder.extract_features(batch)
        else:
            h_drug = self.drug_encoder(batch)

        # Target encoding — ESM-2 reads batch, ProteinCNN reads batch["target_seq"]
        if hasattr(self.target_encoder, "extract_features"):
            try:
                h_target = self.target_encoder.extract_features(
                    batch, return_sequence=return_sequence
                )
            except TypeError:
                h_target = self.target_encoder.extract_features(batch)
        else:
            h_target = self.target_encoder.encode_sequence(batch["target_seq"])

        return h_drug, h_target

    def forward(
        self,
        batch: Dict[str, Any],
        return_attention: bool = False,
        return_sequence: Optional[bool] = None,
    ) -> Any:
        """Predict binding affinity for (Drug, Target) pairs.

        Args:
            batch: Dict containing at minimum:
                "drug_graph"  : torch_geometric.data.Batch (from DTADataModule)
                "target_seq"  : LongTensor [B, max_len]    (from AminoAcidTokenizer)
            return_attention: If True and supported, returns (affinity, attention_dict)
            return_sequence : If True and supported by encoders, uses token-level sequences

        Returns:
            FloatTensor [B, out_dim] — normalised affinity predictions,
            or (affinity, attn_dict) if return_attention=True.
        """
        h_drug, h_target = self.extract_features(batch, return_sequence=return_sequence)

        # Pocket-Specific Slicing: selectively slice target representations to binding pocket residues
        pocket_indices = batch.get("pocket_indices")
        if pocket_indices is not None and h_target.dim() == 3:
            from tdc_studio.features.pocket_extractor import slice_pocket_embeddings
            h_target = slice_pocket_embeddings(h_target, pocket_indices)

        extra_kwargs: Dict[str, Any] = {}
        for k in ("pocket_coords", "residue_importance", "target_padding_mask", "drug_padding_mask"):
            if k in batch:
                val = batch[k]
                if (
                    pocket_indices is not None
                    and k in ("target_padding_mask", "residue_importance", "pocket_coords")
                    and val is not None
                    and hasattr(val, "shape")
                ):
                    target_dim_check = val.shape[-2] if k == "pocket_coords" else val.shape[-1]
                    if target_dim_check > len(pocket_indices):
                        from tdc_studio.features.pocket_extractor import slice_pocket_embeddings
                        val = slice_pocket_embeddings(val, pocket_indices)
                extra_kwargs[k] = val


        if return_attention:
            try:
                return self.fusion(h_drug, h_target, return_attention=True, **extra_kwargs)
            except TypeError:
                try:
                    return self.fusion(h_drug, h_target, return_attention=True)
                except TypeError:
                    pass
        try:
            return self.fusion(h_drug, h_target, **extra_kwargs)
        except TypeError:
            return self.fusion(h_drug, h_target)  # [B, out_dim]

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        domain_logits: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        lambda_domain: float = 0.1,
    ) -> torch.Tensor:
        """Compute MSE affinity loss (+ optional domain adversarial loss in Phase C).

        Args:
            preds         : [B, out_dim] predicted affinity.
            targets       : [B, out_dim] or [B] ground-truth affinity (normalised).
            mask          : Optional [B, out_dim] mask for multi-task affinity (Task F-3).
            domain_logits : [B, 2] from DomainAdversarialHead (Phase C only).
            domain_labels : [B] 0=source / 1=target domain (Phase C only).
            lambda_domain : Weight for domain loss (default 0.1).

        Returns:
            Scalar loss tensor.
        """
        if mask is not None:
            preds = preds.view_as(targets)
            diff_sq = (preds - targets.float()) ** 2
            masked_diff = diff_sq * mask.float()
            valid_count = torch.clamp(mask.float().sum(), min=1.0)
            affinity_loss = masked_diff.sum() / valid_count
        else:
            if preds.shape[-1] == 1 and targets.dim() == 1:
                preds = preds.squeeze(-1)
            affinity_loss = nn.functional.mse_loss(preds, targets.float())

        if domain_logits is not None and domain_labels is not None:
            domain_loss = nn.functional.cross_entropy(domain_logits, domain_labels)
            return affinity_loss + lambda_domain * domain_loss

        return affinity_loss
