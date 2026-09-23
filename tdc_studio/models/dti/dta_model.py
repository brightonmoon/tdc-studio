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
from tdc_studio.models.dti.fusion import BilinearAttentionFusion
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


@MODELS.register("graph_dta")
class GraphDTAModel(BaseTherapeuticsModel):
    """Drug-Target Affinity model with GNN drug encoder + CNN target encoder.

    Config keys (all nested under top-level config dict):
        drug_encoder  : dict — passed to _build_drug_encoder()
            type        : "gine" (Phase A) | "chembert_encoder" (Phase B)
            hidden_dim  : 256
            num_layers  : 5
        target_encoder: dict — passed to _build_target_encoder()
            type        : "protein_cnn" (Phase A) | "esm2_encoder" (Phase B)
            out_dim     : 256
        fusion        : dict — passed to BilinearAttentionFusion
            hidden_dim  : 512
        use_domain_adaptation: bool — False (Phase A), True (Phase C)
    """

    def __init__(self, config: Dict[str, Any]):
        # task_type="dta" triggers CI+MSE metrics in BaseTherapeuticsModel
        super().__init__({**config, "task_type": "dta"})

        # ── Drug encoder (GINEModel reused from ADMET) ──
        drug_enc_cfg = config.get("drug_encoder", {"type": "gine", "hidden_dim": 256, "num_layers": 5})
        self.drug_encoder: nn.Module = _build_drug_encoder(drug_enc_cfg)
        drug_out_dim = drug_enc_cfg.get("hidden_dim", 256) * 2  # GINEModel: mean+sum concat

        # ── Target encoder (ProteinCNNEncoder new) ──
        target_enc_cfg = config.get("target_encoder", {"type": "protein_cnn", "out_dim": 256})
        self.target_encoder: nn.Module = _build_target_encoder(target_enc_cfg)
        target_out_dim = target_enc_cfg.get("out_dim", 256)

        # ── Fusion head (BilinearAttentionFusion) ──
        fusion_cfg = config.get("fusion", {"hidden_dim": 512})
        fusion_full_cfg = {
            **fusion_cfg,
            "drug_dim":   drug_out_dim,
            "target_dim": target_out_dim,
            "out_dim":    1,
        }
        self.fusion: BilinearAttentionFusion = BilinearAttentionFusion(fusion_full_cfg)

        # ── Phase C placeholder: domain adversarial head ──
        self.use_domain_adaptation: bool = config.get("use_domain_adaptation", False)
        self.domain_head: Optional[nn.Module] = None  # set externally in Phase C

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def extract_features(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (h_drug, h_target) before fusion.

        Used by:
        - Phase C: DomainAdversarialHead receives h_drug for distribution alignment.
        - Interpretability: visualise drug/target representation spaces.

        Returns:
            h_drug   : [B, drug_out_dim]   from drug encoder
            h_target : [B, target_out_dim] from target encoder
        """
        # Drug encoding — GINEModel reads batch["drug_graph"]
        h_drug = self.drug_encoder.extract_features(batch)   # [B, hidden*2]

        # Target encoding — ProteinCNNEncoder reads batch["target_seq"]
        h_target = self.target_encoder.encode_sequence(batch["target_seq"])  # [B, out_dim]

        return h_drug, h_target

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Predict binding affinity for (Drug, Target) pairs.

        Args:
            batch: Dict containing at minimum:
                "drug_graph"  : torch_geometric.data.Batch (from DTADataModule)
                "target_seq"  : LongTensor [B, max_len]    (from AminoAcidTokenizer)

        Returns:
            FloatTensor [B, 1] — normalised affinity predictions.
        """
        h_drug, h_target = self.extract_features(batch)
        return self.fusion(h_drug, h_target)  # [B, 1]

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        domain_logits: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        lambda_domain: float = 0.1,
    ) -> torch.Tensor:
        """Compute MSE affinity loss (+ optional domain adversarial loss in Phase C).

        Args:
            preds         : [B, 1] or [B] predicted affinity.
            targets       : [B] ground-truth affinity (normalised).
            domain_logits : [B, 2] from DomainAdversarialHead (Phase C only).
            domain_labels : [B] 0=source / 1=target domain (Phase C only).
            lambda_domain : Weight for domain loss (default 0.1).

        Returns:
            Scalar loss tensor.
        """
        affinity_loss = nn.functional.mse_loss(
            preds.squeeze(-1), targets.float()
        )

        if domain_logits is not None and domain_labels is not None:
            domain_loss = nn.functional.cross_entropy(domain_logits, domain_labels)
            return affinity_loss + lambda_domain * domain_loss

        return affinity_loss
