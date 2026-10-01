"""Hybrid Retrosynthesis Policy combining Neural Seq2Seq and Rule-based Templates."""

from typing import Any, Dict, List, Optional, Tuple

from tdc_studio.core.registry import MODELS
from tdc_studio.models.retrosynthesis.base import BaseRetroModel
from tdc_studio.models.retrosynthesis.rule_policy import RuleRetroPolicy
from tdc_studio.models.retrosynthesis.seq2seq_retro import Seq2SeqRetroModel


@MODELS.register("hybrid_retro_policy")
class HybridRetroPolicy(BaseRetroModel):
    """2-Tier Hybrid Retrosynthesis Engine.

    Tier 1: Neural Seq2Seq Transformer model for high generalizability and novel reactions.
    Tier 2: Deterministic RDKit SMARTS Rule Policy for 100% validity guarantee and rapid fallback.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        neural_model: Optional[BaseRetroModel] = None,
        rule_policy: Optional[RuleRetroPolicy] = None,
    ):
        super().__init__(config)
        self.neural_model = neural_model or Seq2SeqRetroModel(self.config.get("neural", {}))
        self.rule_policy = rule_policy or RuleRetroPolicy(self.config.get("rule", {}))
        self.neural_weight = float(self.config.get("neural_weight", 0.7))
        self.rule_weight = float(self.config.get("rule_weight", 0.3))

    def predict_reactants(
        self,
        product_smiles: str,
        top_k: int = 5,
        reaction_type: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Predict candidates using neural model first, filling and blending with rule policy."""
        seen = set()
        merged_candidates: List[Tuple[str, float]] = []

        # 1. Query neural model
        try:
            neural_preds = self.neural_model.predict_reactants(
                product_smiles, top_k=top_k, reaction_type=reaction_type
            )
            for reactants, score in neural_preds:
                if reactants not in seen:
                    seen.add(reactants)
                    merged_candidates.append((reactants, round(score * self.neural_weight, 4)))
        except Exception:
            pass

        # 2. Query rule-based policy to complement / fallback
        rule_preds = self.rule_policy.predict_reactants(
            product_smiles, top_k=top_k, reaction_type=reaction_type
        )
        for reactants, score in rule_preds:
            if reactants not in seen:
                seen.add(reactants)
                merged_candidates.append((reactants, round(score * self.rule_weight, 4)))

        # Sort by final blended score descending
        merged_candidates.sort(key=lambda x: x[1], reverse=True)
        return merged_candidates[:top_k]
