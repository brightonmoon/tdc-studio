"""Wet-Lab Experiment Recommender for Closed-Loop Drug Discovery.

Ranks and selects the "Top 10 Priority Candidates" for chemical synthesis and binding assay:
1. Conformal Prediction uncertainty quantification (exact 95% confidence intervals).
2. Bayesian Acquisition scoring (Expected Improvement / UCB).
3. Chemical Diversity filtering (MaxMin Tanimoto distance to avoid redundant analogues).
4. Automated priority tiering and rationale generation for medicinal chemists.
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from tdc_studio.active_learning.acquisition import compute_acquisition_score
from tdc_studio.active_learning.diversity import calculate_tanimoto_similarity, maxmin_diversity_picker

logger = logging.getLogger("tdc_studio.active_learning.recommender")


@dataclass
class WetLabCandidate:
    """Individual compound recommended for laboratory synthesis and assay validation."""
    rank: int
    smiles: str
    predicted_affinity: float       # Predicted pKd / pKi
    lower_95: float                 # 95% Conformal Lower Bound
    upper_95: float                 # 95% Conformal Upper Bound
    conformal_uncertainty: float    # Interval half-width q_hat
    acquisition_score: float        # EI / UCB metric
    min_tanimoto_distance: float    # Distance to other selected candidates (1 - Tanimoto)
    synthesis_priority: str         # "Urgent (Top 1-3)", "High (Top 4-6)", "Standard (Top 7-10)"
    rationale: str                  # Explainable decision context for wet-lab scientists

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActiveLearningRecommender:
    """Active Learning engine recommending Top-K molecules for experimental synthesis."""

    def __init__(
        self,
        strategy: str = "ei",
        confidence_level: float = 0.95,
        tradeoff_lambda: float = 0.4,
        min_diversity_distance: float = 0.35,
    ):
        self.strategy = strategy
        self.confidence_level = confidence_level
        self.tradeoff_lambda = tradeoff_lambda
        self.min_diversity_distance = min_diversity_distance

    def recommend(
        self,
        candidate_smiles: List[str],
        predictions: Union[List[float], np.ndarray],
        uncertainties: Union[List[float], np.ndarray, float],
        current_best: Optional[float] = None,
        top_k: int = 10,
        target_name: Optional[str] = None,
    ) -> List[WetLabCandidate]:
        """Rank and recommend Top-K wet-lab candidates from a screening pool.

        Args:
            candidate_smiles: Pool of unmeasured candidate SMILES strings.
            predictions: Model predicted affinities (e.g. pKd) for each SMILES.
            uncertainties: Conformal margin q_hat (per-sample array or single scalar).
            current_best: Highest confirmed experimental affinity to date (for EI).
            top_k: Number of candidates to select (default 10).
            target_name: Optional protein/target name for reporting.

        Returns:
            List of WetLabCandidate objects sorted by recommendation priority.
        """
        n = len(candidate_smiles)
        if n == 0:
            return []

        preds = np.asarray(predictions, dtype=np.float64).flatten()
        if np.ndim(uncertainties) == 0:
            q_arr = np.full(n, float(uncertainties), dtype=np.float64)
        else:
            q_arr = np.asarray(uncertainties, dtype=np.float64).flatten()
            if len(q_arr) < n:
                q_arr = np.pad(q_arr, (0, n - len(q_arr)), mode="edge")

        # 1. Compute Bayesian Acquisition Scores (EI, UCB, or Greedy)
        acq_scores = compute_acquisition_score(
            pred_mean=preds,
            q_hat=q_arr,
            current_best=current_best,
            strategy=self.strategy,
        )
        if np.ndim(acq_scores) == 0:
            acq_scores = np.array([float(acq_scores)])
        acq_scores_list = [float(s) for s in acq_scores]

        # 2. Select diverse Top-K using MaxMin Picker
        selected_indices = maxmin_diversity_picker(
            smiles_list=candidate_smiles,
            acquisition_scores=acq_scores_list,
            top_k=min(top_k, n),
            min_distance=self.min_diversity_distance,
            tradeoff_lambda=self.tradeoff_lambda,
        )

        # 3. Assemble WetLabCandidate records
        results: List[WetLabCandidate] = []
        selected_smiles = [candidate_smiles[idx] for idx in selected_indices]

        for rank, idx in enumerate(selected_indices, start=1):
            smi = candidate_smiles[idx]
            p_mean = float(preds[idx])
            q_val = float(q_arr[idx])
            low_95 = p_mean - q_val
            up_95 = p_mean + q_val
            acq = float(acq_scores[idx])

            # Compute min distance to other selected candidates
            other_smiles = [s for j, s in enumerate(selected_smiles) if j != (rank - 1)]
            if other_smiles:
                min_dist = min(1.0 - calculate_tanimoto_similarity(smi, o_smi) for o_smi in other_smiles)
            else:
                min_dist = 1.0

            # Priority tiering
            if rank <= 3:
                priority = "Urgent (Top 1-3)"
            elif rank <= 6:
                priority = "High (Top 4-6)"
            else:
                priority = "Standard (Top 7-10)"

            # Rationale summary
            target_str = f" for {target_name}" if target_name else ""
            if self.strategy.lower().startswith("ei"):
                rationale = (
                    f"High Expected Improvement ({acq:.3f}) with pKd {p_mean:.2f} "
                    f"[{low_95:.2f} ~ {up_95:.2f}] and structural novelty{target_str}."
                )
            else:
                rationale = (
                    f"Balanced UCB potential ({acq:.3f}) with pKd {p_mean:.2f} "
                    f"[95% CI: {low_95:.2f} ~ {up_95:.2f}]{target_str}."
                )

            results.append(
                WetLabCandidate(
                    rank=rank,
                    smiles=smi,
                    predicted_affinity=round(p_mean, 3),
                    lower_95=round(low_95, 3),
                    upper_95=round(up_95, 3),
                    conformal_uncertainty=round(q_val, 3),
                    acquisition_score=round(acq, 4),
                    min_tanimoto_distance=round(float(min_dist), 3),
                    synthesis_priority=priority,
                    rationale=rationale,
                )
            )

        return results


def recommend_top_wetlab_candidates(
    candidate_smiles: List[str],
    predictions: Union[List[float], np.ndarray],
    uncertainties: Union[List[float], np.ndarray, float],
    current_best: Optional[float] = None,
    top_k: int = 10,
    strategy: str = "ei",
    target_name: Optional[str] = None,
) -> List[WetLabCandidate]:
    """Convenience helper function for generating Top-K wet-lab recommendations."""
    recommender = ActiveLearningRecommender(strategy=strategy)
    return recommender.recommend(
        candidate_smiles=candidate_smiles,
        predictions=predictions,
        uncertainties=uncertainties,
        current_best=current_best,
        top_k=top_k,
        target_name=target_name,
    )
