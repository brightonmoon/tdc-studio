"""Evaluation metrics for Single-Step and Multi-Step retrosynthesis benchmarks."""

import time
from typing import Any, Dict, List, Optional, Sequence

from rdkit import Chem

from tdc_studio.core.registry import EVALUATORS
from tdc_studio.models.retrosynthesis.base import BaseRetroModel
from tdc_studio.models.retrosynthesis.forward_verifier import ForwardVerifier
from tdc_studio.retrosynthesis.planner import RetroPlanner


def _canonicalize_reactants(reactants_str: str) -> Optional[str]:
    """Canonicalize and sort reactant fragments for permutation-invariant exact matching."""
    if not reactants_str:
        return None
    frags = [f.strip() for f in reactants_str.split(".") if f.strip()]
    cleaned = []
    for f in frags:
        mol = Chem.MolFromSmiles(f)
        if mol is None:
            return None
        cleaned.append(Chem.MolToSmiles(mol, canonical=True))
    return ".".join(sorted(cleaned))


def compute_top_k_exact_match(
    predictions: List[List[str]],
    targets: List[str],
    k_list: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Compute Top-k Exact Match Accuracy for chemical retrosynthesis.

    Args:
        predictions: List of ranked candidate reactant strings for each sample.
        targets: Ground-truth reactant strings for each sample.
        k_list: Top-k thresholds to evaluate (e.g. 1, 3, 5, 10).

    Returns:
        Dictionary mapping metric names (e.g., 'top_1', 'top_5') to accuracy percentages (0~100%).
    """
    n_samples = len(targets)
    if n_samples == 0:
        return {f"top_{k}": 0.0 for k in k_list}

    hits = {k: 0 for k in k_list}

    for ranked_preds, target_str in zip(predictions, targets):
        canon_target = _canonicalize_reactants(target_str)
        if not canon_target:
            continue

        canon_preds = []
        for p in ranked_preds:
            cp = _canonicalize_reactants(p)
            if cp:
                canon_preds.append(cp)

        for k in k_list:
            if canon_target in canon_preds[:k]:
                hits[k] += 1

    return {f"top_{k}": round((hits[k] / n_samples) * 100.0, 2) for k in k_list}


def compute_invalid_smiles_rate(predictions: List[List[str]]) -> float:
    """Calculate the percentage of generated candidate fragments that fail RDKit parsing."""
    total_mols = 0
    invalid_mols = 0

    for ranked in predictions:
        for pred_str in ranked:
            for frag in pred_str.split("."):
                frag_s = frag.strip()
                if not frag_s:
                    continue
                total_mols += 1
                if Chem.MolFromSmiles(frag_s) is None:
                    invalid_mols += 1

    if total_mols == 0:
        return 0.0
    return round((invalid_mols / total_mols) * 100.0, 2)


@EVALUATORS.register("retrosynthesis_evaluator")
class RetroBenchmarkEvaluator:
    """Unified benchmark evaluator for single-step retrosynthesis models."""

    def __init__(self, verifier: Optional[ForwardVerifier] = None):
        self.verifier = verifier or ForwardVerifier()

    def evaluate(
        self,
        model: BaseRetroModel,
        test_samples: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Run standard USPTO-50K evaluation across test samples."""
        predictions: List[List[str]] = []
        targets: List[str] = []
        class_records: Dict[int, Dict[str, Any]] = {}

        for sample in test_samples:
            prod = sample["input"]
            target_react = sample["output"]
            rxn_type = int(sample.get("reaction_type", 0))

            targets.append(target_react)
            preds = model.predict_reactants(prod, top_k=top_k)
            pred_strs = [p[0] for p in preds]
            predictions.append(pred_strs)

            if rxn_type not in class_records:
                class_records[rxn_type] = {"preds": [], "targets": []}
            class_records[rxn_type]["preds"].append(pred_strs)
            class_records[rxn_type]["targets"].append(target_react)

        overall_metrics = compute_top_k_exact_match(predictions, targets, k_list=(1, 3, 5, 10))
        overall_metrics["invalid_rate"] = compute_invalid_smiles_rate(predictions)
        overall_metrics["num_samples"] = len(targets)

        # Per-class breakdown
        class_breakdown = {}
        for c_id, data in class_records.items():
            c_metrics = compute_top_k_exact_match(data["preds"], data["targets"], k_list=(1, 5, 10))
            class_breakdown[f"class_{c_id}"] = c_metrics

        overall_metrics["class_breakdown"] = class_breakdown
        return overall_metrics


def evaluate_multistep_routes(
    planner: RetroPlanner,
    target_molecules: List[str],
    max_depth: int = 5,
    timeout_sec: float = 3.0,
) -> Dict[str, float]:
    """Benchmark multi-step route search success rate, length, yield, and latency."""
    n_total = len(target_molecules)
    if n_total == 0:
        return {}

    solved_count = 0
    total_depth = 0
    cum_yield_sum = 0.0
    total_time = 0.0

    for target in target_molecules:
        t0 = time.time()
        route = planner.plan_route(target, max_depth=max_depth, timeout_sec=timeout_sec)
        dt = time.time() - t0
        total_time += dt

        if route.solved:
            solved_count += 1
            total_depth += route.total_depth
            cum_yield_sum += route.cumulative_yield

    avg_depth = round(total_depth / max(1, solved_count), 2)
    avg_yield = round(cum_yield_sum / max(1, solved_count), 2)
    avg_time = round(total_time / n_total, 3)
    success_rate = round((solved_count / n_total) * 100.0, 2)

    return {
        "search_success_rate": success_rate,
        "avg_route_depth": avg_depth,
        "avg_cumulative_yield": avg_yield,
        "avg_latency_sec": avg_time,
        "num_evaluated": n_total,
    }
