"""Unit tests for Phase 4: Retrosynthesis benchmark metrics and evaluators."""

from tdc_studio.data.retrosyn import get_mock_retrosyn_dataset
from tdc_studio.evaluation.retro_metrics import (
    RetroBenchmarkEvaluator,
    compute_invalid_smiles_rate,
    compute_top_k_exact_match,
    evaluate_multistep_routes,
)
from tdc_studio.models.retrosynthesis.rule_policy import RuleRetroPolicy
from tdc_studio.retrosynthesis.planner import RetroPlanner


def test_compute_top_k_exact_match():
    # 2 samples
    targets = [
        "c1ccc(C(=O)O)cc1.CN",
        "c1ccc(Br)cc1.OB(O)c1ccccc1",
    ]
    # Sample 1 hits at rank 1, Sample 2 hits at rank 2
    predictions = [
        ["CN.c1ccc(C(=O)O)cc1", "c1ccccc1"],
        ["c1ccccc1", "OB(O)c1ccccc1.c1ccc(Br)cc1"],
    ]

    metrics = compute_top_k_exact_match(predictions, targets, k_list=(1, 2, 5))
    assert metrics["top_1"] == 50.0  # 1 out of 2
    assert metrics["top_2"] == 100.0  # 2 out of 2
    assert metrics["top_5"] == 100.0


def test_compute_invalid_smiles_rate():
    # Mix of valid and invalid SMILES
    predictions = [
        ["c1ccccc1", "invalid_smi_xyz"],
        ["CC(=O)O", "C1CC1"],
    ]
    rate = compute_invalid_smiles_rate(predictions)
    assert rate == 25.0  # 1 invalid out of 4 total molecules

    # All valid
    valid_preds = [["CC", "CCO"], ["c1ccccc1"]]
    assert compute_invalid_smiles_rate(valid_preds) == 0.0


def test_retro_benchmark_evaluator():
    evaluator = RetroBenchmarkEvaluator()
    model = RuleRetroPolicy()

    # Use 10 verified mock samples
    df = get_mock_retrosyn_dataset(n_samples=10)
    samples = df.to_dict(orient="records")

    metrics = evaluator.evaluate(model, samples, top_k=5)
    assert "top_1" in metrics
    assert "top_5" in metrics
    assert "invalid_rate" in metrics
    assert "class_breakdown" in metrics
    assert metrics["num_samples"] == 10
    assert metrics["invalid_rate"] == 0.0


def test_evaluate_multistep_routes():
    planner = RetroPlanner(policy_type="rule", max_depth=3, timeout_sec=2.0)
    test_targets = [
        "c1ccc(C(=O)NC)cc1",  # Benzoic acid + Methylamine (both in stock)
        "c1ccc(-c2ccccc2)cc1",  # Suzuki coupling (both in stock)
    ]

    metrics = evaluate_multistep_routes(planner, test_targets)
    assert "search_success_rate" in metrics
    assert "avg_route_depth" in metrics
    assert "avg_cumulative_yield" in metrics
    assert metrics["num_evaluated"] == 2
    assert metrics["search_success_rate"] == 100.0
