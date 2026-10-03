"""Tests for TherapeuticsEvaluator and domain metrics."""

import numpy as np
import pytest
import torch

from tdc_studio.evaluation.evaluator import (
    TherapeuticsEvaluator,
    evaluate_all,
    evaluate_predictions,
    is_metric_higher_better,
)


def test_evaluator_regression_metrics():
    preds = np.array([1.0, 2.0, 3.0, 4.0])
    targets = np.array([1.1, 2.2, 2.9, 3.8])

    evaluator = TherapeuticsEvaluator(task_type="regression")

    mae = evaluator.compute(preds, targets, "mae")
    assert 0.0 < mae < 0.3

    rmse = evaluator.compute(preds, targets, "rmse")
    assert 0.0 < rmse < 0.3

    pearson = evaluator.compute(preds, targets, "pearson")
    assert 0.9 < pearson <= 1.0

    spearman = evaluator.compute(preds, targets, "spearman")
    assert 0.9 < spearman <= 1.0

    r2 = evaluator.compute(preds, targets, "r2")
    assert 0.9 < r2 <= 1.0

    comp = evaluator.compute(preds, targets, "composite")
    assert isinstance(comp, float)
    # mae ~ 0.15, rmse ~ 0.15, r2 ~ 0.98 -> comp should be negative (good performance)
    assert comp < 0.0


def test_evaluator_with_torch_tensors():
    preds = torch.tensor([[1.0], [2.0], [3.0]])
    targets = torch.tensor([1.0, 2.0, 3.0])

    mae = evaluate_predictions(preds, targets, metric_name="mae", task_type="regression")
    assert pytest.approx(mae, 1e-5) == 0.0


def test_evaluator_classification_metrics():
    # Logits / probabilities
    probs = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])

    evaluator = TherapeuticsEvaluator(task_type="binary_classification")

    auc = evaluator.compute(probs, labels, "roc_auc")
    assert auc == 1.0

    pr_auc = evaluator.compute(probs, labels, "pr_auc")
    assert pr_auc == 1.0

    acc = evaluator.compute(probs, labels, "accuracy")
    assert acc == 1.0

    f1 = evaluator.compute(probs, labels, "f1")
    assert f1 == 1.0


def test_evaluator_edge_cases():
    # Single class targets in test / dry-run batch
    preds = np.array([0.5, 0.6])
    targets = np.array([1, 1])

    evaluator = TherapeuticsEvaluator(task_type="binary_classification")
    # Should safely return 0.5 without raising ValueError
    auc = evaluator.compute(preds, targets, "roc_auc")
    assert auc == 0.5


def test_compute_all_suite():
    preds = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.0, 2.0, 3.0])

    reg_metrics = evaluate_all(preds, targets, task_type="regression")
    assert "mae" in reg_metrics
    assert "rmse" in reg_metrics
    assert "pearson" in reg_metrics
    assert "spearman" in reg_metrics
    assert "r2" in reg_metrics
    assert "composite" in reg_metrics

    cls_preds = np.array([0.8, 0.2])
    cls_targets = np.array([1, 0])
    cls_metrics = evaluate_all(cls_preds, cls_targets, task_type="binary_classification")
    assert "roc_auc" in cls_metrics
    assert "accuracy" in cls_metrics


def test_is_metric_higher_better():
    assert is_metric_higher_better("roc_auc") is True
    assert is_metric_higher_better("ROC-AUC") is True
    assert is_metric_higher_better("accuracy") is True
    assert is_metric_higher_better("pearson") is True
    assert is_metric_higher_better("ci") is True
    assert is_metric_higher_better("concordance_index") is True
    assert is_metric_higher_better("mae") is False
    assert is_metric_higher_better("rmse") is False
    assert is_metric_higher_better("mse") is False
    assert is_metric_higher_better("composite") is False


def test_fast_concordance_index_precision_and_speed():
    """Verify O(N log N) CI on N=10,000 runs within 0.5s without subsampling error."""
    import time

    rng = np.random.default_rng(42)
    n = 10000
    y_true = rng.uniform(2.0, 10.0, size=n)
    y_pred = y_true + rng.normal(0.0, 1.5, size=n)

    evaluator = TherapeuticsEvaluator(task_type="dta")
    t0 = time.perf_counter()
    ci = evaluator.compute(y_pred, y_true, "ci")
    t1 = time.perf_counter()

    elapsed = t1 - t0
    assert 0.75 < ci < 0.90
    assert elapsed < 0.6, f"Concordance Index took {elapsed:.3f}s, expected < 0.6s"


def test_concordance_index_ties_and_edge_cases():
    evaluator = TherapeuticsEvaluator(task_type="dta")

    # Perfect ranking
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    yp = np.array([10.0, 20.0, 30.0, 40.0])
    assert evaluator.compute(yp, yt, "ci") == 1.0

    # Inverted ranking
    yp_rev = np.array([40.0, 30.0, 20.0, 10.0])
    assert evaluator.compute(yp_rev, yt, "ci") == 0.0

    # Ties in predictions (should be awarded 0.5)
    yp_ties = np.array([10.0, 10.0, 30.0, 40.0])
    ci_ties = evaluator.compute(yp_ties, yt, "ci")
    assert 0.8 < ci_ties < 1.0

    # All identical predictions
    yp_all_same = np.array([5.0, 5.0, 5.0, 5.0])
    assert evaluator.compute(yp_all_same, yt, "ci") == 0.5

    # All identical targets (no valid comparison pairs)
    yt_all_same = np.array([3.0, 3.0, 3.0, 3.0])
    assert evaluator.compute(yp, yt_all_same, "ci") == 0.5


def test_dta_compute_all_suite():
    evaluator = TherapeuticsEvaluator(task_type="dta")
    yt = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    yp = np.array([5.1, 5.9, 7.2, 7.8, 9.1])

    dta_metrics = evaluator.compute_all(yp, yt, task_type="dta")
    assert "ci" in dta_metrics
    assert "mse" in dta_metrics
    assert "rmse" in dta_metrics
    assert "pearson" in dta_metrics
    assert dta_metrics["ci"] > 0.9
    assert dta_metrics["mse"] < 0.05
