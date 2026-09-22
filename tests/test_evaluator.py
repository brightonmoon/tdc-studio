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
    assert is_metric_higher_better("mae") is False
    assert is_metric_higher_better("rmse") is False
    assert is_metric_higher_better("mse") is False
    assert is_metric_higher_better("composite") is False
