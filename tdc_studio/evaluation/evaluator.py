"""Therapeutics domain evaluation metrics and unified Evaluator."""

from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from tdc_studio.core.registry import EVALUATORS

# Set of metrics where higher value indicates better model performance
HIGHER_IS_BETTER_METRICS = {
    "roc_auc",
    "roc-auc",
    "auc",
    "pr_auc",
    "pr-auc",
    "average_precision",
    "accuracy",
    "acc",
    "f1",
    "f1_score",
    "f1_macro",
    "f1_micro",
    "balanced_acc",
    "balanced_accuracy",
    "pearson",
    "pearsonr",
    "spearman",
    "spearmanr",
    "r2",
    "r2_score",
    "ci",
    "concordance_index",
    "c_index",
}


def _fast_fenwick_concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute exact Concordance Index in O(N log N) using a Binary Indexed Tree.

    Handles ties in y_true (pairs with y_true[i] == y_true[j] are excluded)
    and ties in y_pred (given weight 0.5 per standard Harrell's C-index).
    """
    n = len(y_true)
    if n < 2:
        return 0.5

    # Unique sorted values of y_pred to build 1-based ranks
    unique_preds, pred_ranks = np.unique(y_pred, return_inverse=True)
    m = len(unique_preds)

    # Sort indices by y_true ascending
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    ranks_sorted = pred_ranks[sort_idx] + 1  # 1-indexed for BIT

    # Binary Indexed Tree (Fenwick tree)
    tree = np.zeros(m + 1, dtype=np.int64)

    def _update(idx: int, val: int = 1) -> None:
        while idx <= m:
            tree[idx] += val
            idx += idx & (-idx)

    def _query(idx: int) -> int:
        s = 0
        while idx > 0:
            s += int(tree[idx])
            idx -= idx & (-idx)
        return s

    concordant = 0.0
    discordant = 0.0
    tied_pred = 0.0

    # Process samples in groups of identical y_true to exclude ties in ground-truth
    i = 0
    total_processed = 0
    while i < n:
        j = i
        while j < n and y_true_sorted[j] == y_true_sorted[i]:
            j += 1

        # Query BIT for each element in the group against all strictly smaller y_true
        for k in range(i, j):
            r = int(ranks_sorted[k])
            strictly_smaller = _query(r - 1)
            up_to_r = _query(r)
            equal_r = up_to_r - strictly_smaller
            strictly_greater = total_processed - up_to_r

            concordant += strictly_smaller
            discordant += strictly_greater
            tied_pred += equal_r

        # Insert all elements of this group into BIT
        for k in range(i, j):
            _update(int(ranks_sorted[k]), 1)
        total_processed += j - i
        i = j

    valid_pairs = concordant + discordant + tied_pred
    if valid_pairs == 0:
        return 0.5
    return float((concordant + 0.5 * tied_pred) / valid_pairs)


def _concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Concordance Index (CI) for Drug-Target Affinity prediction in O(N log N).

    Priority:
    1. lifelines.utils.concordance_index (if installed)
    2. tdc.Evaluator('c-index') (if installed)
    3. _fast_fenwick_concordance_index (native high-speed NumPy implementation)

    Args:
        y_true: Ground-truth affinity values (e.g. pKd, Kd).
        y_pred: Predicted affinity values.

    Returns:
        CI score in [0, 1]. 0.5 = random baseline, 1.0 = perfect ranking.
    """
    try:
        from lifelines.utils import concordance_index

        return float(concordance_index(y_true, y_pred))
    except (ImportError, Exception):
        pass

    try:
        from tdc import Evaluator

        eval_fn = Evaluator(name="c-index")
        return float(eval_fn(y_true.tolist(), y_pred.tolist()))
    except (ImportError, Exception):
        pass

    return _fast_fenwick_concordance_index(y_true, y_pred)


def _to_numpy(data: Union[torch.Tensor, np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert input to a 1D float numpy array."""
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        arr = np.array(data)

    arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = np.expand_dims(arr, 0)
    return arr.astype(np.float64)


def is_metric_higher_better(metric_name: str) -> bool:
    """Check if the evaluation metric should be maximized."""
    clean_name = metric_name.lower().replace(" ", "_").replace("-", "_")
    return clean_name in HIGHER_IS_BETTER_METRICS or clean_name in {
        m.replace("-", "_") for m in HIGHER_IS_BETTER_METRICS
    }


@EVALUATORS.register("therapeutics_evaluator")
class TherapeuticsEvaluator:
    """Unified evaluator for molecular property regression and classification tasks."""

    def __init__(
        self,
        default_metric: Optional[str] = None,
        task_type: str = "regression",
    ):
        self.default_metric = default_metric
        self.task_type = task_type

    def compute(
        self,
        preds: Union[torch.Tensor, np.ndarray, Sequence[float]],
        targets: Union[torch.Tensor, np.ndarray, Sequence[float]],
        metric_name: Optional[str] = None,
    ) -> float:
        """Compute a single evaluation metric."""
        metric = (metric_name or self.default_metric or "mae").lower().replace(" ", "_")
        y_pred = _to_numpy(preds)
        y_true = _to_numpy(targets)

        if len(y_pred) != len(y_true):
            raise ValueError(
                f"Predictions length ({len(y_pred)}) does not match targets length ({len(y_true)})."
            )

        # Defensively filter out NaN or Inf values
        valid = np.isfinite(y_pred) & np.isfinite(y_true)
        if not np.all(valid):
            y_pred = y_pred[valid]
            y_true = y_true[valid]

        if len(y_pred) == 0:
            return 0.0

        # Regression Metrics
        if metric in ("mae", "mean_absolute_error"):
            return float(mean_absolute_error(y_true, y_pred))

        if metric in ("mse", "mean_squared_error"):
            return float(mean_squared_error(y_true, y_pred))

        if metric in ("rmse", "root_mean_squared_error"):
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))

        if metric in ("r2", "r2_score"):
            if len(y_true) < 2 or np.all(y_true == y_true[0]):
                return 0.0
            return float(r2_score(y_true, y_pred))

        if metric in ("ci", "concordance_index", "c_index"):
            return _concordance_index(y_true, y_pred)

        if metric in ("composite", "composite_score", "balanced", "balanced_regression"):
            mae_val = float(mean_absolute_error(y_true, y_pred))
            rmse_val = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            if len(y_true) < 2 or np.all(y_true == y_true[0]):
                r2_val = 0.0
            else:
                r2_val = float(r2_score(y_true, y_pred))
            r2_clamped = max(-1.0, min(1.0, r2_val))
            # Minimize MAE and RMSE while maximizing R2 (lower is better)
            return float(mae_val + 0.5 * rmse_val - r2_clamped)

        if metric in ("pearson", "pearsonr", "pcc"):
            if len(y_true) < 2 or np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
                return 0.0
            r_val, _ = pearsonr(y_pred, y_true)
            return float(0.0 if np.isnan(r_val) else r_val)

        if metric in ("spearman", "spearmanr", "scc"):
            if len(y_true) < 2 or np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
                return 0.0
            rho_val, _ = spearmanr(y_pred, y_true)
            return float(0.0 if np.isnan(rho_val) else rho_val)

        # Classification Metrics
        if metric in ("roc_auc", "roc-auc", "auc"):
            # Requires at least one positive and one negative sample
            if len(np.unique(y_true)) < 2:
                return 0.5
            # Convert raw logits to probabilities if outside [0, 1]
            if np.any(y_pred < 0.0) or np.any(y_pred > 1.0):
                probs = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -50.0, 50.0)))
            else:
                probs = y_pred
            return float(roc_auc_score(y_true, probs))

        if metric in ("pr_auc", "pr-auc", "average_precision"):
            if len(np.unique(y_true)) < 2:
                return 0.0
            if np.any(y_pred < 0.0) or np.any(y_pred > 1.0):
                probs = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -50.0, 50.0)))
            else:
                probs = y_pred
            return float(average_precision_score(y_true, probs))

        if metric in ("accuracy", "acc"):
            binary_preds = (
                (y_pred >= 0.5).astype(int)
                if np.all((y_pred >= 0.0) & (y_pred <= 1.0))
                else (y_pred >= 0.0).astype(int)
            )
            return float(accuracy_score(y_true.astype(int), binary_preds))

        if metric in ("balanced_acc", "balanced_accuracy"):
            binary_preds = (
                (y_pred >= 0.5).astype(int)
                if np.all((y_pred >= 0.0) & (y_pred <= 1.0))
                else (y_pred >= 0.0).astype(int)
            )
            return float(balanced_accuracy_score(y_true.astype(int), binary_preds))

        if metric in ("f1", "f1_score"):
            binary_preds = (
                (y_pred >= 0.5).astype(int)
                if np.all((y_pred >= 0.0) & (y_pred <= 1.0))
                else (y_pred >= 0.0).astype(int)
            )
            return float(f1_score(y_true.astype(int), binary_preds, zero_division=0))

        # Default fallback to MAE
        return float(mean_absolute_error(y_true, y_pred))

    def compute_all(
        self,
        preds: Union[torch.Tensor, np.ndarray, Sequence[float]],
        targets: Union[torch.Tensor, np.ndarray, Sequence[float]],
        task_type: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute a standard suite of metrics for the given task type."""
        task = task_type or self.task_type
        metrics: Dict[str, float] = {}

        if task == "regression":
            metrics["mae"] = self.compute(preds, targets, "mae")
            metrics["rmse"] = self.compute(preds, targets, "rmse")
            metrics["pearson"] = self.compute(preds, targets, "pearson")
            metrics["spearman"] = self.compute(preds, targets, "spearman")
            metrics["r2"] = self.compute(preds, targets, "r2")
            metrics["composite"] = self.compute(preds, targets, "composite")
        elif task in ("dta", "drug_target_affinity"):
            # Primary metrics for Drug Cold Split evaluation
            metrics["ci"] = self.compute(preds, targets, "ci")  # 1순위: Cold Drug CI
            metrics["mse"] = self.compute(preds, targets, "mse")  # 1순위: Cold Drug MSE
            metrics["rmse"] = self.compute(preds, targets, "rmse")  # 보조
            metrics["pearson"] = self.compute(preds, targets, "pearson")
        elif task in ("binary_classification", "classification"):
            metrics["roc_auc"] = self.compute(preds, targets, "roc_auc")
            metrics["pr_auc"] = self.compute(preds, targets, "pr_auc")
            metrics["accuracy"] = self.compute(preds, targets, "accuracy")
            metrics["f1"] = self.compute(preds, targets, "f1")
        else:
            metrics["accuracy"] = self.compute(preds, targets, "accuracy")

        return metrics


def evaluate_predictions(
    preds: Union[torch.Tensor, np.ndarray, Sequence[float]],
    targets: Union[torch.Tensor, np.ndarray, Sequence[float]],
    metric_name: str,
    task_type: str = "regression",
) -> float:
    """Convenience function to compute a single evaluation metric."""
    evaluator = TherapeuticsEvaluator(default_metric=metric_name, task_type=task_type)
    return evaluator.compute(preds, targets, metric_name=metric_name)


def evaluate_all(
    preds: Union[torch.Tensor, np.ndarray, Sequence[float]],
    targets: Union[torch.Tensor, np.ndarray, Sequence[float]],
    task_type: str = "regression",
) -> Dict[str, float]:
    """Convenience function to compute all standard metrics for a task."""
    evaluator = TherapeuticsEvaluator(task_type=task_type)
    return evaluator.compute_all(preds, targets, task_type=task_type)
