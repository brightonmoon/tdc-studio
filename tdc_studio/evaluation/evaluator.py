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



def _concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Concordance Index (CI) for Drug-Target Affinity prediction.

    CI measures the probability that two randomly chosen drug-target pairs
    are ranked correctly relative to each other by the model.

    Formula: CI = #{(i,j): y_true[i] > y_true[j] and y_pred[i] > y_pred[j]} / #{(i,j): y_true[i] != y_true[j]}

    Args:
        y_true: Ground-truth affinity values (e.g. pKd, Kd).
        y_pred: Predicted affinity values.

    Returns:
        CI score in [0, 1]. 0.5 = random baseline, 1.0 = perfect ranking.
    """
    n = len(y_true)
    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                tied += 1
                continue
            if y_true[i] > y_true[j]:
                if y_pred[i] > y_pred[j]:
                    concordant += 1
                elif y_pred[i] < y_pred[j]:
                    discordant += 1
                else:
                    tied += 1
            else:
                if y_pred[i] < y_pred[j]:
                    concordant += 1
                elif y_pred[i] > y_pred[j]:
                    discordant += 1
                else:
                    tied += 1

    total_pairs = concordant + discordant + tied
    if total_pairs == 0:
        return 0.5
    return float(concordant / (concordant + discordant)) if (concordant + discordant) > 0 else 0.5


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
            # For large datasets, subsample to keep O(n^2) tractable (max 2000 pairs).
            if len(y_true) > 2000:
                rng = np.random.default_rng(seed=0)
                idx = rng.choice(len(y_true), size=2000, replace=False)
                return _concordance_index(y_true[idx], y_pred[idx])
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
            metrics["ci"] = self.compute(preds, targets, "ci")         # 1순위: Cold Drug CI
            metrics["mse"] = self.compute(preds, targets, "mse")       # 1순위: Cold Drug MSE
            metrics["rmse"] = self.compute(preds, targets, "rmse")     # 보조
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
