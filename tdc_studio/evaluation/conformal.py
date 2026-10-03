"""Split Conformal Prediction for Distribution-Free Uncertainty Quantification (UQ).

Provides:
- ConformalCalibrator: Inductive Conformal Prediction for regression with exact coverage guarantees.
- ConformalPrediction: Dataclass capturing mean, 95% lower/upper bounds, and Applicability Domain flags.

Theoretical guarantee:
    P(y_new in [y_hat - q, y_hat + q]) >= 1 - alpha
under exchangeability of calibration and test data, with no distributional assumptions.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger("tdc_studio.evaluation.conformal")


@dataclass
class ConformalPrediction:
    """Prediction interval and domain validity produced by ConformalCalibrator."""

    mean: float
    lower: float
    upper: float
    interval_width: float
    confidence_level: float
    is_in_domain: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConformalCalibrator:
    """Split Conformal Calibrator for regression models (DTA / ADMET).

    Args:
        alpha: Miscoverage rate (default 0.05 for 95% confidence intervals).
        ad_threshold_multiplier: Multiplier on calibration max residual for Applicability Domain check (default 1.5).
    """

    def __init__(self, alpha: float = 0.05, ad_threshold_multiplier: float = 1.5):
        self.alpha = alpha
        self.ad_threshold_multiplier = ad_threshold_multiplier
        self.q_hat: Optional[float] = None
        self.cal_size: int = 0
        self.max_residual: float = 0.0
        self.median_residual: float = 0.0
        self.calibrated: bool = False

    def calibrate(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List[float]],
        y_pred: Union[np.ndarray, torch.Tensor, List[float]],
        alpha: Optional[float] = None,
    ) -> float:
        """Calibrate non-conformity scores on hold-out validation/calibration set.

        Args:
            y_true: Ground truth target values.
            y_pred: Model predictions on calibration samples.
            alpha: Optional miscoverage rate override.

        Returns:
            Calculated quantile q_hat.
        """
        if alpha is not None:
            self.alpha = alpha

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy().flatten()
        else:
            y_true = np.asarray(y_true).flatten()

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy().flatten()
        else:
            y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")

        n = len(y_true)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Non-conformity score: absolute prediction residual R_i = |y_i - y_hat_i|
        residuals = np.abs(y_true - y_pred)
        self.cal_size = n
        self.median_residual = float(np.median(residuals))
        self.max_residual = float(np.max(residuals))

        # Standard Conformal Quantile calculation: ceil((n + 1) * (1 - alpha)) / n
        level = np.clip(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 0.0, 1.0)
        self.q_hat = float(
            np.quantile(residuals, level, method="higher" if hasattr(np, "quantile") else "linear")
        )
        self.calibrated = True

        logger.info(
            "ConformalCalibrator calibrated on N=%d samples (alpha=%.2f, q_hat=%.4f, median_res=%.4f)",
            n,
            self.alpha,
            self.q_hat,
            self.median_residual,
        )
        return self.q_hat

    def predict_interval(
        self,
        y_pred: Union[float, np.ndarray, torch.Tensor, List[float]],
        distance_to_domain: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[ConformalPrediction, List[ConformalPrediction]]:
        """Compute conformal prediction intervals for one or multiple predictions.

        Args:
            y_pred: Predicted target value(s).
            distance_to_domain: Optional outlier distance metric to evaluate Applicability Domain.

        Returns:
            ConformalPrediction dataclass or List of ConformalPrediction.
        """
        if not self.calibrated or self.q_hat is None:
            # Fallback default quantile if not explicitly calibrated
            q = 0.80  # Default ~0.8 pKd margin (~6-fold affinity variation)
            is_cal = False
        else:
            q = self.q_hat
            is_cal = True

        is_scalar = isinstance(y_pred, (int, float)) or (
            isinstance(y_pred, torch.Tensor) and y_pred.numel() == 1
        )
        if is_scalar:
            val = float(y_pred.item() if isinstance(y_pred, torch.Tensor) else y_pred)
            in_domain = True
            if distance_to_domain is not None:
                ad_limit = self.max_residual * self.ad_threshold_multiplier if is_cal else 2.5
                in_domain = float(distance_to_domain) <= ad_limit

            return ConformalPrediction(
                mean=val,
                lower=val - q,
                upper=val + q,
                interval_width=2.0 * q,
                confidence_level=1.0 - self.alpha,
                is_in_domain=in_domain,
            )

        if isinstance(y_pred, torch.Tensor):
            vals = y_pred.detach().cpu().numpy().flatten()
        else:
            vals = np.asarray(y_pred).flatten()

        results = []
        ad_limit = self.max_residual * self.ad_threshold_multiplier if is_cal else 2.5
        for idx, v in enumerate(vals):
            in_dom = True
            if distance_to_domain is not None:
                d = float(
                    distance_to_domain[idx]
                    if hasattr(distance_to_domain, "__getitem__")
                    else distance_to_domain
                )
                in_dom = d <= ad_limit

            results.append(
                ConformalPrediction(
                    mean=float(v),
                    lower=float(v - q),
                    upper=float(v + q),
                    interval_width=float(2.0 * q),
                    confidence_level=1.0 - self.alpha,
                    is_in_domain=in_dom,
                )
            )
        return results

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist calibration parameters to JSON."""
        state = {
            "alpha": self.alpha,
            "q_hat": self.q_hat,
            "cal_size": self.cal_size,
            "max_residual": self.max_residual,
            "median_residual": self.median_residual,
            "ad_threshold_multiplier": self.ad_threshold_multiplier,
            "calibrated": self.calibrated,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ConformalCalibrator":
        """Load calibrated parameters from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        calibrator = cls(
            alpha=state.get("alpha", 0.05),
            ad_threshold_multiplier=state.get("ad_threshold_multiplier", 1.5),
        )
        calibrator.q_hat = state.get("q_hat")
        calibrator.cal_size = state.get("cal_size", 0)
        calibrator.max_residual = state.get("max_residual", 0.0)
        calibrator.median_residual = state.get("median_residual", 0.0)
        calibrator.calibrated = state.get("calibrated", False)
        return calibrator
