"""Bayesian Acquisition Functions powered by Conformal Prediction Uncertainty.

Translates distribution-free Conformal Prediction intervals [y_hat - q, y_hat + q]
into Bayesian Optimization acquisition metrics (Expected Improvement, UCB, Greedy)
to guide wet-lab experimental synthesis candidate selection.
"""

import math
from typing import Optional, Union

import numpy as np


def _normal_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard normal cumulative distribution function Phi(x)."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _normal_pdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard normal probability density function phi(x)."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * (x**2))


def conformal_expected_improvement(
    pred_mean: Union[float, np.ndarray],
    q_hat: Union[float, np.ndarray],
    current_best: float,
    confidence_level: float = 0.95,
) -> Union[float, np.ndarray]:
    """Compute Expected Improvement (EI) using Conformal Prediction uncertainty.

    In Bayesian Optimization for maximizing affinity (e.g. pKd):
        EI(x) = E[max(0, y - y_best)]
    Under 95% coverage, effective standard error sigma ~= q_hat / z_(1 - alpha/2) (z ~= 1.96).

    Args:
        pred_mean: Predicted affinity (pKd / score) from model.
        q_hat: Conformal non-conformity quantile (half-width of 95% CI).
        current_best: Highest confirmed experimental affinity achieved to date.
        confidence_level: Conformal confidence level (default 0.95).

    Returns:
        Expected Improvement score (>= 0.0). Higher is better.
    """
    mu = np.asarray(pred_mean, dtype=np.float64)
    q = np.asarray(q_hat, dtype=np.float64)

    # Derive equivalent gaussian sigma from conformal interval width: 2 * q = 2 * z * sigma
    z_score = 1.95996 if abs(confidence_level - 0.95) < 0.01 else 2.5758
    sigma = np.maximum(q / z_score, 1e-6)

    diff = mu - current_best
    z = diff / sigma

    cdf_vals = _normal_cdf(z)
    pdf_vals = _normal_pdf(z)

    ei = diff * cdf_vals + sigma * pdf_vals
    ei = np.maximum(ei, 0.0)

    if np.ndim(pred_mean) == 0:
        return float(ei)
    return ei


def conformal_upper_confidence_bound(
    pred_mean: Union[float, np.ndarray],
    q_hat: Union[float, np.ndarray],
    beta: float = 1.0,
) -> Union[float, np.ndarray]:
    """Compute Upper Confidence Bound (UCB) using Conformal Prediction.

    Balances exploitation (high predicted affinity) and exploration (high uncertainty):
        UCB(x) = mu(x) + beta * q_hat

    Args:
        pred_mean: Predicted affinity.
        q_hat: Conformal uncertainty margin.
        beta: Exploration weight multiplier (default 1.0).

    Returns:
        UCB acquisition value.
    """
    mu = np.asarray(pred_mean, dtype=np.float64)
    q = np.asarray(q_hat, dtype=np.float64)
    ucb = mu + beta * q
    if np.ndim(pred_mean) == 0:
        return float(ucb)
    return ucb


def compute_acquisition_score(
    pred_mean: Union[float, np.ndarray],
    q_hat: Union[float, np.ndarray],
    current_best: Optional[float] = None,
    strategy: str = "ei",
    beta: float = 1.0,
) -> Union[float, np.ndarray]:
    """Unified acquisition score dispatcher ('ei', 'ucb', 'greedy')."""
    strat = strategy.lower().strip()
    if strat in ("ei", "expected_improvement"):
        best = current_best if current_best is not None else float(np.percentile(pred_mean, 80))
        return conformal_expected_improvement(pred_mean, q_hat, current_best=best)
    elif strat in ("ucb", "upper_confidence_bound"):
        return conformal_upper_confidence_bound(pred_mean, q_hat, beta=beta)
    elif strat in ("greedy", "mean"):
        return np.asarray(pred_mean, dtype=np.float64)
    else:
        raise ValueError(
            f"Unknown acquisition strategy '{strategy}'. Choose 'ei', 'ucb', or 'greedy'."
        )
