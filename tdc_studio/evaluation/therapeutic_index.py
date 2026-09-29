"""Therapeutic Index (TI) and Preclinical Safety Margin Quantification Engine.

Quantifies on-target drug efficacy (Kd) relative to cardiotoxicity (hERG IC50)
and hepatotoxicity (DILI) liabilities into a unified clinical progression score.
"""

import math
from typing import Any, Dict, Tuple


def calculate_therapeutic_index(
    kd_nm: float,
    herg_ic50_nm: float,
) -> Tuple[float, str]:
    """Calculate Therapeutic Index (TI) on logarithmic scale and assign risk tier.

    TI = log10(IC50_hERG / Kd) = pKd - pIC50

    Tiers:
    - Safe (Wide Window): TI >= 2.0 (>= 100-fold safety margin)
    - Moderate Risk:      1.0 <= TI < 2.0 (10 ~ 100-fold margin)
    - Critical Hazard:    TI < 1.0 (< 10-fold margin, cardiotoxicity alert)

    Args:
        kd_nm: Predicted On-Target binding affinity Kd in nanomolar (nM).
        herg_ic50_nm: Predicted hERG cardiotoxicity IC50 in nanomolar (nM).

    Returns:
        Tuple of (TI_float, tier_string).
    """
    safe_kd = max(float(kd_nm), 1e-4)
    safe_herg = max(float(herg_ic50_nm), 1e-4)

    ratio = safe_herg / safe_kd
    ti = math.log10(ratio)
    ti_rounded = round(ti, 3)

    if ti >= 2.0:
        tier = "Safe (Wide Window)"
    elif ti >= 1.0:
        tier = "Moderate Risk"
    else:
        tier = "Critical Hazard"

    return ti_rounded, tier


def calibrate_herg_ic50_from_probability(
    herg_prob: float,
    cutoff_nm: float = 10000.0,
) -> Tuple[float, str]:
    """Calibrate continuous hERG IC50 (nM) from classification probability.

    Applies pharmacological log-logistic transformation centered at the TDC
    standard benchmark cutoff (10 uM = 10,000 nM):
        IC50 = cutoff_nm * ((1 - P) / P)

    Args:
        herg_prob: Predicted probability of being a hERG blocker (0.0 to 1.0).
        cutoff_nm: Assay threshold in nM (default 10,000 nM).

    Returns:
        Tuple of (herg_ic50_nm, decision_tier).
    """
    p = min(max(float(herg_prob), 0.001), 0.999)

    ic50_nm = cutoff_nm * ((1.0 - p) / p)
    # Clamp to realistic biophysical assay window [1.0 nM, 10,000,000 nM]
    ic50_clamped = min(max(ic50_nm, 1.0), 10_000_000.0)
    ic50_rounded = round(ic50_clamped, 2)

    if p < 0.3:
        decision = "Low Cardiotoxicity Risk"
    elif p < 0.7:
        decision = "Moderate Risk"
    else:
        decision = "High Risk (hERG Blocker)"

    return ic50_rounded, decision


def calculate_dili_penalty(dili_prob: float) -> Tuple[float, str]:
    """Calculate graduated hepatotoxicity penalty (0 to 40 points) from DILI probability.

    Args:
        dili_prob: Predicted probability of Drug-Induced Liver Injury (0.0 to 1.0).

    Returns:
        Tuple of (penalty_points, decision_tier).
    """
    p = min(max(float(dili_prob), 0.0), 1.0)

    if p < 0.3:
        penalty = 0.0
        decision = "Low Hepatotoxicity Risk"
    elif p < 0.5:
        # Scaled penalty 0 to 15
        penalty = 15.0 * ((p - 0.3) / 0.2)
        decision = "Moderate Hepatotoxicity Risk"
    else:
        # Scaled penalty 15 to 40
        penalty = 15.0 + 25.0 * ((p - 0.5) / 0.5)
        decision = "Hepatotoxicity Risk (DILI+)"

    return round(penalty, 2), decision


def calculate_clinical_progression_score(
    kd_nm: float,
    herg_ic50_nm: float,
    dili_prob: float,
    drug_likeness_score: float = 80.0,
    herg_prob: float = 0.5,
) -> Dict[str, Any]:
    """Calculate overall Preclinical Progression Score (0 to 100) and recommendation tier.

    Score formulation:
    - Base TI Score (0 to 55 pts): 27.5 * min(2.0, max(0.0, TI))
    - Potency Score (0 to 25 pts): Kd-driven potency bonus
    - Drug-Likeness Bonus (0 to 20 pts): 0.20 * drug_likeness_score
    - DILI Penalty (0 to 40 pts): Deducted for liver injury liability

    Decisions:
    - Recommended (Pass):   Score >= 70.0, TI >= 2.0, dili_prob < 0.5
    - Caution (Moderate):   50.0 <= Score < 70.0 and TI >= 1.0
    - Rejected (Critical):  Score < 50.0 or TI < 1.0

    Returns:
        Dict with comprehensive scores, tiers, and radar dimensions.
    """
    ti, ti_tier = calculate_therapeutic_index(kd_nm=kd_nm, herg_ic50_nm=herg_ic50_nm)
    dili_penalty, dili_decision = calculate_dili_penalty(dili_prob=dili_prob)

    # 1. Therapeutic Index score (0 to 55)
    ti_score = 27.5 * min(2.0, max(0.0, ti))

    # 2. On-target potency score (0 to 25)
    kd_val = float(kd_nm)
    if kd_val <= 10.0:
        potency_score = 25.0
    elif kd_val <= 100.0:
        potency_score = 20.0
    elif kd_val <= 1000.0:
        potency_score = 12.0
    else:
        potency_score = 4.0

    # 3. Drug-likeness bonus (0 to 20)
    dl_score = min(max(float(drug_likeness_score), 0.0), 100.0)
    dl_bonus = 0.20 * dl_score

    # 4. Total clinical progression score
    raw_score = ti_score + potency_score + dl_bonus - dili_penalty
    total_score = min(max(raw_score, 0.0), 100.0)

    # Enforce strict liability ceiling: candidates with moderate or critical toxicity risks
    # cannot exceed the threshold for uninhibited clinical progression
    if ti < 1.0:
        total_score = min(total_score, 49.9)
    elif ti < 2.0 or dili_prob >= 0.5:
        total_score = min(total_score, 69.9)

    total_score = round(total_score, 1)

    # 5. Clinical decision assignment
    if total_score >= 70.0 and ti >= 2.0 and dili_prob < 0.5:
        decision = "Recommended (Pass)"
    elif total_score >= 50.0 and ti >= 1.0:
        decision = "Caution (Moderate Risk)"
    else:
        decision = "Rejected (Critical Hazard)"

    # 6. Safety radar normalized coordinates (0 to 100)
    norm_window = min(max(ti / 2.0 * 100.0, 0.0), 100.0)
    norm_potency = min(max(potency_score / 25.0 * 100.0, 0.0), 100.0)
    cardiac_safety = round(min(max((1.0 - herg_prob) * 100.0, 0.0), 100.0), 1)
    hepatic_safety = round(min(max((1.0 - dili_prob) * 100.0, 0.0), 100.0), 1)

    safety_radar = {
        "therapeutic_window": round(norm_window, 1),
        "target_potency": round(norm_potency, 1),
        "cardiac_safety": cardiac_safety,
        "hepatic_safety": hepatic_safety,
        "drug_likeness": round(dl_score, 1),
    }

    return {
        "therapeutic_index": ti,
        "ti_tier": ti_tier,
        "dili_penalty": dili_penalty,
        "dili_decision": dili_decision,
        "clinical_progression_score": total_score,
        "clinical_decision": decision,
        "safety_radar": safety_radar,
    }
