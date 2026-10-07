"""Therapeutic Index (TI) and Preclinical Safety Margin Quantification Engine.

Quantifies on-target drug efficacy (Kd) relative to cardiotoxicity (hERG IC50)
and hepatotoxicity (DILI) liabilities into a unified clinical progression score.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger("tdc_studio.evaluation.therapeutic_index")


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


# ==============================================================================
# Clinical Developability & Multi-Pillar Therapeutic Index Engine
# ==============================================================================



@dataclass
class ComponentScores:
    """Breakdown of the 4 clinical developability score pillars (0~25 pts each)."""

    potency: float = 0.0
    safety_window: float = 0.0
    organ_toxicology: float = 0.0
    human_pk: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "potency": round(self.potency, 2),
            "safety_window": round(self.safety_window, 2),
            "organ_toxicology": round(self.organ_toxicology, 2),
            "human_pk": round(self.human_pk, 2),
        }


@dataclass
class TherapeuticIndexProfile:
    """Comprehensive pharmacological Therapeutic Index and Clinical Developability Profile."""

    smiles: str
    canonical_smiles: str
    target_kd_nm: float
    target_pkd: float
    herg_ic50_nm: float
    herg_safety_margin: float
    herg_therapeutic_window_log10: float
    herg_risk_tier: str  # "Safe (Margin >= 100x)", "Borderline (30x-100x)", "High Risk (<30x)"
    dili_risk_probability: float
    clintox_risk_probability: float
    ames_mutagenicity_probability: float
    clinical_developability_score: float  # 0 ~ 100
    developability_tier: str  # Tier 1, Tier 2, Tier 3
    component_scores: ComponentScores
    pbpk_cmax_total_ug_ml: Optional[float] = None
    pbpk_cmax_free_ug_ml: Optional[float] = None
    in_vivo_herg_margin: Optional[float] = None
    target_name: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to serializable dictionary."""
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "target_kd_nm": round(self.target_kd_nm, 3),
            "target_pkd": round(self.target_pkd, 3),
            "herg_ic50_nm": round(self.herg_ic50_nm, 1),
            "herg_safety_margin": round(self.herg_safety_margin, 2),
            "herg_therapeutic_window_log10": round(self.herg_therapeutic_window_log10, 3),
            "herg_risk_tier": self.herg_risk_tier,
            "dili_risk_probability": round(self.dili_risk_probability, 4),
            "clintox_risk_probability": round(self.clintox_risk_probability, 4),
            "ames_mutagenicity_probability": round(self.ames_mutagenicity_probability, 4),
            "clinical_developability_score": round(self.clinical_developability_score, 1),
            "developability_tier": self.developability_tier,
            "component_scores": self.component_scores.to_dict(),
            "pbpk_cmax_total_ug_ml": (
                round(self.pbpk_cmax_total_ug_ml, 4)
                if self.pbpk_cmax_total_ug_ml is not None
                else None
            ),
            "pbpk_cmax_free_ug_ml": (
                round(self.pbpk_cmax_free_ug_ml, 5)
                if self.pbpk_cmax_free_ug_ml is not None
                else None
            ),
            "in_vivo_herg_margin": (
                round(self.in_vivo_herg_margin, 2) if self.in_vivo_herg_margin is not None else None
            ),
            "target_name": self.target_name,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


class TherapeuticIndexEngine:
    """Engine for evaluating Therapeutic Index, Safety Margins, and Clinical Developability."""

    def __init__(
        self,
        admet_pipeline: Optional[Any] = None,
        dti_pipeline: Optional[Any] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.admet_pipeline = admet_pipeline
        self.dti_pipeline = dti_pipeline

    def _ensure_admet_pipeline(self) -> Any:
        """Lazily initialize UnifiedADMETPipeline if needed."""
        if self.admet_pipeline is None:
            from tdc_studio.serving.unified_pipeline import (
                UnifiedADMETPipeline,
                get_unified_pipeline,
                set_unified_pipeline,
            )

            pipe = get_unified_pipeline()
            if pipe is None:
                pipe = UnifiedADMETPipeline(device=self.device)
                set_unified_pipeline(pipe)
            self.admet_pipeline = pipe
        return self.admet_pipeline

    @staticmethod
    def calibrate_herg_ic50(prob_blocker: float) -> float:
        """Calibrate hERG IC50 in nM from blocker probability.

        P(blocker) = 0.50 corresponds to the TDC standard threshold of 10 uM (10,000 nM).
        P(blocker) -> 0.05 corresponds to ~30 uM (safe non-blocker).
        P(blocker) -> 0.95 corresponds to ~300 nM (potent nanomolar blocker).
        """
        # Clamp probability to avoid numerical instability
        p = max(0.01, min(0.99, float(prob_blocker)))
        # log10(IC50_uM) ~= 1.0 - 2.0 * (p - 0.5)
        # For p=0.5 -> log10(uM) = 1.0 -> 10 uM = 10,000 nM
        log10_u_m = 1.0 - 2.0 * (p - 0.5)
        ic50_nm = (10.0**log10_u_m) * 1000.0
        # Clamp to realistic physical range [50 nM, 100,000 nM]
        return max(50.0, min(100000.0, ic50_nm))

    @staticmethod
    def calculate_potency_score(target_kd_nm: float) -> float:
        """Score on-target potency pillar (0 ~ 25 pts)."""
        if target_kd_nm <= 1.0:
            return 25.0
        elif target_kd_nm <= 10.0:
            # Linear scale from 25 to 22
            return 22.0 + (10.0 - target_kd_nm) / 9.0 * 3.0
        elif target_kd_nm <= 50.0:
            # Linear scale from 22 to 18
            return 18.0 + (50.0 - target_kd_nm) / 40.0 * 4.0
        elif target_kd_nm <= 200.0:
            # Linear scale from 18 to 12
            return 12.0 + (200.0 - target_kd_nm) / 150.0 * 6.0
        elif target_kd_nm <= 1000.0:
            # Linear scale from 12 to 5
            return 5.0 + (1000.0 - target_kd_nm) / 800.0 * 7.0
        else:
            return max(0.0, 5.0 * math.exp(-(target_kd_nm - 1000.0) / 2000.0))

    @staticmethod
    def calculate_safety_window_score(margin_herg: float) -> float:
        """Score safety window pillar (0 ~ 25 pts).

        ICH S7B guidelines: margin >= 100x is ideal; < 30x is red flag.
        """
        if margin_herg >= 100.0:
            return 25.0
        elif margin_herg >= 50.0:
            # 20 ~ 25 pts
            return 20.0 + (margin_herg - 50.0) / 50.0 * 5.0
        elif margin_herg >= 30.0:
            # 14 ~ 20 pts
            return 14.0 + (margin_herg - 30.0) / 20.0 * 6.0
        elif margin_herg >= 10.0:
            # 5 ~ 14 pts
            return 5.0 + (margin_herg - 10.0) / 20.0 * 9.0
        else:
            return max(0.0, margin_herg / 10.0 * 5.0)

    @staticmethod
    def calculate_organ_toxicology_score(
        prob_dili: float,
        prob_ames: float,
        prob_clintox: float,
    ) -> float:
        """Score organ & regulatory toxicology pillar (0 ~ 25 pts)."""
        base = 25.0
        # AMES mutagenicity penalty (often severe project stopper)
        if prob_ames > 0.5:
            base -= 10.0 * (prob_ames - 0.5) / 0.5
        # DILI hepatotoxicity penalty
        if prob_dili > 0.5:
            base -= 8.0 * (prob_dili - 0.5) / 0.5
        # ClinTox clinical failure penalty
        if prob_clintox > 0.5:
            base -= 7.0 * (prob_clintox - 0.5) / 0.5
        return max(0.0, min(25.0, base))

    @staticmethod
    def calculate_pk_score(
        half_life_hr: float,
        extraction_ratio_eh: Optional[float],
        caco2_val: Optional[float] = None,
        solubility_val: Optional[float] = None,
    ) -> float:
        """Score human PK and oral druggability pillar (0 ~ 25 pts)."""
        score = 0.0

        # 1. Elimination half-life (max 10 pts)
        if 2.0 <= half_life_hr <= 24.0:
            score += 10.0  # Ideal once/twice-daily dosing regimen
        elif 1.0 <= half_life_hr < 2.0:
            score += 6.0
        elif 24.0 < half_life_hr <= 48.0:
            score += 7.0
        elif half_life_hr > 48.0:
            score += 3.0  # Excessive accumulation risk
        else:
            score += 2.0  # Too rapid clearance

        # 2. Hepatic extraction ratio E_H (max 8 pts)
        eh = extraction_ratio_eh if extraction_ratio_eh is not None else 0.4
        if eh <= 0.3:
            score += 8.0  # Low extraction, high oral bioavailability potential
        elif eh <= 0.7:
            score += 5.0  # Intermediate
        else:
            score += 1.0  # High first-pass loss

        # 3. Permeability & Solubility (max 7 pts)
        perm_ok = caco2_val is None or caco2_val >= -5.5
        sol_ok = solubility_val is None or solubility_val >= -4.0
        if perm_ok and sol_ok:
            score += 7.0
        elif perm_ok or sol_ok:
            score += 4.0
        else:
            score += 1.0

        return max(0.0, min(25.0, score))

    def compute(
        self,
        smiles: str,
        target_kd_nm: Optional[float] = None,
        target_pkd: Optional[float] = None,
        target_sequence: Optional[str] = None,
        herg_ic50_nm: Optional[float] = None,
        dose_mg: float = 100.0,
        custom_admet_profile: Optional[Any] = None,
    ) -> TherapeuticIndexProfile:
        """Compute the full Therapeutic Index and Clinical Developability Profile.

        Args:
            smiles: Input candidate SMILES.
            target_kd_nm: Known/assumed on-target binding affinity Kd in nM.
            target_pkd: Known/assumed on-target binding affinity pKd (-log10 Kd).
            target_sequence: Target amino acid sequence (used for DTI prediction if provided).
            herg_ic50_nm: Known explicit experimental hERG IC50 in nM.
            dose_mg: Reference human oral dose for PBPK exposure scaling (default: 100 mg).
            custom_admet_profile: Optional precomputed UnifiedADMETProfile.

        Returns:
            TherapeuticIndexProfile with scores, margins, tiers, warnings, and recommendations.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: '{smiles}'")
        canon_smiles = Chem.MolToSmiles(mol, canonical=True)
        mw = Descriptors.MolWt(mol)

        warnings: List[str] = []
        recommendations: List[str] = []

        # ----------------------------------------------------------------------
        # 1. Determine Target Potency (Kd & pKd)
        # ----------------------------------------------------------------------
        resolved_kd: float
        resolved_pkd: float

        if target_kd_nm is not None and target_kd_nm > 0:
            resolved_kd = float(target_kd_nm)
            resolved_pkd = 9.0 - math.log10(resolved_kd)
        elif target_pkd is not None:
            resolved_pkd = float(target_pkd)
            resolved_kd = 10.0 ** (9.0 - resolved_pkd)
        elif target_sequence and self.dti_pipeline is not None:
            try:
                res = self.dti_pipeline.predict_affinity(
                    [canon_smiles], [target_sequence], return_kd_nm=True
                )
                resolved_pkd = float(res["predictions_pkd"][0])
                kd_list = res.get("kd_nm")
                if kd_list and kd_list[0] is not None:
                    resolved_kd = float(kd_list[0])
                else:
                    resolved_kd = 10.0 ** (9.0 - resolved_pkd)
            except Exception as e:
                logger.warning("DTI prediction failed: %s; falling back to 10.0 nM reference.", e)
                resolved_kd = 10.0
                resolved_pkd = 8.0
                warnings.append(f"DTI inference failed ({e}); using 10.0 nM reference potency.")
        else:
            # Default reference: 10.0 nM (standard clinical lead target potency)
            resolved_kd = 10.0
            resolved_pkd = 8.0
            warnings.append(
                "No target Kd or sequence provided; assumed clinical reference potency Kd = 10.0 nM."
            )

        # ----------------------------------------------------------------------
        # 2. ADMET Profile & Organ Toxicity Indicators
        # ----------------------------------------------------------------------
        if custom_admet_profile is not None:
            admet_prof = custom_admet_profile
        else:
            pipe = self._ensure_admet_pipeline()
            admet_prof = pipe.predict_single(canon_smiles)

        def _get_prob(cluster_dict: dict, key: str, default: float = 0.2) -> float:
            if key in cluster_dict:
                ind = cluster_dict[key]
                if getattr(ind, "probability", None) is not None:
                    return float(ind.probability)
                if getattr(ind, "value", None) is not None:
                    return float(ind.value)
            return default

        prob_herg = _get_prob(admet_prof.toxicity, "herg", default=0.25)
        prob_dili = _get_prob(admet_prof.toxicity, "dili", default=0.15)
        prob_ames = _get_prob(admet_prof.toxicity, "ames", default=0.10)
        prob_clintox = _get_prob(admet_prof.toxicity, "clintox", default=0.08)

        # Calibrate or assign hERG IC50
        if herg_ic50_nm is not None and herg_ic50_nm > 0:
            resolved_herg_ic50 = float(herg_ic50_nm)
        else:
            resolved_herg_ic50 = self.calibrate_herg_ic50(prob_herg)

        # ----------------------------------------------------------------------
        # 3. hERG Safety Window & Risk Tier
        # ----------------------------------------------------------------------
        herg_margin = resolved_herg_ic50 / max(resolved_kd, 1e-6)
        ti_window_log10 = math.log10(max(herg_margin, 1e-4))

        if herg_margin >= 100.0:
            herg_tier = "Safe (Margin >= 100x)"
        elif herg_margin >= 30.0:
            herg_tier = "Borderline (Caution 30x-100x)"
            warnings.append(
                f"hERG safety margin ({herg_margin:.1f}x) is borderline (< 100x). "
                "Consider bioisosteric modulation of basic centers."
            )
            recommendations.append(
                "Attenuate basic amine pKa or introduce electron-withdrawing groups to widen the hERG window."
            )
        else:
            herg_tier = "High Risk (Cardiotoxicity Margin < 30x)"
            warnings.append(
                f"CRITICAL: Narrow hERG safety margin ({herg_margin:.1f}x < 30x). "
                "High risk of QT prolongation and clinical ventricular arrhythmias."
            )
            recommendations.append(
                "Immediate lead optimization required: replace basic nitrogens with bioisosteres (amides/ethers/polar heterocycles)."
            )

        # ----------------------------------------------------------------------
        # 4. PBPK In Vivo Exposure & Free Drug Concentration
        # ----------------------------------------------------------------------
        cmax_total_ug_ml = None
        cmax_free_ug_ml = None
        in_vivo_herg_margin = None
        half_life_hr = 4.0
        extraction_ratio_eh = 0.35

        pbpk = getattr(admet_prof, "pbpk", None)
        if pbpk is not None:
            vdss_l_kg = getattr(pbpk, "vdss_l_kg", 2.0)
            fu = getattr(pbpk, "unbound_fraction_fu", 0.1)
            half_life_hr = getattr(pbpk, "half_life_hr", 4.0)
            eh = getattr(pbpk, "extraction_ratio_eh", 0.35)
            extraction_ratio_eh = eh if eh is not None else 0.35
            f_max = getattr(pbpk, "f_max_oral", 0.65)
            if f_max is None:
                f_max = max(0.1, 1.0 - extraction_ratio_eh)

            # Cmax_total (ug/mL) estimate for 70 kg human
            # Dose * F_max / (Vdss * 70 kg)
            cmax_total_ug_ml = (dose_mg * f_max) / max(0.1, vdss_l_kg * 70.0)
            cmax_free_ug_ml = cmax_total_ug_ml * fu

            # Convert free Cmax to nM for direct comparison with IC50
            # Cmax (ug/mL) = (Cmax * 1000 mg/L) / MW (g/mol) = mol/L * 10^3 = mmol/L
            # Cmax_nM = (Cmax_ug_ml * 10^6) / MW
            cmax_free_nm = (cmax_free_ug_ml * 1e6) / max(10.0, mw)
            in_vivo_herg_margin = resolved_herg_ic50 / max(cmax_free_nm, 1e-4)

            if in_vivo_herg_margin < 30.0:
                warnings.append(
                    f"In vivo free hERG margin ({in_vivo_herg_margin:.1f}x) is below FDA recommended 30x threshold at {dose_mg} mg dose."
                )

        # Additional toxicology warnings
        if prob_dili > 0.5:
            warnings.append(
                f"High predicted hepatotoxicity liability (DILI probability: {prob_dili:.2f})."
            )
            recommendations.append(
                "Inspect potential reactive metabolites or quinone-forming structural alerts."
            )
        if prob_ames > 0.5:
            warnings.append(
                f"Positive AMES mutagenicity predicted ({prob_ames:.2f}). Potential IND-enabling regulatory block."
            )
            recommendations.append(
                "Remove mutagenic aromatic amines or nitro groups using bioisosteric replacements."
            )
        if prob_clintox > 0.5:
            warnings.append(
                f"Elevated risk of clinical trial toxicity failure (ClinTox: {prob_clintox:.2f})."
            )

        # ----------------------------------------------------------------------
        # 5. Clinical Developability Index (CDI, 0~100)
        # ----------------------------------------------------------------------
        caco2_val = None
        if "caco2_wang" in admet_prof.absorption:
            caco2_val = getattr(admet_prof.absorption["caco2_wang"], "value", None)
        sol_val = None
        if "solubility_aqsoldb" in admet_prof.absorption:
            sol_val = getattr(admet_prof.absorption["solubility_aqsoldb"], "value", None)

        score_potency = self.calculate_potency_score(resolved_kd)
        score_safety = self.calculate_safety_window_score(herg_margin)
        score_tox = self.calculate_organ_toxicology_score(prob_dili, prob_ames, prob_clintox)
        score_pk = self.calculate_pk_score(half_life_hr, extraction_ratio_eh, caco2_val, sol_val)

        total_score = score_potency + score_safety + score_tox + score_pk
        total_score = max(0.0, min(100.0, total_score))

        if total_score >= 80.0:
            dev_tier = "Tier 1: High Clinical Potential"
        elif total_score >= 60.0:
            dev_tier = "Tier 2: Moderate Potential / Lead Optimization Required"
        else:
            dev_tier = "Tier 3: High Liability Risk"

        component_scores = ComponentScores(
            potency=score_potency,
            safety_window=score_safety,
            organ_toxicology=score_tox,
            human_pk=score_pk,
        )

        return TherapeuticIndexProfile(
            smiles=smiles,
            canonical_smiles=canon_smiles,
            target_kd_nm=resolved_kd,
            target_pkd=resolved_pkd,
            herg_ic50_nm=resolved_herg_ic50,
            herg_safety_margin=herg_margin,
            herg_therapeutic_window_log10=ti_window_log10,
            herg_risk_tier=herg_tier,
            dili_risk_probability=prob_dili,
            clintox_risk_probability=prob_clintox,
            ames_mutagenicity_probability=prob_ames,
            clinical_developability_score=total_score,
            developability_tier=dev_tier,
            component_scores=component_scores,
            pbpk_cmax_total_ug_ml=cmax_total_ug_ml,
            pbpk_cmax_free_ug_ml=cmax_free_ug_ml,
            in_vivo_herg_margin=in_vivo_herg_margin,
            warnings=warnings,
            recommendations=recommendations,
        )
