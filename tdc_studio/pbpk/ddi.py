"""Mechanism-Based CYP Drug-Drug Interaction (DDI) Simulation Module.

Implements FDA/ICH M12 Drug Interaction Guidance static mechanistic models:
- Predicts Victim Drug AUC Fold Change upon co-administration:
    AUC_ratio = 1 / ((1 - f_m) + (f_m / (1 + [I] / Ki)))
- Estimates systemic unbound inhibitor concentration [I]_u = fu * C_max
- Estimates hepatic inlet unbound concentration [I]_inlet,u = fu * (C_max + (ka * Fa * Dose) / Qh)
- Evaluates DDI risk against 5 major CYP isoforms (1A2, 2C9, 2C19, 2D6, 3A4)
- Identifies potential contraindications and clinical warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StandardVictimDrug:
    """Standard clinical probe substrate representing a CYP isoform."""

    name: str
    target_cyp: str  # e.g., "CYP3A4", "CYP2C9"
    fraction_metabolized_fm: float  # Fraction of clearance via target enzyme (f_m)
    clinical_risk_class: str  # Narrow therapeutic index, sensitive index substrate, etc.
    description: str


# FDA Clinical DDI Guidance Standard Sensitive Index Substrates
STANDARD_VICTIM_DRUGS: Dict[str, StandardVictimDrug] = {
    "midazolam": StandardVictimDrug(
        name="Midazolam",
        target_cyp="CYP3A4",
        fraction_metabolized_fm=0.92,
        clinical_risk_class="Sensitive Index Substrate",
        description="Standard FDA probe substrate for CYP3A4-mediated clearance.",
    ),
    "warfarin": StandardVictimDrug(
        name="S-Warfarin",
        target_cyp="CYP2C9",
        fraction_metabolized_fm=0.85,
        clinical_risk_class="Narrow Therapeutic Index (NTI)",
        description="Anticoagulant with high risk of severe bleeding upon CYP2C9 inhibition.",
    ),
    "omeprazole": StandardVictimDrug(
        name="Omeprazole",
        target_cyp="CYP2C19",
        fraction_metabolized_fm=0.82,
        clinical_risk_class="Sensitive Index Substrate",
        description="Proton pump inhibitor probe for CYP2C19 metabolism.",
    ),
    "dextromethorphan": StandardVictimDrug(
        name="Dextromethorphan",
        target_cyp="CYP2D6",
        fraction_metabolized_fm=0.90,
        clinical_risk_class="Sensitive Index Substrate",
        description="Standard FDA probe substrate for CYP2D6.",
    ),
    "caffeine": StandardVictimDrug(
        name="Caffeine",
        target_cyp="CYP1A2",
        fraction_metabolized_fm=0.88,
        clinical_risk_class="Probe Substrate",
        description="Standard FDA probe substrate for CYP1A2.",
    ),
}


@dataclass
class SingleCYPDDIRisk:
    """Quantitative DDI evaluation for a single CYP isoform."""

    cyp_enzyme: str  # "CYP3A4", etc.
    is_inhibitor_predicted: bool
    estimated_ki_um: float
    inhibitor_conc_um: float  # [I]_u or [I]_inlet,u
    auc_fold_change: float  # AUC_i / AUC_control
    ddi_classification: str  # "Strong" (>=5x), "Moderate" (2x~5x), "Weak" (1.25x~2x), "No Clinical Risk" (<1.25x)
    affected_victim_drug: str
    clinical_recommendation: str


@dataclass
class DDIProfile:
    """Comprehensive CYP DDI risk profile."""

    smiles: str
    dose_mg: float
    c_max_mg_l: float
    unbound_fraction_fu: float
    perpetrator_risks: List[SingleCYPDDIRisk] = field(default_factory=list)
    has_severe_ddi_risk: bool = False
    contraindicated_drugs: List[str] = field(default_factory=list)
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to serializable dictionary."""
        return {
            "smiles": self.smiles,
            "dose_mg": self.dose_mg,
            "c_max_mg_l": round(self.c_max_mg_l, 4),
            "unbound_fraction_fu": round(self.unbound_fraction_fu, 4),
            "has_severe_ddi_risk": self.has_severe_ddi_risk,
            "contraindicated_drugs": self.contraindicated_drugs,
            "summary_text": self.summary_text,
            "cyp_evaluations": [
                {
                    "cyp_enzyme": r.cyp_enzyme,
                    "is_inhibitor_predicted": r.is_inhibitor_predicted,
                    "estimated_ki_um": round(r.estimated_ki_um, 3),
                    "inhibitor_conc_um": round(r.inhibitor_conc_um, 4),
                    "auc_fold_change": round(r.auc_fold_change, 3),
                    "ddi_classification": r.ddi_classification,
                    "affected_victim_drug": r.affected_victim_drug,
                    "clinical_recommendation": r.clinical_recommendation,
                }
                for r in self.perpetrator_risks
            ],
        }


class DDISimulator:
    """Evaluates mechanism-based pharmacokinetic drug-drug interactions (DDI)."""

    def __init__(self, hepatic_blood_flow_l_hr: float = 90.0):
        # Hepatic blood flow ~1500 mL/min = 90 L/hr
        self.qh_l_hr = hepatic_blood_flow_l_hr

    def evaluate_ddi(
        self,
        smiles: str,
        cyp_inhibition_probs: Dict[str, float],
        c_max_mg_l: float,
        unbound_fraction_fu: float,
        dose_mg: float = 100.0,
        mw_g_mol: float = 350.0,
        use_hepatic_inlet_conc: bool = True,
        co_administered_drugs: Optional[List[str]] = None,
    ) -> DDIProfile:
        """Calculate mechanistic CYP DDI risk and victim AUC fold changes.

        Args:
            smiles: Perpetrator compound SMILES.
            cyp_inhibition_probs: Dictionary mapping CYP names ("CYP1A2", "CYP2C9", "CYP2C19",
                "CYP2D6", "CYP3A4") to inhibition probability (0.0 ~ 1.0).
            c_max_mg_l: Maximum total plasma concentration (mg/L).
            unbound_fraction_fu: Fraction unbound in plasma (0.0 ~ 1.0).
            dose_mg: Clinical oral dose (mg).
            mw_g_mol: Molecular weight of the perpetrator.
            use_hepatic_inlet_conc: Whether to use liver inlet concentration [I]_inlet
                (conservative, recommended by FDA for orally administered drugs).
            co_administered_drugs: Optional list of user-queried victim drugs to specifically check.

        Returns:
            DDIProfile with quantitative fold changes and clinical recommendations.
        """
        fu = max(0.001, min(1.0, unbound_fraction_fu))
        mw = max(100.0, mw_g_mol)

        # Convert C_max (mg/L) to total uM
        c_max_total_um = (c_max_mg_l / mw) * 1000.0
        # Systemic unbound concentration [I]_u (uM)
        i_sys_u = fu * c_max_total_um

        # Liver inlet unbound concentration: [I]_inlet,u = fu * (C_max + (ka * Fa * Dose) / Qh)
        # Assuming typical ka * Fa = 0.8 / hr
        dose_l_hr_um = ((0.8 * dose_mg) / self.qh_l_hr / mw) * 1000.0
        i_inlet_u = fu * (c_max_total_um + dose_l_hr_um)

        conc_i = i_inlet_u if use_hepatic_inlet_conc else i_sys_u

        cyp_keys = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
        probe_map = {
            "CYP1A2": STANDARD_VICTIM_DRUGS["caffeine"],
            "CYP2C9": STANDARD_VICTIM_DRUGS["warfarin"],
            "CYP2C19": STANDARD_VICTIM_DRUGS["omeprazole"],
            "CYP2D6": STANDARD_VICTIM_DRUGS["dextromethorphan"],
            "CYP3A4": STANDARD_VICTIM_DRUGS["midazolam"],
        }

        risks: List[SingleCYPDDIRisk] = []
        contraindicated: List[str] = []
        severe_risk = False

        for cyp in cyp_keys:
            # Check prob from input, supporting keys like "cyp3a4_veith", "cyp3a4", "CYP3A4"
            prob = 0.0
            for k, v in cyp_inhibition_probs.items():
                if cyp.lower() in k.lower():
                    prob = v
                    break

            is_inh = prob >= 0.5

            # Translate probability to an estimated functional Ki (uM)
            # High probability (>0.85) -> potent low Ki (~0.05 - 1.5 uM)
            # Moderate prob (0.5 - 0.85) -> moderate Ki (~2.0 - 15.0 uM)
            # Low prob (<0.5) -> high Ki (>50 uM, non-inhibitor)
            if prob >= 0.85:
                est_ki = max(0.05, (1.0 - prob) * 10.0)
            elif prob >= 0.50:
                est_ki = 1.5 + (0.85 - prob) * 30.0
            else:
                est_ki = 50.0 + (0.50 - prob) * 100.0

            victim = probe_map[cyp]
            fm = victim.fraction_metabolized_fm

            # Static mechanistic formula: AUC_ratio = 1 / ((1 - fm) + (fm / (1 + [I] / Ki)))
            i_over_ki = conc_i / max(0.01, est_ki)
            auc_fold = 1.0 / ((1.0 - fm) + (fm / (1.0 + i_over_ki)))
            auc_fold = max(1.0, auc_fold)

            # FDA Classification
            if auc_fold >= 5.0:
                classification = "Strong Inhibitor"
                rec = (
                    f"Contraindicated or requires severe dose reduction of {victim.name}. "
                    "Avoid co-administration."
                )
                contraindicated.append(victim.name)
                severe_risk = True
            elif auc_fold >= 2.0:
                classification = "Moderate Inhibitor"
                rec = (
                    f"Caution advised with {victim.name}. Consider 50% dose reduction "
                    "and clinical monitoring."
                )
                if "Narrow Therapeutic Index" in victim.clinical_risk_class:
                    contraindicated.append(victim.name)
                    severe_risk = True
            elif auc_fold >= 1.25:
                classification = "Weak Inhibitor"
                rec = f"Mild interaction potential with {victim.name}. Clinical monitoring recommended."
            else:
                classification = "No Clinical Risk"
                rec = f"No significant interaction expected with {victim.name}."

            risks.append(
                SingleCYPDDIRisk(
                    cyp_enzyme=cyp,
                    is_inhibitor_predicted=is_inh,
                    estimated_ki_um=est_ki,
                    inhibitor_conc_um=conc_i,
                    auc_fold_change=auc_fold,
                    ddi_classification=classification,
                    affected_victim_drug=victim.name,
                    clinical_recommendation=rec,
                )
            )

        # Generate summary text
        strong_cyps = [r.cyp_enzyme for r in risks if r.ddi_classification == "Strong Inhibitor"]
        mod_cyps = [r.cyp_enzyme for r in risks if r.ddi_classification == "Moderate Inhibitor"]

        summary_parts = []
        if strong_cyps:
            summary_parts.append(
                f"POTENT INHIBITOR of {', '.join(strong_cyps)} (AUC Fold Change >= 5.0x)."
            )
        if mod_cyps:
            summary_parts.append(
                f"Moderate inhibitor of {', '.join(mod_cyps)} (AUC Fold Change 2.0x - 5.0x)."
            )
        if not strong_cyps and not mod_cyps:
            summary_parts.append(
                "Favorable DDI profile with negligible to weak inhibition across major 5 CYPs."
            )

        summary_text = " ".join(summary_parts)

        return DDIProfile(
            smiles=smiles,
            dose_mg=dose_mg,
            c_max_mg_l=c_max_mg_l,
            unbound_fraction_fu=fu,
            perpetrator_risks=risks,
            has_severe_ddi_risk=severe_risk,
            contraindicated_drugs=contraindicated,
            summary_text=summary_text,
        )
