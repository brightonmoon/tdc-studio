"""PBPK Virtual Population Monte Carlo Simulation Engine.

Simulates inter-individual variability (IIV) across clinical human sub-populations:
- Healthy Adults (reference reference 70kg, log-normal physiological variance)
- Renal Impairment (Mild, Moderate, Severe eGFR reduction)
- Hepatic Impairment (Child-Pugh Class A, B, C)
- Geriatric (Elderly >= 65yo, reduced clearance & altered Vdss)

Computes statistical distribution (Median, 5th-95th percentile CI, %CV)
and Monte Carlo concentration-time PK profiles (Cmax, Tmax, AUC_inf).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from tdc_studio.pbpk.engine import PBPKProfile


class PopulationSubgroup(str, Enum):
    """Clinical sub-population category for PBPK simulation."""

    HEALTHY_ADULTS = "healthy_adults"
    RENAL_MILD = "renal_mild"  # eGFR 60-89 mL/min/1.73m2
    RENAL_MODERATE = "renal_moderate"  # eGFR 30-59 mL/min/1.73m2
    RENAL_SEVERE = "renal_severe"  # eGFR < 30 mL/min/1.73m2
    HEPATIC_CHILD_PUGH_A = "hepatic_child_pugh_a"  # Mild hepatic impairment
    HEPATIC_CHILD_PUGH_B = "hepatic_child_pugh_b"  # Moderate hepatic impairment
    HEPATIC_CHILD_PUGH_C = "hepatic_child_pugh_c"  # Severe hepatic impairment
    GERIATRIC = "geriatric"  # Age >= 65


@dataclass
class SubgroupModifier:
    """Multiplicative modifiers for sub-population physiology relative to standard adult."""

    bw_factor: float = 1.0
    qh_factor: float = 1.0  # Hepatic blood flow modifier
    mppgl_factor: float = 1.0  # Microsomal enzyme capacity modifier
    fu_factor: float = 1.0  # Fraction unbound modifier (e.g. hypoalbuminemia)
    renal_cl_factor: float = 1.0  # Renal filtration clearance factor
    vdss_factor: float = 1.0  # Distribution volume factor


SUBGROUP_MODIFIERS: Dict[PopulationSubgroup, SubgroupModifier] = {
    PopulationSubgroup.HEALTHY_ADULTS: SubgroupModifier(),
    PopulationSubgroup.RENAL_MILD: SubgroupModifier(
        renal_cl_factor=0.75,
    ),
    PopulationSubgroup.RENAL_MODERATE: SubgroupModifier(
        renal_cl_factor=0.45,
        fu_factor=1.15,  # slight reduction in albumin binding
    ),
    PopulationSubgroup.RENAL_SEVERE: SubgroupModifier(
        renal_cl_factor=0.20,
        fu_factor=1.35,  # moderate uremic displacement of plasma binding
    ),
    PopulationSubgroup.HEPATIC_CHILD_PUGH_A: SubgroupModifier(
        qh_factor=0.85,
        mppgl_factor=0.80,
        fu_factor=1.20,
    ),
    PopulationSubgroup.HEPATIC_CHILD_PUGH_B: SubgroupModifier(
        qh_factor=0.65,
        mppgl_factor=0.55,
        fu_factor=1.50,
        renal_cl_factor=0.80,  # hepatorenal syndrome onset
    ),
    PopulationSubgroup.HEPATIC_CHILD_PUGH_C: SubgroupModifier(
        qh_factor=0.45,
        mppgl_factor=0.30,
        fu_factor=2.00,
        renal_cl_factor=0.50,
    ),
    PopulationSubgroup.GERIATRIC: SubgroupModifier(
        bw_factor=0.95,
        qh_factor=0.75,
        mppgl_factor=0.80,
        renal_cl_factor=0.70,
        vdss_factor=1.20,  # increased lipophilic adipose volume
    ),
}


@dataclass
class PKMetricSummary:
    """Summary statistics of a pharmacokinetic parameter across virtual population."""

    mean: float
    sd: float
    cv_pct: float
    median: float
    p5: float
    p25: float
    p75: float
    p95: float

    @classmethod
    def from_array(cls, values: np.ndarray) -> PKMetricSummary:
        """Compute statistical percentiles and moments from numeric array."""
        arr = np.asarray(values, dtype=float)
        mean_val = float(np.mean(arr))
        sd_val = float(np.std(arr))
        cv = (sd_val / mean_val * 100.0) if mean_val > 1e-9 else 0.0
        return cls(
            mean=round(mean_val, 4),
            sd=round(sd_val, 4),
            cv_pct=round(cv, 2),
            median=round(float(np.median(arr)), 4),
            p5=round(float(np.percentile(arr, 5)), 4),
            p25=round(float(np.percentile(arr, 25)), 4),
            p75=round(float(np.percentile(arr, 75)), 4),
            p95=round(float(np.percentile(arr, 95)), 4),
        )


@dataclass
class ConcentrationTimeTrajectory:
    """Monte Carlo population concentration-time profile trajectory."""

    time_hours: List[float]
    p5_ug_ml: List[float]
    median_ug_ml: List[float]
    p95_ug_ml: List[float]
    mean_ug_ml: List[float]


@dataclass
class VirtualPopulationSimulationResult:
    """Comprehensive result of virtual population Monte Carlo simulation."""

    subgroup: str
    n_subjects: int
    dose_mg: float
    vdss_l_kg: PKMetricSummary
    cl_total_l_h_kg: PKMetricSummary
    half_life_hours: PKMetricSummary
    cmax_ug_ml: PKMetricSummary
    tmax_hours: PKMetricSummary
    auc_inf_ug_h_ml: PKMetricSummary
    fraction_unbound: PKMetricSummary
    trajectory: ConcentrationTimeTrajectory

    def to_dict(self) -> Dict[str, Any]:
        """Convert simulation result to dictionary."""
        return {
            "subgroup": self.subgroup,
            "n_subjects": self.n_subjects,
            "dose_mg": self.dose_mg,
            "metrics": {
                "vdss_l_kg": asdict(self.vdss_l_kg),
                "cl_total_l_h_kg": asdict(self.cl_total_l_h_kg),
                "half_life_hours": asdict(self.half_life_hours),
                "cmax_ug_ml": asdict(self.cmax_ug_ml),
                "tmax_hours": asdict(self.tmax_hours),
                "auc_inf_ug_h_ml": asdict(self.auc_inf_ug_h_ml),
                "fraction_unbound": asdict(self.fraction_unbound),
            },
            "trajectory": asdict(self.trajectory),
        }


class VirtualPopulationEngine:
    """Monte Carlo virtual population simulator for mechanistic PBPK profiling."""

    def __init__(self, random_seed: Optional[int] = 42):
        self.rng = np.random.default_rng(random_seed)

    def simulate(
        self,
        smiles: str,
        baseline_profile: PBPKProfile,
        subgroup: PopulationSubgroup = PopulationSubgroup.HEALTHY_ADULTS,
        n_subjects: int = 500,
        dose_mg: float = 100.0,
        ka_per_h: float = 1.2,
        t_max_sim_hours: float = 48.0,
        n_timepoints: int = 60,
    ) -> VirtualPopulationSimulationResult:
        """Run Monte Carlo virtual population trial.

        Args:
            smiles: Chemical structure SMILES.
            baseline_profile: Baseline 70kg adult PBPKProfile.
            subgroup: Clinical sub-population category.
            n_subjects: Number of virtual human subjects (default: 500).
            dose_mg: Single oral dose administered in mg.
            ka_per_h: Absorption rate constant (1/h).
            t_max_sim_hours: Duration of concentration simulation in hours.
            n_timepoints: Resolution of time points for trajectory plotting.

        Returns:
            VirtualPopulationSimulationResult containing statistical metrics and trajectory.
        """
        n_subjects = max(10, min(n_subjects, 5000))
        mod = SUBGROUP_MODIFIERS.get(subgroup, SubgroupModifier())

        # 1. Sample Log-Normal Inter-Individual Variations
        # CV: BW ~ 18%, Vdss ~ 25%, MPPGL ~ 20%
        sigma_bw = math.sqrt(math.log(1.0 + 0.18**2))
        sigma_vd = math.sqrt(math.log(1.0 + 0.25**2))
        sigma_cl = math.sqrt(math.log(1.0 + 0.28**2))

        bw_samples = 70.0 * mod.bw_factor * np.exp(self.rng.normal(-0.5 * sigma_bw**2, sigma_bw, n_subjects))
        vd_samples = (
            baseline_profile.vdss_l_kg
            * mod.vdss_factor
            * np.exp(self.rng.normal(-0.5 * sigma_vd**2, sigma_vd, n_subjects))
        )
        vd_samples = np.clip(vd_samples, 0.05, 50.0)

        # Fraction unbound (logit normal variation with upper bound clipping)
        base_fu = baseline_profile.unbound_fraction_fu * mod.fu_factor
        base_fu = min(max(base_fu, 0.001), 0.999)
        base_logit = math.log(base_fu / (1.0 - base_fu))
        fu_logits = self.rng.normal(base_logit, 0.35, n_subjects)
        fu_samples = 1.0 / (1.0 + np.exp(-fu_logits))
        fu_samples = np.clip(fu_samples, 0.001, 1.0)

        # Scaled systemic clearance incorporating hepatic + renal clearance
        # Assume ~30% renal clearance baseline and ~70% hepatic clearance
        base_cl_total = baseline_profile.cl_total_l_hr_kg
        cl_renal_base = base_cl_total * 0.30 * mod.renal_cl_factor
        cl_hepatic_base = base_cl_total * 0.70 * (mod.qh_factor * mod.mppgl_factor)
        composite_cl_base = cl_renal_base + cl_hepatic_base

        cl_samples = composite_cl_base * np.exp(self.rng.normal(-0.5 * sigma_cl**2, sigma_cl, n_subjects))
        cl_samples = np.clip(cl_samples, 0.005, 20.0)  # L/h/kg

        # Half-life: t_1/2 = (Vdss * ln2) / CL_total
        ln2 = math.log(2.0)
        thalf_samples = (vd_samples * ln2) / cl_samples
        thalf_samples = np.clip(thalf_samples, 0.1, 500.0)

        # Elimination rate constant: k_e = ln2 / t_1/2
        ke_samples = ln2 / thalf_samples

        # 2. Oral PK Curve & Exposure Metrics for 1-Compartment Well-Stirred Model
        # C(t) = (D * F * ka) / (Vd_total * (ka - ke)) * (exp(-ke * t) - exp(-ka * t))
        # Total distribution volume Vd_total = Vd (L/kg) * BW (kg)
        f_oral = baseline_profile.f_max_oral if baseline_profile.f_max_oral is not None else 0.85
        f_oral = max(0.05, min(f_oral, 1.0))

        vd_total_liters = vd_samples * bw_samples  # Liters
        dose_ug = dose_mg * 1000.0  # ug

        # Tmax = ln(ka / ke) / (ka - ke)
        # Avoid division by zero when ka ~= ke
        ke_diff = np.where(np.abs(ka_per_h - ke_samples) < 1e-4, 1e-4, ka_per_h - ke_samples)
        tmax_samples = np.log(ka_per_h / np.maximum(ke_samples, 1e-5)) / ke_diff
        tmax_samples = np.clip(tmax_samples, 0.1, 48.0)

        # Cmax = C(tmax) in ug/mL (ug / L * 1/1000 = ug/mL)
        cmax_samples = (
            (dose_ug * f_oral * ka_per_h)
            / (vd_total_liters * 1000.0 * ke_diff)
            * (np.exp(-ke_samples * tmax_samples) - np.exp(-ka_per_h * tmax_samples))
        )
        cmax_samples = np.maximum(cmax_samples, 1e-6)

        # AUC_inf = (D * F) / (CL_total_whole_body)
        # CL_whole_body (L/h) = cl_samples (L/h/kg) * bw_samples (kg)
        cl_whole_body_l_h = cl_samples * bw_samples
        auc_samples = (dose_ug * f_oral) / (cl_whole_body_l_h * 1000.0)  # ug*h/mL
        auc_samples = np.maximum(auc_samples, 1e-5)

        # 3. Compute Trajectory Time-Course
        time_points = np.linspace(0.0, t_max_sim_hours, n_timepoints)
        # Shape: (n_subjects, n_timepoints)
        # c(t) = factor * (exp(-ke*t) - exp(-ka*t))
        t_matrix = time_points.reshape(1, -1)  # (1, T)
        ke_col = ke_samples.reshape(-1, 1)  # (N, 1)
        diff_col = ke_diff.reshape(-1, 1)  # (N, 1)
        vd_col = vd_total_liters.reshape(-1, 1)  # (N, 1)

        conc_matrix = (
            (dose_ug * f_oral * ka_per_h)
            / (vd_col * 1000.0 * diff_col)
            * (np.exp(-ke_col * t_matrix) - np.exp(-ka_per_h * t_matrix))
        )
        conc_matrix = np.maximum(conc_matrix, 0.0)

        p5_curve = np.percentile(conc_matrix, 5, axis=0).round(4).tolist()
        median_curve = np.median(conc_matrix, axis=0).round(4).tolist()
        p95_curve = np.percentile(conc_matrix, 95, axis=0).round(4).tolist()
        mean_curve = np.mean(conc_matrix, axis=0).round(4).tolist()

        trajectory = ConcentrationTimeTrajectory(
            time_hours=[round(t, 2) for t in time_points],
            p5_ug_ml=p5_curve,
            median_ug_ml=median_curve,
            p95_ug_ml=p95_curve,
            mean_ug_ml=mean_curve,
        )

        return VirtualPopulationSimulationResult(
            subgroup=subgroup.value,
            n_subjects=n_subjects,
            dose_mg=dose_mg,
            vdss_l_kg=PKMetricSummary.from_array(vd_samples),
            cl_total_l_h_kg=PKMetricSummary.from_array(cl_samples),
            half_life_hours=PKMetricSummary.from_array(thalf_samples),
            cmax_ug_ml=PKMetricSummary.from_array(cmax_samples),
            tmax_hours=PKMetricSummary.from_array(tmax_samples),
            auc_inf_ug_h_ml=PKMetricSummary.from_array(auc_samples),
            fraction_unbound=PKMetricSummary.from_array(fu_samples),
            trajectory=trajectory,
        )
