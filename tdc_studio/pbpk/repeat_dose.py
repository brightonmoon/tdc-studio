"""Repeat-Dose Pharmacokinetics and Steady-State Simulation Module.

Implements multi-dose PK accumulation, steady-state metrics, and dynamic time-series:
- Accumulation Ratio: R_ac = 1 / (1 - exp(-k_e * tau))
- Steady-state Peak: C_ss_max
- Steady-state Trough: C_ss_min
- Steady-state Average: C_ss_avg = (F * Dose) / (CL * tau)
- Peak-to-Trough Fluctuation (%PTF)
- Time to 90% and 99% Steady-State (3.32 * t_1/2, 6.64 * t_1/2)
- Multi-dose time series curve (Oral 1-compartment with absorption Bateman equation or IV Bolus)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RepeatDoseParams:
    """Parameters for repeat-dose regimen."""

    dose_mg: float = 100.0  # Dose in milligrams
    interval_hr: float = 24.0  # Dosing interval tau (e.g. 24 for QD, 12 for BID, 8 for TID)
    duration_days: float = 7.0  # Total duration of simulation in days
    route: str = "oral"  # "oral" or "iv_bolus"
    bioavailability_f: float = 0.8  # Fraction absorbed (0.0 ~ 1.0)
    ka_hr_inv: float = 1.2  # Oral absorption rate constant (1/hr)


@dataclass
class SteadyStateMetrics:
    """Summary metrics of drug concentration at steady state."""

    accumulation_ratio_rac: float
    c_ss_max_mg_l: float
    c_ss_min_mg_l: float
    c_ss_avg_mg_l: float
    peak_trough_fluctuation_pct: float
    time_to_90pct_ss_hr: float
    time_to_99pct_ss_hr: float
    doses_to_steady_state: int
    is_safe_against_herg: Optional[bool] = None
    herg_margin_ss_max: Optional[float] = None


@dataclass
class RepeatDoseProfile:
    """Complete multi-dose simulation profile with time series."""

    smiles: str
    regimen: RepeatDoseParams
    steady_state: SteadyStateMetrics
    time_points_hr: List[float] = field(default_factory=list)
    concentrations_mg_l: List[float] = field(default_factory=list)
    trough_concentrations_mg_l: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "smiles": self.smiles,
            "regimen": {
                "dose_mg": self.regimen.dose_mg,
                "interval_hr": self.regimen.interval_hr,
                "duration_days": self.regimen.duration_days,
                "route": self.regimen.route,
                "bioavailability_f": round(self.regimen.bioavailability_f, 3),
                "ka_hr_inv": round(self.regimen.ka_hr_inv, 3),
            },
            "steady_state": {
                "accumulation_ratio_rac": round(self.steady_state.accumulation_ratio_rac, 3),
                "c_ss_max_mg_l": round(self.steady_state.c_ss_max_mg_l, 4),
                "c_ss_min_mg_l": round(self.steady_state.c_ss_min_mg_l, 4),
                "c_ss_avg_mg_l": round(self.steady_state.c_ss_avg_mg_l, 4),
                "peak_trough_fluctuation_pct": round(
                    self.steady_state.peak_trough_fluctuation_pct, 2
                ),
                "time_to_90pct_ss_hr": round(self.steady_state.time_to_90pct_ss_hr, 2),
                "time_to_99pct_ss_hr": round(self.steady_state.time_to_99pct_ss_hr, 2),
                "doses_to_steady_state": self.steady_state.doses_to_steady_state,
                "is_safe_against_herg": self.steady_state.is_safe_against_herg,
                "herg_margin_ss_max": (
                    round(self.steady_state.herg_margin_ss_max, 2)
                    if self.steady_state.herg_margin_ss_max is not None
                    else None
                ),
            },
            "curve_summary": {
                "num_time_points": len(self.time_points_hr),
                "t_final_hr": self.time_points_hr[-1] if self.time_points_hr else 0.0,
                "c_final_mg_l": round(self.concentrations_mg_l[-1], 4)
                if self.concentrations_mg_l
                else 0.0,
            },
        }


class RepeatDoseSimulator:
    """Simulates repeat-dose pharmacokinetics for single or multi-compartment dynamics."""

    def __init__(self, default_weight_kg: float = 70.0):
        self.default_weight_kg = default_weight_kg

    def simulate(
        self,
        smiles: str,
        vdss_l_kg: float,
        half_life_hr: float,
        regimen: Optional[RepeatDoseParams] = None,
        f_max_oral: Optional[float] = None,
        herg_ic50_um: Optional[float] = None,
        mw_g_mol: Optional[float] = 350.0,
        num_points: int = 250,
    ) -> RepeatDoseProfile:
        """Run repeat-dose pharmacokinetic simulation.

        Args:
            smiles: Molecule SMILES representation.
            vdss_l_kg: Steady-state volume of distribution (L/kg).
            half_life_hr: Terminal elimination half-life (hr).
            regimen: Dosing regimen parameters (dose, interval, duration, etc.).
            f_max_oral: Upper bound on oral bioavailability from first-pass hepatic extraction.
            herg_ic50_um: hERG inhibition IC50 in micromolar for safety margin assessment.
            mw_g_mol: Molecular weight for molar concentration conversion.
            num_points: Number of discrete time steps for Cp-t curve.

        Returns:
            RepeatDoseProfile containing steady-state metrics and time series.
        """
        reg = regimen or RepeatDoseParams()

        # Guard against zero or negative values
        vdss = max(0.01, vdss_l_kg)
        t_half = max(0.1, half_life_hr)
        v_total_l = vdss * self.default_weight_kg
        k_e = math.log(2.0) / t_half  # 1/hr
        tau = max(1.0, reg.interval_hr)

        # Determine effective bioavailability F
        f_bio = reg.bioavailability_f
        if reg.route == "iv_bolus":
            f_bio = 1.0
        elif f_max_oral is not None:
            f_bio = min(f_bio, max(0.05, f_max_oral))

        # Accumulation ratio R_ac
        exp_ke_tau = math.exp(-k_e * tau)
        r_ac = 1.0 / (1.0 - exp_ke_tau) if exp_ke_tau < 0.99999 else 100.0

        # Oral absorption rate constant k_a
        k_a = reg.ka_hr_inv
        if abs(k_a - k_e) < 1e-4:
            k_a = k_e * 1.05  # Prevent division by zero in Bateman equation

        # Single dose nominal C0
        dose_mg = max(0.1, reg.dose_mg)
        c0 = (f_bio * dose_mg) / v_total_l

        # Steady-state peak, trough, and average
        if reg.route == "iv_bolus":
            c_ss_max = c0 * r_ac
            c_ss_min = c_ss_max * exp_ke_tau
            c_ss_avg = (f_bio * dose_mg) / ((v_total_l * k_e) * tau)
        else:
            # Oral 1-compartment Bateman peak time: t_max = ln(ka/ke) / (ka - ke)
            t_max_ss = math.log(k_a / k_e) / (k_a - k_e)
            # Steady-state peak via superposition
            r_ka = 1.0 / (1.0 - math.exp(-k_a * tau))
            c_ss_max = (
                c0
                * (k_a / (k_a - k_e))
                * (r_ac * math.exp(-k_e * t_max_ss) - r_ka * math.exp(-k_a * t_max_ss))
            )
            c_ss_max = max(1e-5, c_ss_max)

            # Trough right before next dose (at t = tau)
            c_ss_min = (
                c0
                * (k_a / (k_a - k_e))
                * (r_ac * math.exp(-k_e * tau) - r_ka * math.exp(-k_a * tau))
            )
            c_ss_min = max(0.0, c_ss_min)
            c_ss_avg = (f_bio * dose_mg) / ((v_total_l * k_e) * tau)

        # Fluctuation %PTF
        ptf = ((c_ss_max - c_ss_min) / c_ss_avg) * 100.0 if c_ss_avg > 0 else 0.0

        # Time to steady state
        t_90 = 3.32 * t_half
        t_99 = 6.64 * t_half
        doses_to_ss = max(1, math.ceil(t_90 / tau))

        # hERG Safety Margin check at peak concentration
        herg_margin = None
        is_safe_herg = None
        if herg_ic50_um is not None and herg_ic50_um > 0 and mw_g_mol and mw_g_mol > 0:
            # Convert C_ss_max (mg/L) to uM: (mg/L) / (g/mol) * 1000 = uM
            c_ss_max_um = (c_ss_max / mw_g_mol) * 1000.0
            herg_margin = herg_ic50_um / max(1e-6, c_ss_max_um)
            is_safe_herg = herg_margin >= 30.0  # Standard clinical safety margin criterion (30x)

        ss_metrics = SteadyStateMetrics(
            accumulation_ratio_rac=r_ac,
            c_ss_max_mg_l=c_ss_max,
            c_ss_min_mg_l=c_ss_min,
            c_ss_avg_mg_l=c_ss_avg,
            peak_trough_fluctuation_pct=ptf,
            time_to_90pct_ss_hr=t_90,
            time_to_99pct_ss_hr=t_99,
            doses_to_steady_state=doses_to_ss,
            is_safe_against_herg=is_safe_herg,
            herg_margin_ss_max=herg_margin,
        )

        # Generate multi-dose time series
        total_time_hr = reg.duration_days * 24.0
        dt = total_time_hr / max(10, num_points - 1)
        time_points = [i * dt for i in range(num_points)]
        concentrations: List[float] = []

        total_doses = math.ceil(total_time_hr / tau)

        for t in time_points:
            c_total = 0.0
            # Superposition across all preceding doses
            for d in range(total_doses):
                t_dose = d * tau
                if t >= t_dose:
                    dt_dose = t - t_dose
                    if reg.route == "iv_bolus":
                        c_total += c0 * math.exp(-k_e * dt_dose)
                    else:
                        c_total += (
                            c0
                            * (k_a / (k_a - k_e))
                            * (math.exp(-k_e * dt_dose) - math.exp(-k_a * dt_dose))
                        )
            concentrations.append(max(0.0, c_total))

        # Trough concentrations (sampled at n*tau)
        trough_concs = []
        for d in range(1, total_doses + 1):
            t_trough = d * tau
            if t_trough <= total_time_hr:
                idx = min(range(len(time_points)), key=lambda i: abs(time_points[i] - t_trough))
                trough_concs.append(concentrations[idx])

        return RepeatDoseProfile(
            smiles=smiles,
            regimen=reg,
            steady_state=ss_metrics,
            time_points_hr=time_points,
            concentrations_mg_l=concentrations,
            trough_concentrations_mg_l=trough_concs,
        )
