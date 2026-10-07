"""Unit tests for repeat-dose pharmacokinetics and CYP DDI simulation engines."""

import pytest

from tdc_studio.pbpk.ddi import DDISimulator
from tdc_studio.pbpk.repeat_dose import RepeatDoseParams, RepeatDoseSimulator


def test_repeat_dose_simulation_oral_qd():
    simulator = RepeatDoseSimulator(default_weight_kg=70.0)
    regimen = RepeatDoseParams(
        dose_mg=100.0,
        interval_hr=24.0,  # QD
        duration_days=7.0,
        route="oral",
        bioavailability_f=0.8,
        ka_hr_inv=1.2,
    )

    profile = simulator.simulate(
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        vdss_l_kg=0.5,
        half_life_hr=12.0,
        regimen=regimen,
        herg_ic50_um=15.0,
        mw_g_mol=180.16,
    )

    ss = profile.steady_state
    # With t_half = 12h and tau = 24h, exp(-ke*tau) = exp(-ln2 * 2) = 0.25
    # R_ac = 1 / (1 - 0.25) = 1.333
    assert 1.30 <= ss.accumulation_ratio_rac <= 1.37
    assert ss.c_ss_max_mg_l > ss.c_ss_min_mg_l >= 0.0
    assert ss.c_ss_avg_mg_l > 0.0
    assert ss.peak_trough_fluctuation_pct > 0.0
    assert ss.doses_to_steady_state == 2  # 3.32 * 12 = 39.84h -> ceil(39.84/24) = 2
    # At 100mg dose, C_ss_max is ~14.7 uM, so IC50=15 uM gives margin ~1.02 (< 30x)
    assert ss.is_safe_against_herg is False
    assert ss.herg_margin_ss_max == pytest.approx(1.02, abs=0.05)

    # Re-test with high IC50 (safe drug)
    profile_safe = simulator.simulate(
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        vdss_l_kg=0.5,
        half_life_hr=12.0,
        regimen=regimen,
        herg_ic50_um=500.0,
        mw_g_mol=180.16,
    )
    assert profile_safe.steady_state.is_safe_against_herg is True
    assert profile_safe.steady_state.herg_margin_ss_max > 30.0

    # Check time series
    assert len(profile.time_points_hr) == 250
    assert len(profile.concentrations_mg_l) == 250
    assert profile.concentrations_mg_l[0] == pytest.approx(0.0, abs=1e-5)
    assert max(profile.concentrations_mg_l) > 0.5

    data_dict = profile.to_dict()
    assert "steady_state" in data_dict
    assert "regimen" in data_dict
    assert "curve_summary" in data_dict


def test_repeat_dose_simulation_iv_bid():
    simulator = RepeatDoseSimulator(default_weight_kg=70.0)
    regimen = RepeatDoseParams(
        dose_mg=50.0,
        interval_hr=12.0,  # BID
        duration_days=5.0,
        route="iv_bolus",
    )

    profile = simulator.simulate(
        smiles="Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        vdss_l_kg=1.0,
        half_life_hr=6.0,
        regimen=regimen,
    )

    ss = profile.steady_state
    # t_half = 6h, tau = 12h -> ke*tau = ln2 * 2 -> exp(-ke*tau) = 0.25
    # R_ac = 1 / (1 - 0.25) = 1.333
    assert 1.30 <= ss.accumulation_ratio_rac <= 1.37
    # For IV bolus, C_ss_min = C_ss_max * 0.25
    assert ss.c_ss_min_mg_l == pytest.approx(ss.c_ss_max_mg_l * 0.25, rel=1e-3)


def test_ddi_simulation_strong_and_weak():
    ddi_sim = DDISimulator()

    # Potent CYP3A4 inhibitor (>5x fold change on Midazolam)
    cyp_probs = {
        "CYP3A4": 0.98,
        "CYP2C9": 0.85,
        "CYP2C19": 0.20,
        "CYP2D6": 0.15,
        "CYP1A2": 0.05,
    }

    profile = ddi_sim.evaluate_ddi(
        smiles="O=C(c1ccc(Cl)cc1)c1c[nH]cn1",
        cyp_inhibition_probs=cyp_probs,
        c_max_mg_l=2.5,
        unbound_fraction_fu=0.10,
        dose_mg=200.0,
        mw_g_mol=300.0,
    )

    assert profile.has_severe_ddi_risk is True
    assert "Midazolam" in profile.contraindicated_drugs

    cyp3a4_risk = next(r for r in profile.perpetrator_risks if r.cyp_enzyme == "CYP3A4")
    assert cyp3a4_risk.is_inhibitor_predicted is True
    assert cyp3a4_risk.auc_fold_change >= 2.0  # Significant victim AUC elevation

    cyp1a2_risk = next(r for r in profile.perpetrator_risks if r.cyp_enzyme == "CYP1A2")
    assert cyp1a2_risk.is_inhibitor_predicted is False
    assert cyp1a2_risk.ddi_classification == "No Clinical Risk"
    assert cyp1a2_risk.auc_fold_change < 1.25

    d = profile.to_dict()
    assert len(d["cyp_evaluations"]) == 5
    assert "POTENT INHIBITOR" in d["summary_text"] or "inhibitor" in d["summary_text"]
