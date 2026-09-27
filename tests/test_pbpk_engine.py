"""Unit tests for PBPK Engine and pharmacokinetic parameter derivation."""

import pytest

from tdc_studio.pbpk import PBPKEngine


def test_pbpk_engine_total_clearance():
    engine = PBPKEngine()
    # Drug example: Vdss = 1.0 L/kg, t1/2 = 7.0 hr, PPBR = 90%
    profile = engine.calculate_profile(
        smiles="CC(=O)NC1=CC=C(O)C=C1",  # Paracetamol
        vdss_l_kg=1.0,
        half_life_hr=7.0,
        ppbr_percent=90.0,
    )

    assert profile.smiles == "CC(=O)NC1=CC=C(O)C=C1"
    assert pytest.approx(profile.unbound_fraction_fu, rel=1e-3) == 0.10
    # ke = ln(2) / 7.0 = 0.09902
    assert pytest.approx(profile.ke_hr_inv, rel=1e-3) == 0.09902
    # CL = 1.0 * ln(2) / 7.0 = 0.09902 L/hr/kg
    assert pytest.approx(profile.cl_total_l_hr_kg, rel=1e-3) == 0.09902
    # CL in mL/min/kg = 0.09902 * 1000 / 60 = 1.6503 mL/min/kg
    assert pytest.approx(profile.cl_total_ml_min_kg, rel=1e-3) == 1.6503
    # 70kg whole body = 0.09902 * 70 = 6.931 L/hr
    assert pytest.approx(profile.cl_total_l_hr, rel=1e-3) == 6.931
    # Extraction ratio: 1.6503 / 20.7 = 0.0797 -> Low
    assert profile.extraction_class == "Low"
    assert profile.f_max_oral > 0.90


def test_pbpk_engine_well_stirred_ivive():
    engine = PBPKEngine()
    # High clearance drug: microsome CL_int = 100 uL/min/mg
    profile = engine.calculate_profile(
        smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        vdss_l_kg=0.6,
        half_life_hr=4.0,
        ppbr_percent=35.0,
        cl_int_mic_ul_min_mg=100.0,
    )

    # Scaled CL_int = 100 * 45 * 25.7 / 1000 = 115.65 mL/min/kg
    assert profile.cl_int_mic_scaled_ml_min_kg is not None
    assert pytest.approx(profile.cl_int_mic_scaled_ml_min_kg, rel=1e-2) == 115.65

    # fu = 1.0 - 0.35 = 0.65
    assert pytest.approx(profile.unbound_fraction_fu, rel=1e-3) == 0.65

    # CL_H = (20.7 * 0.65 * 115.65) / (20.7 + 0.65 * 115.65)
    # fu * CL_int = 0.65 * 115.65 = 75.17
    # CL_H = (20.7 * 75.17) / (20.7 + 75.17) = 1556 / 95.87 = 16.23 mL/min/kg
    assert profile.cl_hepatic_ml_min_kg is not None
    assert pytest.approx(profile.cl_hepatic_ml_min_kg, rel=1e-2) == 16.23
    assert profile.extraction_ratio_eh is not None
    assert profile.extraction_class in ("Intermediate", "High")

    d = profile.to_dict()
    assert "cl_total_l_hr_70kg" in d
    assert d["unbound_fraction_fu"] == 0.65
