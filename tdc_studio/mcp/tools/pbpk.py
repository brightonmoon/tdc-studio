"""PBPK simulation and CYP DDI risk evaluation tools for MCP server."""

from __future__ import annotations

import json
from typing import Optional

from mcp.server.mcpserver import MCPServer
from rdkit import Chem
from rdkit.Chem import Descriptors

from tdc_studio.pbpk.ddi import DDISimulator
from tdc_studio.pbpk.repeat_dose import RepeatDoseParams, RepeatDoseSimulator


def register_pbpk_tools(server: MCPServer) -> None:
    """Register PBPK repeat-dose and DDI evaluation tools."""

    @server.tool(
        name="simulate_pbpk_regimen",
        description="Simulate repeat-dose PBPK (QD/BID), steady-state metrics, and hERG safety margin",
    )
    def simulate_pbpk_regimen(
        smiles: str,
        dose_mg: float = 100.0,
        interval_hr: float = 24.0,
        duration_days: float = 7.0,
        route: str = "oral",
        vdss_l_kg: Optional[float] = None,
        half_life_hr: Optional[float] = None,
    ) -> str:
        """Simulate repeat-dose pharmacokinetics to determine steady-state metrics and cardiotoxicity margin.

        Args:
            smiles: Small molecule SMILES string.
            dose_mg: Clinical dose in milligrams (e.g. 100.0).
            interval_hr: Dosing interval tau in hours (24 for QD, 12 for BID, 8 for TID).
            duration_days: Simulation duration in days (e.g. 7.0).
            route: Administration route ("oral" or "iv_bolus").
            vdss_l_kg: Optional volume of distribution (L/kg). If omitted, derived from ADMET.
            half_life_hr: Optional elimination half-life (hr). If omitted, derived from ADMET.

        Returns:
            JSON string with steady-state peak (Css_max), trough (Css_min), accumulation ratio (Rac),
            and clinical hERG safety margin.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return json.dumps({"error": f"Invalid SMILES string: '{smiles}'"}, indent=2)

        mw = float(Descriptors.MolWt(mol))

        # Default or predicted PK parameters
        vdss = vdss_l_kg or 1.2
        t_half = half_life_hr or 8.5
        herg_ic50_um = 12.0  # Default ~12 uM

        try:
            from tdc_studio.serving.unified_pipeline import UnifiedADMETPipeline

            pipe = UnifiedADMETPipeline.from_exported_directory("models/export")
            admet_res = pipe.predict_full(smiles).to_dict()
            pbpk_init = admet_res.get("pbpk_pk_profile", {})
            if vdss_l_kg is None and "vdss_l_kg" in pbpk_init:
                vdss = float(pbpk_init["vdss_l_kg"])
            if half_life_hr is None and "half_life_hr" in pbpk_init:
                t_half = float(pbpk_init["half_life_hr"])
        except Exception:
            pass

        simulator = RepeatDoseSimulator(default_weight_kg=70.0)
        regimen = RepeatDoseParams(
            dose_mg=dose_mg,
            interval_hr=interval_hr,
            duration_days=duration_days,
            route=route,
        )

        profile = simulator.simulate(
            smiles=smiles,
            vdss_l_kg=vdss,
            half_life_hr=t_half,
            regimen=regimen,
            herg_ic50_um=herg_ic50_um,
            mw_g_mol=mw,
        )

        d = profile.to_dict()
        ss = d.get("steady_state", {})

        return json.dumps(
            {
                "smiles": smiles,
                "regimen_tested": {
                    "dose_mg": dose_mg,
                    "interval_hr": interval_hr,
                    "regimen_frequency": "QD (Once Daily)"
                    if interval_hr == 24
                    else ("BID (Twice Daily)" if interval_hr == 12 else f"Q{int(interval_hr)}H"),
                    "route": route,
                },
                "underlying_pk_parameters": {
                    "vdss_l_kg": round(vdss, 3),
                    "half_life_hr": round(t_half, 2),
                    "molecular_weight": round(mw, 2),
                },
                "steady_state_outcome": {
                    "accumulation_ratio_rac": ss.get("accumulation_ratio_rac"),
                    "c_ss_max_mg_l": ss.get("c_ss_max_mg_l"),
                    "c_ss_min_mg_l": ss.get("c_ss_min_mg_l"),
                    "c_ss_avg_mg_l": ss.get("c_ss_avg_mg_l"),
                    "peak_to_trough_fluctuation_pct": ss.get("peak_trough_fluctuation_pct"),
                    "days_to_reach_steady_state": round(
                        ss.get("time_to_90pct_ss_hr", 24.0) / 24.0, 1
                    ),
                },
                "safety_assessment": {
                    "herg_margin_at_css_max": ss.get("herg_margin_ss_max"),
                    "herg_safe_margin_exceeded": ss.get("is_safe_against_herg"),
                    "clinical_comment": "Adequate safety margin (>30x)"
                    if ss.get("is_safe_against_herg")
                    else "Warning: Peak steady-state concentration nears hERG liability threshold",
                },
            },
            indent=2,
        )

    @server.tool(
        name="evaluate_drug_interactions",
        description="Evaluate CYP drug-drug interaction (DDI) risk and victim drug AUC fold changes",
    )
    def evaluate_drug_interactions(
        smiles: str,
        dose_mg: float = 100.0,
        c_max_mg_l: Optional[float] = None,
        unbound_fraction_fu: Optional[float] = None,
    ) -> str:
        """Evaluate mechanism-based clinical drug-drug interaction (DDI) risks across CYP enzymes.

        Args:
            smiles: Small molecule SMILES string.
            dose_mg: Clinical oral dose (mg).
            c_max_mg_l: Optional maximum plasma concentration. If omitted, calculated from PK.
            unbound_fraction_fu: Optional plasma fraction unbound (0.01~1.0). If omitted, estimated from PPBR.

        Returns:
            JSON string containing CYP 5-isoform AUC fold changes on standard probe drugs (Midazolam, Warfarin, etc.)
            and clinical contraindications.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return json.dumps({"error": f"Invalid SMILES string: '{smiles}'"}, indent=2)

        mw = float(Descriptors.MolWt(mol))
        c_max = c_max_mg_l or (dose_mg / 70.0 / 1.2)  # rough ~1.2 mg/L
        fu = unbound_fraction_fu or 0.10

        # Query ADMET CYP predictions
        cyp_probs = {
            "CYP3A4": 0.35,
            "CYP2C9": 0.20,
            "CYP2C19": 0.15,
            "CYP2D6": 0.25,
            "CYP1A2": 0.10,
        }
        try:
            from tdc_studio.serving.unified_pipeline import UnifiedADMETPipeline

            pipe = UnifiedADMETPipeline.from_exported_directory("models/export")
            admet_res = pipe.predict_full(smiles).to_dict()
            ind = admet_res.get("indicators", {})
            for cyp_k in ["cyp3a4", "cyp2c9", "cyp2c19", "cyp2d6", "cyp1a2"]:
                if f"{cyp_k}_inhibition" in ind:
                    cyp_probs[cyp_k.upper()] = float(
                        ind[f"{cyp_k}_inhibition"].get("raw_value", 0.2)
                    )
            if "plasma_protein_binding" in ind and unbound_fraction_fu is None:
                ppbr_pct = float(ind["plasma_protein_binding"].get("raw_value", 90.0))
                fu = max(0.01, (100.0 - ppbr_pct) / 100.0)
        except Exception:
            pass

        simulator = DDISimulator()
        profile = simulator.evaluate_ddi(
            smiles=smiles,
            cyp_inhibition_probs=cyp_probs,
            c_max_mg_l=c_max,
            unbound_fraction_fu=fu,
            dose_mg=dose_mg,
            mw_g_mol=mw,
        )

        return json.dumps(profile.to_dict(), indent=2)
