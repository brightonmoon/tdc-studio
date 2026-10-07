"""ADMET prediction and XAI explanation tools for MCP server."""

from __future__ import annotations

import json
from typing import List

from mcp.server.mcpserver import MCPServer
from rdkit import Chem


def register_admet_tools(server: MCPServer) -> None:
    """Register ADMET prediction and interpretability tools."""

    @server.tool(
        name="predict_admet_profile",
        description="Predict 22-Task ADMET endpoints, liabilities, and safety radar score",
    )
    def predict_admet_profile(smiles: str) -> str:
        """Execute the comprehensive 22-Task ADMET prediction pipeline on a molecule.

        Args:
            smiles: SMILES string of the candidate molecule.

        Returns:
            JSON string containing 22 ADMET predictions, liabilities, and drug-likeness radar score.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return json.dumps({"error": f"Invalid SMILES string: '{smiles}'"}, indent=2)

        try:
            from tdc_studio.serving.unified_pipeline import UnifiedADMETPipeline

            pipeline = UnifiedADMETPipeline.from_exported_directory("models/export", device="cpu")
            profile = pipeline.predict_full(smiles)
            d = profile.to_dict()

            # Identify liabilities (Red flags)
            liabilities: List[str] = []
            indicators = d.get("indicators", {})

            # Safety liabilities
            if indicators.get("herg_cardiotoxicity", {}).get("risk_tier") == "High Risk":
                liabilities.append("hERG Cardiotoxicity High Risk (>0.70)")
            if indicators.get("ames_mutagenicity", {}).get("risk_tier") == "Mutagenic (AMES+)":
                liabilities.append("AMES Mutagenicity Positive")
            if (
                indicators.get("drug_induced_liver_injury", {}).get("risk_tier")
                == "Hepatotoxicity Risk"
            ):
                liabilities.append("Drug-Induced Liver Injury (DILI) Risk")
            if indicators.get("microsomal_clearance", {}).get("risk_tier") == "High Turnover":
                liabilities.append("High Microsomal Clearance (>50 uL/min/mg)")
            if indicators.get("caco2_permeability", {}).get("risk_tier") == "Low Permeability":
                liabilities.append("Low Caco-2 Permeability (Poor oral absorption)")

            result = {
                "smiles": smiles,
                "drug_likeness_radar_score": d.get("radar_summary", {}).get("score_0_to_100", 75),
                "liabilities_detected": liabilities,
                "has_safety_red_flag": len(liabilities) > 0,
                "admet_indicators": indicators,
                "pbpk_initial_estimates": d.get("pbpk_pk_profile", {}),
            }
            return json.dumps(result, indent=2)

        except Exception as e:
            # Fallback heuristic calculation if model weights not fully exported
            from rdkit.Chem import Descriptors

            logp = float(Descriptors.MolLogP(mol))
            mw = float(Descriptors.MolWt(mol))
            tpsa = float(Descriptors.TPSA(mol))

            fallback = {
                "smiles": smiles,
                "drug_likeness_radar_score": 72.0,
                "liabilities_detected": ["Calculated via physicochemical fallback"],
                "has_safety_red_flag": logp > 4.5,
                "physicochemicals": {
                    "mw": round(mw, 2),
                    "logp": round(logp, 2),
                    "tpsa": round(tpsa, 2),
                },
                "note": f"Pipeline fallback triggered: {str(e)}",
            }
            return json.dumps(fallback, indent=2)

    @server.tool(
        name="explain_toxicity_hotspots",
        description="Identify atom-level toxicophore hotspots and suggest bioisosteres",
    )
    def explain_toxicity_hotspots(smiles: str, liability_task: str = "herg") -> str:
        """Pinpoint specific atom/functional group toxicophores and suggest bioisosteres.

        Args:
            smiles: SMILES string of the molecule with liability.
            liability_task: Target liability to explain (e.g. "herg", "ames", "dili").

        Returns:
            JSON string containing toxicophore hotspots and bioisosteric replacement suggestions.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return json.dumps({"error": f"Invalid SMILES string: '{smiles}'"}, indent=2)

        try:
            from tdc_studio.explainability.bioisostere import BioisostereRecommender

            recommender = BioisostereRecommender()
            suggestions = recommender.suggest(smiles, target_property=liability_task)

            # Format suggestions
            rec_list = []
            for s in suggestions[:5]:
                rec_list.append(
                    {
                        "original_fragment": s.get("original_substructure", ""),
                        "replacement_fragment": s.get("suggested_replacement", ""),
                        "rationale": s.get(
                            "rationale",
                            "Reduces lipophilic/basic liability while preserving binding geometry",
                        ),
                        "mutated_smiles": s.get("mutated_smiles", ""),
                    }
                )

            return json.dumps(
                {
                    "smiles": smiles,
                    "liability_investigated": liability_task,
                    "total_recommendations": len(rec_list),
                    "recommended_bioisosteres": rec_list,
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps(
                {
                    "smiles": smiles,
                    "liability_investigated": liability_task,
                    "recommended_bioisosteres": [
                        {
                            "original_fragment": "Basic Aliphatic Amine",
                            "replacement_fragment": "Morpholine / Fluoroethylamine",
                            "rationale": "Reduces pKa and hERG potassium channel binding",
                        },
                        {
                            "original_fragment": "Carboxylic Acid",
                            "replacement_fragment": "Tetrazole / Sulfonamide",
                            "rationale": "Improves cellular permeability and eliminates acyl glucuronide DILI risk",
                        },
                    ],
                    "note": f"Rule-based fallback: {str(e)}",
                },
                indent=2,
            )
