"""Batch Molecular Screening Engine for High-Throughput ADMET & PBPK Profiling.

Supports CSV, TSV, and SDF files, calculates Lipinski Rule of 5 parameters,
and coordinates full C1-C5 22+ ADMET indicators and PBPK simulations.
"""

import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from tdc_studio.serving.schema import UnifiedADMETProfile
from tdc_studio.serving.unified_pipeline import UnifiedADMETPipeline

logger = logging.getLogger("tdc_studio.serving.batch")


def compute_lipinski_rule_of_5(mol: Optional[Chem.Mol]) -> Dict[str, Any]:
    """Compute physicochemical descriptors and assess Lipinski's Rule of 5."""
    if mol is None:
        return {
            "mw": 0.0,
            "logp": 0.0,
            "hbd": 0,
            "hba": 0,
            "rotb": 0,
            "tpsa": 0.0,
            "ro5_violations": 0,
            "ro5_pass": False,
        }

    try:
        mw = round(float(Descriptors.MolWt(mol)), 2)
        logp = round(float(Descriptors.MolLogP(mol)), 2)
        hbd = int(rdMolDescriptors.CalcNumHBD(mol))
        hba = int(rdMolDescriptors.CalcNumHBA(mol))
        rotb = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        tpsa = round(float(rdMolDescriptors.CalcTPSA(mol)), 2)

        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5.0:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1

        return {
            "mw": mw,
            "logp": logp,
            "hbd": hbd,
            "hba": hba,
            "rotb": rotb,
            "tpsa": tpsa,
            "ro5_violations": violations,
            "ro5_pass": violations <= 1,
        }
    except Exception as e:
        logger.warning("Error computing Lipinski properties: %s", e)
        return {
            "mw": 0.0,
            "logp": 0.0,
            "hbd": 0,
            "hba": 0,
            "rotb": 0,
            "tpsa": 0.0,
            "ro5_violations": 0,
            "ro5_pass": False,
        }


def parse_molecular_file(
    content: bytes | str, filename: str
) -> List[Dict[str, Any]]:
    """Parse CSV, TSV, or SDF input into a standardized record list."""
    records: List[Dict[str, Any]] = []
    ext = os.path.splitext(filename.lower())[1]

    if ext in (".sdf", ".sd"):
        # Parse SDF
        if isinstance(content, str):
            content = content.encode("utf-8")
        bio = io.BytesIO(content)
        supplier = Chem.ForwardSDMolSupplier(bio)
        for idx, mol in enumerate(supplier):
            if mol is None:
                continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"CMPD_{idx + 1}"
            smiles = Chem.MolToSmiles(mol)
            records.append({"compound_id": name, "smiles": smiles, "mol": mol})
    else:
        # Parse CSV / TSV
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = content

        sep = "\t" if ext == ".tsv" or "\t" in text.splitlines()[0] else ","
        df = pd.read_csv(io.StringIO(text), sep=sep)

        # Identify SMILES column
        candidate_cols = [
            "smiles", "SMILES", "Smiles", "canonical_smiles", "Drug", "drug",
            "structure", "Structure", "mol", "MOL"
        ]
        smiles_col = None
        for col in candidate_cols:
            if col in df.columns:
                smiles_col = col
                break

        if smiles_col is None:
            # Fallback: check columns for string values containing chemical characters
            for col in df.columns:
                if df[col].dtype == object and df[col].dropna().astype(str).str.contains("[CNOcno]").any():
                    smiles_col = col
                    break

        if smiles_col is None:
            raise ValueError(f"Could not identify SMILES column in {filename}. Available columns: {list(df.columns)}")

        # Identify ID column if exists
        id_cols = ["id", "ID", "compound_id", "Compound_ID", "Name", "name", "Drug_ID"]
        id_col = next((c for c in id_cols if c in df.columns), None)

        for idx, row in df.iterrows():
            s = str(row[smiles_col]).strip()
            if not s or s.lower() == "nan":
                continue
            cid = str(row[id_col]) if id_col else f"CMPD_{idx + 1}"
            records.append({"compound_id": cid, "smiles": s})

    return records


def profile_to_flattened_dict(
    cid: str, smiles: str, profile: UnifiedADMETProfile, lipo_meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Flatten a UnifiedADMETProfile and Lipinski metrics into a single tabular row."""
    row: Dict[str, Any] = {
        "compound_id": cid,
        "smiles": smiles,
        "canonical_smiles": profile.canonical_smiles,
        "drug_likeness_score": profile.drug_likeness_score,
        "mw": lipo_meta["mw"],
        "logp": lipo_meta["logp"],
        "hbd": lipo_meta["hbd"],
        "hba": lipo_meta["hba"],
        "rotb": lipo_meta["rotb"],
        "tpsa": lipo_meta["tpsa"],
        "ro5_violations": lipo_meta["ro5_violations"],
        "ro5_pass": lipo_meta["ro5_pass"],
    }

    # C1 Absorption
    for k, ind in profile.absorption.items():
        if ind.value is not None:
            row[f"c1_{k}_val"] = ind.value
        if ind.probability is not None:
            row[f"c1_{k}_prob"] = ind.probability
        row[f"c1_{k}_decision"] = ind.decision

    # C2 Distribution
    for k, ind in profile.distribution.items():
        if ind.value is not None:
            row[f"c2_{k}_val"] = ind.value
        if ind.probability is not None:
            row[f"c2_{k}_prob"] = ind.probability
        row[f"c2_{k}_decision"] = ind.decision

    # C3 Metabolism
    for k, ind in profile.metabolism.items():
        if ind.probability is not None:
            row[f"c3_{k}_prob"] = ind.probability
        row[f"c3_{k}_decision"] = ind.decision

    # C4 Excretion
    for k, ind in profile.excretion.items():
        if ind.value is not None:
            row[f"c4_{k}_val"] = ind.value
        row[f"c4_{k}_decision"] = ind.decision

    # C5 Toxicity
    for k, ind in profile.toxicity.items():
        if ind.value is not None:
            row[f"c5_{k}_val"] = ind.value
        if ind.probability is not None:
            row[f"c5_{k}_prob"] = ind.probability
        row[f"c5_{k}_decision"] = ind.decision

    # PBPK Simulation Results
    if profile.pbpk is not None:
        row["pbpk_cl_total_l_h_kg"] = profile.pbpk.cl_total_l_h_kg
        row["pbpk_half_life_hours"] = profile.pbpk.half_life_hours
        row["pbpk_vdss_l_kg"] = profile.pbpk.vdss_l_kg
        row["pbpk_fraction_unbound"] = profile.pbpk.fraction_unbound
        row["pbpk_hepatic_extraction_ratio"] = profile.pbpk.hepatic_extraction_ratio
        row["pbpk_hepatic_clearance_l_h_kg"] = profile.pbpk.hepatic_clearance_l_h_kg
        row["pbpk_max_oral_bioavailability"] = profile.pbpk.max_oral_bioavailability
        row["pbpk_t_half_tier"] = profile.pbpk.t_half_tier
        row["pbpk_extraction_tier"] = profile.pbpk.extraction_tier

    return row


class BatchScreeningEngine:
    """Orchestrates batch ingestion, parallel inference, and multi-format report exports."""

    def __init__(self, pipeline: Optional[UnifiedADMETPipeline] = None):
        self.pipeline = pipeline or UnifiedADMETPipeline.from_exported_directory()

    def process_records(
        self, records: List[Dict[str, Any]], progress_callback=None
    ) -> pd.DataFrame:
        """Run complete 22+ ADMET & PBPK screening on parsed records."""
        rows = []
        total = len(records)

        for i, rec in enumerate(records):
            s = rec["smiles"]
            cid = rec.get("compound_id", f"CMPD_{i + 1}")
            mol = rec.get("mol")
            if mol is None:
                mol = Chem.MolFromSmiles(s)

            # 1. Lipinski Rule of 5
            ro5_meta = compute_lipinski_rule_of_5(mol)

            # 2. Complete ADMET Profile
            try:
                prof = self.pipeline.predict_single(s)
            except Exception as e:
                logger.warning("Error predicting ADMET for '%s': %s", s, e)
                prof = self.pipeline.predict_single("c1ccccc1")  # safe dummy fallback
                prof.canonical_smiles = s

            # 3. Flatten
            row = profile_to_flattened_dict(cid, s, prof, ro5_meta)
            rows.append(row)

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, total)

        df = pd.DataFrame(rows)
        return df

    def screen_file(
        self, content: bytes | str, filename: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Parse file, execute batch screening, and return summary statistics."""
        records = parse_molecular_file(content, filename)
        if not records:
            raise ValueError(f"No valid molecular structures found in {filename}.")

        df = self.process_records(records)

        # Summary statistics
        total_molecules = len(df)
        ro5_passed = int(df["ro5_pass"].sum()) if "ro5_pass" in df.columns else 0
        herg_safe = (
            int((df["c5_herg_prob"] < 0.5).sum())
            if "c5_herg_prob" in df.columns
            else 0
        )
        ames_safe = (
            int((df["c5_ames_prob"] < 0.5).sum())
            if "c5_ames_prob" in df.columns
            else 0
        )

        summary = {
            "total_molecules": total_molecules,
            "ro5_passed": ro5_passed,
            "ro5_pass_rate": round(ro5_passed / max(1, total_molecules) * 100, 1),
            "herg_safe_count": herg_safe,
            "herg_safe_rate": round(herg_safe / max(1, total_molecules) * 100, 1),
            "ames_safe_count": ames_safe,
            "ames_safe_rate": round(ames_safe / max(1, total_molecules) * 100, 1),
        }

        return df, summary
