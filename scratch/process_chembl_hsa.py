"""Process and curate ChEMBL Human Serum Albumin (CHEMBL3253) bioactivity records."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from tdc.single_pred import ADME


def canonicalize(s):
    if not s or not isinstance(s, str):
        return None
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None


def main():
    print("Loading ChEMBL HSA JSON files...")
    all_acts = []
    for p in ["scratch/hsa_acts_p1.json", "scratch/hsa_acts_p2.json", "scratch/hsa_acts_p3.json"]:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_acts.extend(data.get("activities", []))

    print(f"Total raw activity records: {len(all_acts)}")

    rows = []
    for a in all_acts:
        smiles = a.get("canonical_smiles")
        val_str = a.get("value")
        act_type = a.get("type")
        units = a.get("units")

        if not smiles or val_str is None:
            continue
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            continue

        c_smi = canonicalize(smiles)
        if c_smi is None:
            continue

        rows.append({
            "Canon_SMILES": c_smi,
            "chembl_id": a.get("molecule_chembl_id"),
            "activity_type": act_type,
            "value": val,
            "units": units,
            "assay_chembl_id": a.get("assay_chembl_id"),
            "assay_description": a.get("assay_description"),
        })

    df = pd.DataFrame(rows)
    print(f"Valid numerical records with SMILES: {len(df)}")
    print(f"Unique canonical SMILES in ChEMBL HSA: {df['Canon_SMILES'].nunique()}")

    print("\n--- Activity Type Distribution ---")
    type_counts = df["activity_type"].value_counts()
    print(type_counts.head(10))

    # Cross-reference with TDC PPBR_AZ
    print("\nLoading TDC PPBR_AZ to audit split leakage...")
    tdc_data = ADME(name="PPBR_AZ")
    tdc_splits = tdc_data.get_split(method="random", seed=42)
    test_smiles = set([canonicalize(s) for s in tdc_splits["test"]["Drug"].tolist() if canonicalize(s)])
    train_smiles = set([canonicalize(s) for s in tdc_splits["train"]["Drug"].tolist() if canonicalize(s)])
    val_smiles = set([canonicalize(s) for s in tdc_splits["valid"]["Drug"].tolist() if canonicalize(s)])

    chembl_smiles = set(df["Canon_SMILES"])
    overlap_test = chembl_smiles & test_smiles
    overlap_train = chembl_smiles & train_smiles
    overlap_val = chembl_smiles & val_smiles
    novel_compounds = chembl_smiles - (test_smiles | val_smiles | train_smiles)

    print(f"ChEMBL overlap with TDC Test (MUST EXCLUDE): {len(overlap_test)}")
    print(f"ChEMBL overlap with TDC Train: {len(overlap_train)}")
    print(f"ChEMBL overlap with TDC Val: {len(overlap_val)}")
    print(f"★ NOVEL UNSEEN COMPOUNDS in ChEMBL HSA: {len(novel_compounds)}")

    # Clean ChEMBL dataset: strictly exclude any TDC test compounds
    clean_df = df[~df["Canon_SMILES"].isin(test_smiles | val_smiles)].copy()
    print(f"\nFinal clean ChEMBL auxiliary dataset size (0% test leakage): {len(clean_df)} records, {clean_df['Canon_SMILES'].nunique()} unique molecules.")

    # Save to data directory
    out_dir = Path("data/external")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chembl_hsa_curated.csv"
    clean_df.to_csv(out_path, index=False)
    print(f"Saved curated ChEMBL HSA dataset to: {out_path}")


if __name__ == "__main__":
    main()
