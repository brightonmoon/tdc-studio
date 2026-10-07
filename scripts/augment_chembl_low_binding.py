"""Acquire and curate low-binding PPB/HSA data from ChEMBL to relieve Hurdle low-binding head sample starvation.

Fetches ChEMBL PPB records using the official chembl_api utility,
strictly excludes any TDC PPBR_AZ validation/test set compounds,
aggregates duplicate SMILES, and generates high-quality low-binding augmented datasets.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
from rdkit import Chem
from tdc.single_pred import ADME

CHEMBL_SCRIPT_PATH = Path(
    r"C:\Users\xps\.gemini\config\plugins\science\skills\chembl_database\scripts\chembl_api.py"
)


def canonicalize(smiles: str) -> str | None:
    """Canonicalize SMILES string without stereochemistry noise."""
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None
    except Exception:
        return None


def fetch_chembl_batches(
    cache_dir: Path, num_batches: int = 15, batch_size: int = 1000
) -> list[Path]:
    """Fetch PPB activity records in batches using chembl_api.py."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    batch_files = []

    for i in range(num_batches):
        offset = i * batch_size
        out_file = cache_dir / f"chembl_ppb_batch_{offset}_{batch_size}.json"
        batch_files.append(out_file)

        if out_file.exists() and out_file.stat().st_size > 1000:
            print(f"Batch at offset {offset} already cached: {out_file.name}")
            continue

        print(f"Fetching ChEMBL PPB batch: offset={offset}, limit={batch_size}...")
        cmd = [
            "uv",
            "run",
            str(CHEMBL_SCRIPT_PATH),
            "activity",
            "--filter",
            "standard_type=PPB",
            "--limit",
            str(batch_size),
            "--offset",
            str(offset),
            "--output",
            str(out_file),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Warning: Failed to fetch offset {offset}: {res.stderr}")
        else:
            print(f"Saved batch to {out_file.name} ({out_file.stat().st_size} bytes)")

    return batch_files


def main():
    print("=" * 90)
    print("=== [Task C2-NEXT-1] ChEMBL PPB LOW-BINDING DATA AUGMENTATION PIPELINE ===")
    print("=" * 90)

    # 1. Load TDC PPBR_AZ to determine test/validation split compounds
    print("[1/5] Loading TDC PPBR_AZ benchmark dataset to prevent data leakage...")
    tdc_data = ADME(name="PPBR_AZ")
    splits = tdc_data.get_split(method="random", seed=42)
    train_df, val_df, test_df = splits["train"], splits["valid"], splits["test"]

    test_smiles = set()
    for s in test_df["Drug"]:
        cs = canonicalize(s)
        if cs:
            test_smiles.add(cs)

    val_smiles = set()
    for s in val_df["Drug"]:
        cs = canonicalize(s)
        if cs:
            val_smiles.add(cs)

    train_smiles = set()
    for s in train_df["Drug"]:
        cs = canonicalize(s)
        if cs:
            train_smiles.add(cs)

    print(
        f"TDC Benchmark Set: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )
    print(
        f"Protected Split SMILES: Test={len(test_smiles)}, Val={len(val_smiles)}, Train={len(train_smiles)}"
    )

    tdc_train_low = (train_df["Y"] < 70.0).sum()
    print(
        f"Original TDC Train Low-Binding (<70%): {tdc_train_low} / {len(train_df)} ({tdc_train_low / len(train_df) * 100:.1f}%)"
    )

    # 2. Fetch batches from ChEMBL (up to 15,000 records)
    print("\n[2/5] Fetching ChEMBL PPB records via ChEMBL API...")
    cache_dir = Path("data/cache/chembl_raw")
    batch_files = fetch_chembl_batches(cache_dir, num_batches=15, batch_size=1000)

    # 3. Parse and standardize records
    print("\n[3/5] Parsing and standardizing bioactivity records...")
    raw_records = []
    for bf in batch_files:
        if not bf.exists():
            continue
        try:
            with open(bf, "r", encoding="utf-8") as f:
                data = json.load(f)
            acts = data.get("activities", [])
            for a in acts:
                smi = a.get("canonical_smiles")
                val_str = a.get("standard_value")
                units = a.get("standard_units")
                if not smi or val_str is None:
                    continue
                if units != "%":
                    continue
                try:
                    val = float(val_str)
                except (ValueError, TypeError):
                    continue

                if not (0.0 <= val <= 100.0):
                    continue

                cs = canonicalize(smi)
                if not cs:
                    continue

                raw_records.append(
                    {
                        "Canon_SMILES": cs,
                        "raw_smiles": smi,
                        "standard_value": val,
                        "molecule_chembl_id": a.get("molecule_chembl_id"),
                        "assay_chembl_id": a.get("assay_chembl_id"),
                        "target_organism": a.get("target_organism"),
                    }
                )
        except Exception as e:
            print(f"Error parsing {bf.name}: {e}")

    df_raw = pd.DataFrame(raw_records)
    print(f"Total valid percentage records parsed: {len(df_raw)}")
    print(f"Unique molecules in raw batch: {df_raw['Canon_SMILES'].nunique()}")

    # 4. Strict leakage prevention & deduplication
    print("\n[4/5] Enforcing 0% leakage against TDC Test/Val splits...")
    leak_test = df_raw["Canon_SMILES"].isin(test_smiles).sum()
    leak_val = df_raw["Canon_SMILES"].isin(val_smiles).sum()
    print(f"Excluded records matching TDC Test: {leak_test}")
    print(f"Excluded records matching TDC Val: {leak_val}")

    clean_df = df_raw[
        ~df_raw["Canon_SMILES"].isin(test_smiles | val_smiles)
    ].copy()

    # Deduplicate by Canon_SMILES using median value
    dedup_df = (
        clean_df.groupby("Canon_SMILES")["standard_value"].median().reset_index()
    )
    dedup_df.columns = ["Drug", "Y"]

    print(
        f"Unique clean novel ChEMBL compounds (0% Test/Val leakage): {len(dedup_df)}"
    )
    print("Value distribution in clean ChEMBL:")
    n_under_50 = (dedup_df["Y"] < 50.0).sum()
    n_under_70 = (dedup_df["Y"] < 70.0).sum()
    n_over_90 = (dedup_df["Y"] >= 90.0).sum()
    print(
        f"  < 50%: {n_under_50} ({n_under_50 / len(dedup_df) * 100:.1f}%)"
    )
    print(
        f"  < 70%: {n_under_70} ({n_under_70 / len(dedup_df) * 100:.1f}%)"
    )
    print(
        f"  >= 90%: {n_over_90} ({n_over_90 / len(dedup_df) * 100:.1f}%)"
    )

    # Filter for low-binding augmented set (< 70%)
    low_binding_df = dedup_df[dedup_df["Y"] < 70.0].copy()
    print(
        f"\n★ Dedicated Low-Binding (<70%) Augmented Compounds: {len(low_binding_df)} compounds!"
    )

    # Also incorporate existing chembl_hsa_processed.csv low-binding if available
    hsa_proc_path = Path("data/external/chembl_hsa_processed.csv")
    if hsa_proc_path.exists():
        hsa_df = pd.read_csv(hsa_proc_path)
        hsa_df["Canon"] = hsa_df["Drug"].apply(canonicalize)
        hsa_clean = hsa_df[
            ~hsa_df["Canon"].isin(test_smiles | val_smiles)
        ].dropna(subset=["Canon"])
        hsa_low = hsa_clean[hsa_clean["Y"] < 70.0][["Canon", "Y"]].rename(
            columns={"Canon": "Drug"}
        )
        combined_low = (
            pd.concat([low_binding_df, hsa_low], ignore_index=True)
            .groupby("Drug")["Y"]
            .median()
            .reset_index()
        )
        low_binding_df = combined_low
        print(
            f"Merged with existing curated HSA: Total Low-Binding (<70%): {len(low_binding_df)} compounds."
        )

    # 5. Export Augmented Datasets
    print("\n[5/5] Exporting augmented datasets...")
    out_dir = Path("data/external")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export pure low-binding augmentation
    low_path = out_dir / "chembl_low_binding_augmented.csv"
    low_binding_df.to_csv(low_path, index=False)
    print(f"Saved: {low_path.resolve()} ({len(low_binding_df)} rows)")

    # Combined augmented training set: TDC Train + Low-binding ChEMBL
    tdc_clean_train = train_df[["Drug", "Y"]].copy()
    tdc_clean_train["Canon"] = tdc_clean_train["Drug"].apply(canonicalize)

    # Add ChEMBL low binding that isn't already in TDC train
    novel_chembl_low = low_binding_df[
        ~low_binding_df["Drug"].isin(set(tdc_clean_train["Canon"]))
    ]
    augmented_train = pd.concat(
        [
            tdc_clean_train[["Canon", "Y"]].rename(columns={"Canon": "Drug"}),
            novel_chembl_low,
        ],
        ignore_index=True,
    )

    aug_path = out_dir / "ppbr_augmented_train_combined.csv"
    augmented_train.to_csv(aug_path, index=False)
    print(f"Saved: {aug_path.resolve()} ({len(augmented_train)} rows)")

    aug_low_count = (augmented_train["Y"] < 70.0).sum()
    print("\n" + "=" * 90)
    print("★ DATA AUGMENTATION SUMMARY ★")
    print(f"Original TDC Train Low-Binding (<70%): {tdc_train_low}")
    print(f"New Augmented Train Low-Binding (<70%): {aug_low_count} (★ +{aug_low_count - tdc_train_low} 확충!)")
    print(f"Total Augmented Train Size: {len(augmented_train)} compounds")
    print("=" * 90)


if __name__ == "__main__":
    main()
