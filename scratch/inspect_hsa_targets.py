"""Inspect curated ChEMBL HSA values and define continuous target."""

import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("data/external/chembl_hsa_curated.csv")
    print("Total rows:", len(df))
    print("Unique SMILES:", df["Canon_SMILES"].nunique())

    for at in ["PB", "PPB", "Kd", "Ka", "Log K'", "Activity", "HSAI", "Ratio"]:
        sub = df[df["activity_type"] == at]
        if len(sub) > 0:
            units = list(sub["units"].dropna().unique())
            print(f"Activity Type: {at:<12} | Count: {len(sub):<5} | Median: {sub['value'].median():.2f} | Units: {units}")


if __name__ == "__main__":
    main()
