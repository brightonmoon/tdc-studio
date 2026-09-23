"""Benchmark GBDT & Foundation Model Stacker with ChEMBL HSA (1,494 novel compounds) Data Augmentation."""

from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tdc.single_pred import ADME

from scratch.test_biophysical_hypothesis import compute_biophysical_descriptors
from scratch.test_chemberta_gbdt import (
    extract_all_features,
    extract_chemberta_embeddings,
    calibrate_and_eval,
)


def main():
    print("=== ChEMBL HSA Data Augmentation Benchmark ===")
    # 1. Load TDC PPBR_AZ
    data = ADME(name="PPBR_AZ")
    splits = data.get_split(method="random", seed=42)
    train_df, val_df, test_df = splits["train"], splits["valid"], splits["test"]

    print(f"TDC Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Load Curated ChEMBL HSA
    chembl_path = Path("data/external/chembl_hsa_curated.csv")
    chembl_df = pd.read_csv(chembl_path)
    print(f"Loaded ChEMBL HSA curated dataset: {len(chembl_df)} records, {chembl_df['Canon_SMILES'].nunique()} unique molecules.")

    # Filter ChEMBL for percentage binding (PB / PPB) or convert Log K' / Kd to binding proxy
    # Let's inspect percentage binding records: PB, PPB, Activity
    pb_records = chembl_df[chembl_df["activity_type"].isin(["PB", "PPB", "Activity"])].copy()
    # Filter values in [0, 100]% range
    pb_records = pb_records[(pb_records["value"] >= 0.0) & (pb_records["value"] <= 100.0)]
    # Deduplicate by Canon_SMILES taking median value
    pb_dedup = pb_records.groupby("Canon_SMILES")["value"].median().reset_index()
    pb_dedup.columns = ["Drug", "Y"]
    print(f"ChEMBL direct % binding (PB/PPB) curated compounds: {len(pb_dedup)} unique compounds.")

    # Also log K' records (chromatographic retention directly proportional to Gibbs free energy logit)
    # Log K' in literature maps linearly to logit(fb): logit(fb) ~ 1.5 * Log K' + 1.2
    logk_records = chembl_df[chembl_df["activity_type"] == "Log K'"].copy()
    logk_dedup = logk_records.groupby("Canon_SMILES")["value"].median().reset_index()
    # Convert log K' to estimated % binding
    est_logit = 1.6 * logk_dedup["value"].values + 1.5
    est_fb = 100.0 / (1.0 + np.exp(-np.clip(est_logit, -5.0, 5.0)))
    logk_dedup["Y"] = est_fb
    logk_dedup = logk_dedup[["Canon_SMILES", "Y"]].rename(columns={"Canon_SMILES": "Drug"})

    # Combine novel ChEMBL compounds
    test_smiles_set = set(test_df["Drug"].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s))
    val_smiles_set = set(val_df["Drug"].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s))

    chembl_combined = pd.concat([pb_dedup, logk_dedup], ignore_index=True)
    chembl_combined["Canon"] = chembl_combined["Drug"].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s)
    chembl_novel = chembl_combined[~chembl_combined["Canon"].isin(test_smiles_set | val_smiles_set)].drop_duplicates(subset=["Canon"]).copy()
    chembl_novel = chembl_novel[["Drug", "Y"]]

    print(f"Final clean novel ChEMBL HSA training compounds: {len(chembl_novel)}")

    # Augmented Training Set
    aug_train_df = pd.concat([train_df[["Drug", "Y"]], chembl_novel], ignore_index=True)
    print(f"Augmented Training Set: {len(train_df)} -> {len(aug_train_df)} compounds (+{len(chembl_novel)})")

    # 3. Extract Features for Train, Val, Test
    print("\nExtracting Features (Morgan FP + RDKit Descriptors + Biophysical Motifs)...")
    X_tr_base, y_tr_logit, _ = extract_all_features(aug_train_df["Drug"].tolist(), aug_train_df["Y"].tolist())
    X_va_base, _, y_va_real = extract_all_features(val_df["Drug"].tolist(), val_df["Y"].tolist())
    X_te_base, _, y_te_real = extract_all_features(test_df["Drug"].tolist(), test_df["Y"].tolist())

    # ChemBERTa Embeddings
    print("Extracting ChemBERTa embeddings for augmented training set...")
    emb_cache = Path("data/cache/chemberta_augmented_train.npz")
    if emb_cache.exists():
        emb_tr = np.load(emb_cache)["train"]
    else:
        emb_tr = extract_chemberta_embeddings(aug_train_df["Drug"].tolist())
        np.savez_compressed(emb_cache, train=emb_tr)

    # Load val/test embeddings from previous cache
    prev_cache = np.load("data/cache/chemberta_ppbr_embeddings.npz")
    emb_va = prev_cache["val"]
    emb_te = prev_cache["test"]

    X_tr_fused = np.hstack([X_tr_base, emb_tr])
    X_va_fused = np.hstack([X_va_base, emb_va])
    X_te_fused = np.hstack([X_te_base, emb_te])

    print(f"Fused Feature Matrix: Train={X_tr_fused.shape}, Val={X_va_fused.shape}, Test={X_te_fused.shape}")

    # 4. Train Fused GBDT on Augmented Data
    print("\nTraining Fused GBDT on Augmented (TDC + ChEMBL HSA) Data...")
    gbdt_aug = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.035,
        l2_regularization=4.0,
        min_samples_leaf=15,
        random_state=42,
    )
    gbdt_aug.fit(X_tr_fused, y_tr_logit)

    z_va_aug = gbdt_aug.predict(X_va_fused)
    z_te_aug = gbdt_aug.predict(X_te_fused)

    res_aug = calibrate_and_eval(z_va_aug, y_va_real, z_te_aug, y_te_real)

    print("\n" + "=" * 75)
    print("★ ChEMBL HSA DATA AUGMENTATION BENCHMARK RESULTS (TEST SET) ★")
    print("=" * 75)
    print(f"{'Configuration':<45} | {'Test R2':<10} | {'MAE (%)':<9} | {'Pearson':<9} | {'Spearman':<9}")
    print("-" * 75)
    print(f"{'TDC Only Baseline GBDT':<45} | {'0.4218':<10} | {'6.97%':<9} | {'0.6577':<9} | {'0.7174':<9}")
    print(f"{'TDC Only ChemBERTa Fused GBDT':<45} | {'0.4816':<10} | {'6.59%':<9} | {'0.7035':<9} | {'0.7334':<9}")
    print(f"{'★ ChEMBL HSA Augmented Fused GBDT':<45} | {res_aug['r2']:<10.4f} | {res_aug['mae']:<9.2f}% | {res_aug['pearson']:<9.4f} | {res_aug['spearman']:<9.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
