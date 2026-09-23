"""Test Standalone GBDT & Hybrid Blending with Biophysical pH 7.4 & Sudlow Descriptors on 8:1:1 Random Split."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Crippen
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tdc.single_pred import ADME

from scratch.test_biophysical_hypothesis import compute_biophysical_descriptors


def extract_features_with_biophysics(smiles_list, labels=None, n_bits=1024):
    features = []
    y_logit = []
    y_real = []

    dummy_mol = Chem.MolFromSmiles("C")
    n_desc = len(Descriptors.CalcMolDescriptors(dummy_mol)) if dummy_mol is not None else 210

    for idx, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s) if (s and isinstance(s, str)) else None
        if mol is None:
            features.append([0.0] * (n_bits + n_desc + 14))
        else:
            # 1. Morgan Fingerprint (1024-bit)
            fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))

            # 2. RDKit 2D Physico-chemical Descriptors
            desc_dict = Descriptors.CalcMolDescriptors(mol)
            desc_vals = []
            for v in desc_dict.values():
                if v is None or np.isnan(v) or np.isinf(v):
                    desc_vals.append(0.0)
                else:
                    desc_vals.append(float(np.clip(v, -100.0, 100.0)))

            # 3. Biophysical pH 7.4 & HSA/AAG Motifs (14 features)
            bio_dict = compute_biophysical_descriptors(s)
            bio_vals = list(bio_dict.values())

            features.append(fp + desc_vals + bio_vals)

        if labels is not None and idx < len(labels):
            raw_y = float(labels[idx])
            y_real.append(raw_y)
            fb = np.clip(raw_y / 100.0, 1e-4, 1.0 - 1e-4)
            y_logit.append(np.log(fb / (1.0 - fb)))

    X = np.array(features, dtype=np.float32)
    y_l = np.array(y_logit, dtype=np.float32) if labels is not None else None
    y_r = np.array(y_real, dtype=np.float32) if labels is not None else None
    return X, y_l, y_r


def main():
    print("Loading TDC PPBR_AZ dataset with 8:1:1 Random Split...")
    data = ADME(name="PPBR_AZ")
    splits = data.get_split(method="random", seed=42)
    train_df = splits["train"]
    val_df = splits["valid"]
    test_df = splits["test"]

    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    X_train, y_train_logit, y_train_real = extract_features_with_biophysics(
        train_df["Drug"].tolist(), train_df["Y"].tolist()
    )
    X_val, y_val_logit, y_val_real = extract_features_with_biophysics(
        val_df["Drug"].tolist(), val_df["Y"].tolist()
    )
    X_test, y_test_logit, y_test_real = extract_features_with_biophysics(
        test_df["Drug"].tolist(), test_df["Y"].tolist()
    )

    print(f"Feature Dimension: {X_train.shape[1]} (1024 FPs + {X_train.shape[1] - 1024 - 14} RDKit Descs + 14 Biophysical Motifs)")

    # 1. Baseline GBDT (without biophysical features)
    gbdt_base = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, l2_regularization=2.0, min_samples_leaf=15, random_state=42)
    gbdt_base.fit(X_train[:, :-14], y_train_logit)
    z_base_test = gbdt_base.predict(X_test[:, :-14])
    z_base_val = gbdt_base.predict(X_val[:, :-14])

    # Calibrate baseline
    def obj_cal(params, z, y):
        a, b = params
        p = 100.0 / (1.0 + np.exp(-np.clip(a * z + b, -40.0, 40.0)))
        return float(np.mean((p - y) ** 2))

    res_base = minimize(obj_cal, [1.0, 0.0], args=(z_base_val, y_val_real), bounds=[(0.1, 3.0), (-2.0, 2.0)], method="L-BFGS-B")
    p_base_test = 100.0 / (1.0 + np.exp(-np.clip(res_base.x[0] * z_base_test + res_base.x[1], -40.0, 40.0)))
    base_r2 = r2_score(y_test_real, p_base_test)
    base_mae = mean_absolute_error(y_test_real, p_base_test)
    base_pr, _ = pearsonr(p_base_test, y_test_real)

    print(f"\n--- Baseline GBDT (FPs + RDKit) ---")
    print(f"Test R² = {base_r2:.4f} | Test MAE = {base_mae:.2f}% | Pearson r = {base_pr:.4f}")

    # 2. Biophysical GBDT (FPs + RDKit + pH 7.4 Ionization / Sudlow)
    gbdt_bio = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, l2_regularization=2.0, min_samples_leaf=15, random_state=42)
    gbdt_bio.fit(X_train, y_train_logit)
    z_bio_test = gbdt_bio.predict(X_test)
    z_bio_val = gbdt_bio.predict(X_val)

    res_bio = minimize(obj_cal, [1.0, 0.0], args=(z_bio_val, y_val_real), bounds=[(0.1, 3.0), (-2.0, 2.0)], method="L-BFGS-B")
    p_bio_test = 100.0 / (1.0 + np.exp(-np.clip(res_bio.x[0] * z_bio_test + res_bio.x[1], -40.0, 40.0)))
    bio_r2 = r2_score(y_test_real, p_bio_test)
    bio_mae = mean_absolute_error(y_test_real, p_bio_test)
    bio_pr, _ = pearsonr(p_bio_test, y_test_real)
    bio_sp, _ = spearmanr(p_bio_test, y_test_real)

    print(f"\n--- Biophysical GBDT (+ pH 7.4 Ionization & Sudlow) ---")
    print(f"Test R² = {bio_r2:.4f} (ΔR² = {bio_r2 - base_r2:+.4f}) | Test MAE = {bio_mae:.2f}% | Pearson r = {bio_pr:.4f} | Spearman ρ = {bio_sp:.4f}")


if __name__ == "__main__":
    main()
