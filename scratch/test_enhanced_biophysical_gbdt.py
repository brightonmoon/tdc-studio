"""Test Enhanced Biophysical Features (18 features) and Asymmetric Calibration on GBDT."""

from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tdc.single_pred import ADME

from tdc_studio.models.hybrid.gbdt_blend import compute_biophysical_motifs

_QUAT_N_SMARTS = Chem.MolFromSmarts("[NX4+;!$([NX4+]([O-])=O)]")


def compute_enhanced_biophysical_motifs(mol):
    """Compute 18 physiological pH 7.4 ionization, 3D non-planarity & HSA/AAG binding motifs."""
    base_14 = compute_biophysical_motifs(mol)
    if mol is None:
        return base_14 + [0.0, 0.0, 0.0, 0.0]

    # 15. Quaternary nitrogen (permanent positive charge that strongly repels albumin pI 4.7)
    n_quat_n = float(len(mol.GetSubstructMatches(_QUAT_N_SMARTS)))

    # 16. Polar Surface Area density (TPSA / MolWt)
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    tpsa_mw_ratio = float(tpsa / max(1.0, mw))

    # 17. 3D Non-planarity (Fraction CSP3) - non-planar rings cannot slip into flat Sudlow Site 1
    fsp3 = float(Descriptors.FractionCSP3(mol))

    # 18. Rotatable bond density
    hac = Descriptors.HeavyAtomCount(mol)
    rot_dens = float(Descriptors.NumRotatableBonds(mol) / max(1.0, hac))

    return base_14 + [n_quat_n, tpsa_mw_ratio, fsp3, rot_dens]


def extract_features(smiles_list, labels=None, n_bits=1024):
    features = []
    y_logit = []
    y_real = []

    dummy_mol = Chem.MolFromSmiles("C")
    n_desc = len(Descriptors.CalcMolDescriptors(dummy_mol)) if dummy_mol is not None else 210

    for idx, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s) if (s and isinstance(s, str)) else None
        if mol is None:
            features.append([0.0] * (n_bits + n_desc + 18))
        else:
            fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
            desc_dict = Descriptors.CalcMolDescriptors(mol)
            desc_vals = [0.0 if (v is None or np.isnan(v) or np.isinf(v)) else float(v) for v in desc_dict.values()]
            bio = compute_enhanced_biophysical_motifs(mol)
            features.append(fp + desc_vals + bio)

        if labels is not None:
            val = float(labels[idx])
            y_real.append(val)
            fb = np.clip(val / 100.0, 1e-4, 1.0 - 1e-4)
            y_logit.append(float(np.log(fb / (1.0 - fb))))

    X = np.array(features, dtype=np.float32)
    y_l = np.array(y_logit, dtype=np.float32) if labels is not None else None
    y_r = np.array(y_real, dtype=np.float32) if labels is not None else None
    return X, y_l, y_r


def main():
    print("=" * 80)
    print("=== TESTING ENHANCED BIOPHYSICAL GBDT WITH 18 FEATURES & ASYMMETRIC LOSS ===")
    print("=" * 80)

    data = ADME(name="PPBR_AZ")
    splits = data.get_split(method="random", seed=42)
    train_df, val_df, test_df = splits["train"], splits["valid"], splits["test"]

    print("Extracting enhanced 18-biophysical features...")
    X_train_base, y_train_logit, y_train_real = extract_features(train_df["Drug"].tolist(), train_df["Y"].tolist())
    X_val_base, y_val_logit, y_val_real = extract_features(val_df["Drug"].tolist(), val_df["Y"].tolist())
    X_test_base, y_test_logit, y_test_real = extract_features(test_df["Drug"].tolist(), test_df["Y"].tolist())

    loaded = np.load("data/cache/chemberta_ppbr_embeddings.npz")
    X_train = np.hstack([X_train_base, loaded["train"]])
    X_val = np.hstack([X_val_base, loaded["val"]])
    X_test = np.hstack([X_test_base, loaded["test"]])

    print(f"Feature Dimensions: {X_train.shape}")

    # 1. Unweighted Enhanced GBDT
    gbdt_std = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=15, random_state=42)
    gbdt_std.fit(X_train, y_train_logit)
    z_std_val = gbdt_std.predict(X_val)
    z_std_test = gbdt_std.predict(X_test)

    # 2. Asymmetric Sample Weighted GBDT (emphasize < 70% low-binding compounds)
    weights = np.ones(len(y_train_real))
    weights[y_train_real < 70] = 2.2
    weights[(y_train_real >= 70) & (y_train_real < 85)] = 1.4

    gbdt_asym = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=15, random_state=42)
    gbdt_asym.fit(X_train, y_train_logit, sample_weight=weights)
    z_asym_val = gbdt_asym.predict(X_val)
    z_asym_test = gbdt_asym.predict(X_test)

    # Parametric Calibration on Validation set
    def calibrate(z_val, z_test):
        def obj(params):
            a, b = params
            p = 100.0 / (1.0 + np.exp(-np.clip(a * z_val + b, -40.0, 40.0)))
            return float(np.mean((p - y_val_real) ** 2))

        res = minimize(obj, [1.0, 0.0], bounds=[(0.2, 3.0), (-3.0, 3.0)], method="L-BFGS-B")
        a_opt, b_opt = res.x
        p_test = 100.0 / (1.0 + np.exp(-np.clip(a_opt * z_test + b_opt, -40.0, 40.0)))
        return p_test, a_opt, b_opt

    p_std_calib, a_std, b_std = calibrate(z_std_val, z_std_test)
    p_asym_calib, a_asym, b_asym = calibrate(z_asym_val, z_asym_test)

    def print_metrics(name, p):
        r2 = r2_score(y_test_real, p)
        mae = mean_absolute_error(y_test_real, p)
        rmse = np.sqrt(mean_squared_error(y_test_real, p))
        pr, _ = pearsonr(p, y_test_real)
        sp, _ = spearmanr(p, y_test_real)
        print(f"{name:<45} | R2: {r2:.4f} | MAE: {mae:.2f}% | RMSE: {rmse:.2f}% | Pearson: {pr:.4f} | Spearman: {sp:.4f}")

    print("\n--- Model Benchmark on 323 Test Compounds ---")
    print_metrics("1. Previous Fused GBDT (14 Bio features)", 100.0 / (1.0 + np.exp(-z_std_test)))
    print_metrics("2. Enhanced Biophysical GBDT (18 Bio features)", p_std_calib)
    print_metrics("3. Enhanced Asymmetric GBDT (18 Bio + Asym Loss)", p_asym_calib)


if __name__ == "__main__":
    main()
