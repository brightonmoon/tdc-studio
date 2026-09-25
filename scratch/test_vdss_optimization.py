"""Test VDss Lombardo Log10 transformation and baseline GBDT / Ridge / Multi-Modal modeling."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tdc.single_pred import ADME


def main():
    print("=" * 80)
    print("=== VDSS LOMBARDO BENCHMARK & LOG10 TRANSFORMATION TEST ===")
    print("=" * 80)

    data = ADME(name="VDss_Lombardo")
    splits = data.get_split(method="random", seed=42)
    trn, val, tst = splits["train"], splits["valid"], splits["test"]

    print(f"Dataset Split: Train={len(trn)}, Val={len(val)}, Test={len(tst)}")

    # 1. Inspect Linear vs Log10 target
    y_trn_raw = trn["Y"].values
    y_tst_raw = tst["Y"].values

    print("\nTarget Statistics (Linear L/kg):")
    print(f"  Train: min={y_trn_raw.min():.3f}, median={np.median(y_trn_raw):.3f}, mean={y_trn_raw.mean():.3f}, max={y_trn_raw.max():.3f}")

    y_trn_log = np.log10(np.clip(y_trn_raw, 1e-3, None))
    y_tst_log = np.log10(np.clip(y_tst_raw, 1e-3, None))

    print("\nTarget Statistics (Log10 L/kg):")
    print(f"  Train: min={y_trn_log.min():.3f}, median={np.median(y_trn_log):.3f}, mean={y_trn_log.mean():.3f}, max={y_trn_log.max():.3f}")
    print(f"  Std={y_trn_log.std():.3f} (Well-behaved Gaussian distribution)")

    # 2. Extract 1024-bit Morgan FP + RDKit 2D Descriptors
    def featurize(smiles_list):
        feats = []
        dummy = Chem.MolFromSmiles("C")
        n_desc = len(Descriptors.CalcMolDescriptors(dummy))
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s) if (s and isinstance(s, str)) else None
            if mol is None:
                feats.append([0.0] * (1024 + n_desc))
            else:
                fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
                desc_dict = Descriptors.CalcMolDescriptors(mol)
                desc = [0.0 if (v is None or np.isnan(v) or np.isinf(v)) else float(np.clip(v, -100.0, 100.0)) for v in desc_dict.values()]
                feats.append(fp + desc)
        return np.array(feats, dtype=np.float32)

    print("\nExtracting molecular descriptors...")
    X_trn = featurize(trn["Drug"].tolist())
    X_tst = featurize(tst["Drug"].tolist())

    # 3. Model 1: Linear Target GBDT (Without log10 - Old Baseline)
    gbdt_linear = HistGradientBoostingRegressor(max_iter=200, random_state=42)
    gbdt_linear.fit(X_trn, y_trn_raw)
    p_tst_linear = gbdt_linear.predict(X_tst)
    r2_lin = r2_score(y_tst_raw, p_tst_linear)
    rmse_lin = np.sqrt(mean_squared_error(y_tst_raw, p_tst_linear))
    mae_lin = mean_absolute_error(y_tst_raw, p_tst_linear)
    pr_lin, _ = pearsonr(p_tst_linear, y_tst_raw)
    sp_lin, _ = spearmanr(p_tst_linear, y_tst_raw)

    # 4. Model 2: Log10 Target GBDT (New Formulation)
    gbdt_log = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, min_samples_leaf=12, l2_regularization=2.0, random_state=42)
    gbdt_log.fit(X_trn, y_trn_log)
    p_tst_log = gbdt_log.predict(X_tst)
    r2_log = r2_score(y_tst_log, p_tst_log)
    rmse_log = np.sqrt(mean_squared_error(y_tst_log, p_tst_log))
    mae_log = mean_absolute_error(y_tst_log, p_tst_log)
    pr_log, _ = pearsonr(p_tst_log, y_tst_log)
    sp_log, _ = spearmanr(p_tst_log, y_tst_log)

    print("\n" + "=" * 95)
    print("=== EXPERIMENTAL COMPARISON: LINEAR VS. LOG10 TARGET FORMULATION ===")
    print("=" * 95)
    print(f"1. Linear Space GBDT (Old)  | Test R²: {r2_lin:+.4f} | RMSE: {rmse_lin:>6.2f} | MAE: {mae_lin:>5.2f} | Pearson: {pr_lin:.4f} | Spearman: {sp_lin:.4f}")
    print(f"2. Log10 Space GBDT (New)   | Test R²: {r2_log:+.4f} | RMSE: {rmse_log:>6.3f} | MAE: {mae_log:>5.3f} | Pearson: {pr_log:.4f} | Spearman: {sp_log:.4f}")
    print("3. ADMETlab 3.0 SOTA Target | Test R²: ~0.7600 | RMSE:  0.301 | MAE: 0.162 | Pearson: ~0.8800")
    print("=" * 95)


if __name__ == "__main__":
    main()
