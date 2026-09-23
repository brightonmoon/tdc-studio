"""5-Fold Out-of-Fold (OOF) Cross-Validation Foundation Stacker with ChemBERTa-77M, Biophysics, and GBDT."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tdc.single_pred import ADME

from scratch.test_chemberta_gbdt import extract_all_features, extract_chemberta_embeddings


def main():
    print("=== 5-Fold OOF Foundation Stacker on TDC PPBR_AZ ===")
    data = ADME(name="PPBR_AZ")
    splits = data.get_split(method="random", seed=42)
    train_df, val_df, test_df = splits["train"], splits["valid"], splits["test"]

    # Combine train + val into development set (1,291 compounds), preserve test set (323 compounds)
    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Development Set: {len(dev_df)} (80% Train + 10% Val), Test Set: {len(test_df)} (10%)")

    # Extract Base Features
    print("Extracting base features + biophysics...")
    X_dev_base, y_dev_logit, y_dev_real = extract_all_features(dev_df["Drug"].tolist(), dev_df["Y"].tolist())
    X_test_base, y_test_logit, y_test_real = extract_all_features(test_df["Drug"].tolist(), test_df["Y"].tolist())

    # ChemBERTa Embeddings
    print("Extracting / loading ChemBERTa-77M embeddings...")
    dev_cache = Path("data/cache/chemberta_dev_embeddings.npz")
    if dev_cache.exists():
        loaded = np.load(dev_cache)
        emb_dev = loaded["dev"]
        emb_test = loaded["test"]
    else:
        emb_dev = extract_chemberta_embeddings(dev_df["Drug"].tolist())
        emb_test = extract_chemberta_embeddings(test_df["Drug"].tolist())
        np.savez_compressed(dev_cache, dev=emb_dev, test=emb_test)

    X_dev_fused = np.hstack([X_dev_base, emb_dev])
    X_test_fused = np.hstack([X_test_base, emb_test])

    print(f"Feature Dimensions: Dev={X_dev_fused.shape}, Test={X_test_fused.shape}")

    # 5-Fold Cross-Validation OOF Generation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_gbdt = np.zeros(len(dev_df))
    oof_ridge = np.zeros(len(dev_df))

    test_preds_gbdt = []
    test_preds_ridge = []

    scaler = StandardScaler()
    emb_dev_scaled = scaler.fit_transform(emb_dev)
    emb_test_scaled = scaler.transform(emb_test)

    print("\nTraining 5-Fold OOF Models...")
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_dev_fused)):
        # GBDT
        gbdt = HistGradientBoostingRegressor(
            max_iter=350, learning_rate=0.04, l2_regularization=3.0, min_samples_leaf=12, random_state=42 + fold
        )
        gbdt.fit(X_dev_fused[trn_idx], y_dev_logit[trn_idx])
        oof_gbdt[val_idx] = gbdt.predict(X_dev_fused[val_idx])
        test_preds_gbdt.append(gbdt.predict(X_test_fused))

        # Ridge ChemBERTa
        ridge = RidgeCV(alphas=np.logspace(-2, 3, 20))
        ridge.fit(emb_dev_scaled[trn_idx], y_dev_logit[trn_idx])
        oof_ridge[val_idx] = ridge.predict(emb_dev_scaled[val_idx])
        test_preds_ridge.append(ridge.predict(emb_test_scaled))

    z_gbdt_test = np.mean(test_preds_gbdt, axis=0)
    z_ridge_test = np.mean(test_preds_ridge, axis=0)

    # Optimize Blending on OOF predictions
    def blend_obj(params):
        w, a, b = params
        w_c = np.clip(w, 0.0, 1.0)
        z_b = w_c * oof_gbdt + (1.0 - w_c) * oof_ridge
        p = 100.0 / (1.0 + np.exp(-np.clip(a * z_b + b, -40.0, 40.0)))
        return float(np.mean((p - y_dev_real) ** 2))

    res = minimize(blend_obj, [0.70, 1.0, 0.0], bounds=[(0.0, 1.0), (0.1, 3.0), (-2.0, 2.0)], method="L-BFGS-B")
    w_opt, a_opt, b_opt = res.x

    # Test Predictions
    z_blend_test = w_opt * z_gbdt_test + (1.0 - w_opt) * z_ridge_test
    p_test_final = 100.0 / (1.0 + np.exp(-np.clip(a_opt * z_blend_test + b_opt, -40.0, 40.0)))

    # Metrics
    r2_final = r2_score(y_test_real, p_test_final)
    mae_final = mean_absolute_error(y_test_real, p_test_final)
    rmse_final = np.sqrt(mean_squared_error(y_test_real, p_test_final))
    pr_final, _ = pearsonr(p_test_final, y_test_real)
    sp_final, _ = spearmanr(p_test_final, y_test_real)

    # Standalone GBDT Test
    p_gbdt_standalone = 100.0 / (1.0 + np.exp(-np.clip(a_opt * z_gbdt_test + b_opt, -40.0, 40.0)))
    r2_gbdt = r2_score(y_test_real, p_gbdt_standalone)

    print("\n" + "=" * 80)
    print("=== 5-FOLD OOF FOUNDATION STACKER FINAL BENCHMARK (TEST SET) ===")
    print("=" * 80)
    print(f"Optimal Stacking Weights: Fused GBDT = {w_opt:.3f}, ChemBERTa Ridge = {1 - w_opt:.3f}")
    print(f"Optimal Calibration: alpha = {a_opt:.3f}, beta = {b_opt:+.3f}")
    print("-" * 80)
    print(f"{'Metric':<25} | {'5-Fold OOF Foundation Stacker':<30} | {'Previous Benchmark':<20}")
    print("-" * 80)
    print(f"{'Test R2':<25} | {r2_final:<30.4f} | {'0.4816':<20}")
    print(f"{'Test MAE (%)':<25} | {mae_final:<29.2f}% | {'6.28%':<20}")
    print(f"{'Test RMSE (%)':<25} | {rmse_final:<29.2f}% | {'11.23%':<20}")
    print(f"{'Pearson Correlation (r)':<25} | {pr_final:<30.4f} | {'0.7047':<20}")
    print(f"{'Spearman Correlation (rho)':<25} | {sp_final:<30.4f} | {'0.7188':<20}")
    print("=" * 80)


if __name__ == "__main__":
    main()
