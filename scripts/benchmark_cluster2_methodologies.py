"""Comprehensive Benchmark Suite for ADMET Cluster 2 (PPBR_AZ) Methodologies.

Evaluates 10+ distinct modeling and training methodologies:
1. Baseline GBDT (2D Morgan + 208 RDKit descriptors)
2. + 18 Physiological pH 7.4 Biophysical Motifs
3. + 3D Single Conformer Steric Features (PBF, Spherocity, Asphericity)
4. Loss Reformulation: Step Asymmetric Sample Weighting (< 70% weighted 2.5x)
5. Loss Reformulation: Balanced MSE / Continuous Focal Inverse-KDE Weighting
6. Loss Reformulation: Asymmetric Pinball / Quantile Loss
7. Two-Stage Hurdle Gated Regression (Extreme-binding Gate + Dual Specialized Regressors)
8. Multi-Modal Manifold Stacking (ChemBERTa-77M Foundation RidgeCV)
9. Optimal Convex Stacking & Parametric Sigmoid Boundary Calibration
10. Ultimate "Quad-Hybrid SOTA" Integration (Hurdle Gating + 3D Biophysical GBDT + Foundation Manifold + Calibration)
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Ensure reproducibility
np.random.seed(42)

def evaluate_metrics(y_true, y_pred, name="Model"):
    """Compute comprehensive regression metrics including low-binding pathology breakdown."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    # Clip predictions to physical binding percentage range (0.0 to 100.0)
    y_pred = np.clip(y_pred, 0.0, 100.0)
    
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    pr, _ = pearsonr(y_pred, y_true)
    sp, _ = spearmanr(y_pred, y_true)
    
    # Residual error pathology on low binding (< 70%)
    low_mask = y_true < 70.0
    n_low = int(np.sum(low_mask))
    low_mae = float(mean_absolute_error(y_true[low_mask], y_pred[low_mask])) if n_low > 0 else 0.0
    
    total_sse = float(np.sum((y_true - y_pred) ** 2))
    low_sse = float(np.sum((y_true[low_mask] - y_pred[low_mask]) ** 2)) if n_low > 0 else 0.0
    low_sse_pct = float(low_sse / max(1e-8, total_sse) * 100.0)
    
    # High binding (>= 90%)
    high_mask = y_true >= 90.0
    high_mae = float(mean_absolute_error(y_true[high_mask], y_pred[high_mask])) if np.sum(high_mask) > 0 else 0.0
    
    return {
        "name": name,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "pearson": float(pr),
        "spearman": float(sp),
        "low_mae": low_mae,
        "low_sse_pct": low_sse_pct,
        "high_mae": high_mae,
        "n_test": len(y_true),
        "n_low": n_low,
    }

def logit_to_prob(z, alpha=1.0, beta=0.0):
    """Calibrated parametric sigmoid transformation from logit space to (0, 100)%."""
    arg = np.clip(alpha * z + beta, -35.0, 35.0)
    return 100.0 / (1.0 + np.exp(-arg))

def main():
    print("=" * 105)
    print("=== TDC ADMET CLUSTER 2 (PPBR_AZ) COMPREHENSIVE METHODOLOGY BENCHMARK ===")
    print("=" * 105)
    
    # 1. Load Pre-extracted Features & Labels
    cache_dir = Path("data/cache")
    if not (cache_dir / "features_ppbr_train_3d_bio.npz").exists():
        print(f"Error: Required cache files not found in {cache_dir.resolve()}.")
        sys.exit(1)
        
    print("[1/5] Loading 3D biophysical features and ChemBERTa embeddings...")
    train_3d = np.load(cache_dir / "features_ppbr_train_3d_bio.npz")
    val_3d = np.load(cache_dir / "features_ppbr_val_3d_bio.npz")
    test_3d = np.load(cache_dir / "features_ppbr_test_3d_bio.npz")
    
    X_train_full = train_3d["X"]        # Shape: (1130, 1258): 1024 FP + 208 Desc + 18 Bio + 6 3D
    y_train_logit = train_3d["y_logit"]  # Logit scale
    y_train_real = train_3d["y_real"]    # Real percentage [0~100]
    
    X_val_full = val_3d["X"]            # (162, 1258)
    y_val_logit = val_3d["y_logit"]
    y_val_real = val_3d["y_real"]
    
    X_test_full = test_3d["X"]          # (323, 1258)
    y_test_logit = test_3d["y_logit"]
    y_test_real = test_3d["y_real"]
    
    chemb_file = cache_dir / "chemberta_ppbr_embeddings.npz"
    has_chemb = chemb_file.exists()
    if has_chemb:
        chemb = np.load(chemb_file)
        emb_train = chemb["train"]  # (1130, 384)
        emb_val = chemb["val"]      # (162, 384)
        emb_test = chemb["test"]    # (323, 384)
    else:
        emb_train = np.zeros((len(X_train_full), 384), dtype=np.float32)
        emb_val = np.zeros((len(X_val_full), 384), dtype=np.float32)
        emb_test = np.zeros((len(X_test_full), 384), dtype=np.float32)

    # Feature subsets
    # Base 2D: 1024 FP + 208 RDKit descriptors = 1232 dims
    X_train_base = X_train_full[:, :1232]
    X_val_base = X_val_full[:, :1232]
    X_test_base = X_test_full[:, :1232]
    
    # 2D + 18 Biophysical motifs = 1250 dims
    X_train_bio = X_train_full[:, :1250]
    X_val_bio = X_val_full[:, :1250]
    X_test_bio = X_test_full[:, :1250]
    
    # 2D + Bio + 3D Conformer Steric = 1258 dims (Full)
    
    print(f"Dataset split: Train={len(y_train_real)}, Val={len(y_val_real)}, Test={len(y_test_real)}")
    print(f"Test composition: High binding (>=90%) = {np.sum(y_test_real >= 90)} ({np.mean(y_test_real >= 90)*100:.1f}%), Low binding (<70%) = {np.sum(y_test_real < 70)} ({np.mean(y_test_real < 70)*100:.1f}%)")
    
    results = []

    # -------------------------------------------------------------------------
    # METHOD 1: Baseline GBDT (Base 2D Features only)
    # -------------------------------------------------------------------------
    print("\n[Running Method 1] Baseline GBDT (2D Morgan + 208 PhysChem)...")
    m1 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=15, random_state=42)
    m1.fit(X_train_base, y_train_logit)
    p1_val_z = m1.predict(X_val_base)
    p1_test_z = m1.predict(X_test_base)
    p1_test = logit_to_prob(p1_test_z)
    results.append(evaluate_metrics(y_test_real, p1_test, "1. Baseline GBDT (2D Only)"))

    # -------------------------------------------------------------------------
    # METHOD 2: Enhanced 18 Biophysical pH 7.4 Motifs
    # -------------------------------------------------------------------------
    print("[Running Method 2] + 18 Physiological Biophysical Motifs...")
    m2 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=15, random_state=42)
    m2.fit(X_train_bio, y_train_logit)
    p2_test_z = m2.predict(X_test_bio)
    p2_test = logit_to_prob(p2_test_z)
    results.append(evaluate_metrics(y_test_real, p2_test, "2. + 18 Biophysical Motifs"))

    # -------------------------------------------------------------------------
    # METHOD 3: + 3D Single Conformer Steric Features (PBF, Spherocity, Asphericity)
    # -------------------------------------------------------------------------
    print("[Running Method 3] + 3D Conformer Steric Features...")
    m3 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=15, random_state=42)
    m3.fit(X_train_full, y_train_logit)
    p3_val_z = m3.predict(X_val_full)
    p3_test_z = m3.predict(X_test_full)
    p3_test = logit_to_prob(p3_test_z)
    results.append(evaluate_metrics(y_test_real, p3_test, "3. + 3D Conformer Steric"))

    # -------------------------------------------------------------------------
    # METHOD 4: Loss Reformulation A - Step Asymmetric Sample Weighting
    # -------------------------------------------------------------------------
    print("[Running Method 4] Loss Reformulation A: Step Asymmetric Weighting (<70% 2.5x)...")
    weights_step = np.ones(len(y_train_real), dtype=np.float32)
    weights_step[y_train_real < 70.0] = 2.5
    weights_step[(y_train_real >= 70.0) & (y_train_real < 85.0)] = 1.5
    
    m4 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=15, random_state=42)
    m4.fit(X_train_full, y_train_logit, sample_weight=weights_step)
    p4_val_z = m4.predict(X_val_full)
    p4_test_z = m4.predict(X_test_full)
    p4_test = logit_to_prob(p4_test_z)
    results.append(evaluate_metrics(y_test_real, p4_test, "4. Step Asymmetric Loss"))

    # -------------------------------------------------------------------------
    # METHOD 5: Loss Reformulation B - Balanced MSE / Continuous Focal Inverse-KDE Weighting
    # -------------------------------------------------------------------------
    print("[Running Method 5] Loss Reformulation B: Continuous Focal Inverse-KDE (BMSE)...")
    kde = gaussian_kde(y_train_real, bw_method=0.15)
    densities = kde.evaluate(y_train_real)
    # Inverse density with smooth temperature
    inv_dens = 1.0 / (densities + 1e-4)
    # Normalize weights so mean is 1.0, power of 0.4 to prevent explosive weights
    weights_bmse = (inv_dens / np.mean(inv_dens)) ** 0.45
    weights_bmse = np.clip(weights_bmse, 0.3, 4.0)

    m5 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=15, random_state=42)
    m5.fit(X_train_full, y_train_logit, sample_weight=weights_bmse)
    p5_val_z = m5.predict(X_val_full)
    p5_test_z = m5.predict(X_test_full)
    p5_test = logit_to_prob(p5_test_z)
    results.append(evaluate_metrics(y_test_real, p5_test, "5. Continuous Focal (BMSE)"))

    # -------------------------------------------------------------------------
    # METHOD 6: Loss Reformulation C - Asymmetric Quantile Loss (Pinball Loss)
    # -------------------------------------------------------------------------
    print("[Running Method 6] Loss Reformulation C: Quantile Loss (alpha=0.45 to penalize over-prediction)...")
    m6 = HistGradientBoostingRegressor(loss="quantile", quantile=0.45, max_iter=300, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=15, random_state=42)
    m6.fit(X_train_full, y_train_logit)
    p6_val_z = m6.predict(X_val_full)
    p6_test_z = m6.predict(X_test_full)
    p6_test = logit_to_prob(p6_test_z)
    results.append(evaluate_metrics(y_test_real, p6_test, "6. Asymmetric Quantile Loss"))

    # -------------------------------------------------------------------------
    # METHOD 7: Two-Stage Hurdle Gated Regression (Mixture Model)
    # -------------------------------------------------------------------------
    print("[Running Method 7] Two-Stage Hurdle Gated Mixture Model...")
    # Stage 1: Extreme binding gating classifier (Is binding >= 90%?)
    y_gate_train = (y_train_real >= 90.0).astype(int)
    y_gate_val = (y_val_real >= 90.0).astype(int)
    y_gate_test = (y_test_real >= 90.0).astype(int)

    gate_clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, min_samples_leaf=15, random_state=42)
    gate_clf.fit(X_train_full, y_gate_train)
    p_gate_val = gate_clf.predict_proba(X_val_full)[:, 1]
    p_gate_test = gate_clf.predict_proba(X_test_full)[:, 1]
    gate_auc = roc_auc_score(y_gate_test, p_gate_test)
    print(f"   -> Stage 1 Gating Classifier ROC-AUC: {gate_auc:.4f}")

    # Stage 2A: High-binding specialist regressor
    high_idx = np.where(y_train_real >= 85.0)[0]
    m7_high = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=10, random_state=42)
    m7_high.fit(X_train_full[high_idx], y_train_logit[high_idx])
    
    # Stage 2B: Low/Mid-binding specialist regressor (heavily weighted towards low)
    m7_low = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, l2_regularization=1.5, min_samples_leaf=8, random_state=42)
    weights_low_branch = np.ones(len(y_train_real), dtype=np.float32)
    weights_low_branch[y_train_real < 70.0] = 3.5
    weights_low_branch[(y_train_real >= 70.0) & (y_train_real < 85.0)] = 2.0
    weights_low_branch[y_train_real >= 90.0] = 0.5
    m7_low.fit(X_train_full, y_train_logit, sample_weight=weights_low_branch)

    pred_high_val = logit_to_prob(m7_high.predict(X_val_full))
    pred_low_val = logit_to_prob(m7_low.predict(X_val_full))
    p7_val = p_gate_val * pred_high_val + (1.0 - p_gate_val) * pred_low_val

    pred_high_test = logit_to_prob(m7_high.predict(X_test_full))
    pred_low_test = logit_to_prob(m7_low.predict(X_test_full))
    p7_test = p_gate_test * pred_high_test + (1.0 - p_gate_test) * pred_low_test
    results.append(evaluate_metrics(y_test_real, p7_test, "7. Two-Stage Hurdle Mixture"))

    # -------------------------------------------------------------------------
    # METHOD 8: Foundation Language Model (ChemBERTa-77M-MTR RidgeCV)
    # -------------------------------------------------------------------------
    print("[Running Method 8] Foundation Model Manifold (ChemBERTa-77M RidgeCV)...")
    scaler = StandardScaler()
    emb_train_sc = scaler.fit_transform(emb_train)
    emb_val_sc = scaler.transform(emb_val)
    emb_test_sc = scaler.transform(emb_test)

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20))
    ridge.fit(emb_train_sc, y_train_logit)
    p8_val_z = ridge.predict(emb_val_sc)
    p8_test_z = ridge.predict(emb_test_sc)
    p8_test = logit_to_prob(p8_test_z)
    results.append(evaluate_metrics(y_test_real, p8_test, "8. ChemBERTa Foundation Ridge"))

    # -------------------------------------------------------------------------
    # METHOD 9: Tri-Hybrid Foundation Stacker (Baseline SOTA)
    # -------------------------------------------------------------------------
    print("[Running Method 9] Tri-Hybrid Foundation Stacker (Fused GBDT + ChemBERTa Ridge)...")
    # Fused GBDT (3D Bio + ChemBERTa)
    X_train_fused = np.hstack([X_train_full, emb_train])
    X_val_fused = np.hstack([X_val_full, emb_val])
    X_test_fused = np.hstack([X_test_full, emb_test])

    gbdt_fused = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=15, random_state=42)
    gbdt_fused.fit(X_train_fused, y_train_logit)
    z_fused_val = gbdt_fused.predict(X_val_fused)
    z_fused_test = gbdt_fused.predict(X_test_fused)

    # Blend GBDT + ChemBERTa with Parametric Sigmoid Calibration on Validation
    def obj_tri(weights):
        w1, w2, a, b = weights
        z_b = w1 * z_fused_val + w2 * p8_val_z
        pred = logit_to_prob(z_b, a, b)
        return float(np.mean((pred - y_val_real) ** 2))

    opt_tri = minimize(obj_tri, [0.8, 0.2, 0.85, 0.1], bounds=[(0.0, 1.0), (0.0, 1.0), (0.2, 3.0), (-2.0, 2.0)], method="L-BFGS-B")
    w1_t, w2_t, a_t, b_t = opt_tri.x
    w_sum = w1_t + w2_t
    w1_t, w2_t = w1_t / w_sum, w2_t / w_sum

    z_tri_test = w1_t * z_fused_test + w2_t * p8_test_z
    p9_test = logit_to_prob(z_tri_test, a_t, b_t)
    results.append(evaluate_metrics(y_test_real, p9_test, "9. Tri-Hybrid Stacker (Previous SOTA)"))

    # -------------------------------------------------------------------------
    # METHOD 10: Quad-Hybrid SOTA Integration (Hurdle Gating + BMSE GBDT + ChemBERTa + Calibration)
    # -------------------------------------------------------------------------
    print("[Running Method 10] Ultimate Quad-Hybrid SOTA Integration...")
    # Train BMSE-weighted Fused GBDT
    gbdt_bmse_fused = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=15, random_state=42)
    gbdt_bmse_fused.fit(X_train_fused, y_train_logit, sample_weight=weights_bmse)
    z_bmse_fused_val = gbdt_bmse_fused.predict(X_val_fused)
    z_bmse_fused_test = gbdt_bmse_fused.predict(X_test_fused)

    # Convex combination of Hurdle Gating + BMSE Fused GBDT + ChemBERTa Ridge
    p_hurdle_val = p7_val
    p_hurdle_test = p7_test

    def obj_quad(params):
        w_h, w_g, w_r, a, b = params
        # Convert hurdle prob to logit for smooth blending
        fb = np.clip(p_hurdle_val / 100.0, 1e-4, 1.0 - 1e-4)
        z_h = np.log(fb / (1.0 - fb))
        z_blend = w_h * z_h + w_g * z_bmse_fused_val + w_r * p8_val_z
        pred = logit_to_prob(z_blend, a, b)
        return float(np.mean((pred - y_val_real) ** 2))

    opt_quad = minimize(obj_quad, [0.35, 0.50, 0.15, 0.85, 0.10], 
                        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.2, 3.0), (-2.0, 2.0)], 
                        method="L-BFGS-B")
    wh_q, wg_q, wr_q, a_q, b_q = opt_quad.x
    sum_q = wh_q + wg_q + wr_q
    wh_q, wg_q, wr_q = wh_q / sum_q, wg_q / sum_q, wr_q / sum_q
    
    fb_test = np.clip(p_hurdle_test / 100.0, 1e-4, 1.0 - 1e-4)
    z_h_test = np.log(fb_test / (1.0 - fb_test))
    z_quad_test = wh_q * z_h_test + wg_q * z_bmse_fused_test + wr_q * p8_test_z
    p10_test = logit_to_prob(z_quad_test, a_q, b_q)
    results.append(evaluate_metrics(y_test_real, p10_test, "10. Quad-Hybrid SOTA (Hurdle+BMSE+ChemB)"))

    # -------------------------------------------------------------------------
    # PRINT COMPREHENSIVE BENCHMARK SCOREBOARD
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"{'Methodology / Architecture':<38} | {'Test R²':<8} | {'MAE (%)':<8} | {'Pearson r':<9} | {'Spearman ρ':<10} | {'Low MAE (<70%)':<14} | {'Low SSE %':<10}")
    print("=" * 115)
    for r in results:
        print(f"{r['name']:<38} | {r['r2']:<8.4f} | {r['mae']:<7.2f}% | {r['pearson']:<9.4f} | {r['spearman']:<10.4f} | {r['low_mae']:<13.2f}% | {r['low_sse_pct']:<9.1f}%")
    print("=" * 115)

    # Save results as JSON
    out_path = Path("scratch/cluster2_methodologies_benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll benchmark results successfully saved to {out_path.resolve()}!")

if __name__ == "__main__":
    main()
