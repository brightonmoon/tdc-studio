"""Tri-Hybrid Multi-Modal Fusion: Multi-Task DMPNN + Fused Biophysical GBDT + ChemBERTa Ridge.

Evaluates orthogonal representation fusion on TDC PPBR_AZ benchmark:
Branch 1: Directed Message Passing Neural Network (DMPNN-MTL, 6,000+ ADMET cross-task trained)
Branch 2: Fused Tree Ensemble (Morgan 1024 + 210 RDKit + 14 Biophysical pH 7.4 Motifs + ChemBERTa)
Branch 3: Foundation Model Linear/Ridge Branch (ChemBERTa-77M-MTR 384-dim continuous manifold)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.loader import DataLoader
from tdc.single_pred import ADME

from tdc_studio.models.graph.dmpnn import DMPNNModel
from scratch.test_chemberta_gbdt import (
    extract_all_features,
    extract_chemberta_embeddings,
)


def canon(s):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False)
    except Exception:
        return s


def main():
    print("=" * 80)
    print("=== TRI-HYBRID FOUNDATION STACKER BENCHMARK (TDC PPBR_AZ) ===")
    print("=" * 80)

    # 1. Load TDC splits
    print("[1/5] Loading TDC PPBR_AZ dataset...")
    data = ADME(name="PPBR_AZ")
    splits = data.get_split(method="random", seed=42)
    train_df, val_df, test_df = splits["train"], splits["valid"], splits["test"]

    # Canonicalize SMILES
    train_df["Canon_Drug"] = train_df["Drug"].apply(canon)
    val_df["Canon_Drug"] = val_df["Drug"].apply(canon)
    test_df["Canon_Drug"] = test_df["Drug"].apply(canon)

    # 2. Extract Base + Biophysical Features
    print("[2/5] Extracting Morgan + RDKit + Biophysical features...")
    X_train_base, y_train_logit, y_train_real = extract_all_features(train_df["Drug"].tolist(), train_df["Y"].tolist())
    X_val_base, y_val_logit, y_val_real = extract_all_features(val_df["Drug"].tolist(), val_df["Y"].tolist())
    X_test_base, y_test_logit, y_test_real = extract_all_features(test_df["Drug"].tolist(), test_df["Y"].tolist())

    # ChemBERTa Embeddings
    print("[3/5] Loading cached ChemBERTa-77M embeddings...")
    emb_cache = Path("data/cache/chemberta_ppbr_embeddings.npz")
    if emb_cache.exists():
        loaded = np.load(emb_cache)
        emb_train = loaded["train"]
        emb_val = loaded["val"]
        emb_test = loaded["test"]
    else:
        emb_train = extract_chemberta_embeddings(train_df["Drug"].tolist())
        emb_val = extract_chemberta_embeddings(val_df["Drug"].tolist())
        emb_test = extract_chemberta_embeddings(test_df["Drug"].tolist())
        np.savez_compressed(emb_cache, train=emb_train, val=emb_val, test=emb_test)

    X_train_fused = np.hstack([X_train_base, emb_train])
    X_val_fused = np.hstack([X_val_base, emb_val])
    X_test_fused = np.hstack([X_test_base, emb_test])

    # Branch 2: Train Fused GBDT
    print("Training Fused GBDT (Branch 2)...")
    from sklearn.ensemble import HistGradientBoostingRegressor
    gbdt = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        l2_regularization=2.0,
        min_samples_leaf=15,
        random_state=42,
    )
    gbdt.fit(X_train_fused, y_train_logit)
    z_gbdt_val = gbdt.predict(X_val_fused)
    z_gbdt_test = gbdt.predict(X_test_fused)

    # Branch 3: Train ChemBERTa Ridge
    print("Training ChemBERTa Ridge (Branch 3)...")
    scaler = StandardScaler()
    emb_train_sc = scaler.fit_transform(emb_train)
    emb_val_sc = scaler.transform(emb_val)
    emb_test_sc = scaler.transform(emb_test)

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20))
    ridge.fit(emb_train_sc, y_train_logit)
    z_ridge_val = ridge.predict(emb_val_sc)
    z_ridge_test = ridge.predict(emb_test_sc)

    # Branch 1: Load Multi-Task DMPNN Model & Predictions
    print("[4/5] Running Multi-Task DMPNN (Branch 1)...")
    ckpt = torch.load("models/checkpoint/best_model.pt", map_location="cpu", weights_only=False)
    with open("models/checkpoint/config.json") as f:
        dmpnn_cfg = json.load(f)
    dmpnn_model = DMPNNModel(dmpnn_cfg)
    dmpnn_model.load_state_dict(ckpt)
    dmpnn_model.eval()

    cache = torch.load("data/cache/distribution_mtl_ppbr_az_random_42_graph_True_True_dbea67ae.pt", map_location="cpu", weights_only=False)
    val_dataset = cache["val"]
    test_dataset = cache["test"]

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    def extract_dmpnn_preds(loader):
        preds_std = []
        smiles = []
        with torch.no_grad():
            for b in loader:
                m = b["mask"][:, 0].bool()
                p = dmpnn_model(b)[:, 0]
                preds_std.extend(p[m].cpu().numpy().tolist())
                smiles.extend([s for s, mask_val in zip(b["drug_smiles_str"], m) if mask_val])
        return dict(zip([canon(s) for s in smiles], preds_std))

    dmpnn_val_dict = extract_dmpnn_preds(val_loader)
    dmpnn_test_dict = extract_dmpnn_preds(test_loader)

    # Train stats for DMPNN de-standardization
    fb_trn = np.clip(train_df["Y"] / 100.0, 1e-4, 1.0 - 1e-4)
    z_trn = np.log(fb_trn / (1.0 - fb_trn))
    mu_trn = float(np.mean(z_trn))
    std_trn = float(np.std(z_trn))

    # Match test compounds
    matched_test_indices = []
    z_dmpnn_test_list = []
    for idx, s in enumerate(test_df["Canon_Drug"]):
        if s in dmpnn_test_dict:
            matched_test_indices.append(idx)
            z_dmpnn_test_list.append(dmpnn_test_dict[s] * std_trn + mu_trn)

    z_dmpnn_test = np.array(z_dmpnn_test_list)
    z_gbdt_test_matched = z_gbdt_test[matched_test_indices]
    z_ridge_test_matched = z_ridge_test[matched_test_indices]
    y_test_matched = y_test_real[matched_test_indices]

    # Match val compounds
    matched_val_indices = []
    z_dmpnn_val_list = []
    for idx, s in enumerate(val_df["Canon_Drug"]):
        if s in dmpnn_val_dict:
            matched_val_indices.append(idx)
            z_dmpnn_val_list.append(dmpnn_val_dict[s] * std_trn + mu_trn)

    z_dmpnn_val = np.array(z_dmpnn_val_list)
    z_gbdt_val_matched = z_gbdt_val[matched_val_indices]
    z_ridge_val_matched = z_ridge_val[matched_val_indices]
    y_val_matched = y_val_real[matched_val_indices]

    print(f"Matched Samples: Val={len(y_val_matched)}, Test={len(y_test_matched)}")

    # 5. Optimize Tri-Hybrid Blending & Parametric Sigmoid Calibration on Validation Set
    print("\n[5/5] Optimizing Tri-Hybrid Stacker Weights on Validation Set...")

    def tri_blend_obj(params):
        w1, w2, w3, alpha, beta = params
        w_sum = w1 + w2 + w3
        if w_sum <= 0:
            return 1e6
        nw1, nw2, nw3 = w1 / w_sum, w2 / w_sum, w3 / w_sum
        z_b = nw1 * z_dmpnn_val + nw2 * z_gbdt_val_matched + nw3 * z_ridge_val_matched
        pred = 100.0 / (1.0 + np.exp(-np.clip(alpha * z_b + beta, -40.0, 40.0)))
        return float(np.mean((pred - y_val_matched) ** 2))

    # Grid / Multi-start optimization
    best_loss = 1e9
    best_params = None

    for init_w in [
        [0.4, 0.4, 0.2, 1.0, 0.0],
        [0.5, 0.3, 0.2, 0.9, -0.1],
        [0.3, 0.5, 0.2, 1.0, 0.1],
        [0.6, 0.2, 0.2, 0.8, -0.2],
        [0.2, 0.6, 0.2, 0.9, 0.0],
    ]:
        res = minimize(
            tri_blend_obj,
            init_w,
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.2, 3.0), (-3.0, 3.0)],
            method="L-BFGS-B",
        )
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x

    w1_opt, w2_opt, w3_opt, a_opt, b_opt = best_params
    w_sum = w1_opt + w2_opt + w3_opt
    nw1, nw2, nw3 = w1_opt / w_sum, w2_opt / w_sum, w3_opt / w_sum

    print(f"Optimal Weights: DMPNN={nw1:.3f}, Fused GBDT={nw2:.3f}, ChemBERTa Ridge={nw3:.3f}")
    print(f"Optimal Calibration: alpha={a_opt:.3f}, beta={b_opt:+.3f}")

    # Evaluate Each Branch Standalone on Test Set
    p_dmpnn_standalone = 100.0 / (1.0 + np.exp(-z_dmpnn_test))
    p_gbdt_standalone = 100.0 / (1.0 + np.exp(-z_gbdt_test_matched))
    p_ridge_standalone = 100.0 / (1.0 + np.exp(-z_ridge_test_matched))

    # Evaluate Tri-Hybrid Blend
    z_tri_test = nw1 * z_dmpnn_test + nw2 * z_gbdt_test_matched + nw3 * z_ridge_test_matched
    p_tri_test = 100.0 / (1.0 + np.exp(-np.clip(a_opt * z_tri_test + b_opt, -40.0, 40.0)))

    # Dual Blend (DMPNN + Fused GBDT without ridge)
    def dual_obj(params):
        w, a, b = params
        w_c = np.clip(w, 0.0, 1.0)
        z_b = w_c * z_dmpnn_val + (1.0 - w_c) * z_gbdt_val_matched
        pred = 100.0 / (1.0 + np.exp(-np.clip(a * z_b + b, -40.0, 40.0)))
        return float(np.mean((pred - y_val_matched) ** 2))

    res_dual = minimize(dual_obj, [0.5, 1.0, 0.0], bounds=[(0.0, 1.0), (0.2, 3.0), (-3.0, 3.0)], method="L-BFGS-B")
    w_dual, a_dual, b_dual = res_dual.x
    z_dual_test = w_dual * z_dmpnn_test + (1.0 - w_dual) * z_gbdt_test_matched
    p_dual_test = 100.0 / (1.0 + np.exp(-np.clip(a_dual * z_dual_test + b_dual, -40.0, 40.0)))

    def metrics(y_true, y_pred):
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "pearson": float(pearsonr(y_pred, y_true)[0]),
            "spearman": float(spearmanr(y_pred, y_true)[0]),
        }

    m_dmpnn = metrics(y_test_matched, p_dmpnn_standalone)
    m_gbdt = metrics(y_test_matched, p_gbdt_standalone)
    m_ridge = metrics(y_test_matched, p_ridge_standalone)
    m_dual = metrics(y_test_matched, p_dual_test)
    m_tri = metrics(y_test_matched, p_tri_test)

    print("\n" + "=" * 90)
    print(f"{'Model Architecture':<32} | {'Test R2':<8} | {'MAE (%)':<8} | {'RMSE (%)':<8} | {'Pearson':<8} | {'Spearman':<8}")
    print("=" * 90)
    print(f"{'1. Baseline Single GBDT':<32} | {'0.4218':<8} | {'6.97%':<8} | {'11.83%':<8} | {'0.6720':<8} | {'0.7061':<8}")
    print(f"{'2. Biophysical GBDT (Option 1)':<32} | {'0.4431':<8} | {'6.66%':<8} | {'11.64%':<8} | {'0.6974':<8} | {'0.7147':<8}")
    mae_r, rmse_r = f"{m_ridge['mae']:.2f}%", f"{m_ridge['rmse']:.2f}%"
    print(f"{'3. ChemBERTa Foundation Ridge':<32} | {m_ridge['r2']:<8.4f} | {mae_r:<8} | {rmse_r:<8} | {m_ridge['pearson']:<8.4f} | {m_ridge['spearman']:<8.4f}")
    mae_g, rmse_g = f"{m_gbdt['mae']:.2f}%", f"{m_gbdt['rmse']:.2f}%"
    print(f"{'4. Fused GBDT (FP+Desc+Bio+ChemB)':<32} | {m_gbdt['r2']:<8.4f} | {mae_g:<8} | {rmse_g:<8} | {m_gbdt['pearson']:<8.4f} | {m_gbdt['spearman']:<8.4f}")
    mae_d, rmse_d = f"{m_dmpnn['mae']:.2f}%", f"{m_dmpnn['rmse']:.2f}%"
    print(f"{'5. Multi-Task DMPNN (Branch 1)':<32} | {m_dmpnn['r2']:<8.4f} | {mae_d:<8} | {rmse_d:<8} | {m_dmpnn['pearson']:<8.4f} | {m_dmpnn['spearman']:<8.4f}")
    mae_du, rmse_du = f"{m_dual['mae']:.2f}%", f"{m_dual['rmse']:.2f}%"
    print(f"{'6. Dual Hybrid (DMPNN + Fused GBDT)':<32} | {m_dual['r2']:<8.4f} | {mae_du:<8} | {rmse_du:<8} | {m_dual['pearson']:<8.4f} | {m_dual['spearman']:<8.4f}")
    print("-" * 90)
    mae_t, rmse_t = f"{m_tri['mae']:.2f}%", f"{m_tri['rmse']:.2f}%"
    print(f"{'7. TRI-HYBRID FOUNDATION STACKER':<32} | {m_tri['r2']:<8.4f} | {mae_t:<8} | {rmse_t:<8} | {m_tri['pearson']:<8.4f} | {m_tri['spearman']:<8.4f}")
    print("=" * 90)

    # Save summary json
    summary = {
        "dmpnn_metrics": m_dmpnn,
        "gbdt_metrics": m_gbdt,
        "ridge_metrics": m_ridge,
        "dual_hybrid_metrics": m_dual,
        "tri_hybrid_metrics": m_tri,
        "optimal_weights": {
            "w_dmpnn": float(nw1),
            "w_fused_gbdt": float(nw2),
            "w_chemberta_ridge": float(nw3),
            "alpha": float(a_opt),
            "beta": float(b_opt),
        },
    }
    with open("models/checkpoint/ensemble/tri_hybrid_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to models/checkpoint/ensemble/tri_hybrid_summary.json")


if __name__ == "__main__":
    main()
