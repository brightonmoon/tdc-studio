"""Test Tri-Hybrid Stacker with Enhanced 18-Biophysical Fused GBDT + DMPNN-MTL + ChemBERTa."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.loader import DataLoader
from tdc.single_pred import ADME

from tdc_studio.models.graph.dmpnn import DMPNNModel
from scratch.test_enhanced_biophysical_gbdt import extract_features


def canon(s):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False)
    except Exception:
        return s


def main():
    print("=" * 80)
    print("=== TRI-HYBRID STACKER WITH ENHANCED 18-BIOPHYSICAL FEATURES ===")
    print("=" * 80)

    data = ADME(name="PPBR_AZ")
    splits = data.get_split(method="random", seed=42)
    train_df, val_df, test_df = splits["train"], splits["valid"], splits["test"]

    train_df["Canon_Drug"] = train_df["Drug"].apply(canon)
    val_df["Canon_Drug"] = val_df["Drug"].apply(canon)
    test_df["Canon_Drug"] = test_df["Drug"].apply(canon)

    # 1. Enhanced 18-Biophysical Feature Extraction
    print("[1/4] Extracting enhanced 18-biophysical features...")
    X_train_base, y_train_logit, y_train_real = extract_features(train_df["Drug"].tolist(), train_df["Y"].tolist())
    X_val_base, y_val_logit, y_val_real = extract_features(val_df["Drug"].tolist(), val_df["Y"].tolist())
    X_test_base, y_test_logit, y_test_real = extract_features(test_df["Drug"].tolist(), test_df["Y"].tolist())

    loaded = np.load("data/cache/chemberta_ppbr_embeddings.npz")
    emb_train = loaded["train"]
    emb_val = loaded["val"]
    emb_test = loaded["test"]

    X_train_fused = np.hstack([X_train_base, emb_train])
    X_val_fused = np.hstack([X_val_base, emb_val])
    X_test_fused = np.hstack([X_test_base, emb_test])

    # 2. Branch 2: Enhanced Fused GBDT
    print("[2/4] Training Enhanced Fused GBDT...")
    gbdt = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=15, random_state=42
    )
    gbdt.fit(X_train_fused, y_train_logit)
    z_gbdt_val = gbdt.predict(X_val_fused)
    z_gbdt_test = gbdt.predict(X_test_fused)

    # 3. Branch 3: ChemBERTa Ridge
    print("[3/4] Training ChemBERTa Ridge...")
    scaler = StandardScaler()
    emb_train_sc = scaler.fit_transform(emb_train)
    emb_val_sc = scaler.transform(emb_val)
    emb_test_sc = scaler.transform(emb_test)

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20))
    ridge.fit(emb_train_sc, y_train_logit)
    z_ridge_val = ridge.predict(emb_val_sc)
    z_ridge_test = ridge.predict(emb_test_sc)

    # 4. Branch 1: Multi-Task DMPNN
    print("[4/4] Loading Multi-Task DMPNN Checkpoint...")
    ckpt = torch.load("models/checkpoint/best_model.pt", map_location="cpu", weights_only=False)
    with open("models/checkpoint/config.json") as f:
        dmpnn_cfg = json.load(f)
    dmpnn_model = DMPNNModel(dmpnn_cfg)
    dmpnn_model.load_state_dict(ckpt)
    dmpnn_model.eval()

    cache = torch.load(
        "data/cache/distribution_mtl_ppbr_az_random_42_graph_True_True_dbea67ae.pt", map_location="cpu", weights_only=False
    )
    val_dataset = cache["val"]
    test_dataset = cache["test"]

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    def extract_dmpnn(loader):
        preds_std, smiles = [], []
        with torch.no_grad():
            for b in loader:
                m = b["mask"][:, 0].bool()
                p = dmpnn_model(b)[:, 0]
                preds_std.extend(p[m].cpu().numpy().tolist())
                smiles.extend([s for s, mask_val in zip(b["drug_smiles_str"], m) if mask_val])
        return dict(zip([canon(s) for s in smiles], preds_std))

    dmpnn_val_dict = extract_dmpnn(val_loader)
    dmpnn_test_dict = extract_dmpnn(test_loader)

    fb_trn = np.clip(train_df["Y"] / 100.0, 1e-4, 1.0 - 1e-4)
    z_trn = np.log(fb_trn / (1.0 - fb_trn))
    mu_trn, std_trn = float(np.mean(z_trn)), float(np.std(z_trn))

    # Match test
    matched_test_idx = []
    z_dmpnn_test_list = []
    for idx, s in enumerate(test_df["Canon_Drug"]):
        if s in dmpnn_test_dict:
            matched_test_idx.append(idx)
            z_dmpnn_test_list.append(dmpnn_test_dict[s] * std_trn + mu_trn)

    z_dmpnn_test = np.array(z_dmpnn_test_list)
    z_gbdt_test_m = z_gbdt_test[matched_test_idx]
    z_ridge_test_m = z_ridge_test[matched_test_idx]
    y_test_m = y_test_real[matched_test_idx]

    # Match val
    matched_val_idx = []
    z_dmpnn_val_list = []
    for idx, s in enumerate(val_df["Canon_Drug"]):
        if s in dmpnn_val_dict:
            matched_val_idx.append(idx)
            z_dmpnn_val_list.append(dmpnn_val_dict[s] * std_trn + mu_trn)

    z_dmpnn_val = np.array(z_dmpnn_val_list)
    z_gbdt_val_m = z_gbdt_val[matched_val_idx]
    z_ridge_val_m = z_ridge_val[matched_val_idx]
    y_val_m = y_val_real[matched_val_idx]

    # Stacking Optimization on Validation Set
    def tri_obj(params):
        w1, w2, w3, a, b = params
        w_sum = w1 + w2 + w3
        if w_sum <= 0:
            return 1e6
        nw1, nw2, nw3 = w1 / w_sum, w2 / w_sum, w3 / w_sum
        z_b = nw1 * z_dmpnn_val + nw2 * z_gbdt_val_m + nw3 * z_ridge_val_m
        pred = 100.0 / (1.0 + np.exp(-np.clip(a * z_b + b, -40.0, 40.0)))
        return float(np.mean((pred - y_val_m) ** 2))

    best_loss = 1e9
    best_params = None
    for init_w in [
        [0.3, 0.5, 0.2, 1.0, 0.0],
        [0.4, 0.4, 0.2, 0.9, 0.0],
        [0.2, 0.6, 0.2, 0.9, 0.1],
        [0.5, 0.3, 0.2, 0.8, -0.1],
    ]:
        res = minimize(
            tri_obj,
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

    print(f"\nOptimal Stacking Weights: DMPNN={nw1:.3f}, Enhanced GBDT={nw2:.3f}, ChemBERTa Ridge={nw3:.3f}")
    print(f"Optimal Calibration: alpha={a_opt:.3f}, beta={b_opt:+.3f}")

    # Dual Blend (DMPNN + Enhanced GBDT)
    def dual_obj(params):
        w, a, b = params
        w_c = np.clip(w, 0.0, 1.0)
        z_b = w_c * z_dmpnn_val + (1.0 - w_c) * z_gbdt_val_m
        pred = 100.0 / (1.0 + np.exp(-np.clip(a * z_b + b, -40.0, 40.0)))
        return float(np.mean((pred - y_val_m) ** 2))

    res_dual = minimize(dual_obj, [0.5, 1.0, 0.0], bounds=[(0.0, 1.0), (0.2, 3.0), (-3.0, 3.0)], method="L-BFGS-B")
    w_dual, a_dual, b_dual = res_dual.x
    z_dual_test = w_dual * z_dmpnn_test + (1.0 - w_dual) * z_gbdt_test_m
    p_dual_test = 100.0 / (1.0 + np.exp(-np.clip(a_dual * z_dual_test + b_dual, -40.0, 40.0)))

    # Tri-Hybrid Test
    z_tri_test = nw1 * z_dmpnn_test + nw2 * z_gbdt_test_m + nw3 * z_ridge_test_m
    p_tri_test = 100.0 / (1.0 + np.exp(-np.clip(a_opt * z_tri_test + b_opt, -40.0, 40.0)))

    def evaluate(y_true, y_pred, name):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        pr, _ = pearsonr(y_pred, y_true)
        sp, _ = spearmanr(y_pred, y_true)
        print(f"{name:<42} | R2: {r2:.4f} | MAE: {mae:.2f}% | RMSE: {rmse:.2f}% | Pearson: {pr:.4f} | Spearman: {sp:.4f}")

    print("\n" + "=" * 90)
    print("=== FINAL TEST BENCHMARK COMPARISON (323 TEST COMPOUNDS) ===")
    print("=" * 90)
    evaluate(y_test_m, 100.0 / (1.0 + np.exp(-z_dmpnn_test)), "1. Multi-Task DMPNN (Branch 1 Standalone)")
    evaluate(y_test_m, 100.0 / (1.0 + np.exp(-z_gbdt_test_m)), "2. Enhanced Fused GBDT (18 Bio Features)")
    evaluate(y_test_m, p_dual_test, "3. Dual Hybrid (DMPNN + Enhanced GBDT)")
    evaluate(y_test_m, p_tri_test, "4. ENHANCED TRI-HYBRID FOUNDATION STACKER")
    print("=" * 90)


if __name__ == "__main__":
    main()
