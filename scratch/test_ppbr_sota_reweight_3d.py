"""SOTA Optimization for PPBR / Distribution Cluster:
- Task A-1: 5-Task Checkpoint (with ChEMBL HSA) + 4-Task Super-Ensemble Integration
- Task A-2: Low-Binding (< 70%) Sample-Reweighted / Density-Focal GBDT + 3D Conformer Steric Features (PBF, Spherocity)
- Parametric Sigmoid Calibration & Tri-Hybrid Blending
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.loader import DataLoader
from tdc.single_pred import ADME

from tdc_studio.models.graph.dmpnn import DMPNNModel
from tdc_studio.models.hybrid.gbdt_blend import compute_biophysical_motifs

_QUAT_N_SMARTS = Chem.MolFromSmarts("[NX4+;!$([NX4+]([O-])=O)]")


def canon(s):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False)
    except Exception:
        return s


def compute_3d_and_biophysical_motifs(mol) -> list[float]:
    """Compute 18 biophysical motifs + 6 3D conformer steric parameters."""
    base_14 = compute_biophysical_motifs(mol)
    if mol is None:
        return base_14 + [0.0] * 10

    # 15. Quaternary nitrogen (permanent positive charge repelling albumin pI 4.7)
    n_quat_n = float(len(mol.GetSubstructMatches(_QUAT_N_SMARTS)))

    # 16. Polar Surface Area density (TPSA / MolWt)
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    tpsa_mw_ratio = float(tpsa / max(1.0, mw))

    # 17. 3D Non-planarity (Fraction CSP3)
    fsp3 = float(Descriptors.FractionCSP3(mol))

    # 18. Rotatable bond density
    hac = Descriptors.HeavyAtomCount(mol)
    rot_dens = float(Descriptors.NumRotatableBonds(mol) / max(1.0, hac))

    # --- 3D Conformer Steric Features (PBF, Spherocity, Shape Factors) ---
    pbf, spherocity, asphericity, eccentricity, inert_factor, rog = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        mol_3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.maxAttempts = 10
        cid = AllChem.EmbedMolecule(mol_3d, params)
        if cid >= 0:
            AllChem.MMFFOptimizeMolecule(mol_3d, confId=cid, maxIters=50)
            pbf = float(rdMolDescriptors.CalcPBF(mol_3d, confId=cid))
            spherocity = float(rdMolDescriptors.CalcSpherocityIndex(mol_3d, confId=cid))
            asphericity = float(rdMolDescriptors.CalcAsphericity(mol_3d, confId=cid))
            eccentricity = float(rdMolDescriptors.CalcEccentricity(mol_3d, confId=cid))
            inert_factor = float(rdMolDescriptors.CalcInertialShapeFactor(mol_3d, confId=cid))
            rog = float(rdMolDescriptors.CalcRadiusOfGyration(mol_3d, confId=cid))
    except Exception:
        pass

    return base_14 + [
        n_quat_n,
        tpsa_mw_ratio,
        fsp3,
        rot_dens,
        pbf,
        spherocity,
        asphericity,
        eccentricity,
        inert_factor,
        rog,
    ]


def extract_all_features(smiles_list, labels=None, n_bits=1024, desc_cache_file=None):
    """Extract 1024 Morgan FP + 210 RDKit Descriptors + 24 Biophysical/3D Steric Features with caching."""
    if desc_cache_file and os.path.exists(desc_cache_file):
        print(f"Loading cached tabular features from {desc_cache_file}...", flush=True)
        data = np.load(desc_cache_file)
        return data["X"], data["y_logit"], data["y_real"]

    features = []
    y_logit = []
    y_real = []

    dummy_mol = Chem.MolFromSmiles("C")
    n_desc = len(Descriptors.CalcMolDescriptors(dummy_mol)) if dummy_mol is not None else 210

    total = len(smiles_list)
    print(f"Extracting 2D/3D & biophysical features for {total} molecules...", flush=True)
    for idx, s in enumerate(smiles_list):
        if (idx + 1) % 300 == 0 or idx == total - 1:
            print(f"  Processed {idx + 1}/{total} compounds...", flush=True)
        mol = Chem.MolFromSmiles(s) if (s and isinstance(s, str)) else None
        if mol is None:
            features.append([0.0] * (n_bits + n_desc + 24))
        else:
            fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
            desc_dict = Descriptors.CalcMolDescriptors(mol)
            desc_vals = [
                0.0 if (v is None or np.isnan(v) or np.isinf(v)) else float(np.clip(v, -100.0, 100.0))
                for v in desc_dict.values()
            ]
            bio_3d = compute_3d_and_biophysical_motifs(mol)
            features.append(fp + desc_vals + bio_3d)

        if labels is not None and idx < len(labels):
            val = float(labels[idx])
            y_real.append(val)
            fb = np.clip(val / 100.0, 1e-4, 1.0 - 1e-4)
            y_logit.append(float(np.log(fb / (1.0 - fb))))

    X = np.array(features, dtype=np.float32)
    y_l = np.array(y_logit, dtype=np.float32) if labels is not None else None
    y_r = np.array(y_real, dtype=np.float32) if labels is not None else None

    if desc_cache_file:
        os.makedirs(os.path.dirname(desc_cache_file), exist_ok=True)
        np.savez_compressed(desc_cache_file, X=X, y_logit=y_l, y_real=y_r)
        print(f"Saved feature cache to {desc_cache_file}.", flush=True)

    return X, y_l, y_r


def load_dmpnn_predictions(ckpt_path, config_path, cache_path, val_smiles_canon, test_smiles_canon, mu_trn, std_trn):
    """Load DMPNN checkpoint and extract de-standardized logit predictions on val and test."""
    print(f"Loading DMPNN from {ckpt_path}...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with open(config_path) as f:
        dmpnn_cfg = json.load(f)
    dmpnn_model = DMPNNModel(dmpnn_cfg)
    dmpnn_model.load_state_dict(ckpt)
    dmpnn_model.eval()

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val_loader = DataLoader(cache["val"], batch_size=64, shuffle=False)
    test_loader = DataLoader(cache["test"], batch_size=64, shuffle=False)

    def extract_dict(loader):
        preds_std, smiles = [], []
        with torch.no_grad():
            for b in loader:
                m = b["mask"][:, 0].bool()
                p = dmpnn_model(b)[:, 0]
                preds_std.extend(p[m].cpu().numpy().tolist())
                smiles.extend([s for s, mask_val in zip(b["drug_smiles_str"], m) if mask_val])
        return dict(zip([canon(s) for s in smiles], preds_std))

    v_dict = extract_dict(val_loader)
    t_dict = extract_dict(test_loader)

    z_val = np.array([v_dict.get(s, 0.0) * std_trn + mu_trn for s in val_smiles_canon])
    z_test = np.array([t_dict.get(s, 0.0) * std_trn + mu_trn for s in test_smiles_canon])

    return z_val, z_test


def evaluate_metrics(y_true, y_pred, name=""):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pr, _ = pearsonr(y_pred, y_true)
    sp, _ = spearmanr(y_pred, y_true)
    low_mask = y_true < 70.0
    low_mae = mean_absolute_error(y_true[low_mask], y_pred[low_mask]) if np.sum(low_mask) > 0 else 0.0
    low_sse = np.sum((y_true[low_mask] - y_pred[low_mask]) ** 2) if np.sum(low_mask) > 0 else 0.0
    tot_sse = np.sum((y_true - y_pred) ** 2)
    low_sse_pct = (low_sse / max(1e-5, tot_sse)) * 100.0

    print(
        f"{name:<48} | R²: {r2:.4f} | MAE: {mae:.2f}% | RMSE: {rmse:.2f}% | Pearson: {pr:.4f} | Spearman: {sp:.4f} | Low(<70) MAE: {low_mae:.2f}% (SSE: {low_sse_pct:.1f}%)",
        flush=True,
    )
    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "pearson": pr,
        "spearman": sp,
        "low_mae": low_mae,
        "low_sse_pct": low_sse_pct,
    }


def main():
    print("=" * 100, flush=True)
    print("=== PPBR SOTA OPTIMIZATION: 5-TASK DMPNN + 3D/BIOPHYSICAL + SAMPLE-REWEIGHTED GBDT ===", flush=True)
    print("=" * 100, flush=True)

    data = ADME(name="PPBR_AZ")
    splits = data.get_split(method="random", seed=42)
    train_df, val_df, test_df = splits["train"], splits["valid"], splits["test"]

    train_df["Canon_Drug"] = train_df["Drug"].apply(canon)
    val_df["Canon_Drug"] = val_df["Drug"].apply(canon)
    test_df["Canon_Drug"] = test_df["Drug"].apply(canon)

    fb_trn = np.clip(train_df["Y"] / 100.0, 1e-4, 1.0 - 1e-4)
    z_trn = np.log(fb_trn / (1.0 - fb_trn))
    mu_trn, std_trn = float(np.mean(z_trn)), float(np.std(z_trn))

    # 1. Feature Extraction (with 3D Steric Conformer Features)
    cache_trn = "data/cache/features_ppbr_train_3d_bio.npz"
    cache_val = "data/cache/features_ppbr_val_3d_bio.npz"
    cache_tst = "data/cache/features_ppbr_test_3d_bio.npz"

    X_train_base, y_train_logit, y_train_real = extract_all_features(
        train_df["Drug"].tolist(), train_df["Y"].tolist(), desc_cache_file=cache_trn
    )
    X_val_base, y_val_logit, y_val_real = extract_all_features(
        val_df["Drug"].tolist(), val_df["Y"].tolist(), desc_cache_file=cache_val
    )
    X_test_base, y_test_logit, y_test_real = extract_all_features(
        test_df["Drug"].tolist(), test_df["Y"].tolist(), desc_cache_file=cache_tst
    )

    chemberta_loaded = np.load("data/cache/chemberta_ppbr_embeddings.npz")
    emb_train = chemberta_loaded["train"]
    emb_val = chemberta_loaded["val"]
    emb_test = chemberta_loaded["test"]

    X_train_fused = np.hstack([X_train_base, emb_train])
    X_val_fused = np.hstack([X_val_base, emb_val])
    X_test_fused = np.hstack([X_test_base, emb_test])

    print(f"Fused Feature Matrix Dimension: {X_train_fused.shape} (2D Descriptors + 3D Conformer + ChemBERTa)", flush=True)

    # 2. Branch 3: ChemBERTa Foundation RidgeCV
    scaler = StandardScaler()
    emb_train_sc = scaler.fit_transform(emb_train)
    emb_val_sc = scaler.transform(emb_val)
    emb_test_sc = scaler.transform(emb_test)

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20))
    ridge.fit(emb_train_sc, y_train_logit)
    z_ridge_val = ridge.predict(emb_val_sc)
    z_ridge_test = ridge.predict(emb_test_sc)

    # 3. Branch 1: DMPNN Predictions (4-Task vs 5-Task vs Super-Ensemble)
    print("\n[Loading DMPNN Graph Checkpoints]...", flush=True)
    # 4-task
    z_dmpnn4_val, z_dmpnn4_test = load_dmpnn_predictions(
        "models/checkpoint/best_model.pt",
        "models/checkpoint/config.json",
        "data/cache/distribution_mtl_ppbr_az_random_42_graph_True_True_dbea67ae.pt",
        val_df["Canon_Drug"].tolist(),
        test_df["Canon_Drug"].tolist(),
        mu_trn,
        std_trn,
    )
    # 5-task (with ChEMBL HSA)
    z_dmpnn5_val, z_dmpnn5_test = load_dmpnn_predictions(
        "models/checkpoint_5tasks/best_model.pt",
        "models/checkpoint_5tasks/config.json",
        "data/cache/distribution_mtl_ppbr_az_random_42_graph_True_True_38423764.pt",
        val_df["Canon_Drug"].tolist(),
        test_df["Canon_Drug"].tolist(),
        mu_trn,
        std_trn,
    )

    # Super-Ensemble DMPNN (4-task + 5-task)
    z_dmpnn_super_val = 0.5 * z_dmpnn4_val + 0.5 * z_dmpnn5_val
    z_dmpnn_super_test = 0.5 * z_dmpnn4_test + 0.5 * z_dmpnn5_test

    # 4. Task A-2: Low-Binding Sample Reweighting
    print("\n[Evaluating Low-Binding (<70%) Sample Reweighting Strategies]...", flush=True)
    # Baseline GBDT (Unweighted)
    gbdt_unweighted = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=15, random_state=42
    )
    gbdt_unweighted.fit(X_train_fused, y_train_logit)
    z_gbdt_unw_val = gbdt_unweighted.predict(X_val_fused)
    z_gbdt_unw_test = gbdt_unweighted.predict(X_test_fused)

    # Strategy 1: Stepwise Asymmetric Reweighting
    weights_step = np.ones(len(y_train_real), dtype=np.float32)
    weights_step[y_train_real < 70.0] = 2.4
    weights_step[(y_train_real >= 70.0) & (y_train_real < 85.0)] = 1.4

    gbdt_step = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=15, random_state=42
    )
    gbdt_step.fit(X_train_fused, y_train_logit, sample_weight=weights_step)
    z_gbdt_step_val = gbdt_step.predict(X_val_fused)
    z_gbdt_step_test = gbdt_step.predict(X_test_fused)

    # Strategy 2: Smooth Inverse Density Reweighting via KDE
    kde = gaussian_kde(y_train_real)
    densities = kde(y_train_real)
    weights_kde = (1.0 / (densities + 0.005)) ** 0.5
    weights_kde = weights_kde / np.mean(weights_kde)
    weights_kde = np.clip(weights_kde, 0.5, 4.0)

    gbdt_kde = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.04, l2_regularization=2.0, min_samples_leaf=15, random_state=42
    )
    gbdt_kde.fit(X_train_fused, y_train_logit, sample_weight=weights_kde)
    z_gbdt_kde_val = gbdt_kde.predict(X_val_fused)
    z_gbdt_kde_test = gbdt_kde.predict(X_test_fused)

    # 5. Tri-Hybrid Blending & Parametric Calibration Function
    def optimize_and_evaluate(z_gnn_val, z_gnn_test, z_g_val, z_g_test, z_r_val, z_r_test, label):
        def obj(params):
            w1, w2, w3, a, b = params
            ws = w1 + w2 + w3
            if ws <= 0:
                return 1e6
            nw1, nw2, nw3 = w1 / ws, w2 / ws, w3 / ws
            zb = nw1 * z_gnn_val + nw2 * z_g_val + nw3 * z_r_val
            pred = 100.0 / (1.0 + np.exp(-np.clip(a * zb + b, -40.0, 40.0)))
            return float(np.mean((pred - y_val_real) ** 2))

        best_loss = 1e9
        best_p = None
        for init in [
            [0.2, 0.6, 0.2, 0.9, 0.1],
            [0.3, 0.5, 0.2, 0.9, 0.0],
            [0.15, 0.70, 0.15, 0.85, 0.05],
            [0.4, 0.4, 0.2, 1.0, 0.0],
        ]:
            res = minimize(
                obj,
                init,
                bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.2, 3.0), (-3.0, 3.0)],
                method="L-BFGS-B",
            )
            if res.fun < best_loss:
                best_loss = res.fun
                best_p = res.x

        w1_o, w2_o, w3_o, a_o, b_o = best_p
        ws = w1_o + w2_o + w3_o
        nw1, nw2, nw3 = w1_o / ws, w2_o / ws, w3_o / ws

        z_tri_t = nw1 * z_gnn_test + nw2 * z_g_test + nw3 * z_r_test
        p_tri_t = 100.0 / (1.0 + np.exp(-np.clip(a_o * z_tri_t + b_o, -40.0, 40.0)))
        print(f"\n[{label}] Optimal Weights: GNN={nw1:.3f}, GBDT={nw2:.3f}, ChemBERTa={nw3:.3f} | α={a_o:.3f}, β={b_o:+.3f}", flush=True)
        return evaluate_metrics(y_test_real, p_tri_t, label), p_tri_t, (nw1, nw2, nw3, a_o, b_o)

    print("\n" + "=" * 105, flush=True)
    print("=== STANDALONE AND HYBRID BENCHMARK RESULTS (323 TEST COMPOUNDS) ===", flush=True)
    print("=" * 105, flush=True)

    evaluate_metrics(y_test_real, 100.0 / (1.0 + np.exp(-z_dmpnn4_test)), "1. Standalone 4-Task DMPNN")
    evaluate_metrics(y_test_real, 100.0 / (1.0 + np.exp(-z_dmpnn5_test)), "2. Standalone 5-Task DMPNN (with ChEMBL HSA)")
    evaluate_metrics(y_test_real, 100.0 / (1.0 + np.exp(-z_dmpnn_super_test)), "3. Standalone Super-DMPNN (4-Task + 5-Task)")
    evaluate_metrics(y_test_real, 100.0 / (1.0 + np.exp(-z_gbdt_unw_test)), "4. Standalone Fused GBDT (Unweighted)")
    evaluate_metrics(y_test_real, 100.0 / (1.0 + np.exp(-z_gbdt_step_test)), "5. Standalone Fused GBDT (Step Asym Weighted)")
    evaluate_metrics(y_test_real, 100.0 / (1.0 + np.exp(-z_gbdt_kde_test)), "6. Standalone Fused GBDT (Smooth KDE Weighted)")

    print("-" * 105, flush=True)
    res_base, p_base, params_base = optimize_and_evaluate(
        z_dmpnn4_val, z_dmpnn4_test, z_gbdt_unw_val, z_gbdt_unw_test, z_ridge_val, z_ridge_test,
        "Tri-Hybrid Baseline (4-Task DMPNN + Unw GBDT)"
    )
    res_5task, p_5task, params_5task = optimize_and_evaluate(
        z_dmpnn5_val, z_dmpnn5_test, z_gbdt_unw_val, z_gbdt_unw_test, z_ridge_val, z_ridge_test,
        "Tri-Hybrid + 5-Task DMPNN (Task A-1)"
    )
    res_step, p_step, params_step = optimize_and_evaluate(
        z_dmpnn_super_val, z_dmpnn_super_test, z_gbdt_step_val, z_gbdt_step_test, z_ridge_val, z_ridge_test,
        "Tri-Hybrid Super-DMPNN + Step-Weighted GBDT (Task A-1 + A-2)"
    )
    res_kde, p_kde, params_kde = optimize_and_evaluate(
        z_dmpnn_super_val, z_dmpnn_super_test, z_gbdt_kde_val, z_gbdt_kde_test, z_ridge_val, z_ridge_test,
        "★ SOTA Tri-Hybrid Super-DMPNN + KDE-Weighted GBDT + 3D Steric (Task A-1 + A-2)"
    )
    print("=" * 105, flush=True)

    # Save SOTA checkpoint and summary
    best_results = {
        "benchmark_summary": {
            "tri_hybrid_baseline": res_base,
            "tri_hybrid_5task": res_5task,
            "tri_hybrid_step_weighted": res_step,
            "tri_hybrid_kde_weighted_sota": res_kde,
        },
        "optimal_params_sota": {
            "w_gnn": float(params_kde[0]),
            "w_gbdt": float(params_kde[1]),
            "w_chemberta": float(params_kde[2]),
            "alpha": float(params_kde[3]),
            "beta": float(params_kde[4]),
        },
    }
    out_dir = Path("models/export")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "ppbr_tri_hybrid_sota_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=2)
    print(f"\n[Saved SOTA benchmark summary to {summary_path}]", flush=True)


if __name__ == "__main__":
    main()
