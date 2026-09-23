"""Tri-Hybrid Multi-Modal Foundation Stacker:
DMPNN Graph Topological Representations + ChemBERTa-77M Pretrained Foundation Embeddings + GBDT Descriptors & Biophysical Motifs.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Crippen
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
from tdc.single_pred import ADME

from tdc_studio.core.registry import DATASETS, MODELS
from tdc_studio.cli import load_yaml, _batch_to_device
from scratch.test_biophysical_hypothesis import compute_biophysical_descriptors
from scratch.test_chemberta_gbdt import extract_all_features


def main():
    print("=== Tri-Hybrid Multi-Modal Foundation Stacking Pipeline ===")
    config_path = "configs/config_distribution_mtl_random.yaml"
    cfg = load_yaml(config_path)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    data_cls = DATASETS.get(data_cfg.get("type", "admet_cluster_loader"))
    data_params = {k: v for k, v in data_cfg.items() if k not in ("type", "name", "batch_size")}
    data_module = data_cls(**data_params)
    data_module.prepare_data()
    train_loader, val_loader, test_loader = data_module.setup_loaders(batch_size=64)

    primary_task = "ppbr_az"
    task_names = data_module.task_names
    primary_idx = task_names.index(primary_task)

    stat = data_module.task_stats.get(primary_task, {"mean": 0.0, "std": 1.0})
    mean, std = stat["mean"], stat["std"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Run DMPNN Checkpoints on Val & Test
    ckpt_paths = [
        "models/checkpoint/ensemble/model_seed_101.pt",
        "models/checkpoint/ensemble/model_seed_42.pt",
    ]
    model_cls = MODELS.get(model_cfg.get("type", "dmpnn_mtl"))

    all_dmpnn_v = []
    all_dmpnn_t = []
    v_labels_real = None
    t_labels_real = None

    for p in ckpt_paths:
        if not os.path.exists(p):
            continue
        print(f"Loading DMPNN checkpoint: {p}")
        ckpt = torch.load(p, map_location=device, weights_only=False)
        m_cfg = ckpt.get("config", model_cfg)
        model = model_cls(m_cfg).to(device)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        model.eval()

        v_preds, v_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                dev_batch = _batch_to_device(batch, device)
                preds = model(dev_batch)
                mask = dev_batch["mask"][:, primary_idx].bool() if "mask" in dev_batch else torch.ones(preds.shape[0], dtype=torch.bool)
                if mask.any():
                    v_preds.append(preds[mask, primary_idx].detach().cpu())
                    v_labels.append(dev_batch["labels"][mask, primary_idx].detach().cpu())
        vp = torch.cat(v_preds, dim=0).numpy() * std + mean
        all_dmpnn_v.append(vp)

        if v_labels_real is None:
            vy = torch.cat(v_labels, dim=0).numpy() * std + mean
            v_labels_real = 100.0 / (1.0 + np.exp(-np.clip(vy, -40.0, 40.0)))

        t_preds, t_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                dev_batch = _batch_to_device(batch, device)
                preds = model(dev_batch)
                mask = dev_batch["mask"][:, primary_idx].bool() if "mask" in dev_batch else torch.ones(preds.shape[0], dtype=torch.bool)
                if mask.any():
                    t_preds.append(preds[mask, primary_idx].detach().cpu())
                    t_labels.append(dev_batch["labels"][mask, primary_idx].detach().cpu())
        tp = torch.cat(t_preds, dim=0).numpy() * std + mean
        all_dmpnn_t.append(tp)

        if t_labels_real is None:
            ty = torch.cat(t_labels, dim=0).numpy() * std + mean
            t_labels_real = 100.0 / (1.0 + np.exp(-np.clip(ty, -40.0, 40.0)))

    z_dmpnn_val = np.mean(all_dmpnn_v, axis=0)
    z_dmpnn_test = np.mean(all_dmpnn_t, axis=0)

    # 2. Extract ChemBERTa + Fused Features
    data = ADME(name="PPBR_AZ")
    splits = data.get_split(method="random", seed=42)
    train_df, val_df, test_df = splits["train"], splits["valid"], splits["test"]

    X_base_train, y_tr_logit, y_tr_real = extract_all_features(train_df["Drug"].tolist(), train_df["Y"].tolist())
    X_base_val, _, _ = extract_all_features(val_df["Drug"].tolist(), val_df["Y"].tolist())
    X_base_test, _, _ = extract_all_features(test_df["Drug"].tolist(), test_df["Y"].tolist())

    cache_file = Path("data/cache/chemberta_ppbr_embeddings.npz")
    loaded = np.load(cache_file)
    emb_train = loaded["train"]
    emb_val = loaded["val"]
    emb_test = loaded["test"]

    X_fused_train = np.hstack([X_base_train, emb_train])
    X_fused_val = np.hstack([X_base_val, emb_val])
    X_fused_test = np.hstack([X_base_test, emb_test])

    # 3. Fit GBDT
    gbdt = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.04, l2_regularization=3.0, min_samples_leaf=12, random_state=42)
    gbdt.fit(X_fused_train, y_tr_logit)
    z_gbdt_val = gbdt.predict(X_fused_val)
    z_gbdt_test = gbdt.predict(X_fused_test)

    # 4. Fit ChemBERTa Ridge
    scaler = StandardScaler()
    emb_tr_s = scaler.fit_transform(emb_train)
    emb_va_s = scaler.transform(emb_val)
    emb_te_s = scaler.transform(emb_test)
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20))
    ridge.fit(emb_tr_s, y_tr_logit)
    z_ridge_val = ridge.predict(emb_va_s)
    z_ridge_test = ridge.predict(emb_te_s)

    # Align lengths if needed
    N_v = min(len(z_dmpnn_val), len(z_gbdt_val), len(v_labels_real))
    N_t = min(len(z_dmpnn_test), len(z_gbdt_test), len(t_labels_real))

    z_dmpnn_v, z_gbdt_v, z_ridge_v, vy = z_dmpnn_val[:N_v], z_gbdt_val[:N_v], z_ridge_val[:N_v], v_labels_real[:N_v]
    z_dmpnn_t, z_gbdt_t, z_ridge_t, ty = z_dmpnn_test[:N_t], z_gbdt_test[:N_t], z_ridge_test[:N_t], t_labels_real[:N_t]

    # 5. Optimize Tri-Hybrid Blending (DMPNN + Fused GBDT + ChemBERTa Ridge) + Calibration
    def tri_obj(params):
        w1, w2, a, b = params
        w1_c = np.clip(w1, 0.0, 1.0)
        w2_c = np.clip(w2, 0.0, 1.0 - w1_c)
        w3_c = 1.0 - w1_c - w2_c
        z_tri = w1_c * z_dmpnn_v + w2_c * z_gbdt_v + w3_c * z_ridge_v
        p = 100.0 / (1.0 + np.exp(-np.clip(a * z_tri + b, -40.0, 40.0)))
        return float(np.mean((p - vy) ** 2))

    res_tri = minimize(
        tri_obj,
        [0.50, 0.35, 1.0, 0.0],
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.1, 3.0), (-2.0, 2.0)],
        method="L-BFGS-B"
    )
    w1_opt, w2_opt, a_opt, b_opt = res_tri.x
    w1_opt = np.clip(w1_opt, 0.0, 1.0)
    w2_opt = np.clip(w2_opt, 0.0, 1.0 - w1_opt)
    w3_opt = 1.0 - w1_opt - w2_opt

    z_tri_test = w1_opt * z_dmpnn_t + w2_opt * z_gbdt_t + w3_opt * z_ridge_t
    p_tri_test = 100.0 / (1.0 + np.exp(-np.clip(a_opt * z_tri_test + b_opt, -40.0, 40.0)))

    r2_tri = float(r2_score(ty, p_tri_test))
    mae_tri = float(mean_absolute_error(ty, p_tri_test))
    rmse_tri = float(np.sqrt(mean_squared_error(ty, p_tri_test)))
    pr_tri, _ = pearsonr(p_tri_test, ty)
    sp_tri, _ = spearmanr(p_tri_test, ty)

    # Standalone metrics for comparison
    def eval_standalone(z, name):
        def obj(params):
            a, b = params
            p = 100.0 / (1.0 + np.exp(-np.clip(a * z[:N_v] + b, -40.0, 40.0)))
            return float(np.mean((p - vy) ** 2))
        res = minimize(obj, [1.0, 0.0], bounds=[(0.1, 3.0), (-2.0, 2.0)], method="L-BFGS-B")
        a, b = res.x
        p = 100.0 / (1.0 + np.exp(-np.clip(a * z[:N_t] + b, -40.0, 40.0)))
        return {
            "r2": float(r2_score(ty, p)),
            "mae": float(mean_absolute_error(ty, p)),
            "rmse": float(np.sqrt(mean_squared_error(ty, p))),
            "pearson": float(pearsonr(p, ty)[0]),
            "spearman": float(spearmanr(p, ty)[0]),
        }

    m_dmpnn = eval_standalone(z_dmpnn_test, "DMPNN")
    m_gbdt = eval_standalone(z_gbdt_test, "Fused GBDT")
    m_ridge = eval_standalone(z_ridge_test, "ChemBERTa Ridge")

    print("\n" + "=" * 80)
    print("★ FINAL TRI-HYBRID FOUNDATION BENCHMARK RESULTS (PPBR AZ, Random 8:1:1) ★")
    print("=" * 80)
    print(f"{'Model Architecture':<40} | {'Test R2':<10} | {'MAE (%)':<9} | {'RMSE (%)':<10} | {'Pearson':<9} | {'Spearman':<9}")
    print("-" * 80)
    print(f"{'DMPNN Multi-Task Graph Model':<40} | {m_dmpnn['r2']:<10.4f} | {m_dmpnn['mae']:<9.2f}% | {m_dmpnn['rmse']:<10.2f}% | {m_dmpnn['pearson']:<9.4f} | {m_dmpnn['spearman']:<9.4f}")
    print(f"{'ChemBERTa-77M Ridge Projection':<40} | {m_ridge['r2']:<10.4f} | {m_ridge['mae']:<9.2f}% | {m_ridge['rmse']:<10.2f}% | {m_ridge['pearson']:<9.4f} | {m_ridge['spearman']:<9.4f}")
    print(f"{'Fused GBDT (FP+RDKit+Bio+ChemBERTa)':<40} | {m_gbdt['r2']:<10.4f} | {m_gbdt['mae']:<9.2f}% | {m_gbdt['rmse']:<10.2f}% | {m_gbdt['pearson']:<9.4f} | {m_gbdt['spearman']:<9.4f}")
    print("-" * 80)
    print(f"{'★ TRI-HYBRID FOUNDATION STACKER':<40} | {r2_tri:<10.4f} | {mae_tri:<9.2f}% | {rmse_tri:<10.2f}% | {pr_tri:<9.4f} | {sp_tri:<9.4f}")
    print(f"Optimal Stacking Weights: DMPNN={w1_opt:.3f}, Fused GBDT={w2_opt:.3f}, ChemBERTa Ridge={w3_opt:.3f}")
    print(f"Optimal Calibration: alpha={a_opt:.3f}, beta={b_opt:+.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
