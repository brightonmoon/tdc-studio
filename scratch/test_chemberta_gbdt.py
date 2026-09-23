"""Test Foundation Model (ChemBERTa-77M) Embedding Fusion with GBDT & Ridge on PPBR AZ."""

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
from transformers import AutoTokenizer, AutoModel
from tdc.single_pred import ADME

from scratch.test_biophysical_hypothesis import compute_biophysical_descriptors


def extract_chemberta_embeddings(smiles_list, model_name="DeepChem/ChemBERTa-77M-MTR", batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = [s if (s and isinstance(s, str)) else "C" for s in smiles_list[i : i + batch_size]]
        encoded = tokenizer(batch_smiles, padding=True, truncation=True, max_length=256, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state  # [B, L, 384]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).detach().cpu().numpy()
            all_embeddings.append(mean_pooled)

    return np.vstack(all_embeddings)


def extract_all_features(smiles_list, labels=None, n_bits=1024):
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


def calibrate_and_eval(z_val, y_val_real, z_test, y_test_real):
    def obj_cal(params):
        a, b = params
        p = 100.0 / (1.0 + np.exp(-np.clip(a * z_val + b, -40.0, 40.0)))
        return float(np.mean((p - y_val_real) ** 2))

    res = minimize(obj_cal, [1.0, 0.0], bounds=[(0.1, 3.0), (-2.0, 2.0)], method="L-BFGS-B")
    a, b = res.x
    p_test = 100.0 / (1.0 + np.exp(-np.clip(a * z_test + b, -40.0, 40.0)))
    r2 = float(r2_score(y_test_real, p_test))
    mae = float(mean_absolute_error(y_test_real, p_test))
    rmse = float(np.sqrt(mean_squared_error(y_test_real, p_test)))
    pr, _ = pearsonr(p_test, y_test_real)
    sp, _ = spearmanr(p_test, y_test_real)
    return {"r2": r2, "mae": mae, "rmse": rmse, "pearson": pr, "spearman": sp, "alpha": a, "beta": b}


def main():
    print("Loading TDC PPBR_AZ dataset...")
    data = ADME(name="PPBR_AZ")
    splits = data.get_split(method="random", seed=42)
    train_df, val_df, test_df = splits["train"], splits["valid"], splits["test"]

    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 1. Base + Biophysical Features
    print("Extracting Morgan FPs + RDKit Descriptors + Biophysical Motifs...")
    X_base_train, y_train_logit, y_train_real = extract_all_features(train_df["Drug"].tolist(), train_df["Y"].tolist())
    X_base_val, y_val_logit, y_val_real = extract_all_features(val_df["Drug"].tolist(), val_df["Y"].tolist())
    X_base_test, y_test_logit, y_test_real = extract_all_features(test_df["Drug"].tolist(), test_df["Y"].tolist())

    # 2. ChemBERTa Foundation Model Embeddings (Cached)
    cache_file = Path("data/cache/chemberta_ppbr_embeddings.npz")
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        print(f"Loading cached ChemBERTa embeddings from {cache_file}...")
        loaded = np.load(cache_file)
        emb_train = loaded["train"]
        emb_val = loaded["val"]
        emb_test = loaded["test"]
    else:
        print("Extracting ChemBERTa-77M-MTR 384-dimensional embeddings...")
        emb_train = extract_chemberta_embeddings(train_df["Drug"].tolist())
        emb_val = extract_chemberta_embeddings(val_df["Drug"].tolist())
        emb_test = extract_chemberta_embeddings(test_df["Drug"].tolist())
        np.savez_compressed(cache_file, train=emb_train, val=emb_val, test=emb_test)
        print(f"Saved ChemBERTa embeddings to {cache_file}")

    print(f"Embedding shape: {emb_train.shape}")

    # Combine Base Features + ChemBERTa
    X_fused_train = np.hstack([X_base_train, emb_train])
    X_fused_val = np.hstack([X_base_val, emb_val])
    X_fused_test = np.hstack([X_base_test, emb_test])

    print(f"Total Fused Dimension: {X_fused_train.shape[1]} (1024 FP + 210 RDKit + 14 Bio + 384 ChemBERTa)")

    # -------------------------------------------------------------
    # Evaluation 1: Baseline GBDT (FPs + RDKit)
    # -------------------------------------------------------------
    gbdt_base = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, l2_regularization=2.0, min_samples_leaf=15, random_state=42)
    gbdt_base.fit(X_base_train[:, :-14], y_train_logit)
    res_base = calibrate_and_eval(
        gbdt_base.predict(X_base_val[:, :-14]), y_val_real,
        gbdt_base.predict(X_base_test[:, :-14]), y_test_real
    )
    print("\n[1] Baseline GBDT (FPs + RDKit):")
    print(f"    R² = {res_base['r2']:.4f} | MAE = {res_base['mae']:.2f}% | Pearson = {res_base['pearson']:.4f} | Spearman = {res_base['spearman']:.4f}")

    # -------------------------------------------------------------
    # Evaluation 2: Biophysical GBDT (+ pH 7.4 & Sudlow)
    # -------------------------------------------------------------
    gbdt_bio = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, l2_regularization=2.0, min_samples_leaf=15, random_state=42)
    gbdt_bio.fit(X_base_train, y_train_logit)
    res_bio = calibrate_and_eval(
        gbdt_bio.predict(X_base_val), y_val_real,
        gbdt_bio.predict(X_base_test), y_test_real
    )
    print("\n[2] Biophysical GBDT (+ pH 7.4 & Sudlow):")
    print(f"    R² = {res_bio['r2']:.4f} | MAE = {res_bio['mae']:.2f}% | Pearson = {res_bio['pearson']:.4f} | Spearman = {res_bio['spearman']:.4f}")

    # -------------------------------------------------------------
    # Evaluation 3: Foundation Model Fused GBDT (+ ChemBERTa-77M)
    # -------------------------------------------------------------
    gbdt_fused = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.04, l2_regularization=3.0, min_samples_leaf=12, random_state=42)
    gbdt_fused.fit(X_fused_train, y_train_logit)
    res_fused = calibrate_and_eval(
        gbdt_fused.predict(X_fused_val), y_val_real,
        gbdt_fused.predict(X_fused_test), y_test_real
    )
    print("\n[3] Fused GBDT (FPs + RDKit + Biophysics + ChemBERTa-77M Foundation):")
    print(f"    R2 = {res_fused['r2']:.4f} (Delta R2 = {res_fused['r2'] - res_base['r2']:+.4f}) | MAE = {res_fused['mae']:.2f}% | Pearson = {res_fused['pearson']:.4f} | Spearman = {res_fused['spearman']:.4f}")

    # -------------------------------------------------------------
    # Evaluation 4: Dual-Branch Stacking (GBDT Trees + Ridge Latent Projection on ChemBERTa)
    # -------------------------------------------------------------
    scaler = StandardScaler()
    emb_tr_scaled = scaler.fit_transform(emb_train)
    emb_va_scaled = scaler.transform(emb_val)
    emb_te_scaled = scaler.transform(emb_test)

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20))
    ridge.fit(emb_tr_scaled, y_train_logit)
    z_ridge_val = ridge.predict(emb_va_scaled)
    z_ridge_test = ridge.predict(emb_te_scaled)

    # Blend GBDT + Ridge ChemBERTa
    z_gbdt_val = gbdt_fused.predict(X_fused_val)
    z_gbdt_test = gbdt_fused.predict(X_fused_test)

    def blend_obj(params):
        w, a, b = params
        w_c = np.clip(w, 0.0, 1.0)
        zb = w_c * z_gbdt_val + (1.0 - w_c) * z_ridge_val
        p = 100.0 / (1.0 + np.exp(-np.clip(a * zb + b, -40.0, 40.0)))
        return float(np.mean((p - y_val_real) ** 2))

    res_bl = minimize(blend_obj, [0.75, 1.0, 0.0], bounds=[(0.0, 1.0), (0.1, 3.0), (-2.0, 2.0)], method="L-BFGS-B")
    w_opt, a_opt, b_opt = res_bl.x
    zb_test = w_opt * z_gbdt_test + (1.0 - w_opt) * z_ridge_test
    p_blend_test = 100.0 / (1.0 + np.exp(-np.clip(a_opt * zb_test + b_opt, -40.0, 40.0)))

    bl_r2 = float(r2_score(y_test_real, p_blend_test))
    bl_mae = float(mean_absolute_error(y_test_real, p_blend_test))
    bl_rmse = float(np.sqrt(mean_squared_error(y_test_real, p_blend_test)))
    bl_pr, _ = pearsonr(p_blend_test, y_test_real)
    bl_sp, _ = spearmanr(p_blend_test, y_test_real)

    print("\n[4] Dual-Branch Foundation Stacker (Fused GBDT + ChemBERTa Ridge):")
    print(f"    Optimal Weights: GBDT={w_opt:.3f}, ChemBERTa Ridge={1-w_opt:.3f}")
    print(f"    Test R2 = {bl_r2:.4f} (Delta R2 = {bl_r2 - res_base['r2']:+.4f}) | Test MAE = {bl_mae:.2f}% | RMSE = {bl_rmse:.2f}% | Pearson = {bl_pr:.4f} | Spearman = {bl_sp:.4f}")


if __name__ == "__main__":
    main()
