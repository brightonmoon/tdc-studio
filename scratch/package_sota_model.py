"""Export and package SOTA PPBR Tri-Hybrid model for production serving.

Saves:
- models/export/ppbr_tri_hybrid_sota.pt
- models/export/model.pt
- models/export/config.json
- models/export/export_manifest.json
- models/export/training_meta.json

Validates FastAPI serving pipeline and latency.
"""

import json
import os
import time
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from fastapi.testclient import TestClient

from api import app
from tdc_studio.serving.exporter import export_tri_hybrid_package
from tdc_studio.serving.tri_hybrid_pipeline import (
    TriHybridInferencePipeline,
    load_tri_hybrid_from_package,
)


def main():
    print("=" * 80)
    print("=== PACKAGING SOTA TRI-HYBRID PPBR MODEL FOR PRODUCTION ===")
    print("=" * 80)

    # 1. Load cached features
    print("[1/5] Loading precomputed feature matrices...")
    trn = np.load("data/cache/features_ppbr_train_3d_bio.npz")
    val = np.load("data/cache/features_ppbr_val_3d_bio.npz")
    tst = np.load("data/cache/features_ppbr_test_3d_bio.npz")

    chemberta_loaded = np.load("data/cache/chemberta_ppbr_embeddings.npz")
    emb_trn = chemberta_loaded["train"]
    emb_val = chemberta_loaded["val"]
    emb_tst = chemberta_loaded["test"]

    X_train_fused = np.hstack([trn["X"], emb_trn])
    y_train_logit = trn["y_logit"]
    y_train_real = trn["y_real"]

    # 2. Train SOTA 3D-Biophysical GBDT with Low-Binding Step Reweighting
    print("[2/5] Training SOTA GBDT Regressor (1,642 features, 3D Conformer Steric + Low-Binding Weights)...")
    weights = np.ones(len(y_train_real), dtype=np.float32)
    weights[y_train_real < 70.0] = 2.4
    weights[(y_train_real >= 70.0) & (y_train_real < 85.0)] = 1.4

    gbdt_model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        l2_regularization=2.0,
        min_samples_leaf=15,
        random_state=42,
    )
    gbdt_model.fit(X_train_fused, y_train_logit, sample_weight=weights)

    # 3. Train ChemBERTa RidgeCV
    print("[3/5] Fitting ChemBERTa Foundation RidgeCV...")
    scaler = StandardScaler()
    emb_trn_sc = scaler.fit_transform(emb_trn)
    ridge_model = RidgeCV(alphas=np.logspace(-2, 3, 20))
    ridge_model.fit(emb_trn_sc, y_train_logit)

    # 4. Package SOTA Model Bundle
    print("[4/5] Packaging SOTA model artifacts to models/export/...")
    output_dir = "models/export"
    os.makedirs(output_dir, exist_ok=True)

    # Load SOTA summary parameters
    with open("models/export/ppbr_tri_hybrid_sota_summary.json") as f:
        meta_summary = json.load(f)

    opt_p = meta_summary["optimal_params_sota"]
    blending_weights = (opt_p["w_gnn"], opt_p["w_gbdt"], opt_p["w_chemberta"])
    calibration_params = (opt_p["alpha"], opt_p["beta"])

    dmpnn_ckpt_path = "models/checkpoint/best_model.pt"
    dmpnn_cfg_path = "models/checkpoint/config.json"
    dmpnn_stats = (3.238, 1.488)  # mu_trn, std_trn

    manifest = export_tri_hybrid_package(
        dmpnn_checkpoint_path=dmpnn_ckpt_path,
        dmpnn_config_path=dmpnn_cfg_path,
        gbdt_model=gbdt_model,
        ridge_model=ridge_model,
        scaler=scaler,
        blending_weights=blending_weights,
        calibration_params=calibration_params,
        dmpnn_stats=dmpnn_stats,
        output_dir=output_dir,
        benchmark_meta=meta_summary["benchmark_summary"],
    )

    print(f"  Artifacts saved to {output_dir}:")
    print(f"    - {manifest['weights_file']} (Full SOTA weights)")
    print(f"    - {manifest['config_file']}")
    print("    - export_manifest.json")
    print("    - training_meta.json")

    # 5. Validate Serving Pipeline and Latency
    print("[5/5] Validating FastAPI Serving & Inference Pipeline...")
    pipeline = load_tri_hybrid_from_package(output_dir, device="cpu")

    test_drugs = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("Warfarin", "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O"),
        ("Diazepam", "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13"),
        ("Propranolol", "CC(C)NCC(O)COc1cccc2ccccc12"),
    ]
    smiles_batch = [d[1] for d in test_drugs]

    t0 = time.perf_counter()
    preds = pipeline.predict(smiles_batch)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nInference Verification ({len(smiles_batch)} drugs in {elapsed_ms:.1f}ms):")
    detailed = pipeline.predict_detailed(smiles_batch)
    for (name, sm), d in zip(test_drugs, detailed):
        print(f"  - {name:<12} | Predicted PPBR: {d['ppbr_percent']:>5.2f}% | Tier: {d['decision']}")

    # 6. Test FastAPI Endpoint
    client = TestClient(app)
    resp = client.post("/predict", json={"smiles": smiles_batch})
    assert resp.status_code == 200, f"API failed: {resp.text}"
    api_preds = resp.json()["predictions"]
    assert len(api_preds) == len(smiles_batch)
    print(f"\nFastAPI Endpoint /predict verified successfully! Response: {api_preds}")
    print("=" * 80)
    print("=== SOTA PACKAGING & PIPELINE VERIFICATION COMPLETE ===")
    print("=" * 80)


if __name__ == "__main__":
    main()
