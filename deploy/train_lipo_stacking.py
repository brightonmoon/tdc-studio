"""Task E-4: Lipophilicity AstraZeneca 24-dim Motif + GBDT + ChemBERTa Stacking.

Aims for Test R^2 >= 0.85 (Pearson r >= 0.93) on the official Bemis-Murcko scaffold benchmark.
"""

import argparse
import json
import logging
import os

import numpy as np
from tdc.single_pred import ADME

from tdc_studio.models.hybrid.lipo_stacker import LipophilicityStacker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_lipo_stacking")


def main():
    parser = argparse.ArgumentParser(description="Lipophilicity Stacking Training")
    parser.add_argument("--dry-run", action="store_true", help="Execute on subset")
    parser.add_argument("--export-dir", type=str, default="models/export/lipophilicity_stacker")
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "catboost", "histgbdt"],
        help="Tree model backend ('catboost', 'histgbdt', or 'auto')",
    )
    args = parser.parse_args()

    if "__file__" in globals() and __file__:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        repo_root = os.getcwd()
    if os.path.exists(repo_root):
        os.chdir(repo_root)

    os.makedirs(args.export_dir, exist_ok=True)

    print("\n[Lipo Pipeline] Loading Lipophilicity AstraZeneca dataset (4,200 compounds)...", flush=True)
    data = ADME(name="Lipophilicity_AstraZeneca")
    split = data.get_split(method="scaffold", seed=42)

    train_df = split["train"]
    val_df = split["valid"]
    test_df = split["test"]

    if args.dry_run:
        train_df = train_df.head(80)
        val_df = val_df.head(40)
        test_df = test_df.head(40)

    smiles_train = train_df["Drug"].tolist()
    y_train = np.asarray(train_df["Y"], dtype=np.float32)

    smiles_val = val_df["Drug"].tolist()
    y_val = np.asarray(val_df["Y"], dtype=np.float32)

    smiles_test = test_df["Drug"].tolist()
    y_test = np.asarray(test_df["Y"], dtype=np.float32)

    print(f"[Lipo Pipeline] Initializing LipophilicityStacker with 24-dim Biophysical Motifs and {args.model_type} backend...", flush=True)
    stacker = LipophilicityStacker(
        gbdt_params={
            "iterations": 500,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 2.5,
            "max_iter": 400,
            "l2_regularization": 1.5,
            "random_state": 42,
            "random_seed": 42,
        },
        use_chemberta=not args.dry_run,
        model_type=args.model_type,
    )

    # Optional Branch 3: Pretrained 4-Layer D-MPNN Predictions (from models/checkpoint_5tasks)
    dmpnn_chk = "models/checkpoint_5tasks/best_model.pt"
    if not os.path.exists(dmpnn_chk):
        for cand in [
            os.path.join(repo_root, dmpnn_chk),
            os.path.join("/content/tdc-studio", dmpnn_chk),
        ]:
            if os.path.exists(cand):
                dmpnn_chk = cand
                break

    gnn_val = None
    gnn_test = None

    if os.path.exists(dmpnn_chk):
        try:
            print(f"[Lipo Pipeline] Loading pretrained D-MPNN from {dmpnn_chk} for Tri-Hybrid Stacking...", flush=True)
            import torch
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            from sklearn.linear_model import RidgeCV
            from torch_geometric.data import Data

            from tdc_studio.cli import load_yaml
            from tdc_studio.core.registry import MODELS
            from tdc_studio.data.collate import molecule_collate_fn
            from tdc_studio.data.transforms import SmilesToGraphTransform

            cfg_path = "configs/config_distribution_mtl_5tasks.yaml"
            if not os.path.exists(cfg_path):
                for cand in [
                    os.path.join(repo_root, cfg_path),
                    os.path.join("/content/tdc-studio", cfg_path),
                ]:
                    if os.path.exists(cand):
                        cfg_path = cand
                        break

            cfg = load_yaml(cfg_path)
            model_cls = MODELS.get(cfg["model"]["type"])
            dmpnn_model = model_cls(cfg["model"])
            weights = torch.load(dmpnn_chk, map_location="cpu", weights_only=False)
            dmpnn_model.load_state_dict(weights)
            dmpnn_model.eval()

            g_trans = SmilesToGraphTransform()

            def extract_dmpnn_batch(smiles_l):
                preds = []
                batch_size = 64
                for i in range(0, len(smiles_l), batch_size):
                    batch_smiles = smiles_l[i : i + batch_size]
                    batch_items = []
                    for sm in batch_smiles:
                        g = g_trans(sm)
                        if g is None:
                            g = Data(
                                x=torch.zeros((1, 14)),
                                edge_index=torch.empty((2, 0), dtype=torch.long),
                                edge_attr=torch.empty((0, 6), dtype=torch.float),
                            )
                        mol = Chem.MolFromSmiles(sm) if sm else None
                        if mol is not None:
                            desc_dict = Descriptors.CalcMolDescriptors(mol)
                            desc_vals = [
                                0.0
                                if (v is None or np.isnan(v) or np.isinf(v))
                                else float(np.clip(v, -100.0, 100.0))
                                for v in desc_dict.values()
                            ][:210]
                            if len(desc_vals) < 210:
                                desc_vals = desc_vals + [0.0] * (210 - len(desc_vals))
                        else:
                            desc_vals = [0.0] * 210
                        batch_items.append({
                            "drug_graph": g,
                            "descriptors": torch.tensor(desc_vals, dtype=torch.float32),
                            "drug_smiles_str": sm,
                        })
                    collated = molecule_collate_fn(batch_items)
                    with torch.no_grad():
                        out = dmpnn_model(collated)
                        preds.extend(out[:, 3].cpu().numpy())
                return np.array(preds, dtype=np.float32)

            print("[Lipo Pipeline] Extracting D-MPNN representations for train, val, test splits...", flush=True)
            raw_trn = extract_dmpnn_batch(smiles_train)
            raw_val = extract_dmpnn_batch(smiles_val)
            raw_tst = extract_dmpnn_batch(smiles_test)

            calib = RidgeCV()
            calib.fit(raw_trn.reshape(-1, 1), y_train)
            gnn_val = calib.predict(raw_val.reshape(-1, 1))
            gnn_test = calib.predict(raw_tst.reshape(-1, 1))
            print("[Lipo Pipeline] Successfully calibrated D-MPNN Branch.", flush=True)
        except Exception as e:
            print(f"[Lipo Pipeline Warning] Could not load D-MPNN predictions ({e}), continuing with 2-Branch stacker.", flush=True)
            gnn_val = None
            gnn_test = None

    print(f"\n[Lipo Pipeline] Fitting stacker on {len(smiles_train)} training samples with validation calibration...", flush=True)
    stacker.fit(
        smiles_train=smiles_train,
        y_train=y_train,
        val_data=(smiles_val, y_val, gnn_val),
    )

    print(f"\n[Lipo Pipeline] Evaluating stacker on {len(smiles_test)} test samples...", flush=True)
    test_metrics = stacker.evaluate(
        smiles_test=smiles_test,
        y_test=y_test,
        gnn_preds_test=gnn_test,
    )

    print("=================================================================", flush=True)
    print("[SOTA] FINAL LIPOPHILICITY TEST METRICS (Scaffold Split):", flush=True)
    print(f"   R^2          : {test_metrics['r2']:.4f} (Target: >= 0.85)", flush=True)
    print(f"   Pearson r    : {test_metrics['pearson_r']:.4f} (Target: >= 0.93)", flush=True)
    print(f"   Spearman rho : {test_metrics['spearman_rho']:.4f}", flush=True)
    print(f"   MAE          : {test_metrics['mae']:.4f}", flush=True)
    print(f"   RMSE         : {test_metrics['rmse']:.4f}", flush=True)
    print(f"   Stack Weights: {test_metrics['weights']}", flush=True)
    print("=================================================================", flush=True)

    with open(os.path.join(args.export_dir, "lipophilicity_stacking_summary.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    import joblib
    save_path = os.path.join(args.export_dir, "lipo_stacker_model.joblib")
    joblib.dump(stacker, save_path)
    logger.info("Successfully saved Lipophilicity Stacker model to: %s", save_path)


if __name__ == "__main__":
    main()
