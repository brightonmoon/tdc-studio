"""Fast Test-Only Evaluation of ACC, MCC, AUROC for Cluster 3."""

import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from tdc_studio.cli import _batch_to_device, load_yaml
from tdc_studio.data.admet_cluster import ADMETClusterDataModule
from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.models.graph.dmpnn import DMPNNModel


def evaluate_test_only(config_path: str, checkpoint_path: str):
    cfg = load_yaml(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    print("\n=======================================================")
    print(f"Fast Evaluating: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=======================================================")

    data_params = {k: v for k, v in data_cfg.items() if k not in ("type", "name", "batch_size")}
    data_module = ADMETClusterDataModule(**data_params)
    data_module.prepare_data()

    # CRITICAL: Build ONLY test dataset (avoiding hours of featurizing 10,000+ train molecules)
    test_df = data_module.splits["test"]
    print(f"Test split compound count: {len(test_df)}")
    test_dataset = data_module._build_dataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=molecule_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg["task_type"] = "multi_task"
    model = DMPNNModel(model_cfg).to(device)

    chk = torch.load(checkpoint_path, map_location=device)
    state_dict = chk.get("state_dict", chk) if isinstance(chk, dict) else chk
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_labels, all_masks = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            dev_batch = _batch_to_device(batch, device)
            preds = model(dev_batch)
            all_preds.append(preds.detach().cpu())
            all_labels.append(dev_batch["labels"].detach().cpu())
            all_masks.append(dev_batch["mask"].detach().cpu())

    cat_preds = torch.cat(all_preds, dim=0).numpy()
    cat_labels = torch.cat(all_labels, dim=0).numpy()
    cat_masks = torch.cat(all_masks, dim=0).numpy()

    task_names = data_module.task_names
    results = {}

    for i, t_name in enumerate(task_names):
        mask = cat_masks[:, i]
        y_true = cat_labels[mask, i]
        y_prob = cat_preds[mask, i]

        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            continue

        y_pred = (y_prob >= 0.5).astype(int)

        auc = float(roc_auc_score(y_true, y_prob))
        acc = float(accuracy_score(y_true, y_pred))
        bacc = float(balanced_accuracy_score(y_true, y_pred))
        mcc = float(matthews_corrcoef(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        results[t_name] = {
            "num_test_samples": int(len(y_true)),
            "pos_ratio": round(float(np.mean(y_true)), 4),
            "roc_auc": round(auc, 4),
            "acc": round(acc, 4),
            "balanced_acc": round(bacc, 4),
            "mcc": round(mcc, 4),
            "f1": round(f1, 4),
        }
        print(f"Task: {t_name:38s} | N={len(y_true):4d} | AUC={auc:.4f} | ACC={acc:.4f} | B-ACC={bacc:.4f} | MCC={mcc:.4f} | F1={f1:.4f}")

    return results


if __name__ == "__main__":
    out = {}
    s1_cfg = "configs/config_cyp450_mtl.yaml"
    s1_ckpt = "models/checkpoint_cyp450_stage1/best_model.pt"
    if os.path.exists(s1_ckpt):
        out["stage1_joint"] = evaluate_test_only(s1_cfg, s1_ckpt)

    s2_cfg = "configs/config_cyp450_stage2_substrates.yaml"
    s2_ckpt = "models/checkpoint_cyp450_stage2/best_model.pt"
    if os.path.exists(s2_ckpt):
        out["stage2_substrates"] = evaluate_test_only(s2_cfg, s2_ckpt)

    with open("scratch/cluster3_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n[SUCCESS] Summary written to scratch/cluster3_metrics_summary.json")
