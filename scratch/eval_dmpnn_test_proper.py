"""Verify DMPNN Multi-Task model checkpoint predictions and calibration on validation & test sets."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
from torch_geometric.loader import DataLoader
from tdc.single_pred import ADME

from tdc_studio.models.graph.dmpnn import DMPNNModel
from tdc_studio.data.transforms import SmilesToGraphTransform, RDKit2DDescriptorsTransform
from tdc_studio.data.base import MolecularDataset


def main():
    print("=== Evaluating Multi-Task DMPNN Checkpoint (best_model.pt) ===")
    ckpt_path = Path("models/checkpoint/best_model.pt")
    config_path = Path("models/checkpoint/config.json")
    meta_path = Path("models/checkpoint/training_meta.json")

    with open(config_path) as f:
        config = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    print("Config:", config)
    print("Training Meta Best Epoch:", meta.get("best_epoch"), "Val R2:", meta.get("best_metric_value"))

    # Load Model
    model = DMPNNModel(config)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # Load Cached Cluster Dataset
    cache_path = Path("data/cache/distribution_mtl_ppbr_az_random_42_graph_True_True_dbea67ae.pt")
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    val_dataset = cached["val"]
    test_dataset = cached["test"]
    train_dataset = cached["train"]

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    def get_dataset_preds_and_labels(dataset, name="Set"):
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        all_preds = []
        all_labels = []
        all_masks = []
        all_smiles = []

        with torch.no_grad():
            for batch in loader:
                preds = model(batch)
                all_preds.append(preds[:, 0].cpu().numpy())
                all_labels.append(batch["labels"][:, 0].cpu().numpy())
                all_masks.append(batch["mask"][:, 0].cpu().numpy())
                all_smiles.extend(batch["drug_smiles_str"])

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        masks = np.concatenate(all_masks).astype(bool)

        valid_preds = preds[masks]
        valid_labels = labels[masks]
        valid_smiles = [s for s, m in zip(all_smiles, masks) if m]

        print(f"\n--- {name} Results ({len(valid_preds)} compounds with ppbr_az) ---")
        pr, _ = pearsonr(valid_preds, valid_labels)
        sp, _ = spearmanr(valid_preds, valid_labels)
        r2 = r2_score(valid_labels, valid_preds)
        mae = mean_absolute_error(valid_labels, valid_preds)
        print(f"Standardized Target Space: R2={r2:.4f}, MAE={mae:.4f}, Pearson={pr:.4f}, Spearman={sp:.4f}")
        return valid_smiles, valid_preds, valid_labels

    val_smiles, val_preds_std, val_labels_std = get_dataset_preds_and_labels(val_dataset, "Validation")
    test_smiles, test_preds_std, test_labels_std = get_dataset_preds_and_labels(test_dataset, "Test")


if __name__ == "__main__":
    main()
