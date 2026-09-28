"""Task E-1: hERG 2-Stage Transfer Training (Karim 13.4k Pretrain -> Wang 648 Finetune).

Executes on Google Colab GPU (Account: munhyoungdo@gmail.com, Session: tdc-studio-safety)
or locally with --dry-run.
"""

import argparse
import json
import logging
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
from tdc.single_pred import Tox
from torch.utils.data import DataLoader

from tdc_studio.core.registry import MODELS
from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.transforms import SmilesToGraphTransform
from tdc_studio.models.loss.focal_loss import BinaryFocalLossWithLogits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_herg_2stage")


class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, labels, graph_transform, desc_transform=None):
        self.smiles = list(smiles_list)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.graph_transform = graph_transform
        self.desc_transform = desc_transform

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        s = self.smiles[idx]
        g = self.graph_transform(s)
        if g is None:
            from torch_geometric.data import Data
            g = Data(
                x=torch.zeros((1, 14), dtype=torch.float),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 6), dtype=torch.float),
            )
        g.y = torch.tensor([self.labels[idx]], dtype=torch.float)
        return {"drug_graph": g, "label": float(self.labels[idx])}


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            logits = model(batch)
            if logits.ndim > 1 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = batch["labels"].view(-1).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets)

    y_true = np.array(all_targets)
    y_prob = np.array(all_preds)
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "roc_auc": float(auc),
        "accuracy": float(acc),
        "balanced_acc": float(bacc),
        "f1": float(f1),
        "mcc": float(mcc),
    }


def main():
    parser = argparse.ArgumentParser(description="hERG 2-Stage Transfer Training")
    parser.add_argument("--dry-run", action="store_true", help="Execute 1 batch smoke test")
    parser.add_argument("--epochs-stage1", type=int, default=30, help="Stage 1 Karim epochs")
    parser.add_argument("--epochs-stage2", type=int, default=25, help="Stage 2 Wang epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr-stage1", type=float, default=0.001, help="Stage 1 LR")
    parser.add_argument("--lr-stage2", type=float, default=0.00005, help="Stage 2 LR")
    parser.add_argument("--export-dir", type=str, default="models/export/herg_champion")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    os.makedirs(args.export_dir, exist_ok=True)

    transform = SmilesToGraphTransform()

    # =========================================================================
    # Stage 1: Pre-training on hERG_Karim (13.4k compounds)
    # =========================================================================
    logger.info("=== Stage 1: Loading hERG_Karim (13,445 compounds) ===")
    karim_data = Tox(name="hERG_Karim")
    karim_split = karim_data.get_split(method="scaffold", seed=42)

    train_df = karim_split["train"]
    val_df = karim_split["valid"]
    test_df = karim_split["test"]

    if args.dry_run:
        train_df = train_df.head(64)
        val_df = val_df.head(32)
        test_df = test_df.head(32)
        args.epochs_stage1 = 1
        args.epochs_stage2 = 1

    train_ds = MoleculeDataset(train_df["Drug"], train_df["Y"], transform)
    val_ds = MoleculeDataset(val_df["Drug"], val_df["Y"], transform)
    test_ds = MoleculeDataset(test_df["Drug"], test_df["Y"], transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=molecule_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=molecule_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=molecule_collate_fn)

    # Initialize D-MPNN model
    model_cfg = {
        "type": "dmpnn",
        "in_dim": 14,
        "edge_dim": 6,
        "hidden_dim": 512,
        "num_layers": 4,
        "dropout": 0.20,
        "use_descriptors": False,
    }
    model_cls = MODELS.get("dmpnn")
    model = model_cls(model_cfg).to(device)

    loss_fn = BinaryFocalLossWithLogits(gamma=1.5, alpha=0.65)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_stage1, weight_decay=1e-4)

    best_val_auc = 0.0
    stage1_pt = os.path.join(args.export_dir, "herg_karim_pretrained.pt")

    logger.info("Starting Stage 1 training for %d epochs...", args.epochs_stage1)
    for epoch in range(1, args.epochs_stage1 + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate_model(model, val_loader, device)
        logger.info("[Stage 1 Epoch %02d] Loss: %.4f | Val AUC: %.4f, Acc: %.4f",
                    epoch, total_loss / max(1, len(train_loader)), val_metrics["roc_auc"], val_metrics["accuracy"])

        if val_metrics["roc_auc"] >= best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            torch.save(model.state_dict(), stage1_pt)

    # Load best Stage 1 weights and evaluate on Karim test set
    if os.path.exists(stage1_pt):
        model.load_state_dict(torch.load(stage1_pt, map_location=device))
    karim_test_metrics = evaluate_model(model, test_loader, device)
    logger.info(">>> Stage 1 (Karim) Test Metrics: AUROC = %.4f, ACC = %.4f, MCC = %.4f <<<",
                karim_test_metrics["roc_auc"], karim_test_metrics["accuracy"], karim_test_metrics["mcc"])

    # =========================================================================
    # Stage 2: Staged Fine-tuning on hERG (Wang et al., 648 compounds)
    # =========================================================================
    logger.info("=== Stage 2: Transferring to hERG Wang et al. (648 compounds) ===")
    wang_data = Tox(name="hERG")
    wang_split = wang_data.get_split(method="scaffold", seed=42)

    w_train_df = wang_split["train"]
    w_val_df = wang_split["valid"]
    w_test_df = wang_split["test"]

    if args.dry_run:
        w_train_df = w_train_df.head(32)
        w_val_df = w_val_df.head(16)
        w_test_df = w_test_df.head(16)

    w_train_ds = MoleculeDataset(w_train_df["Drug"], w_train_df["Y"], transform)
    w_val_ds = MoleculeDataset(w_val_df["Drug"], w_val_df["Y"], transform)
    w_test_ds = MoleculeDataset(w_test_df["Drug"], w_test_df["Y"], transform)

    w_train_loader = DataLoader(w_train_ds, batch_size=32, shuffle=True, collate_fn=molecule_collate_fn)
    w_val_loader = DataLoader(w_val_ds, batch_size=32, shuffle=False, collate_fn=molecule_collate_fn)
    w_test_loader = DataLoader(w_test_ds, batch_size=32, shuffle=False, collate_fn=molecule_collate_fn)

    # Phase 2A: Freeze GNN backbone for first 5 epochs
    logger.info("Phase 2A: Freezing GNN backbone, fine-tuning classification head only...")
    for name, param in model.named_parameters():
        if "out" not in name and "mlp" not in name and "head" not in name:
            param.requires_grad = False

    optimizer_head = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

    for epoch in range(1, min(6, args.epochs_stage2 + 1)):
        model.train()
        for batch in w_train_loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            optimizer_head.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            optimizer_head.step()

    # Phase 2B: Unfreeze all parameters and train with gentle LR
    logger.info("Phase 2B: Unfreezing full network with lr = %.6f...", args.lr_stage2)
    for param in model.parameters():
        param.requires_grad = True

    optimizer_full = torch.optim.AdamW(model.parameters(), lr=args.lr_stage2, weight_decay=1e-4)
    best_wang_auc = 0.0
    stage2_pt = os.path.join(args.export_dir, "herg_wang_finetuned.pt")

    for epoch in range(6, args.epochs_stage2 + 1):
        model.train()
        for batch in w_train_loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            optimizer_full.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            optimizer_full.step()

        w_val_metrics = evaluate_model(model, w_val_loader, device)
        if w_val_metrics["roc_auc"] >= best_wang_auc:
            best_wang_auc = w_val_metrics["roc_auc"]
            torch.save(model.state_dict(), stage2_pt)

    if os.path.exists(stage2_pt):
        model.load_state_dict(torch.load(stage2_pt, map_location=device))
    wang_test_metrics = evaluate_model(model, w_test_loader, device)

    logger.info("=================================================================")
    logger.info("🏆 FINAL hERG WANG TEST METRICS (Scaffold Split):")
    logger.info("   AUROC        : %.4f (Target: >= 0.88 ~ 0.90)", wang_test_metrics["roc_auc"])
    logger.info("   Accuracy     : %.4f", wang_test_metrics["accuracy"])
    logger.info("   Balanced ACC : %.4f", wang_test_metrics["balanced_acc"])
    logger.info("   F1-Score     : %.4f", wang_test_metrics["f1"])
    logger.info("   MCC          : %.4f", wang_test_metrics["mcc"])
    logger.info("=================================================================")

    # Save summary json
    summary = {
        "stage1_karim": karim_test_metrics,
        "stage2_wang": wang_test_metrics,
        "model_config": model_cfg,
    }
    with open(os.path.join(args.export_dir, "herg_2stage_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
