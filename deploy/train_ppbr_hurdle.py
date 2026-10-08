"""Deployable Training & Evaluation Script for PPBR SOTA Two-Stage Hurdle D-MPNN.

Architecture:
- Two-Stage Internalized Hurdle D-MPNN (Directed Message Passing on Bonds)
- 210 RDKit Physico-chemical 2D Descriptors
- Internalized Gating Classifier Head (High-binding detection >= 90%)
- High-binding Specialist Regressor (>= 85%)
- Low/Mid-binding Specialist Regressor (< 85%, weighted for rare < 70% samples)
- Smooth Gated Mixture Prediction: p_gate * pred_high + (1 - p_gate) * pred_low

Data:
- ChEMBL Augmented Training Set: data/external/ppbr_augmented_train_combined.csv
  (1,130 TDC Train + 1,214 Low-Binding ChEMBL compounds strictly disjoint from Val/Test)
- Official TDC Benchmark Split: 161 Validation, 323 Test compounds (zero data leakage)

Execution:
- Google Colab GPU via: powershell -ExecutionPolicy Bypass -File .\\scripts\\colab_exec.ps1 -Session tdc-studio-admet -FilePath deploy\\train_ppbr_hurdle.py
- Local Smoke / Dry Run via: python deploy/train_ppbr_hurdle.py --dry-run
"""

# ruff: noqa: E402

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Extract bundle if executed via colab exec on remote VM
_bundle_b64 = globals().get("BUNDLE_B64", None)
if _bundle_b64:
    import base64
    import io
    import zipfile

    print("[BUNDLE] Extracting local workspace bundle...")
    data = base64.b64decode(_bundle_b64)
    workspace = os.path.abspath("tdc-studio")
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        zf.extractall(workspace)
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    os.chdir(workspace)
    print(f"[BUNDLE] Unpacked workspace at: {workspace}")
else:
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from tdc.single_pred import ADME
from torch.utils.data import DataLoader, Dataset
import yaml

from concurrent.futures import ThreadPoolExecutor
from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.transforms import RDKit2DDescriptorsTransform, SmilesToGraphTransform
from tdc_studio.features.boltzmann_conformers import (
    FEATURE_NAMES as BOLTZMANN_FEATURE_NAMES,
    compute_boltzmann_conformer_features,
)
from tdc_studio.models.graph.dmpnn_hurdle import DMPNNHurdleModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_ppbr_hurdle")


def build_boltzmann_cache(
    smiles_list: List[str],
    cache_path: Path = Path("data/cache/boltzmann_3d_features.json"),
    max_workers: int = 6,
) -> Dict[str, List[float]]:
    """Build or load precomputed Boltzmann 3D conformer ensemble features."""
    cache: Dict[str, List[float]] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info(
                "Loaded %d cached Boltzmann 3D conformer records from %s",
                len(cache),
                cache_path.name,
            )
        except Exception as e:
            logger.warning("Failed to load Boltzmann cache: %s. Recomputing.", e)

    missing = [s for s in set(smiles_list) if s not in cache]
    if missing:
        logger.info(
            "Extracting Boltzmann 3D features for %d compounds using %d threads...",
            len(missing),
            max_workers,
        )
        t0 = time.time()

        def extract_one(s: str) -> Tuple[str, List[float]]:
            try:
                feat = compute_boltzmann_conformer_features(s, num_confs=5)
                return s, [float(feat[k]) for k in BOLTZMANN_FEATURE_NAMES]
            except Exception:
                return s, [0.0] * len(BOLTZMANN_FEATURE_NAMES)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for s, vec in executor.map(extract_one, missing):
                cache[s] = vec

        logger.info("Completed Boltzmann 3D feature extraction in %.2f seconds", time.time() - t0)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            logger.info("Saved updated Boltzmann cache to %s", cache_path.name)
        except Exception as e:
            logger.warning("Failed to save Boltzmann cache: %s", e)

    return cache


class PPBRDataset(Dataset):
    """In-memory cached Dataset for PPBR molecular graphs, 2D and 3D Boltzmann descriptors."""

    def __init__(
        self,
        smiles_list: List[str],
        targets: List[float],
        graph_transform: SmilesToGraphTransform,
        desc_transform: RDKit2DDescriptorsTransform,
        desc_dim: int = 210,
        boltzmann_cache: Optional[Dict[str, List[float]]] = None,
    ):
        self.samples = []
        valid_count = 0
        b_dim = 8 if boltzmann_cache is not None else 0
        base_desc_dim = desc_dim - b_dim

        for s, y in zip(smiles_list, targets):
            g = graph_transform(s)
            d = desc_transform(s)
            if g is None:
                continue

            if d is None:
                d = torch.zeros(base_desc_dim, dtype=torch.float32)

            if boltzmann_cache is not None:
                b_feats = boltzmann_cache.get(s, [0.0] * 8)
                b_tensor = torch.tensor(b_feats, dtype=torch.float32)
                d = torch.cat([d, b_tensor], dim=-1)

            self.samples.append(
                {
                    "drug_graph": g,
                    "descriptors": d,
                    "label": float(y),
                    "smiles": s,
                }
            )
            valid_count += 1

        logger.info("Successfully constructed %d/%d valid samples", valid_count, len(smiles_list))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, p_gate: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics for regression and gating classification."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    p_gate = np.asarray(p_gate, dtype=np.float64)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    if len(y_true) > 1 and np.std(y_pred) > 1e-8:
        pr, _ = pearsonr(y_true, y_pred)
        sr, _ = spearmanr(y_true, y_pred)
    else:
        pr, sr = 0.0, 0.0

    # Gate classification metrics (threshold: y >= 90.0%)
    gate_true = (y_true >= 90.0).astype(int)
    if len(np.unique(gate_true)) > 1:
        gate_auc = float(roc_auc_score(gate_true, p_gate))
    else:
        gate_auc = 0.5
    gate_acc = float(accuracy_score(gate_true, (p_gate >= 0.5).astype(int)))

    # Stratified MAE subsets
    high_mask = y_true >= 85.0
    high_mae = (
        float(mean_absolute_error(y_true[high_mask], y_pred[high_mask]))
        if high_mask.sum() > 0
        else 0.0
    )

    low_mask = y_true < 70.0
    low_mae = (
        float(mean_absolute_error(y_true[low_mask], y_pred[low_mask]))
        if low_mask.sum() > 0
        else 0.0
    )

    mid_mask = (y_true >= 70.0) & (y_true < 85.0)
    mid_mae = (
        float(mean_absolute_error(y_true[mid_mask], y_pred[mid_mask]))
        if mid_mask.sum() > 0
        else 0.0
    )

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "pearson_r": round(float(pr), 4),
        "spearman_rho": round(float(sr), 4),
        "gate_auc": round(gate_auc, 4),
        "gate_acc": round(gate_acc, 4),
        "high_binding_mae": round(high_mae, 4),
        "low_binding_mae": round(low_mae, 4),
        "mid_binding_mae": round(mid_mae, 4),
    }


def evaluate(model: DMPNNHurdleModel, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate model on a DataLoader using mixture prediction and gating probabilities."""
    model.eval()
    all_targets = []
    all_pred_mixture = []
    all_p_gate = []
    all_pred_high = []
    all_pred_low = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            out = model(batch, return_dict=True)

            all_pred_mixture.extend(out["pred_mixture"].squeeze(-1).cpu().numpy())
            all_p_gate.extend(out["gate_prob"].squeeze(-1).cpu().numpy())
            all_pred_high.extend(out["pred_high"].squeeze(-1).cpu().numpy())
            all_pred_low.extend(out["pred_low"].squeeze(-1).cpu().numpy())
            all_targets.extend(batch["labels"].view(-1).cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_pred_mixture)
    p_gate = np.array(all_p_gate)

    metrics = compute_metrics(y_true, y_pred, p_gate)
    preds_dict = {
        "y_true": y_true,
        "y_pred": y_pred,
        "p_gate": p_gate,
        "pred_high": np.array(all_pred_high),
        "pred_low": np.array(all_pred_low),
    }
    return metrics, preds_dict


def parse_args():
    parser = argparse.ArgumentParser(description="PPBR SOTA Hurdle D-MPNN Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_ppbr_dmpnn_hurdle.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override maximum epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/export/ppbr_dmpnn_hurdle",
        help="Directory to save model checkpoint and metrics summary",
    )
    parser.add_argument(
        "--use-augmented",
        dest="use_augmented",
        action="store_true",
        default=True,
        help="Use ChEMBL augmented training set",
    )
    parser.add_argument(
        "--no-augmented",
        dest="use_augmented",
        action="store_false",
        help="Use standard unaugmented TDC training set only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute fast smoke test (1 epoch on 32 samples)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--use-boltzmann-3d",
        dest="use_boltzmann_3d",
        action="store_true",
        default=True,
        help="Incorporate Boltzmann 3D steric ensemble descriptors",
    )
    parser.add_argument(
        "--no-boltzmann-3d",
        dest="use_boltzmann_3d",
        action="store_false",
        help="Disable Boltzmann 3D steric ensemble descriptors",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device ('cuda', 'cpu', or auto-detect)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("  ★ TDC-Studio: PPBR SOTA Two-Stage Hurdle D-MPNN Multi-Task Training")
    print("=" * 80)

    # 1. Load Configuration
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Setup Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Using device: %s", device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("GPU Hardware: %s (%.2f GB VRAM)", gpu_name, vram_gb)

    # 3. Hyperparameters
    batch_size = args.batch_size or int(cfg.get("data", {}).get("batch_size", 32))
    max_epochs = args.epochs or int(cfg.get("max_epochs", 60))
    learning_rate = args.lr or float(cfg.get("lr", 4.0e-4))
    min_lr = float(cfg.get("min_lr", 1.0e-6))
    weight_decay = float(cfg.get("weight_decay", 1.0e-5))
    early_stopping_patience = int(cfg.get("early_stopping", 20))

    if args.dry_run:
        logger.info("[DRY-RUN] Setting max_epochs=1, batch_size=16 for smoke testing")
        max_epochs = 1
        batch_size = 16

    # 4. Load Data
    logger.info("Loading official TDC PPBR_AZ dataset...")
    tdc_data = ADME(name="PPBR_AZ", path="data")
    split_method = cfg.get("data", {}).get("split_type", "random")
    seed = int(cfg.get("data", {}).get("seed", 42))
    splits = tdc_data.get_split(method=split_method, seed=seed)

    val_df = splits["valid"].copy()
    test_df = splits["test"].copy()

    # Determine Training Set
    aug_path_str = cfg.get("data", {}).get("augmented_data_path", "data/external/ppbr_augmented_train_combined.csv")
    aug_path = Path(aug_path_str)

    if args.use_augmented and aug_path.exists():
        logger.info("★ Loading augmented training dataset from: %s", aug_path.resolve())
        train_df = pd.read_csv(aug_path)
    else:
        logger.info("Loading default TDC PPBR_AZ training dataset (%d samples)", len(splits["train"]))
        train_df = splits["train"].copy()

    if args.dry_run:
        train_df = train_df.head(32)
        val_df = val_df.head(16)
        test_df = test_df.head(16)

    logger.info("Dataset sizes -> Train: %d | Val: %d | Test: %d", len(train_df), len(val_df), len(test_df))

    # 5. Extract Molecular Representations & Build Datasets
    logger.info("Initializing Graph and RDKit Descriptors transforms...")
    graph_transform = SmilesToGraphTransform(extended=False)
    desc_transform = RDKit2DDescriptorsTransform(fill_na=0.0)
    sample_desc = desc_transform("CC")
    base_desc_dim = (
        len(sample_desc)
        if sample_desc is not None
        else int(cfg.get("model", {}).get("descriptor_dim", 210))
    )

    boltzmann_cache = None
    if args.use_boltzmann_3d:
        all_smiles = list(
            set(train_df["Drug"].tolist() + val_df["Drug"].tolist() + test_df["Drug"].tolist())
        )
        boltzmann_cache = build_boltzmann_cache(all_smiles)
        total_desc_dim = base_desc_dim + 8
        logger.info("Integrated 8-dim Boltzmann 3D steric ensemble descriptors")
    else:
        total_desc_dim = base_desc_dim

    logger.info("Adaptive Total Descriptor Dimension (2D + 3D Boltzmann): %d", total_desc_dim)

    t0 = time.time()
    logger.info("Building Training Dataset...")
    train_dataset = PPBRDataset(
        smiles_list=train_df["Drug"].tolist(),
        targets=train_df["Y"].tolist(),
        graph_transform=graph_transform,
        desc_transform=desc_transform,
        desc_dim=total_desc_dim,
        boltzmann_cache=boltzmann_cache,
    )

    logger.info("Building Validation Dataset...")
    val_dataset = PPBRDataset(
        smiles_list=val_df["Drug"].tolist(),
        targets=val_df["Y"].tolist(),
        graph_transform=graph_transform,
        desc_transform=desc_transform,
        desc_dim=total_desc_dim,
        boltzmann_cache=boltzmann_cache,
    )

    logger.info("Building Test Dataset...")
    test_dataset = PPBRDataset(
        smiles_list=test_df["Drug"].tolist(),
        targets=test_df["Y"].tolist(),
        graph_transform=graph_transform,
        desc_transform=desc_transform,
        desc_dim=total_desc_dim,
        boltzmann_cache=boltzmann_cache,
    )
    logger.info("Feature extraction completed in %.2f seconds", time.time() - t0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=molecule_collate_fn,
        drop_last=(len(train_dataset) > batch_size),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=molecule_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=molecule_collate_fn,
    )

    # 6. Instantiate Model, Optimizer, and Scheduler
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["descriptor_dim"] = total_desc_dim
    model = DMPNNHurdleModel(model_cfg).to(device)
    logger.info(
        "Instantiated DMPNNHurdleModel: depth=%s, hidden_dim=%s, descriptor_dim=%s",
        model_cfg.get("depth", 3),
        model_cfg.get("hidden_dim", 300),
        total_desc_dim,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=min_lr
    )

    # 7. Setup Experiment Tracking (W&B)
    wandb_run = None
    tracking_cfg = cfg.get("tracking", {})
    if tracking_cfg.get("enabled", False) and not args.no_wandb and not args.dry_run:
        try:
            import wandb

            project_name = tracking_cfg.get("project", "tdc-learning")
            entity_name = tracking_cfg.get("entity", "tdc-studio")
            wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
            wandb_run = wandb.init(
                project=project_name,
                entity=entity_name,
                name="ppbr_dmpnn_hurdle_sota",
                mode=wandb_mode,
                config={
                    "model_config": model_cfg,
                    "train_size": len(train_dataset),
                    "val_size": len(val_dataset),
                    "test_size": len(test_dataset),
                    "max_epochs": max_epochs,
                    "batch_size": batch_size,
                    "lr": learning_rate,
                    "use_augmented": args.use_augmented,
                },
            )
            logger.info("W&B experiment tracking initialized (project: %s, entity: %s)", project_name, entity_name)
        except Exception as e:
            logger.warning("W&B initialization skipped: %s. Continuing with local logging.", e)

    # 8. Training Loop
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_mae = float("inf")
    best_metrics = {}
    patience_counter = 0

    logger.info("Starting training loop for %d epochs...", max_epochs)
    train_start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        gate_losses = []
        high_losses = []
        low_losses = []
        mix_losses = []

        for batch in train_loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            targets = batch["labels"].view(-1)

            optimizer.zero_grad()
            out = model(batch, return_dict=True)
            loss, loss_metrics = model.loss_fn(out, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            gate_losses.append(loss_metrics.get("loss_gate", 0.0))
            high_losses.append(loss_metrics.get("loss_high", 0.0))
            low_losses.append(loss_metrics.get("loss_low", 0.0))
            mix_losses.append(loss_metrics.get("loss_mixture", 0.0))

        avg_train_loss = total_loss / max(1, len(train_loader))
        avg_gate_loss = np.mean(gate_losses)
        avg_high_loss = np.mean(high_losses)
        avg_low_loss = np.mean(low_losses)
        avg_mix_loss = np.mean(mix_losses)

        # Validation Step
        val_metrics, _ = evaluate(model, val_loader, device)
        val_mae = val_metrics["mae"]
        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %02d/%02d | TrLoss: %.4f (G:%.3f H:%.3f L:%.3f Mix:%.3f) | Val MAE: %.4f | Val R2: %.4f | Val Pr: %.4f | Gate AUC: %.4f | LR: %.2e",
            epoch,
            max_epochs,
            avg_train_loss,
            avg_gate_loss,
            avg_high_loss,
            avg_low_loss,
            avg_mix_loss,
            val_mae,
            val_metrics["r2"],
            val_metrics["pearson_r"],
            val_metrics["gate_auc"],
            current_lr,
        )

        if wandb_run:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss_total": avg_train_loss,
                    "train/loss_gate": avg_gate_loss,
                    "train/loss_high": avg_high_loss,
                    "train/loss_low": avg_low_loss,
                    "train/loss_mixture": avg_mix_loss,
                    "val/mae": val_mae,
                    "val/rmse": val_metrics["rmse"],
                    "val/r2": val_metrics["r2"],
                    "val/pearson_r": val_metrics["pearson_r"],
                    "val/spearman_rho": val_metrics["spearman_rho"],
                    "val/gate_auc": val_metrics["gate_auc"],
                    "val/gate_acc": val_metrics["gate_acc"],
                    "val/high_mae": val_metrics["high_binding_mae"],
                    "val/low_mae": val_metrics["low_binding_mae"],
                    "lr": current_lr,
                }
            )

        # Save Best Checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_metrics = dict(val_metrics)
            patience_counter = 0

            best_pt_path = out_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config": model_cfg,
                    "best_val_metrics": best_metrics,
                },
                best_pt_path,
            )
            logger.info("  -> [BEST CHECKPOINT] Saved to %s (Val MAE: %.4f)", best_pt_path.name, val_mae)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", early_stopping_patience)
                break

    logger.info("Training finished in %.2f seconds.", time.time() - train_start_time)

    # 9. Final Benchmark Evaluation on Official Test Set
    best_pt_path = out_dir / "best_model.pt"
    if best_pt_path.exists():
        logger.info("Loading best checkpoint for final evaluation...")
        checkpoint = torch.load(best_pt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_preds = evaluate(model, test_loader, device)

    print("\n" + "=" * 80)
    print("🏆 FINAL PPBR BENCHMARK TEST METRICS (Official TDC Benchmark Split, N=323):")
    print(f"   ★ Test MAE          : {test_metrics['mae']:.4f} (Primary TDC Metric)")
    print(f"   ★ Test R^2          : {test_metrics['r2']:.4f} (Target SOTA >= 0.600)")
    print(f"   ★ Pearson r         : {test_metrics['pearson_r']:.4f} (Target >= 0.78)")
    print(f"   ★ Spearman rho      : {test_metrics['spearman_rho']:.4f}")
    print(f"   ★ Gate AUC          : {test_metrics['gate_auc']:.4f} (Separating >=90% extreme binding)")
    print(f"   ★ Gate Accuracy     : {test_metrics['gate_acc'] * 100:.2f}%")
    print(f"   ★ High-Binding MAE  : {test_metrics['high_binding_mae']:.4f} (Subset >= 85%)")
    print(f"   ★ Low-Binding MAE   : {test_metrics['low_binding_mae']:.4f} (Subset < 70%)")
    print(f"   ★ Mid-Binding MAE   : {test_metrics['mid_binding_mae']:.4f} (Subset 70-85%)")
    print("=" * 80 + "\n")

    # 10. Export Benchmark Summary and Artifacts
    summary = {
        "dataset": "PPBR_AZ",
        "model_architecture": "DMPNNHurdleModel",
        "training_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "augmented_training": args.use_augmented,
        "best_val_metrics": best_metrics,
        "final_test_metrics": test_metrics,
    }

    summary_path = out_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Benchmark summary exported to: %s", summary_path.resolve())

    if wandb_run:
        wandb.summary.update({f"test/{k}": v for k, v in test_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
