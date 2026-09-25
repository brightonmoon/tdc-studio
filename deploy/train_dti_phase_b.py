"""Training script for DTI / DTA Phase B Foundation Models on Google Colab GPU.

Dataset: BindingDB_Kd (Cold-Drug Split)
Target Encoders:
  - Drug: ChemBERTa-77M-MTR (384 -> 256 dim)
  - Target: facebook/esm2_t12_35M_UR50D (480 -> 256 dim) [fallback: t6_8M]
  - Head: BilinearAttentionFusion
Target Metrics:
  - Cold-Drug CI >= 0.70
  - Cold-Drug MSE <= 0.75
"""

# ruff: noqa: E402

import argparse
import os
import sys
import time
from pathlib import Path

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

import numpy as np
import torch
import yaml

# Import pretrained encoders to register in MODELS
import tdc_studio.models.dti.pretrained_encoders  # noqa: F401
from tdc_studio.data.multi_pred import DTADataModule
from tdc_studio.evaluation.evaluator import TherapeuticsEvaluator
from tdc_studio.models.dti.dta_model import GraphDTAModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train DTI Phase B Foundation Model on Colab")
    parser.add_argument("--config", type=str, default="configs/config_dti_phase_b.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output-dir", type=str, default="models/dti/phase_b", help="Dir to save best checkpoint")
    parser.add_argument("--esm-model", type=str, default=None, help="Override ESM-2 model name")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("  ★ TDC-Studio DTI Phase B: Foundation Models Training (Colab GPU)")
    print("=" * 70)

    # 1. Load Configuration
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Hardware: {gpu_name} ({total_mem:.1f} GB VRAM)")

    # Overrides
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 32)
    max_epochs = args.epochs or cfg.get("training", {}).get("max_epochs", 30)
    base_lr = args.lr or float(cfg.get("training", {}).get("learning_rate", 1.0e-4))
    weight_decay = float(cfg.get("training", {}).get("weight_decay", 1.0e-4))
    patience = int(cfg.get("training", {}).get("early_stopping_patience", 7))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.esm_model:
        cfg["model"]["target_encoder"]["model_name"] = args.esm_model

    # 2. DataModule Setup
    print("\n[Step 1] Loading TDC BindingDB_Kd dataset with Cold-Drug split...")
    dataset_cfg = cfg.get("dataset", {})
    data_module = DTADataModule(
        dataset_name=dataset_cfg.get("name", "BindingDB_Kd"),
        split_type=dataset_cfg.get("split", "cold_drug"),
        seed=dataset_cfg.get("seed", 42),
        log_transform=dataset_cfg.get("log_transform", True),
        aa_max_length=dataset_cfg.get("aa_max_length", 1024),
    )
    data_module.prepare_data()
    train_loader, val_loader, test_loader = data_module.setup_loaders(batch_size=batch_size)
    print(f"Dataset split sizes: {data_module.split_info}")
    print(f"Target scaler stats: mean={data_module.y_mean:.4f}, std={data_module.y_std:.4f}")

    # 3. Model Initialization
    print("\n[Step 2] Initializing GraphDTAModel with Pretrained Encoders...")
    model_cfg = cfg.get("model", {})
    model = GraphDTAModel(model_cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,} (Backbone initially frozen)")

    # 4. Optimizer & Scaler Setup
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    evaluator = TherapeuticsEvaluator(task_type="dta")

    # 5. Training Loop
    print("\n[Step 3] Starting Staged Training Loop...")
    best_val_ci = -1.0
    best_val_mse = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Staged Training: Unfreeze ChemBERTa top layers after epoch 2
        if epoch == 3 and hasattr(model.drug_encoder, "unfreeze"):
            print("  [Stage 2] Unfreezing ChemBERTa encoder for fine-tuning...")
            model.drug_encoder.unfreeze(last_n_layers=2)
            # Recreate optimizer to include newly unfrozen parameters with lower lr
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.drug_encoder.parameters(), "lr": 2.0e-5},
                    {"params": model.fusion.parameters(), "lr": base_lr},
                    {"params": model.target_encoder.proj.parameters(), "lr": base_lr},
                ],
                weight_decay=weight_decay,
            )

        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            dev_batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            targets = dev_batch["labels"].float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                preds = model(dev_batch)
                loss = model.compute_loss(preds, targets)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item())
            train_batches += 1

            if train_batches % 100 == 0 or train_batches == len(train_loader):
                print(
                    f"  [Epoch {epoch:02d} | Step {train_batches:04d}/{len(train_loader):04d}] "
                    f"Loss: {loss.item():.4f} | Running Avg: {train_loss_sum / train_batches:.4f}",
                    flush=True,
                )

        avg_train_loss = train_loss_sum / max(1, train_batches)
        scheduler.step()

        # Validation Epoch
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                dev_batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    preds = model(dev_batch).squeeze(-1)
                val_preds.extend(preds.cpu().numpy().tolist())
                val_targets.extend(dev_batch["labels"].cpu().numpy().tolist())

        val_metrics = evaluator.compute_all(np.array(val_preds), np.array(val_targets), task_type="dta")
        val_ci = val_metrics.get("ci", 0.0)
        val_mse = val_metrics.get("mse", 1.0)
        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val CI: {val_ci:.4f} | Val MSE: {val_mse:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Check for Best Checkpoint (Primary Metric: CI)
        if val_ci > best_val_ci:
            best_val_ci = val_ci
            best_val_mse = val_mse
            best_epoch = epoch
            patience_counter = 0

            checkpoint_file = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_ci": best_val_ci,
                    "val_mse": best_val_mse,
                    "config": cfg,
                },
                checkpoint_file,
            )
            print(f"  ★ Best model saved (Val CI: {best_val_ci:.4f}, Val MSE: {best_val_mse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement in Val CI for {patience} epochs.")
                break

    # 6. Test Set Final Evaluation (Cold-Drug Split)
    print("\n" + "=" * 70)
    print(f"  ★ Evaluating Best Checkpoint (Epoch {best_epoch}) on COLD-DRUG Test Split")
    print("=" * 70)

    best_checkpoint = output_dir / "best_model.pt"
    if best_checkpoint.is_file():
        ckpt = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from: {best_checkpoint}")

    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            dev_batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                preds = model(dev_batch).squeeze(-1)
            test_preds.extend(preds.cpu().numpy().tolist())
            test_targets.extend(dev_batch["labels"].cpu().numpy().tolist())

    test_metrics = evaluator.compute_all(np.array(test_preds), np.array(test_targets), task_type="dta")

    ci_val = test_metrics.get("ci", 0.0)
    mse_val = test_metrics.get("mse", 0.0)
    rmse_val = test_metrics.get("rmse", 0.0)
    pearson_val = test_metrics.get("pearson", 0.0)

    ci_pass = ci_val >= 0.70
    mse_pass = mse_val <= 0.75

    print("\n══════════════════════════════════════════════════════════════════════")
    print("  ★ Phase B Cold-Drug Benchmark Results Summary")
    print("══════════════════════════════════════════════════════════════════════")
    print(f"  • Primary Metric:   Concordance Index (CI) = {ci_val:.4f}  [Target >= 0.70] -> {'✓ PASSED' if ci_pass else '✗ FAILED'}")
    print(f"  • Secondary Metric: Mean Squared Error (MSE) = {mse_val:.4f} [Target <= 0.75] -> {'✓ PASSED' if mse_pass else '✗ FAILED'}")
    print(f"  • Secondary Metric: Root MSE (RMSE)          = {rmse_val:.4f}")
    print(f"  • Secondary Metric: Pearson Correlation (r)  = {pearson_val:.4f}")
    print("══════════════════════════════════════════════════════════════════════")

    summary_file = output_dir / "benchmark_summary.yaml"
    with open(summary_file, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "best_epoch": best_epoch,
                "test_metrics": test_metrics,
                "ci_passed": bool(ci_pass),
                "mse_passed": bool(mse_pass),
            },
            f,
            default_flow_style=False,
        )
    print(f"Summary saved to: {summary_file}")

    if not (ci_pass and mse_pass):
        print("\n[Warning] Benchmark target metrics were not fully met. Review hyperparameters.")
        return 1

    print("\n[Success] Phase B Benchmark Goals Successfully Achieved!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
