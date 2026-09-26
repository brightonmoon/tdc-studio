"""Training pipeline for DTI Phase C: Cross-Attention Fusion & ChemBERTa Fine-Tuning.

Dataset: BindingDB_Kd (Cold-Drug Split)
Model Architecture:
  - Drug Encoder: ChemBERTa-77M-MTR (Staged Top-N Layer Unfreezing)
  - Target Encoder: facebook/esm2_t12_35M_UR50D (Frozen Backbone + In-Memory Cache)
  - Fusion Head: CrossAttentionFusion (Bidirectional Multi-Head Cross-Attention)
Target Metrics:
  - Cold-Drug CI >= 0.76 (Baseline Phase B: 0.7464)
  - Cold-Drug MSE <= 0.65 (Baseline Phase B: 0.6609)
"""

# ruff: noqa: E402

import argparse
import json
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

import torch
import yaml

# Import pretrained encoders and fusion to register in MODELS
import tdc_studio.models.dti.fusion  # noqa: F401
import tdc_studio.models.dti.pretrained_encoders  # noqa: F401
from tdc_studio.data.multi_pred import DTADataModule
from tdc_studio.evaluation.evaluator import TherapeuticsEvaluator
from tdc_studio.models.dti.dta_model import GraphDTAModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train DTI Phase C Cross-Attention Model on Colab")
    parser.add_argument("--config", type=str, default="configs/config_dti_phase_c.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output-dir", type=str, default="models/dti/phase_c", help="Dir to save best checkpoint")
    parser.add_argument("--stage2-epoch", type=int, default=None, help="Epoch to start Stage 2 unfreezing")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("  ★ TDC-Studio DTI Phase C: Cross-Attention Fusion & Staged Fine-Tuning")
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

    # Hyperparameters
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 32)
    max_epochs = args.epochs or cfg.get("training", {}).get("max_epochs", 18)
    base_lr = args.lr or float(cfg.get("training", {}).get("learning_rate", 1.0e-4))
    weight_decay = float(cfg.get("training", {}).get("weight_decay", 1.0e-4))
    patience = int(cfg.get("training", {}).get("early_stopping_patience", 6))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    staged_cfg = cfg.get("staged_training", {})
    stage1_epochs = args.stage2_epoch or staged_cfg.get("stage1_epochs", 3)
    stage2_unfreeze_layers = staged_cfg.get("stage2_unfreeze_layers", 2)
    stage2_backbone_lr = float(staged_cfg.get("stage2_backbone_lr", 1.0e-5))
    stage2_head_lr = float(staged_cfg.get("stage2_head_lr", 5.0e-5))

    # 2. DataModule Setup
    print("\n[Step 1] Loading BindingDB_Kd Cold-Drug Split...")
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
    print(f"Split sizes: {data_module.split_info}")
    print(f"Scaler stats: mean={data_module.y_mean:.4f}, std={data_module.y_std:.4f}")

    # Save scaler.json for immediate serving integration
    scaler_info = {
        "y_mean": float(data_module.y_mean),
        "y_std": float(data_module.y_std),
        "log_transform": bool(dataset_cfg.get("log_transform", True)),
        "dataset_name": dataset_cfg.get("name", "BindingDB_Kd"),
    }
    with open(output_dir / "scaler.json", "w", encoding="utf-8") as f:
        json.dump(scaler_info, f, indent=2)

    # 3. Model Initialization
    print("\n[Step 2] Initializing GraphDTAModel with CrossAttentionFusion...")
    model_cfg = cfg.get("model", {})
    model = GraphDTAModel(model_cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable (Stage 1)  : {trainable_params:,}")

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
    print("\n[Step 3] Commencing Staged Training...")
    best_val_ci = -1.0
    best_val_mse = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Check for Stage 2 Transition
        if epoch == stage1_epochs + 1 and hasattr(model.drug_encoder, "unfreeze"):
            print(f"\n  ★ [Stage 2] Unfreezing ChemBERTa top {stage2_unfreeze_layers} layers!")
            model.drug_encoder.unfreeze(last_n_layers=stage2_unfreeze_layers)

            # Differential learning rate
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.drug_encoder.parameters(), "lr": stage2_backbone_lr},
                    {"params": model.fusion.parameters(), "lr": stage2_head_lr},
                    {"params": model.target_encoder.proj.parameters(), "lr": stage2_head_lr},
                ],
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(max_epochs - stage1_epochs), eta_min=1e-6
            )
            unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable Parameters (Stage 2): {unfrozen_params:,}")

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

        scheduler.step()
        avg_train_loss = train_loss_sum / max(1, train_batches)

        # Validation
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

        current_lr = optimizer.param_groups[0]["lr"]
        stage_tag = f"Stage {1 if epoch <= stage1_epochs else 2}"
        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} [{stage_tag}] [{elapsed:.1f}s] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val CI: {val_ci:.4f} | Val MSE: {val_mse:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_ci > best_val_ci:
            best_val_ci = val_ci
            best_val_mse = val_mse
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint & config
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
            # Also save standardized config.json
            with open(output_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(model_cfg, f, indent=2)

            print(f"  ★ Best checkpoint saved (Val CI: {best_val_ci:.4f}, Val MSE: {best_val_mse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement in Val CI for {patience} epochs.")
                break

    # 6. Test Set Final Evaluation (Cold-Drug Split)
    print("\n" + "=" * 70)
    print(f"  ★ Final Evaluation: Best Checkpoint (Epoch {best_epoch}) on COLD-DRUG Split")
    print("=" * 70)

    best_checkpoint = output_dir / "best_model.pt"
    if best_checkpoint.is_file():
        ckpt = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {best_checkpoint}")

    model.eval()
    test_preds, test_targets = [], []
    attention_sample = None

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            dev_batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if i == 0 and hasattr(model, "forward"):
                    # Extract XAI attention weights from first batch
                    preds, attn_dict = model(dev_batch, return_attention=True)
                    attention_sample = {k: v.cpu().numpy()[:2] for k, v in attn_dict.items()}
                else:
                    preds = model(dev_batch)
                preds_sq = preds.squeeze(-1)

            test_preds.extend(preds_sq.cpu().numpy().tolist())
            test_targets.extend(dev_batch["labels"].cpu().numpy().tolist())

    test_metrics = evaluator.compute_all(np.array(test_preds), np.array(test_targets), task_type="dta")

    ci_val = test_metrics.get("ci", 0.0)
    mse_val = test_metrics.get("mse", 0.0)
    rmse_val = test_metrics.get("rmse", 0.0)
    pearson_val = test_metrics.get("pearson", 0.0)

    target_ci = float(cfg.get("evaluation", {}).get("target_thresholds", {}).get("ci", 0.76))
    target_mse = float(cfg.get("evaluation", {}).get("target_thresholds", {}).get("mse", 0.65))

    ci_pass = ci_val >= target_ci
    mse_pass = mse_val <= target_mse

    print("\n" + "=" * 70)
    print("  * Phase C Cross-Attention Cold-Drug Benchmark Results Summary")
    print("=" * 70)
    print(f"  - Primary Metric:   Concordance Index (CI) = {ci_val:.4f}  [Target >= {target_ci:.2f}] -> {'[OK] PASSED' if ci_pass else '[FAIL]'}")
    print(f"  - Secondary Metric: Mean Squared Error (MSE) = {mse_val:.4f} [Target <= {target_mse:.2f}] -> {'[OK] PASSED' if mse_pass else '[FAIL]'}")
    print(f"  - Secondary Metric: Root MSE (RMSE)          = {rmse_val:.4f}")
    print(f"  - Secondary Metric: Pearson Correlation (r)  = {pearson_val:.4f}")
    print(f"  - XAI Attention Map: {'[OK] Extracted' if attention_sample is not None else '[None]'}")
    print("=" * 70)


    summary_file = output_dir / "benchmark_summary.yaml"
    with open(summary_file, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "phase": "Phase C (Cross-Attention + Staged Fine-Tuning)",
                "best_epoch": best_epoch,
                "test_metrics": test_metrics,
                "ci_passed": bool(ci_pass),
                "mse_passed": bool(mse_pass),
                "has_xai_attention": attention_sample is not None,
            },
            f,
            default_flow_style=False,
        )
    print(f"Benchmark summary saved to: {summary_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
