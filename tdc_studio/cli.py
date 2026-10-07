"""Command-Line Interface (CLI) for TDC-Studio."""

import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import typer
import yaml
from rich.console import Console

# Reconfigure Windows standard streams to UTF-8 to prevent cp949 UnicodeEncodeError
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from tdc_studio.core.registry import auto_import_modules

# Ensure all submodules are imported and registered
auto_import_modules("tdc_studio")

app = typer.Typer(
    name="tdc-studio",
    help="TDC-Studio: Modular MLOps Studio for Therapeutics Data Commons",
    add_completion=False,
)
remote_app = typer.Typer(help="Cloud GPU remote execution commands via Google Colab CLI")
app.add_typer(remote_app, name="remote")
retro_app = typer.Typer(help="AI Retrosynthesis single-step prediction and multi-step planning")
app.add_typer(retro_app, name="retrosynthesis")

console = Console()


def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def _batch_to_device(batch: dict, device: Any) -> dict:
    """Move tensor and PyG graph structures in batch to target device."""
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@app.command()
def train(
    config: str = typer.Option("configs/config.yaml", help="Path to main YAML config"),
    epochs: Optional[int] = typer.Option(None, help="Override max epochs"),
    checkpoint_dir: str = typer.Option("./models/checkpoint", help="Directory to save checkpoints"),
    eval_metric: Optional[str] = typer.Option(
        None, help="Evaluation metric to track for best model (e.g. mae, rmse, roc_auc)"
    ),
    early_stopping: Optional[int] = typer.Option(
        None, help="Patience epochs for early stopping (e.g. 5)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Run 1 step for validation without training"
    ),
    remote: bool = typer.Option(
        True,
        "--remote/--local",
        help="Execute on remote Google Colab GPU (default: True, use --local for local CPU)",
    ),
    session: Optional[str] = typer.Option(
        "tdc-studio", "-s", "--session", help="Colab session ID (default: tdc-studio)"
    ),
):
    """Train a model with validation evaluation, checkpointing, and W&B tracking."""
    if remote and not dry_run and os.environ.get("TDC_REMOTE_EXECUTION", "0") != "1":
        console.print(
            f"[bold cyan]Delegating training pipeline to Google Colab GPU session '{session}' via `remote exec`...[/bold cyan]"
        )
        from tdc_studio.remote.colab_runner import ColabRunner

        runner = ColabRunner()
        remote_cmd = f"tdc-studio train --local --config {config} --checkpoint-dir {checkpoint_dir}"
        if epochs is not None:
            remote_cmd += f" --epochs {epochs}"
        if eval_metric is not None:
            remote_cmd += f" --eval-metric {eval_metric}"
        if early_stopping is not None:
            remote_cmd += f" --early-stopping {early_stopping}"
        if dry_run:
            remote_cmd += " --dry-run"

        retcode = runner.run_remote_exec(command_to_run=remote_cmd, session=session)
        if retcode != 0:
            raise typer.Exit(retcode)
        return

    import torch

    from tdc_studio.core.registry import DATASETS, MODELS
    from tdc_studio.evaluation.evaluator import TherapeuticsEvaluator, is_metric_higher_better
    from tdc_studio.serving.exporter import export_model_checkpoint, load_model_from_checkpoint
    from tdc_studio.tracking.wandb_tracker import WandBTracker

    cfg = load_yaml(config)
    # Support staged/hierarchical configs (e.g. hERG standalone phase1_pretraining)
    if "phase1_pretraining" in cfg:
        tracking_cfg = cfg.get("tracking", {})
        cfg = dict(cfg["phase1_pretraining"])
        if "tracking" not in cfg:
            cfg["tracking"] = tracking_cfg

    console.print(f"[bold green]Starting Training Pipeline[/bold green] with config: {config}")

    data_cfg = cfg.get("data", {})
    if not data_cfg and "dataset" in cfg:
        data_cfg = dict(cfg["dataset"])
        if "type" not in data_cfg:
            data_cfg["type"] = "dta_loader"
        if "dataset_name" not in data_cfg and "name" in data_cfg:
            data_cfg["dataset_name"] = data_cfg["name"]
        if "split_type" not in data_cfg and "split" in data_cfg:
            data_cfg["split_type"] = data_cfg["split"]

    model_cfg = cfg.get("model", {})
    dataset_name = data_cfg.get("dataset_name", data_cfg.get("name", "dataset"))

    # 1. Load Data
    data_type = data_cfg.get("type", data_cfg.get("name"))
    data_cls = DATASETS.get(data_type)
    if "params" in data_cfg and isinstance(data_cfg["params"], dict):
        data_params = dict(data_cfg["params"])
    else:
        data_params = {k: v for k, v in data_cfg.items() if k not in ("type", "name", "batch_size")}
    if dry_run and "max_samples" not in data_params:
        data_params["max_samples"] = 40
    data_module = data_cls(**data_params)

    data_module.prepare_data()
    batch_size = int(data_cfg.get("batch_size") or cfg.get("training", {}).get("batch_size") or 32)
    train_loader, val_loader, test_loader = data_module.setup_loaders(batch_size=batch_size)

    task_type = data_module.task_type
    primary_task = (
        getattr(data_module, "primary_task", None)
        or data_cfg.get("primary_task")
        or (data_module.task_names[0] if getattr(data_module, "task_names", None) else None)
    )
    target_metric = (
        eval_metric
        or data_cfg.get("metric_name")
        or getattr(data_module, "metric_name", None)
        or (
            "val_loss"
            if task_type == "multi_task"
            else (
                "ci"
                if task_type in ("dta", "multi_dta")
                else ("mae" if task_type == "regression" else "roc_auc")
            )
        )
    )
    higher_is_better = (
        is_metric_higher_better(target_metric) if target_metric != "val_loss" else False
    )
    evaluator = TherapeuticsEvaluator(default_metric=target_metric, task_type=task_type)

    # 2. Build Model & Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg["task_type"] = task_type
    model_type = model_cfg.get("type", model_cfg.get("name"))
    model_cls = MODELS.get(model_type)
    model = model_cls(model_cfg).to(device)

    # Transfer Learning: Load pre-trained weights if specified in config
    pretrained_path = cfg.get("pretrained_checkpoint")
    if pretrained_path and os.path.exists(pretrained_path):
        console.print(f"[bold cyan]Loading pre-trained weights from: {pretrained_path}[/bold cyan]")
        try:
            chk = torch.load(pretrained_path, map_location=device, weights_only=True)
        except Exception:
            chk = torch.load(pretrained_path, map_location=device)
        state_dict = chk.get("state_dict", chk) if isinstance(chk, dict) else chk
        model_dict = model.state_dict()
        matched = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        console.print(
            f"[bold green]Successfully transferred {len(matched)}/{len(model_dict)} tensors from backbone![/bold green]"
        )

    # Initial backbone freezing for Stage 2 fine-tuning
    freeze_epochs = int(cfg.get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0:
        for p_name, param in model.named_parameters():
            if not p_name.startswith("task_heads"):
                param.requires_grad = False
        console.print(
            f"[bold yellow]Backbone weights frozen for first {freeze_epochs} epoch(s). Training task heads only.[/bold yellow]"
        )

    lr = float(cfg.get("lr", cfg.get("training", {}).get("learning_rate", 1e-3)))
    weight_decay = float(cfg.get("weight_decay", cfg.get("training", {}).get("weight_decay", 1e-4)))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    use_pcgrad = bool(cfg.get("use_pcgrad", False) or model_cfg.get("use_pcgrad", False))
    if use_pcgrad:
        from tdc_studio.models.loss.pcgrad import PCGrad

        optimizer = PCGrad(optimizer)
        console.print(
            "[bold magenta]PCGrad (Projecting Conflicting Gradients) optimizer active for multi-task learning![/bold magenta]"
        )

    console.print(
        f"Device: [cyan]{device}[/cyan] | Task: [cyan]{task_type}[/cyan] | Target Metric: [yellow]{target_metric}[/yellow] (higher_is_better={higher_is_better})"
    )

    # 3. Setup Tracking
    tracking_cfg = cfg.get("tracking", {})
    tracker = WandBTracker(tracking_cfg)
    tracker.init_run(
        name=f"train_{dataset_name}",
        config=cfg,
        group=f"{dataset_name}_train",
    )

    if checkpoint_dir == "./models/checkpoint" and "export" in cfg and "save_dir" in cfg["export"]:
        checkpoint_dir = cfg["export"]["save_dir"]

    max_epochs = (
        1
        if dry_run
        else (epochs or cfg.get("max_epochs", cfg.get("training", {}).get("max_epochs", 5)))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, max_epochs), eta_min=float(cfg.get("min_lr", 1e-6))
    )
    early_stopping_patience = (
        early_stopping
        if early_stopping is not None
        else cfg.get("early_stopping", cfg.get("training", {}).get("early_stopping_patience", None))
    )
    best_metric = -float("inf") if higher_is_better else float("inf")
    best_epoch = 0
    patience_counter = 0

    console.print(f"Training for {max_epochs} epoch(s) (dry_run={dry_run})...")

    try:
        for epoch in range(max_epochs):
            # Staged Training support (e.g. for ChemBERTa top-layers in DTI Phase C)
            staged_cfg = cfg.get("staged_training", {})
            s1_epochs = int(staged_cfg.get("stage1_epochs", 0))
            if (
                s1_epochs > 0
                and epoch == s1_epochs
                and hasattr(model, "drug_encoder")
                and hasattr(model.drug_encoder, "unfreeze")
            ):
                s2_layers = int(staged_cfg.get("stage2_unfreeze_layers", 2))
                s2_bb_lr = float(staged_cfg.get("stage2_backbone_lr", 1e-6))
                s2_head_lr = float(staged_cfg.get("stage2_head_lr", 5e-5))
                s2_wd = float(staged_cfg.get("stage2_weight_decay", 0.05))

                console.print(
                    f"[bold green]★ Stage 2 Transition: Unfreezing Drug Encoder top {s2_layers} layers at epoch {epoch + 1} (LR: {s2_bb_lr:.2e}, WD: {s2_wd})[/bold green]"
                )
                model.drug_encoder.unfreeze(last_n_layers=s2_layers)
                target_opt = optimizer._optim if hasattr(optimizer, "_optim") else optimizer
                existing_params = {p for pg in target_opt.param_groups for p in pg["params"]}
                new_params = [
                    p
                    for p in model.drug_encoder.parameters()
                    if p.requires_grad and p not in existing_params
                ]
                if new_params:
                    target_opt.add_param_group(
                        {"params": new_params, "lr": s2_bb_lr, "weight_decay": s2_wd}
                    )
                for pg in target_opt.param_groups[:-1]:
                    pg["lr"] = s2_head_lr

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, max_epochs - epoch),
                    eta_min=float(cfg.get("min_lr", 1e-7)),
                )

            # --- Train Epoch ---
            model.train()
            train_loss_sum = 0.0
            train_batches = 0

            for batch in train_loader:
                dev_batch = _batch_to_device(batch, device)
                optimizer.zero_grad()
                preds = model(dev_batch)
                mask = dev_batch.get("mask")

                if use_pcgrad and hasattr(model, "loss_fn") and model.loss_fn is not None:
                    loss_res = model.compute_loss(preds, dev_batch["labels"], mask=mask, return_per_task=True)
                    if isinstance(loss_res, tuple) and len(loss_res) == 3:
                        loss, _, task_loss_list = loss_res
                    else:
                        loss = loss_res
                        task_loss_list = []

                    if not torch.isfinite(loss):
                        optimizer.zero_grad()
                        continue

                    if task_loss_list:
                        optimizer.pc_backward(task_loss_list)
                    else:
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                else:
                    if mask is not None:
                        loss = model.compute_loss(preds, dev_batch["labels"], mask=mask)
                    else:
                        loss = model.compute_loss(preds, dev_batch["labels"])

                    if not torch.isfinite(loss):
                        optimizer.zero_grad()
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss_sum += float(loss.item())
                train_batches += 1
                if dry_run:
                    break

            avg_train_loss = train_loss_sum / max(1, train_batches)

            # --- Validation Epoch ---
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            val_preds_list = []
            val_labels_list = []
            val_masks_list = []

            with torch.no_grad():
                for batch in val_loader:
                    dev_batch = _batch_to_device(batch, device)
                    preds = model(dev_batch)
                    mask = dev_batch.get("mask")
                    if mask is not None:
                        loss = model.compute_loss(preds, dev_batch["labels"], mask=mask)
                        val_masks_list.append(mask.detach().cpu())
                    else:
                        loss = model.compute_loss(preds, dev_batch["labels"])

                    val_loss_sum += float(loss.item())
                    val_batches += 1
                    val_preds_list.append(preds.detach().cpu())
                    val_labels_list.append(dev_batch["labels"].detach().cpu())
                    if dry_run:
                        break

            avg_val_loss = val_loss_sum / max(1, val_batches)
            val_preds_cat = torch.cat(val_preds_list, dim=0) if val_preds_list else torch.tensor([])
            val_labels_cat = (
                torch.cat(val_labels_list, dim=0) if val_labels_list else torch.tensor([])
            )

            if task_type == "multi_task":
                current_val_metric = avg_val_loss
                all_val_metrics = {"val_loss": avg_val_loss}
                if hasattr(data_module, "task_names"):
                    val_masks_cat = (
                        torch.cat(val_masks_list, dim=0) if val_masks_list else torch.tensor([])
                    )
                    for t_idx, t_name in enumerate(data_module.task_names):
                        t_type = data_module.task_types[t_idx]
                        if (
                            val_masks_cat.numel() > 0
                            and val_masks_cat.ndim >= 2
                            and val_masks_cat.size(1) > t_idx
                        ):
                            t_valid = val_masks_cat[:, t_idx]
                        elif (
                            val_labels_cat.numel() > 0
                            and val_labels_cat.ndim >= 2
                            and val_labels_cat.size(1) > t_idx
                        ):
                            t_valid = ~torch.isnan(val_labels_cat[:, t_idx])
                        else:
                            t_valid = torch.tensor([], dtype=torch.bool)
                        if (
                            int(t_valid.sum().item()) > 0
                            and val_preds_cat.ndim >= 2
                            and val_labels_cat.ndim >= 2
                        ):
                            t_p = val_preds_cat[t_valid, t_idx]
                            t_y = val_labels_cat[t_valid, t_idx]
                            finite_mask = torch.isfinite(t_p) & torch.isfinite(t_y)
                            if not finite_mask.all():
                                t_p = t_p[finite_mask]
                                t_y = t_y[finite_mask]
                            if t_p.numel() == 0:
                                continue

                            if (
                                getattr(data_module, "standardize_target", False)
                                and t_type == "regression"
                            ):
                                stat = getattr(data_module, "task_stats", {}).get(
                                    t_name, {"mean": 0.0, "std": 1.0}
                                )
                                t_p = t_p * stat["std"] + stat["mean"]
                                t_y = t_y * stat["std"] + stat["mean"]

                            t_trans = getattr(data_module, "task_transforms", {}).get(t_name, None)
                            if t_trans == "logit":
                                t_p = 100.0 * torch.sigmoid(t_p)
                                t_y = 100.0 * torch.sigmoid(t_y)
                            elif t_trans == "log10":
                                pass
                            elif "ppbr" in str(t_name).lower():
                                t_p = torch.clamp(t_p, min=0.0, max=100.0)
                                t_y = torch.clamp(t_y, min=0.0, max=100.0)

                            t_metric_name = "mae" if t_type == "regression" else "roc_auc"
                            all_val_metrics[f"{t_name}_{t_metric_name}"] = evaluator.compute(
                                t_p, t_y, t_metric_name
                            )
                            if t_type == "regression":
                                all_val_metrics[f"{t_name}_r2"] = evaluator.compute(t_p, t_y, "r2")
                                all_val_metrics[f"{t_name}_rmse"] = evaluator.compute(
                                    t_p, t_y, "rmse"
                                )
                                all_val_metrics[f"{t_name}_pearson"] = evaluator.compute(
                                    t_p, t_y, "pearson"
                                )
                                all_val_metrics[f"{t_name}_spearman"] = evaluator.compute(
                                    t_p, t_y, "spearman"
                                )
                                all_val_metrics[f"{t_name}_composite"] = evaluator.compute(
                                    t_p, t_y, "composite"
                                )

                if primary_task and f"{primary_task}_{target_metric}" in all_val_metrics:
                    current_val_metric = all_val_metrics[f"{primary_task}_{target_metric}"]
                elif target_metric in all_val_metrics:
                    current_val_metric = all_val_metrics[target_metric]
            else:
                eval_p = val_preds_cat.squeeze(-1) if val_preds_cat.ndim > 1 else val_preds_cat
                eval_y = val_labels_cat.squeeze(-1) if val_labels_cat.ndim > 1 else val_labels_cat
                if getattr(data_module, "standardize_target", False) and task_type == "regression":
                    mean = getattr(data_module, "target_mean", 0.0)
                    std = getattr(data_module, "target_std", 1.0)
                    eval_p = eval_p * std + mean
                    eval_y = eval_y * std + mean

                if getattr(data_module, "target_transform", None) == "logit":
                    eval_p = 100.0 * torch.sigmoid(eval_p)
                    eval_y = 100.0 * torch.sigmoid(eval_y)
                elif "ppbr" in str(dataset_name).lower() or "ppbr" in str(target_metric).lower():
                    eval_p = torch.clamp(eval_p, min=0.0, max=100.0)
                    eval_y = torch.clamp(eval_y, min=0.0, max=100.0)

                current_val_metric = evaluator.compute(
                    eval_p,
                    eval_y,
                    target_metric,
                )
                all_val_metrics = evaluator.compute_all(
                    eval_p,
                    eval_y,
                    task_type,
                )

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            console.print(
                f"Epoch {epoch + 1}/{max_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val {target_metric.upper()}: [bold cyan]{current_val_metric:.4f}[/bold cyan] | "
                f"LR: {current_lr:.6f}"
            )

            # Track metrics
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr,
                **{f"val_{k}": v for k, v in all_val_metrics.items()},
            }
            tracker.log_metrics(log_dict, step=epoch + 1)

            # Check for best model improvement
            improved = (
                current_val_metric > best_metric
                if higher_is_better
                else current_val_metric < best_metric
            )

            if improved and not dry_run:
                best_metric = current_val_metric
                best_epoch = epoch + 1
                patience_counter = 0

                meta_info = {
                    "dataset_name": dataset_name,
                    "best_epoch": best_epoch,
                    "target_metric": target_metric,
                    "best_metric_value": best_metric,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "all_val_metrics": all_val_metrics,
                }
                export_model_checkpoint(
                    model=model,
                    model_config=model_cfg,
                    output_dir=checkpoint_dir,
                    checkpoint_name="best_model.pt",
                    extra_meta=meta_info,
                )
                console.print(
                    f"  [bold green]★ Best checkpoint saved to '{checkpoint_dir}' "
                    f"({target_metric}={best_metric:.4f})[/bold green]"
                )

                if tracking_cfg.get("enabled", False):
                    tracker.log_artifact(
                        checkpoint_dir,
                        name=f"{dataset_name}_best_checkpoint",
                        artifact_type="model",
                    )
            else:
                patience_counter += 1
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    console.print(
                        f"[yellow]Early stopping triggered after {patience_counter} epochs without improvement.[/yellow]"
                    )
                    break

            if dry_run:
                break

        # --- Test Set Evaluation (if available and not dry_run) ---
        if not dry_run and test_loader is not None and len(test_loader) > 0:
            console.print("\n[bold cyan]Evaluating on Test Set with Best Checkpoint...[/bold cyan]")
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            eval_model = model
            if os.path.exists(best_model_path):
                eval_model = load_model_from_checkpoint(checkpoint_dir).to(device)

            eval_model.eval()
            test_preds_list = []
            test_labels_list = []
            test_masks_list = []
            with torch.no_grad():
                for batch in test_loader:
                    dev_batch = _batch_to_device(batch, device)
                    preds = eval_model(dev_batch)
                    test_preds_list.append(preds.detach().cpu())
                    test_labels_list.append(dev_batch["labels"].detach().cpu())
                    if "mask" in dev_batch:
                        test_masks_list.append(dev_batch["mask"].detach().cpu())

            test_preds_cat = (
                torch.cat(test_preds_list, dim=0) if test_preds_list else torch.tensor([])
            )
            test_labels_cat = (
                torch.cat(test_labels_list, dim=0) if test_labels_list else torch.tensor([])
            )

            if task_type == "multi_task":
                all_test_metrics = {}
                test_masks_cat = (
                    torch.cat(test_masks_list, dim=0) if test_masks_list else torch.tensor([])
                )
                for t_idx, t_name in enumerate(data_module.task_names):
                    t_type = data_module.task_types[t_idx]
                    if test_masks_cat.numel() > 0:
                        t_valid = test_masks_cat[:, t_idx]
                    else:
                        t_valid = ~torch.isnan(test_labels_cat[:, t_idx])
                    if int(t_valid.sum().item()) > 0:
                        t_p = test_preds_cat[t_valid, t_idx]
                        t_y = test_labels_cat[t_valid, t_idx]
                        if (
                            getattr(data_module, "standardize_target", False)
                            and t_type == "regression"
                        ):
                            stat = getattr(data_module, "task_stats", {}).get(
                                t_name, {"mean": 0.0, "std": 1.0}
                            )
                            t_p = t_p * stat["std"] + stat["mean"]
                            t_y = t_y * stat["std"] + stat["mean"]

                        t_trans = getattr(data_module, "task_transforms", {}).get(t_name, None)
                        if t_trans == "logit":
                            t_p = 100.0 * torch.sigmoid(t_p)
                            t_y = 100.0 * torch.sigmoid(t_y)
                        elif t_trans == "log10":
                            pass
                        elif "ppbr" in str(t_name).lower():
                            t_p = torch.clamp(t_p, min=0.0, max=100.0)
                            t_y = torch.clamp(t_y, min=0.0, max=100.0)

                        if t_type == "regression":
                            all_test_metrics[f"{t_name}_r2"] = evaluator.compute(t_p, t_y, "r2")
                            all_test_metrics[f"{t_name}_rmse"] = evaluator.compute(t_p, t_y, "rmse")
                            all_test_metrics[f"{t_name}_mae"] = evaluator.compute(t_p, t_y, "mae")
                            all_test_metrics[f"{t_name}_pearson"] = evaluator.compute(
                                t_p, t_y, "pearson"
                            )
                            all_test_metrics[f"{t_name}_spearman"] = evaluator.compute(
                                t_p, t_y, "spearman"
                            )
                        else:
                            all_test_metrics[f"{t_name}_roc_auc"] = evaluator.compute(
                                t_p, t_y, "roc_auc"
                            )

                target_key = (
                    f"{primary_task}_{target_metric}"
                    if primary_task and f"{primary_task}_{target_metric}" in all_test_metrics
                    else target_metric
                )
                test_metric_val = all_test_metrics.get(target_key, 0.0)

                console.print(
                    f"[bold green]Test Results ({target_metric.upper()}): {test_metric_val:.4f}[/bold green]"
                )
                console.print(f"Detailed Test Metrics: {all_test_metrics}")

                if primary_task == "caco2_wang":
                    from rich.table import Table

                    c_r2 = all_test_metrics.get("caco2_wang_r2", 0.0)
                    c_rmse = all_test_metrics.get("caco2_wang_rmse", 0.0)
                    c_mae = all_test_metrics.get("caco2_wang_mae", 0.0)
                    c_pr = all_test_metrics.get("caco2_wang_pearson", 0.0)
                    c_sp = all_test_metrics.get("caco2_wang_spearman", 0.0)

                    table = Table(title="★ Multi-Task Benchmark Results: CACO2_WANG")
                    table.add_column("Metric", style="bold")
                    table.add_column("Phase 4 MTL (Test)", style="bold cyan")
                    table.add_column("Literature SOTA", style="bold green")
                    table.add_row("R²", f"{c_r2:.4f}", "0.743±0.018")
                    table.add_row("RMSE", f"{c_rmse:.4f}", "0.325±0.013")
                    table.add_row("MAE", f"{c_mae:.4f}", "0.242±0.011")
                    table.add_row("Pearson (r)", f"{c_pr:.4f}", "~0.86")
                    table.add_row("Spearman (ρ)", f"{c_sp:.4f}", "~0.83")
                    console.print(table)

                if primary_task == "ppbr_az" or dataset_name == "distribution_mtl":
                    from rich.table import Table

                    p_mae = all_test_metrics.get("ppbr_az_mae", 0.0)
                    p_rmse = all_test_metrics.get("ppbr_az_rmse", 0.0)
                    b_auc = all_test_metrics.get("bbb_martins_roc_auc", 0.0)
                    v_r2 = all_test_metrics.get("vdss_lombardo_r2", 0.0)
                    v_pr = all_test_metrics.get("vdss_lombardo_pearson", 0.0)
                    v_rmse = all_test_metrics.get("vdss_lombardo_rmse", 0.0)
                    v_mae = all_test_metrics.get("vdss_lombardo_mae", 0.0)
                    l_r2 = all_test_metrics.get("lipophilicity_astrazeneca_r2", 0.0)
                    l_pr = all_test_metrics.get("lipophilicity_astrazeneca_pearson", 0.0)

                    p_r2 = all_test_metrics.get("ppbr_az_r2", 0.0)
                    p_pr = all_test_metrics.get("ppbr_az_pearson", 0.0)
                    p_sp = all_test_metrics.get("ppbr_az_spearman", 0.0)

                    table = Table(
                        title="★ Cluster 2 (Plasma Distribution) Multi-Task Benchmark Results"
                    )
                    table.add_column("Task Endpoint", style="bold")
                    table.add_column("Metric", style="bold cyan")
                    table.add_column("Our Model (Test)", style="bold green")
                    table.add_column("TDC Benchmark / SOTA", style="yellow")
                    table.add_row(
                        "PPBR (AZ)",
                        "R² (Pearson r, Spearman ρ)",
                        f"{p_r2:.4f} (r={p_pr:.4f}, ρ={p_sp:.4f})",
                        "ADMETlab: 0.733",
                    )
                    table.add_row(
                        "PPBR (AZ)",
                        "MAE (%) [RMSE]",
                        f"{p_mae:.2f}% [{p_rmse:.2f}%]",
                        "7.4% ~ 8.6%",
                    )
                    table.add_row("BBB Martins", "ROC-AUC", f"{b_auc:.4f}", "0.908±0.012")
                    table.add_row(
                        "VDss Lombardo",
                        "R² (Pearson r)",
                        f"{v_r2:.4f} (r={v_pr:.4f})",
                        "0.760 (r~0.88)",
                    )
                    table.add_row(
                        "VDss Lombardo",
                        "MAE [RMSE] (log10)",
                        f"{v_mae:.3f} [{v_rmse:.3f}]",
                        "0.162 [0.301]",
                    )
                    table.add_row(
                        "Lipophilicity",
                        "R² (Pearson r)",
                        f"{l_r2:.4f} (r={l_pr:.4f})",
                        "0.650 (r~0.80)",
                    )
                    if "solubility_aqsoldb_r2" in all_test_metrics:
                        s_r2 = all_test_metrics["solubility_aqsoldb_r2"]
                        s_pr = all_test_metrics.get("solubility_aqsoldb_pearson", 0.0)
                        table.add_row(
                            "Solubility (AqSolDB)",
                            "R² (Pearson r)",
                            f"{s_r2:.4f} (r={s_pr:.4f})",
                            "SOTA: ~0.78",
                        )
                    if "caco2_wang_r2" in all_test_metrics:
                        c_r2 = all_test_metrics["caco2_wang_r2"]
                        c_pr = all_test_metrics.get("caco2_wang_pearson", 0.0)
                        table.add_row(
                            "Caco-2 (Wang)",
                            "R² (Pearson r)",
                            f"{c_r2:.4f} (r={c_pr:.4f})",
                            "SOTA: ~0.74",
                        )
                    console.print(table)

                tracker.log_metrics({f"test_{k}": v for k, v in all_test_metrics.items()})
            else:
                test_preds_cat = (
                    test_preds_cat.squeeze(-1) if test_preds_cat.ndim > 1 else test_preds_cat
                )
                test_labels_cat = (
                    test_labels_cat.squeeze(-1) if test_labels_cat.ndim > 1 else test_labels_cat
                )
                if getattr(data_module, "standardize_target", False) and task_type == "regression":
                    mean = getattr(data_module, "target_mean", 0.0)
                    std = getattr(data_module, "target_std", 1.0)
                    test_preds_eval = test_preds_cat * std + mean
                    test_labels_eval = test_labels_cat * std + mean
                else:
                    test_preds_eval = test_preds_cat
                    test_labels_eval = test_labels_cat

                if getattr(data_module, "target_transform", None) == "logit":
                    test_preds_eval = 100.0 * torch.sigmoid(test_preds_eval)
                    test_labels_eval = 100.0 * torch.sigmoid(test_labels_eval)
                elif "ppbr" in str(dataset_name).lower() or "ppbr" in str(target_metric).lower():
                    test_preds_eval = torch.clamp(test_preds_eval, min=0.0, max=100.0)
                    test_labels_eval = torch.clamp(test_labels_eval, min=0.0, max=100.0)

                test_metric_val = evaluator.compute(
                    test_preds_eval, test_labels_eval, target_metric
                )
                all_test_metrics = evaluator.compute_all(
                    test_preds_eval, test_labels_eval, task_type
                )

                console.print(
                    f"[bold green]Test Results ({target_metric.upper()}): {test_metric_val:.4f}[/bold green]"
                )
                console.print(f"Detailed Test Metrics: {all_test_metrics}")

                if task_type in ("dta", "drug_target_affinity"):
                    from rich.table import Table

                    d_ci = all_test_metrics.get("ci", 0.0)
                    d_mse = all_test_metrics.get("mse", 0.0)
                    d_rmse = all_test_metrics.get("rmse", 0.0)
                    d_pr = all_test_metrics.get("pearson", 0.0)

                    table = Table(title="★ DTA (Drug-Target Affinity) Benchmark Results")
                    table.add_column("Metric", style="bold")
                    table.add_column("Our Model (Test)", style="bold cyan")
                    table.add_column("Target Threshold", style="bold green")
                    table.add_row("Concordance Index (CI)", f"{d_ci:.4f}", ">= 0.7700")
                    table.add_row("Mean Squared Error (MSE)", f"{d_mse:.4f}", "<= 0.5500")
                    table.add_row("Root MSE (RMSE)", f"{d_rmse:.4f}", "~ 0.74")
                    table.add_row("Pearson (r)", f"{d_pr:.4f}", "~ 0.69")
                    console.print(table)

                tracker.log_metrics({f"test_{k}": v for k, v in all_test_metrics.items()})

    finally:
        tracker.finish()

    console.print(
        f"[bold green]Training Pipeline Finished![/bold green] "
        f"(Best Epoch: {best_epoch}, Best {target_metric.upper()}: {best_metric:.4f})"
    )


@app.command()
def tune(
    config: str = typer.Option("configs/config.yaml", help="Path to main YAML config"),
    n_trials: int = typer.Option(10, "--n-trials", help="Number of Optuna trials"),
    save_path: str = typer.Option(
        "configs/best_hpo_params.yaml", help="Path to save best hyperparameters YAML"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run 1 step per trial for testing"),
):
    """Run Optuna Hyperparameter Optimization (HPO)."""
    import json

    from tdc_studio.automl.tuner import StudioTuner

    cfg = load_yaml(config)
    console.print(f"[bold cyan]Launching Optuna HPO[/bold cyan] ({n_trials} trials)...")

    tuner = StudioTuner(
        data_cfg=cfg.get("data", {}),
        model_cfg=cfg.get("model", {}),
        hpo_cfg=cfg.get("hpo", {}),
        tracking_cfg=cfg.get("tracking", {}),
        dry_run=dry_run,
    )
    result = tuner.tune(n_trials=n_trials)
    console.print("[bold green]Tuning Finished![/bold green]")
    console.print(f"Best Trial #{result['best_trial_number']}: Metric = {result['best_value']:.4f}")
    console.print(f"Best Hyperparameters: {result['best_params']}")

    if not dry_run:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, "w", encoding="utf-8") as f:
            yaml.dump(result["best_params"], f, default_flow_style=False)
        console.print(f"Saved best parameters YAML to: [cyan]{save_file}[/cyan]")

        json_path = Path("models/hpo/best_params.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        console.print(f"Saved full study summary JSON to: [cyan]{json_path}[/cyan]")


@app.command()
def ensemble(
    config: str = typer.Option("configs/config_caco2_dmpnn.yaml", help="Path to main YAML config"),
    n_models: int = typer.Option(5, "--n-models", help="Number of ensemble models (default 5)"),
    seeds: str = typer.Option("42,43,44,45,46", "--seeds", help="Comma-separated seeds for models"),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Epochs per ensemble model"),
    checkpoint_dir: str = typer.Option(
        "./models/checkpoint/ensemble", help="Dir to save ensemble checkpoints"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run 1 step per model for testing"),
    remote: bool = typer.Option(
        True,
        "--remote/--local",
        help="Execute on remote Google Colab GPU (default: True, use --local for local CPU)",
    ),
    session: Optional[str] = typer.Option(
        "tdc-studio", "-s", "--session", help="Colab session ID (default: tdc-studio)"
    ),
):
    """Train an ensemble of models with multiple random seeds and compute consensus predictions."""
    if remote and not dry_run and os.environ.get("TDC_REMOTE_EXECUTION", "0") != "1":
        console.print(
            f"[bold cyan]Delegating ensemble pipeline to Google Colab GPU session '{session}' via `remote exec`...[/bold cyan]"
        )
        from tdc_studio.remote.colab_runner import ColabRunner

        runner = ColabRunner()
        remote_cmd = (
            f"tdc-studio ensemble --local --config {config} --n-models {n_models} --seeds {seeds}"
            f" --checkpoint-dir {checkpoint_dir}"
        )
        if epochs is not None:
            remote_cmd += f" --epochs {epochs}"
        if dry_run:
            remote_cmd += " --dry-run"

        retcode = runner.run_remote_exec(command_to_run=remote_cmd, session=session)
        if retcode != 0:
            raise typer.Exit(retcode)
        return

    import json

    import numpy as np
    import torch
    from rich.table import Table

    from tdc_studio.core.registry import DATASETS, MODELS
    from tdc_studio.evaluation.evaluator import TherapeuticsEvaluator, is_metric_higher_better
    from tdc_studio.tracking.wandb_tracker import WandBTracker

    cfg = load_yaml(config)
    seed_list = [int(s.strip()) for s in seeds.split(",")][:n_models]
    console.print(
        f"[bold green]Starting Ensemble Pipeline[/bold green] with {len(seed_list)} models (Seeds: {seed_list})..."
    )

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    dataset_name = data_cfg.get("dataset_name", "dataset")

    # 1. Load Data once
    data_type = data_cfg.get("type", data_cfg.get("name"))
    data_cls = DATASETS.get(data_type)
    data_params = (
        dict(data_cfg.get("params", {}))
        if "params" in data_cfg and isinstance(data_cfg["params"], dict)
        else {k: v for k, v in data_cfg.items() if k not in ("type", "name", "batch_size")}
    )
    data_module = data_cls(**data_params)
    data_module.prepare_data()

    batch_size = data_cfg.get("batch_size", 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_type = data_module.task_type
    primary_task = (
        getattr(data_module, "primary_task", None)
        or data_cfg.get("primary_task")
        or (data_module.task_names[0] if getattr(data_module, "task_names", None) else None)
    )
    primary_idx = (
        data_module.task_names.index(primary_task)
        if (primary_task and hasattr(data_module, "task_names"))
        else 0
    )
    eval_task_type = (
        data_module.task_types[primary_idx]
        if hasattr(data_module, "task_types")
        else ("regression" if task_type == "regression" else "binary_classification")
    )
    target_metric = (
        data_cfg.get("metric_name")
        or getattr(data_module, "metric_name", None)
        or ("composite" if eval_task_type == "regression" else "roc_auc")
    )
    higher_is_better = is_metric_higher_better(target_metric)
    evaluator = TherapeuticsEvaluator(default_metric=target_metric, task_type=eval_task_type)

    tracking_cfg = cfg.get("tracking", {})
    tracker = WandBTracker(tracking_cfg)
    tracker.init_run(
        name=f"ensemble_{dataset_name}",
        config={**cfg, "ensemble_seeds": seed_list, "n_models": len(seed_list)},
        group=f"{dataset_name}_ensemble",
    )

    max_epochs = 1 if dry_run else (epochs or cfg.get("max_epochs", 80))
    patience = 20

    all_test_preds = []
    all_val_preds = []
    model_metrics_list = []
    test_labels_real = None
    val_labels_real = None

    try:
        for idx, seed in enumerate(seed_list):
            console.print(
                f"\n[bold cyan]─── Training Ensemble Model #{idx + 1}/{len(seed_list)} (Seed: {seed}) ───[/bold cyan]"
            )
            import random

            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            train_loader, val_loader, test_loader = data_module.setup_loaders(batch_size=batch_size)

            curr_model_cfg = {**model_cfg, "task_type": task_type}
            model_cls = MODELS.get(curr_model_cfg.get("type", curr_model_cfg.get("name")))
            model = model_cls(curr_model_cfg).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, max_epochs), eta_min=1e-6
            )

            best_metric = -float("inf") if higher_is_better else float("inf")
            best_epoch = 0
            patience_counter = 0
            best_state_dict = None

            for epoch in range(max_epochs):
                model.train()
                for batch in train_loader:
                    dev_batch = _batch_to_device(batch, device)
                    optimizer.zero_grad()
                    preds = model(dev_batch)
                    mask = dev_batch.get("mask")
                    if mask is not None:
                        loss = model.compute_loss(preds, dev_batch["labels"], mask=mask)
                    else:
                        loss = model.compute_loss(preds, dev_batch["labels"])

                    if not torch.isfinite(loss):
                        optimizer.zero_grad()
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if dry_run:
                        break

                # Validation
                model.eval()
                v_preds_list = []
                v_labels_list = []
                v_masks_list = []
                with torch.no_grad():
                    for batch in val_loader:
                        dev_batch = _batch_to_device(batch, device)
                        preds = model(dev_batch)
                        if task_type == "multi_task":
                            v_preds_list.append(preds[:, primary_idx].detach().cpu())
                            v_labels_list.append(dev_batch["labels"][:, primary_idx].detach().cpu())
                            if "mask" in dev_batch and dev_batch["mask"] is not None:
                                v_masks_list.append(
                                    dev_batch["mask"][:, primary_idx].detach().cpu()
                                )
                        else:
                            v_preds_list.append(preds.squeeze(-1).detach().cpu())
                            v_labels_list.append(dev_batch["labels"].detach().cpu())
                            if "mask" in dev_batch and dev_batch["mask"] is not None:
                                v_masks_list.append(dev_batch["mask"].squeeze(-1).detach().cpu())
                        if dry_run:
                            break

                v_preds_cat = torch.cat(v_preds_list, dim=0) if v_preds_list else torch.tensor([])
                v_labels_cat = (
                    torch.cat(v_labels_list, dim=0) if v_labels_list else torch.tensor([])
                )
                v_masks_cat = torch.cat(v_masks_list, dim=0) if v_masks_list else torch.tensor([])

                if (
                    getattr(data_module, "standardize_target", False)
                    and eval_task_type == "regression"
                ):
                    if task_type == "multi_task":
                        stat = getattr(data_module, "task_stats", {}).get(
                            primary_task, {"mean": 0.0, "std": 1.0}
                        )
                        mean = stat["mean"]
                        std = stat["std"]
                    else:
                        mean = getattr(data_module, "target_mean", 0.0)
                        std = getattr(data_module, "target_std", 1.0)
                    eval_p = v_preds_cat * std + mean
                    eval_y = v_labels_cat * std + mean
                else:
                    eval_p = v_preds_cat
                    eval_y = v_labels_cat

                is_logit = False
                if task_type == "multi_task":
                    is_logit = (
                        getattr(data_module, "task_transforms", {}).get(primary_task) == "logit"
                    )
                else:
                    is_logit = getattr(data_module, "target_transform", None) == "logit"

                if is_logit:
                    eval_p = 100.0 * torch.sigmoid(eval_p)
                    eval_y = 100.0 * torch.sigmoid(eval_y)
                elif "ppbr" in str(primary_task or dataset_name).lower():
                    eval_p = torch.clamp(eval_p, min=0.0, max=100.0)
                    eval_y = torch.clamp(eval_y, min=0.0, max=100.0)

                if v_masks_cat.numel() > 0:
                    val_mask = v_masks_cat.bool() & ~torch.isnan(eval_y) & ~torch.isnan(eval_p)
                else:
                    val_mask = ~torch.isnan(eval_y) & ~torch.isnan(eval_p)

                if val_mask.sum() > 0:
                    val_metric = evaluator.compute(
                        eval_p[val_mask], eval_y[val_mask], target_metric
                    )
                else:
                    val_metric = 0.0
                scheduler.step()

                improved = (
                    val_metric > best_metric if higher_is_better else val_metric < best_metric
                )
                if improved:
                    best_metric = val_metric
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience and not dry_run:
                        break

                if dry_run:
                    break

            console.print(
                f"  Model #{idx + 1} Best Epoch: {best_epoch} (Val {target_metric.upper()}={best_metric:.4f})"
            )

            # Load best weights
            if best_state_dict is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

            # Evaluate Model on Validation Set (with best checkpoint)
            model.eval()
            best_v_preds_list = []
            best_v_labels_list = []
            best_v_masks_list = []
            with torch.no_grad():
                for batch in val_loader:
                    dev_batch = _batch_to_device(batch, device)
                    preds = model(dev_batch)
                    if task_type == "multi_task":
                        best_v_preds_list.append(preds[:, primary_idx].detach().cpu())
                        best_v_labels_list.append(
                            dev_batch["labels"][:, primary_idx].detach().cpu()
                        )
                        if "mask" in dev_batch and dev_batch["mask"] is not None:
                            best_v_masks_list.append(
                                dev_batch["mask"][:, primary_idx].detach().cpu()
                            )
                    else:
                        best_v_preds_list.append(preds.squeeze(-1).detach().cpu())
                        best_v_labels_list.append(dev_batch["labels"].detach().cpu())
                        if "mask" in dev_batch and dev_batch["mask"] is not None:
                            best_v_masks_list.append(dev_batch["mask"].squeeze(-1).detach().cpu())
                    if dry_run:
                        break

            v_preds_cat = (
                torch.cat(best_v_preds_list, dim=0) if best_v_preds_list else torch.tensor([])
            )
            v_labels_cat = (
                torch.cat(best_v_labels_list, dim=0) if best_v_labels_list else torch.tensor([])
            )
            v_masks_cat = (
                torch.cat(best_v_masks_list, dim=0) if best_v_masks_list else torch.tensor([])
            )

            # Evaluate Model on Test Set (with best checkpoint)
            t_preds_list = []
            t_labels_list = []
            t_masks_list = []
            with torch.no_grad():
                for batch in test_loader:
                    dev_batch = _batch_to_device(batch, device)
                    preds = model(dev_batch)
                    if task_type == "multi_task":
                        t_preds_list.append(preds[:, primary_idx].detach().cpu())
                        t_labels_list.append(dev_batch["labels"][:, primary_idx].detach().cpu())
                        if "mask" in dev_batch and dev_batch["mask"] is not None:
                            t_masks_list.append(dev_batch["mask"][:, primary_idx].detach().cpu())
                    else:
                        t_preds_list.append(preds.squeeze(-1).detach().cpu())
                        t_labels_list.append(dev_batch["labels"].detach().cpu())
                        if "mask" in dev_batch and dev_batch["mask"] is not None:
                            t_masks_list.append(dev_batch["mask"].squeeze(-1).detach().cpu())
                    if dry_run:
                        break

            t_preds_cat = torch.cat(t_preds_list, dim=0) if t_preds_list else torch.tensor([])
            t_labels_cat = torch.cat(t_labels_list, dim=0) if t_labels_list else torch.tensor([])
            t_masks_cat = torch.cat(t_masks_list, dim=0) if t_masks_list else torch.tensor([])

            if getattr(data_module, "standardize_target", False) and eval_task_type == "regression":
                if task_type == "multi_task":
                    stat = getattr(data_module, "task_stats", {}).get(
                        primary_task, {"mean": 0.0, "std": 1.0}
                    )
                    mean = stat["mean"]
                    std = stat["std"]
                else:
                    mean = getattr(data_module, "target_mean", 0.0)
                    std = getattr(data_module, "target_std", 1.0)
                real_t_p = t_preds_cat * std + mean
                real_t_y = t_labels_cat * std + mean
                real_v_p = v_preds_cat * std + mean
                real_v_y = v_labels_cat * std + mean
            else:
                real_t_p = t_preds_cat
                real_t_y = t_labels_cat
                real_v_p = v_preds_cat
                real_v_y = v_labels_cat

            if is_logit:
                real_t_p = 100.0 * torch.sigmoid(real_t_p)
                real_t_y = 100.0 * torch.sigmoid(real_t_y)
                real_v_p = 100.0 * torch.sigmoid(real_v_p)
                real_v_y = 100.0 * torch.sigmoid(real_v_y)
            elif "ppbr" in str(primary_task or dataset_name).lower():
                real_t_p = torch.clamp(real_t_p, min=0.0, max=100.0)
                real_t_y = torch.clamp(real_t_y, min=0.0, max=100.0)
                real_v_p = torch.clamp(real_v_p, min=0.0, max=100.0)
                real_v_y = torch.clamp(real_v_y, min=0.0, max=100.0)

            if t_masks_cat.numel() > 0:
                t_valid = t_masks_cat.bool() & ~torch.isnan(real_t_y) & ~torch.isnan(real_t_p)
            else:
                t_valid = ~torch.isnan(real_t_y) & ~torch.isnan(real_t_p)

            if v_masks_cat.numel() > 0:
                v_valid = v_masks_cat.bool() & ~torch.isnan(real_v_y) & ~torch.isnan(real_v_p)
            else:
                v_valid = ~torch.isnan(real_v_y) & ~torch.isnan(real_v_p)

            eval_t_p = real_t_p[t_valid]
            eval_t_y = real_t_y[t_valid]
            eval_v_p = real_v_p[v_valid]
            eval_v_y = real_v_y[v_valid]

            test_labels_real = eval_t_y
            val_labels_real = eval_v_y

            all_test_preds.append(eval_t_p)
            all_val_preds.append(eval_v_p)

            m_metrics = evaluator.compute_all(eval_t_p, eval_t_y, eval_task_type)
            model_metrics_list.append(m_metrics)
            console.print(
                f"  Model #{idx + 1} Test Results ({eval_task_type}): R2={m_metrics.get('r2', 0):.4f} | "
                f"RMSE={m_metrics.get('rmse', 0):.4f} | MAE={m_metrics.get('mae', 0):.4f} | "
                f"Pearson={m_metrics.get('pearson', 0):.4f}"
            )

            # Save individual model checkpoint
            os.makedirs(checkpoint_dir, exist_ok=True)
            chk_path = os.path.join(checkpoint_dir, f"model_seed_{seed}.pt")
            torch.save(
                {"model_state": model.state_dict(), "seed": seed, "config": curr_model_cfg},
                chk_path,
            )

        # 2. Ensemble Averaging
        if all_test_preds and test_labels_real is not None:
            ens_test_preds = torch.stack(all_test_preds, dim=0).mean(dim=0)
            ens_test_metrics = evaluator.compute_all(
                ens_test_preds, test_labels_real, eval_task_type
            )

            ens_val_preds = torch.stack(all_val_preds, dim=0).mean(dim=0)
            ens_val_metrics = evaluator.compute_all(ens_val_preds, val_labels_real, eval_task_type)

            table = Table(
                title=f"★ Ensemble Benchmark Results: {dataset_name.upper()} (N={len(seed_list)} Models)"
            )
            table.add_column("Metric", style="bold cyan")
            for idx, s in enumerate(seed_list):
                table.add_column(f"M#{idx + 1} ({s})", justify="center")
            table.add_column("Individual Mean ± Std", justify="center", style="yellow")
            table.add_column("★ Ensemble", justify="center", style="bold green")
            table.add_column("Literature SOTA", justify="center", style="magenta")

            lit_refs = {
                "r2": "0.743 ± 0.018",
                "rmse": "0.325 ± 0.013",
                "mae": "0.242 ± 0.011",
                "pearson": "~0.86",
                "spearman": "~0.83",
            }

            def _fmt(vals):
                return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"

            row_metrics = ["r2", "rmse", "mae", "pearson", "spearman"]
            for m in row_metrics:
                m_vals = [ml.get(m, 0) for ml in model_metrics_list]
                indiv_cells = [f"{v:.4f}" for v in m_vals]
                table.add_row(
                    m.upper(),
                    *indiv_cells,
                    _fmt(m_vals),
                    f"[bold green]{ens_test_metrics.get(m, 0):.4f}[/bold green]",
                    lit_refs.get(m, "N/A"),
                )

            console.print("\n")
            console.print(table)

            tracker.log_metrics({f"ensemble_test_{k}": v for k, v in ens_test_metrics.items()})
            tracker.log_metrics({f"ensemble_val_{k}": v for k, v in ens_val_metrics.items()})

            summary_path = os.path.join(checkpoint_dir, "ensemble_summary.json")
            summary_data = {
                "dataset": dataset_name,
                "n_models": len(seed_list),
                "seeds": seed_list,
                "individual_models": model_metrics_list,
                "ensemble_test_metrics": ens_test_metrics,
                "ensemble_val_metrics": ens_val_metrics,
                "literature_benchmark": lit_refs,
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)
            console.print(f"\n[green]Saved ensemble summary to: {summary_path}[/green]")
    finally:
        tracker.finish()


@app.command()
def blend(
    config: str = typer.Option(
        "configs/config_distribution_mtl_random.yaml", help="Path to main YAML config"
    ),
    checkpoint_dir: str = typer.Option(
        "./models/checkpoint/ensemble", help="Directory containing trained model checkpoints"
    ),
    seeds: str = typer.Option(
        "42,101,202,303,404", "--seeds", help="Comma-separated seeds for ensemble models"
    ),
    gbdt_iter: int = typer.Option(300, help="Max iterations for GBDT"),
    gbdt_lr: float = typer.Option(0.05, help="Learning rate for GBDT"),
    gbdt_l2: float = typer.Option(2.0, help="L2 regularization for GBDT"),
    output_summary: str = typer.Option(
        "hybrid_blend_summary.json", help="Summary filename to save in checkpoint_dir"
    ),
):
    """Multi-Modal Hybrid Stacking (DMPNN Graph + GBDT Molecular Descriptors) with Parametric Calibration."""
    import json
    import os

    import numpy as np
    import torch
    from rich.table import Table

    from tdc_studio.core.registry import DATASETS, MODELS
    from tdc_studio.models.hybrid.gbdt_blend import GBDTDMPNNBlender, extract_molecular_features
    from tdc_studio.tracking.wandb_tracker import WandBTracker

    cfg = load_yaml(config)
    console.print(
        f"[bold green]Starting Hybrid Stacking & Calibration Pipeline[/bold green] (Config: {config})"
    )

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    # 1. Setup Data
    data_type = data_cfg.get("type", "admet_cluster_loader")
    data_cls = DATASETS.get(data_type)
    if "params" in data_cfg and isinstance(data_cfg["params"], dict):
        data_params = dict(data_cfg["params"])
    else:
        data_params = {k: v for k, v in data_cfg.items() if k not in ("type", "name", "batch_size")}

    data_module = data_cls(**data_params)
    data_module.prepare_data()
    batch_size = data_cfg.get("batch_size", 64)
    train_loader, val_loader, test_loader = data_module.setup_loaders(batch_size=batch_size)

    primary_task = getattr(data_module, "primary_task", "ppbr_az")
    task_names = getattr(data_module, "task_names", [primary_task])
    primary_idx = task_names.index(primary_task) if primary_task in task_names else 0
    is_logit = getattr(data_module, "task_transforms", {}).get(primary_task) == "logit"

    stat = getattr(data_module, "task_stats", {}).get(primary_task, {"mean": 0.0, "std": 1.0})
    mean = stat["mean"]
    std = stat["std"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(
        f"Device: [cyan]{device}[/cyan] | Primary Task: [bold cyan]{primary_task}[/bold cyan] | Logit Space: [yellow]{is_logit}[/yellow]"
    )

    # 2. Extract Training Features and Fit GBDT
    train_df = data_module.splits["train"]
    primary_train = train_df.dropna(subset=[primary_task])
    smiles_train = primary_train["Canon_SMILES"].tolist()
    y_train_raw = primary_train[primary_task].values.astype(float)

    console.print(
        f"[bold cyan]Extracting molecular features (1024-bit Morgan FP + 210 RDKit Descriptors) for {len(smiles_train)} training compounds...[/bold cyan]"
    )
    X_train, y_train_logit, _ = extract_molecular_features(
        smiles_train, y_train_raw, n_bits=1024, transform="logit" if is_logit else None
    )

    blender = GBDTDMPNNBlender(
        max_iter=gbdt_iter,
        learning_rate=gbdt_lr,
        l2_regularization=gbdt_l2,
    )
    console.print("[bold cyan]Fitting GBDT in thermodynamic Gibbs logit space...[/bold cyan]")
    blender.fit_gbdt(X_train, y_train_logit)

    # 3. Locate DMPNN Checkpoints (support multiple comma-separated dirs)
    dir_list = [d.strip() for d in checkpoint_dir.split(",") if d.strip()]
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    found_ckpts = []
    for c_dir in dir_list:
        for s in seed_list:
            p = os.path.join(c_dir, f"model_seed_{s}.pt")
            if os.path.exists(p):
                found_ckpts.append((s, p))

    if not found_ckpts:
        for c_dir in dir_list:
            for candidate in [
                os.path.join(c_dir, "best_model.pt"),
                "./models/checkpoint/best_model.pt",
            ]:
                if os.path.exists(candidate) and candidate not in [x[1] for x in found_ckpts]:
                    found_ckpts.append((0, candidate))
                    break

    if not found_ckpts:
        raise FileNotFoundError(
            f"No model checkpoints found in '{checkpoint_dir}' or standard paths."
        )

    console.print(
        f"[bold green]Found {len(found_ckpts)} DMPNN checkpoints across {len(dir_list)} directory/directories for evaluation.[/bold green]"
    )

    # 4. Run DMPNN Forward Passes on Validation and Test Sets
    all_dmpnn_v_logits = []
    all_dmpnn_t_logits = []
    v_labels_real = None
    t_labels_real = None
    v_smiles_list = []
    t_smiles_list = []

    model_cls = MODELS.get(model_cfg.get("type", "dmpnn_mtl"))

    for seed, chk_path in found_ckpts:
        console.print(f"Evaluating DMPNN Checkpoint (Seed {seed}): [yellow]{chk_path}[/yellow]")
        ckpt = torch.load(chk_path, map_location=device, weights_only=False)
        loaded_cfg = ckpt.get("config", model_cfg)
        model = model_cls(loaded_cfg).to(device)
        state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state_dict)
        model.eval()

        # Validation forward
        v_preds, v_labels, curr_v_smiles = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                dev_batch = _batch_to_device(batch, device)
                preds = model(dev_batch)
                if "mask" in dev_batch and dev_batch["mask"] is not None:
                    mask = dev_batch["mask"][:, primary_idx].bool()
                else:
                    mask = torch.ones(preds.shape[0], dtype=torch.bool, device=preds.device)

                if mask.any():
                    p = preds[mask, primary_idx].detach().cpu()
                    y = dev_batch["labels"][mask, primary_idx].detach().cpu()
                    v_preds.append(p)
                    v_labels.append(y)
                    if "drug_smiles_str" in batch:
                        curr_v_smiles.extend(
                            [
                                batch["drug_smiles_str"][i]
                                for i in range(len(batch["drug_smiles_str"]))
                                if mask[i].item()
                            ]
                        )

        v_p_cat = torch.cat(v_preds, dim=0).numpy()
        v_y_cat = torch.cat(v_labels, dim=0).numpy()
        # Unstandardize to get Gibbs logit space
        v_z = v_p_cat * std + mean
        all_dmpnn_v_logits.append(v_z)

        if v_labels_real is None:
            if is_logit:
                v_labels_real = 100.0 / (1.0 + np.exp(-np.clip(v_y_cat * std + mean, -40.0, 40.0)))
            else:
                v_labels_real = np.clip(v_y_cat * std + mean, 0.0, 100.0)
            if curr_v_smiles:
                v_smiles_list = curr_v_smiles
            else:
                val_df = data_module.splits["valid"]
                v_smiles_list = val_df.dropna(subset=[primary_task])["Canon_SMILES"].tolist()[
                    : len(v_z)
                ]

        # Test forward
        t_preds, t_labels, curr_t_smiles = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                dev_batch = _batch_to_device(batch, device)
                preds = model(dev_batch)
                if "mask" in dev_batch and dev_batch["mask"] is not None:
                    mask = dev_batch["mask"][:, primary_idx].bool()
                else:
                    mask = torch.ones(preds.shape[0], dtype=torch.bool, device=preds.device)

                if mask.any():
                    p = preds[mask, primary_idx].detach().cpu()
                    y = dev_batch["labels"][mask, primary_idx].detach().cpu()
                    t_preds.append(p)
                    t_labels.append(y)
                    if "drug_smiles_str" in batch:
                        curr_t_smiles.extend(
                            [
                                batch["drug_smiles_str"][i]
                                for i in range(len(batch["drug_smiles_str"]))
                                if mask[i].item()
                            ]
                        )

        t_p_cat = torch.cat(t_preds, dim=0).numpy()
        t_y_cat = torch.cat(t_labels, dim=0).numpy()
        t_z = t_p_cat * std + mean
        all_dmpnn_t_logits.append(t_z)

        if t_labels_real is None:
            if is_logit:
                t_labels_real = 100.0 / (1.0 + np.exp(-np.clip(t_y_cat * std + mean, -40.0, 40.0)))
            else:
                t_labels_real = np.clip(t_y_cat * std + mean, 0.0, 100.0)
            if curr_t_smiles:
                t_smiles_list = curr_t_smiles
            else:
                test_df = data_module.splits["test"]
                t_smiles_list = test_df.dropna(subset=[primary_task])["Canon_SMILES"].tolist()[
                    : len(t_z)
                ]

    # 5. Average Ensemble DMPNN Logits
    z_dmpnn_val = np.mean(all_dmpnn_v_logits, axis=0)
    z_dmpnn_test = np.mean(all_dmpnn_t_logits, axis=0)

    # 6. Extract GBDT Features for Val and Test and Predict Logits
    console.print(
        f"[bold cyan]Extracting validation molecular features ({len(v_smiles_list)} samples)...[/bold cyan]"
    )
    X_val, _, _ = extract_molecular_features(
        v_smiles_list, None, n_bits=1024, transform="logit" if is_logit else None
    )
    z_gbdt_val = blender.predict_gbdt(X_val)

    console.print(
        f"[bold cyan]Extracting test molecular features ({len(t_smiles_list)} samples)...[/bold cyan]"
    )
    X_test, _, _ = extract_molecular_features(
        t_smiles_list, None, n_bits=1024, transform="logit" if is_logit else None
    )
    z_gbdt_test = blender.predict_gbdt(X_test)

    # 7. Fit Calibration and Blending on Validation Set
    console.print(
        "[bold yellow]Optimizing Blending Weight (w) and Calibration Parameters (alpha, beta) on Validation Set...[/bold yellow]"
    )
    calib_res = blender.fit_calibration_and_blend(z_dmpnn_val, z_gbdt_val, v_labels_real)
    console.print(
        f"[bold green]Optimal Validation Parameters:[/bold green] "
        f"w_dmpnn={calib_res['optimal_w_dmpnn']:.3f}, w_gbdt={calib_res['optimal_w_gbdt']:.3f}, "
        f"alpha={calib_res['optimal_alpha']:.3f}, beta={calib_res['optimal_beta']:.3f} | "
        f"Val R²={calib_res['val_r2']:.4f}, Val MAE={calib_res['val_mae']:.2f}%"
    )

    # 8. Evaluate on Test Set
    console.print("[bold green]Evaluating Standalone vs. Hybrid Models on Test Set...[/bold green]")
    test_results = blender.evaluate_test(
        z_dmpnn_test,
        z_gbdt_test,
        t_labels_real,
        z_dmpnn_val=z_dmpnn_val,
        z_gbdt_val=z_gbdt_val,
        y_val_real=v_labels_real,
    )

    # 9. Format Results Table
    d_raw = test_results["dmpnn_raw_metrics"]
    d_cal = test_results.get("dmpnn_calibrated_metrics", d_raw)
    g_raw = test_results["gbdt_raw_metrics"]
    g_cal = test_results.get("gbdt_calibrated_metrics", g_raw)
    hyb = test_results["hybrid_metrics"]

    table = Table(title="★ Multi-Modal Hybrid Stacking & Calibration Benchmark Results (PPBR AZ)")
    table.add_column("Model Architecture", style="bold")
    table.add_column("Test R²", justify="center", style="bold cyan")
    table.add_column("Test MAE (%)", justify="center", style="green")
    table.add_column("Test RMSE (%)", justify="center")
    table.add_column("Pearson (r)", justify="center", style="yellow")
    table.add_column("Spearman (ρ)", justify="center")

    table.add_row(
        "DMPNN Ensemble (Raw Sigmoid)",
        f"{d_raw['r2']:.4f}",
        f"{d_raw['mae']:.2f}%",
        f"{d_raw['rmse']:.2f}%",
        f"{d_raw['pearson']:.4f}",
        f"{d_raw['spearman']:.4f}",
    )
    table.add_row(
        "DMPNN Ensemble (Calibrated)",
        f"{d_cal['r2']:.4f}",
        f"{d_cal['mae']:.2f}%",
        f"{d_cal['rmse']:.2f}%",
        f"{d_cal['pearson']:.4f}",
        f"{d_cal['spearman']:.4f}",
    )
    table.add_row(
        "GBDT Descriptors (Raw Sigmoid)",
        f"{g_raw['r2']:.4f}",
        f"{g_raw['mae']:.2f}%",
        f"{g_raw['rmse']:.2f}%",
        f"{g_raw['pearson']:.4f}",
        f"{g_raw['spearman']:.4f}",
    )
    table.add_row(
        "GBDT Descriptors (Calibrated)",
        f"{g_cal['r2']:.4f}",
        f"{g_cal['mae']:.2f}%",
        f"{g_cal['rmse']:.2f}%",
        f"{g_cal['pearson']:.4f}",
        f"{g_cal['spearman']:.4f}",
    )
    table.add_row(
        "★ Hybrid Stacker (DMPNN + GBDT + Calibrated)",
        f"[bold green]{hyb['r2']:.4f}[/bold green]",
        f"[bold green]{hyb['mae']:.2f}%[/bold green]",
        f"[bold green]{hyb['rmse']:.2f}%[/bold green]",
        f"[bold green]{hyb['pearson']:.4f}[/bold green]",
        f"[bold green]{hyb['spearman']:.4f}[/bold green]",
    )
    table.add_row(
        "Literature / ADMETlab Benchmark",
        "0.60 ~ 0.73",
        "7.4% ~ 8.6%",
        "11% ~ 13%",
        "~0.75",
        "~0.73",
    )
    console.print(table)

    # 10. Save Summary
    save_dir = dir_list[0] if "dir_list" in locals() and dir_list else checkpoint_dir
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, output_summary)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    console.print(f"[bold green]Saved Hybrid Benchmark Summary to: {summary_path}[/bold green]")

    # 11. Log to W&B
    tracking_cfg = dict(cfg.get("tracking", {}))
    if tracking_cfg.get("enabled", False):
        tracking_cfg["run_name"] = "ppbr_hybrid_blend_calibration"
        tracking_cfg["config"] = cfg
        tracker = WandBTracker(tracking_cfg)
        try:
            tracker.log_metrics(
                {
                    "test_hybrid_r2": hyb["r2"],
                    "test_hybrid_mae": hyb["mae"],
                    "test_hybrid_rmse": hyb["rmse"],
                    "test_hybrid_pearson": hyb["pearson"],
                    "test_hybrid_spearman": hyb["spearman"],
                    "test_dmpnn_raw_r2": d_raw["r2"],
                    "test_dmpnn_cal_r2": d_cal["r2"],
                    "test_gbdt_raw_r2": g_raw["r2"],
                    "test_gbdt_cal_r2": g_cal["r2"],
                    "optimal_w_dmpnn": blender.optimal_w,
                    "optimal_w_gbdt": 1.0 - blender.optimal_w,
                    "optimal_alpha": blender.optimal_alpha,
                    "optimal_beta": blender.optimal_beta,
                }
            )
        finally:
            tracker.finish()


@app.command()
def export(
    checkpoint_dir: str = typer.Option(
        "./models/checkpoint",
        help="Directory containing trained checkpoint (best_model.pt or model.pt, config.json)",
    ),
    output_dir: str = typer.Option("./models/export", help="Target output directory for serving"),
    checkpoint_name: Optional[str] = typer.Option(
        None, help="Specific weights file name inside checkpoint_dir"
    ),
):
    """Export and package trained model for production serving."""
    from tdc_studio.serving.exporter import export_production_package

    console.print(
        f"[bold cyan]Packaging model for production[/bold cyan] from '{checkpoint_dir}' to '{output_dir}'..."
    )
    manifest = export_production_package(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        checkpoint_name=checkpoint_name,
    )

    console.print(f"[bold green]Model successfully exported to: {output_dir}[/bold green]")
    console.print(
        f"Architecture: [yellow]{manifest.get('model_type')}[/yellow] | "
        f"Task: [yellow]{manifest.get('task_type')}[/yellow] | "
        f"Parameters: [yellow]{manifest.get('parameter_count'):,}[/yellow]"
    )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host address"),
    port: int = typer.Option(8000, help="Port to listen on"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    model_dir: Optional[str] = typer.Option(
        None,
        "--model-dir",
        help="Path to exported model directory (contains model.pt and config.json)",
    ),
):
    """Start the FastAPI inference microservice with optional model directory."""
    import uvicorn

    if model_dir:
        abs_model_dir = str(Path(model_dir).resolve())
        os.environ["MODEL_DIR"] = abs_model_dir
        console.print(
            f"[bold green]Starting TDC-Studio Serving API & Biomedical Dashboard[/bold green] on http://{host}:{port}..."
        )
    console.print(
        f"  👉 [bold cyan]Interactive Web Dashboard[/bold cyan]: http://localhost:{port}/"
    )
    console.print(
        f"  👉 [bold dim]Swagger API Documentation[/bold dim]: http://localhost:{port}/docs"
    )
    uvicorn.run("tdc_studio.serving.app:app", host=host, port=port, workers=workers)


@app.command("ui")
def ui_serve(
    host: str = typer.Option("127.0.0.1", help="Host address"),
    port: int = typer.Option(8000, help="Port to listen on"),
    model_dir: Optional[str] = typer.Option(
        None, "--model-dir", help="Path to exported model directory"
    ),
):
    """Launch the interactive biomedical web dashboard."""
    serve(host=host, port=port, workers=1, model_dir=model_dir)


@app.command("batch-predict")
def batch_predict_cli(
    input_file: str = typer.Argument(..., help="Path to input file (CSV, TSV, or SDF)"),
    output_file: Optional[str] = typer.Option(
        None, "-o", "--output", help="Path to output file (default: <input>_admet_results.<format>)"
    ),
    export_format: str = typer.Option("csv", "--format", help="Output format ('csv' or 'xlsx')"),
    model_dir: Optional[str] = typer.Option(
        "models/export", "--model-dir", help="Path to exported model directory"
    ),
):
    """Run batch 22+ ADMET, Lipinski Rule of 5, and PBPK screening on molecular libraries."""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
    from rich.table import Table

    from tdc_studio.serving.batch_engine import BatchScreeningEngine
    from tdc_studio.serving.unified_pipeline import UnifiedADMETPipeline

    in_path = Path(input_file).resolve()
    if not in_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input file '{in_path}' does not exist.")
        raise typer.Exit(code=1)

    console.print("[bold cyan]TDC-Studio High-Throughput Batch Screening[/bold cyan]")
    console.print(f"  📁 Reading molecular library from: [yellow]{in_path}[/yellow]")

    pipeline = UnifiedADMETPipeline.from_exported_directory(model_dir)
    engine = BatchScreeningEngine(pipeline=pipeline)

    with open(in_path, "rb") as f:
        content = f.read()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Screening compounds...", total=100)
        df, summary = engine.screen_file(content, in_path.name)
        progress.update(task, completed=100)

    # Determine output path
    if output_file is None:
        stem = in_path.stem
        out_ext = ".xlsx" if export_format.lower() == "xlsx" else ".csv"
        out_path = in_path.parent / f"{stem}_admet_results{out_ext}"
    else:
        out_path = Path(output_file).resolve()

    if str(out_path).lower().endswith(".xlsx"):
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    console.print("\n[bold green]✅ Batch screening complete![/bold green]")
    console.print(f"  💾 Results exported to: [bold underline]{out_path}[/bold underline]")

    # Print summary table
    table = Table(title="Batch Screening Summary Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right", style="green")

    total = summary["total_molecules"]
    table.add_row("Total Compounds Screened", str(total), "100.0%")
    table.add_row(
        "Lipinski Rule of 5 Compliant", str(summary["ro5_passed"]), f"{summary['ro5_pass_rate']}%"
    )
    table.add_row(
        "Low Cardiotoxicity (hERG < 0.5)",
        str(summary["herg_safe_count"]),
        f"{summary['herg_safe_rate']}%",
    )
    table.add_row(
        "Non-Mutagenic (AMES < 0.5)",
        str(summary["ames_safe_count"]),
        f"{summary['ames_safe_rate']}%",
    )
    console.print(table)


@app.command("ti")
@app.command("therapeutic-index")
def therapeutic_index_cli(
    smiles: str = typer.Argument(..., help="Candidate drug molecule SMILES"),
    kd: Optional[float] = typer.Option(None, "--kd", help="On-target binding affinity Kd in nM"),
    target_seq: Optional[str] = typer.Option(
        None, "--target-seq", help="Target amino acid sequence"
    ),
    dose: float = typer.Option(100.0, "--dose", help="Reference oral dose in mg"),
    herg_ic50: Optional[float] = typer.Option(None, "--herg-ic50", help="Explicit hERG IC50 in nM"),
    model_dir: Optional[str] = typer.Option(
        "models/export", "--model-dir", help="Path to exported models"
    ),
):
    """Evaluate Therapeutic Index, hERG Safety Window, and Clinical Developability."""
    from rich.panel import Panel
    from rich.table import Table

    from tdc_studio.evaluation.therapeutic_index import TherapeuticIndexEngine
    from tdc_studio.serving.unified_pipeline import UnifiedADMETPipeline

    console.print(
        "\n[bold cyan]🧪 TDC-Studio Therapeutic Index & Clinical Developability Engine[/bold cyan]"
    )
    console.print(f"  Molecule: [bold yellow]{smiles}[/bold yellow]")

    pipeline = UnifiedADMETPipeline.from_exported_directory(model_dir)
    engine = TherapeuticIndexEngine(admet_pipeline=pipeline)

    profile = engine.compute(
        smiles=smiles,
        target_kd_nm=kd,
        target_sequence=target_seq,
        herg_ic50_nm=herg_ic50,
        dose_mg=dose,
    )

    # Format colors
    score = profile.clinical_developability_score
    score_color = "green" if score >= 80 else ("yellow" if score >= 60 else "red")
    tier_color = (
        "green"
        if "Safe" in profile.herg_risk_tier
        else ("yellow" if "Borderline" in profile.herg_risk_tier else "red")
    )

    table = Table(title="Pharmacological Safety Margin & Potency Profile", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Assessment", style="bold")

    table.add_row(
        "Target Potency (Kd)", f"{profile.target_kd_nm:.2f} nM", f"pKd = {profile.target_pkd:.2f}"
    )
    table.add_row(
        "hERG IC50 (Potassium Channel)", f"{profile.herg_ic50_nm:.1f} nM", "TDC Blocker Calibration"
    )
    table.add_row(
        "hERG Safety Margin (IC50 / Kd)",
        f"{profile.herg_safety_margin:.1f}x",
        f"[{tier_color}]{profile.herg_risk_tier}[/{tier_color}]",
    )
    table.add_row(
        "Therapeutic Window (log10)",
        f"{profile.herg_therapeutic_window_log10:.2f}",
        "Target Window >= 2.0",
    )
    table.add_row(
        "DILI Hepatotoxicity Risk",
        f"{profile.dili_risk_probability:.2%}",
        "[green]Low[/green]" if profile.dili_risk_probability < 0.5 else "[red]High Risk[/red]",
    )
    table.add_row(
        "ClinTox Clinical Failure Risk",
        f"{profile.clintox_risk_probability:.2%}",
        "[green]Low[/green]" if profile.clintox_risk_probability < 0.5 else "[red]High Risk[/red]",
    )
    table.add_row(
        "AMES Mutagenicity Risk",
        f"{profile.ames_mutagenicity_probability:.2%}",
        "[green]Negative[/green]"
        if profile.ames_mutagenicity_probability < 0.5
        else "[red]Positive (Alert)[/red]",
    )

    if profile.pbpk_cmax_free_ug_ml is not None:
        table.add_row(
            f"PBPK Cmax (Free, {dose}mg dose)",
            f"{profile.pbpk_cmax_free_ug_ml:.4f} ug/mL",
            "Unbound in vivo systemic exposure",
        )
    if profile.in_vivo_herg_margin is not None:
        table.add_row(
            "In Vivo Free hERG Margin",
            f"{profile.in_vivo_herg_margin:.1f}x",
            "FDA S7B recommends >= 30x",
        )

    console.print(table)

    comp = profile.component_scores
    console.print(
        Panel(
            f"[bold {score_color}]Clinical Developability Index (CDI): {score:.1f} / 100[/bold {score_color}] "
            f"([bold]{profile.developability_tier}[/bold])\n\n"
            f"  • Potency Pillar: [cyan]{comp.potency:.1f} / 25[/cyan]\n"
            f"  • Safety Window Pillar: [cyan]{comp.safety_window:.1f} / 25[/cyan]\n"
            f"  • Organ Toxicology Pillar: [cyan]{comp.organ_toxicology:.1f} / 25[/cyan]\n"
            f"  • Human PK Druggability Pillar: [cyan]{comp.human_pk:.1f} / 25[/cyan]",
            title="[bold]Summary Developability Score[/bold]",
            border_style=score_color,
        )
    )

    if profile.warnings:
        console.print("\n[bold yellow]⚠️ Pharmacological & Regulatory Warnings:[/bold yellow]")
        for w in profile.warnings:
            console.print(f"  • [yellow]{w}[/yellow]")

    if profile.recommendations:
        console.print("\n[bold green]💡 Medicinal Chemistry Recommendations:[/bold green]")
        for r in profile.recommendations:
            console.print(f"  • [green]{r}[/green]")
    console.print()


switch_app = typer.Typer(help="Manage and switch Colab CLI accounts (tokens)")
remote_app.add_typer(switch_app, name="switch")


@remote_app.command("accounts")
@switch_app.command("list")
def list_accounts():
    """List all registered Colab accounts and display the active one."""
    from rich.table import Table

    from tdc_studio.remote.colab_account import ColabAccountManager

    mgr = ColabAccountManager()
    active = mgr.get_active_account()
    accounts = mgr.list_accounts()

    table = Table(title="Google Colab CLI Accounts")
    table.add_column("Status", style="bold")
    table.add_column("Account Name", style="cyan")
    table.add_column("Token Path", style="dim")

    for acc in accounts:
        if acc["is_active"]:
            status = "[bold green]★ ACTIVE[/bold green]"
            name = f"[bold green]{acc['name']}[/bold green]"
        else:
            status = "  Standby"
            name = acc["name"]
        table.add_row(status, name, acc["path"])

    console.print(table)
    console.print(f"\nCurrently active account: [bold green]{active or 'None'}[/bold green]")


@switch_app.command("use")
def switch_account(account: str = typer.Argument(..., help="Account name to activate")):
    """Switch active Colab credentials to the specified account."""
    from tdc_studio.remote.colab_account import ColabAccountManager

    mgr = ColabAccountManager()
    try:
        mgr.use_account(account)
        console.print(
            f"[bold green]Successfully switched active Colab account to:[/bold green] [cyan]{account}[/cyan]"
        )
    except Exception as e:
        console.print(f"[bold red]Failed to switch account:[/bold red] {e}")
        raise typer.Exit(1)


@switch_app.command("save")
def save_account(
    account: str = typer.Argument(..., help="Name to save current active credentials as"),
):
    """Save currently active credentials as a named account."""
    from tdc_studio.remote.colab_account import ColabAccountManager

    mgr = ColabAccountManager()
    try:
        path = mgr.save_account(account)
        console.print(
            f"[bold green]Saved current Colab credentials as:[/bold green] [cyan]{account}[/cyan] ({path})"
        )
    except Exception as e:
        console.print(f"[bold red]Failed to save account:[/bold red] {e}")
        raise typer.Exit(1)


@switch_app.command("new")
def new_account(
    account: str = typer.Argument(..., help="Name for the newly authenticated account"),
):
    """Authenticate a new Google account via OAuth browser flow and save credentials."""
    from tdc_studio.remote.colab_account import ColabAccountManager

    mgr = ColabAccountManager()
    try:
        success = mgr.new_account(account)
        if success:
            console.print(
                f"[bold green]Successfully created and registered account:[/bold green] [cyan]{account}[/cyan]"
            )
        else:
            console.print("[bold yellow]Authentication flow was aborted or failed.[/bold yellow]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error during account creation:[/bold red] {e}")
        raise typer.Exit(1)


@switch_app.command("delete")
def delete_account(account: str = typer.Argument(..., help="Account name to delete")):
    """Delete saved credentials for an account."""
    from tdc_studio.remote.colab_account import ColabAccountManager

    mgr = ColabAccountManager()
    try:
        mgr.delete_account(account)
        console.print(
            f"[bold green]Deleted credentials for account:[/bold green] [cyan]{account}[/cyan]"
        )
    except Exception as e:
        console.print(f"[bold red]Failed to delete account:[/bold red] {e}")
        raise typer.Exit(1)


@remote_app.command("status")
def remote_status():
    """Check Google Colab GPU session status, active accounts, and CLI connectivity."""
    import shutil
    import subprocess

    from rich.table import Table

    from tdc_studio.remote.colab_account import ColabAccountManager

    console.print("[bold cyan]Google Colab Remote Environment Status[/bold cyan]\n")

    colab_path = shutil.which("colab")
    if not colab_path:
        console.print("[bold red]Colab CLI ('colab') is NOT found on PATH.[/bold red]")
        console.print("Please install via: [yellow]pip install google-colab-cli[/yellow]")
        raise typer.Exit(1)

    mgr = ColabAccountManager()
    active_acc = mgr.get_active_account()
    accounts = mgr.list_accounts()

    console.print(f"Colab CLI Executable: [green]{colab_path}[/green]")
    console.print(
        f"Active Google Account: [bold yellow]{active_acc or 'None / Not Authenticated'}[/bold yellow]\n"
    )

    # Check Active Colab Sessions
    try:
        res = subprocess.run(["colab", "sessions"], capture_output=True, text=True, check=False)
        output = (res.stdout or "") + (res.stderr or "")
        lines = [line.strip() for line in output.splitlines() if line.strip()]

        table = Table(title="Colab GPU Sessions")
        table.add_column("Session Identifier", style="cyan")
        table.add_column("Status / State", style="bold")
        table.add_column("Details", style="dim")

        found_session = False
        for line in lines:
            if "no active sessions" in line.lower():
                table.add_row(
                    "None",
                    "[yellow]Inactive[/yellow]",
                    "No active sessions currently running on server",
                )
                found_session = True
                break
            elif "pruned" in line.lower():
                continue
            else:
                table.add_row(
                    line,
                    "[bold green]Online / Active[/bold green]",
                    "Available for remote execution",
                )
                found_session = True

        if not found_session:
            table.add_row("None", "[yellow]Inactive[/yellow]", "No active sessions found")

        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Failed to check Colab sessions:[/bold red] {e}")

    if accounts:
        acc_summary = ", ".join(
            [
                f"[bold green]{a['name']}[/bold green]"
                if a["is_active"]
                else f"[cyan]{a['name']}[/cyan]"
                for a in accounts
            ]
        )
        console.print(f"\nRegistered Accounts ({len(accounts)}): {acc_summary}")


@remote_app.command("run")
def remote_run(
    command: str = typer.Option(
        "tdc-studio train --config configs/config.yaml", help="Command to run on Colab"
    ),
    gpu: str = typer.Option("t4", help="GPU tier: t4, l4, a100, v100"),
    account: Optional[str] = typer.Option(None, help="Colab account to use for this execution"),
    auto_switch: bool = typer.Option(
        True, "--auto-switch/--no-auto-switch", help="Auto-switch account if GPU quota is exceeded"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command without executing"),
):
    """Dispatch training or HPO job to Google Colab Cloud GPU via google-colab-cli."""
    from tdc_studio.remote.colab_runner import ColabRunner

    runner = ColabRunner(gpu_type=gpu, account=account, auto_switch_on_quota=auto_switch)
    console.print(f"[bold cyan]Dispatching remote job to Google Colab (GPU: {gpu})[/bold cyan]...")
    cmd = runner.build_run_command(command)
    console.print(f"Generated Colab CLI Command: [yellow]{' '.join(cmd)}[/yellow]")

    returncode = runner.run_remote_job(command, dry_run=dry_run)
    if returncode == 0:
        console.print(
            "[bold green]Remote Colab Job completed successfully (or dry-run passed).[/bold green]"
        )
    else:
        console.print(f"[bold red]Remote Colab Job failed with code {returncode}.[/bold red]")


@remote_app.command("exec")
def remote_exec(
    command: Optional[str] = typer.Option(
        None, help="Command payload for deploy/colab_runner_job.py"
    ),
    session: Optional[str] = typer.Option(None, "-s", "--session", help="Active Colab session ID"),
    script: Optional[str] = typer.Option(
        None, "-f", "--script", help="Local python script to execute"
    ),
    script_args: Optional[List[str]] = typer.Option(
        None, "--arg", help="Arguments to inject and pass to the script"
    ),
    account: Optional[str] = typer.Option(None, help="Colab account to use"),
    timeout: float = typer.Option(7200.0, "--timeout", help="Execution timeout in seconds"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command without executing"),
):
    """Execute code or a local script directly on an active Google Colab instance via `colab exec`."""
    from tdc_studio.remote.colab_runner import ColabRunner

    runner = ColabRunner(account=account)
    console.print(
        "[bold cyan]Executing remote job on Colab instance via `colab exec`[/bold cyan]..."
    )

    returncode = runner.run_remote_exec(
        command_to_run=command or "tdc-studio train --config configs/config.yaml",
        session=session,
        script_file=script,
        script_args=script_args,
        timeout=timeout,
        dry_run=dry_run,
    )
    if returncode == 0:
        console.print(
            "[bold green]Remote Colab Exec completed successfully (or dry-run passed).[/bold green]"
        )
    else:
        console.print(f"[bold red]Remote Colab Exec failed with code {returncode}.[/bold red]")


@remote_app.command("export-notebook")
def export_notebook(
    output: str = typer.Option("tdc_studio_colab.ipynb", help="Output .ipynb path"),
    repo_url: str = typer.Option(
        "https://github.com/your-org/tdc-studio.git", help="Git repo to clone on Colab"
    ),
    command: str = typer.Option(
        "tdc-studio train --config configs/config.yaml", help="Command to run"
    ),
):
    """Export a standalone Google Colab notebook (.ipynb) for browser execution."""
    from tdc_studio.remote.notebook_gen import export_notebook_file

    export_notebook_file(output, repo_url=repo_url, run_command=command)
    console.print(f"[bold green]Generated Google Colab notebook at:[/bold green] {output}")


@retro_app.command("single-step")
def retro_single_step_cli(
    smiles: str = typer.Option(..., "--smiles", "-s", help="Target molecule product SMILES"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of precursor candidate sets"),
    reaction_type: Optional[int] = typer.Option(
        None, "--reaction-type", "-t", help="USPTO reaction class (1-10)"
    ),
):
    """Predict candidate precursor reactant sets for a target molecule."""
    from tdc_studio.models.retrosynthesis.rule_policy import RuleRetroPolicy

    policy = RuleRetroPolicy()
    candidates = policy.predict_reactants(smiles, top_k=top_k, reaction_type=reaction_type)

    console.print(f"[bold cyan]🎯 Target Molecule:[/bold cyan] {smiles}")
    console.print(f"[bold green]Top-{len(candidates)} Retrosynthetic Precursors:[/bold green]")
    for idx, (reactants, score) in enumerate(candidates, start=1):
        console.print(f"  {idx}. [yellow]{reactants}[/yellow] (confidence: {score:.3f})")


@retro_app.command("plan")
def retro_plan_cli(
    smiles: str = typer.Option(
        ..., "--smiles", "-s", help="Target molecule SMILES to plan pathway for"
    ),
    top_k: int = typer.Option(
        3,
        "--top-k",
        "-k",
        help="Number of candidate routes to find (1 = optimal only, >1 = alternative routes)",
    ),
    banned: Optional[str] = typer.Option(
        None, "--banned", help="Comma-separated SMILES to ban/exclude from commercial stock"
    ),
    min_diversity: float = typer.Option(
        0.25, "--min-diversity", help="Minimum diversity distance between routes"
    ),
    max_depth: int = typer.Option(5, "--max-depth", "-d", help="Maximum search tree depth"),
    timeout: float = typer.Option(5.0, "--timeout", help="Search timeout in seconds"),
    compare: bool = typer.Option(
        True, "--compare", help="Display comparative Markdown table of all routes"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Optional JSON output file path"
    ),
    render_mermaid: bool = typer.Option(False, "--render-mermaid", help="Print Mermaid diagram"),
):
    """Plan multi-step chemical synthesis route from commercial stock reagents."""
    import json

    from tdc_studio.retrosynthesis.planner import RetroPlanner

    banned_list = [s.strip() for s in banned.split(",") if s.strip()] if banned else None

    planner = RetroPlanner(policy_type="rule", max_depth=max_depth, timeout_sec=timeout)
    routes = planner.plan_routes(
        target_smiles=smiles,
        top_k=top_k,
        diversity_threshold=min_diversity,
        banned_smiles=banned_list,
        timeout_sec=timeout,
    )

    champion = routes[0] if routes else None

    if compare and len(routes) > 1:
        console.print("\n[bold cyan]📊 Multi-Route Comparison Summary:[/bold cyan]")
        console.print(planner.render_comparison_table(routes))
        console.print("\n" + "=" * 70 + "\n")

    if champion:
        tree_str = planner.render_tree(champion)
        console.print(tree_str)

    if render_mermaid:
        console.print("\n[bold magenta]Mermaid Diagram(s):[/bold magenta]")
        if len(routes) > 1:
            console.print(planner.render_multi_mermaid(routes))
        elif champion:
            console.print(planner.render_mermaid(champion))

    if output:
        out_payload = {
            "target_smiles": smiles,
            "routes_count": len(routes),
            "routes": [r.to_dict() for r in routes],
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)
        console.print(f"[bold green]Saved routes JSON to:[/bold green] {output}")


@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport protocol: 'stdio' or 'sse'"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host IP for SSE transport"),
    port: int = typer.Option(8000, "--port", "-p", help="Port number for SSE transport"),
):
    """Launch the TDC-Studio Model Context Protocol (MCP) Server for AI Agent collaboration."""
    from tdc_studio.mcp.server import run_mcp_server

    console.print(f"[bold green]🚀 Launching TDC-Studio MCP Server ({transport})...[/bold green]")
    run_mcp_server(transport=transport, host=host, port=port)


@app.command()
def dossier(
    smiles: str = typer.Argument(..., help="Candidate molecule SMILES string"),
    target_name: str = typer.Option("Target", "--target", "-t", help="Target gene symbol or name"),
    target_seq: Optional[str] = typer.Option(None, "--sequence", "-s", help="Target protein amino acid sequence"),
    dose_mg: float = typer.Option(100.0, "--dose", "-d", help="Clinical oral dose in mg"),
    output_dir: str = typer.Option("reports", "--output-dir", "-o", help="Output directory for reports"),
):
    """Generate an ICH CTD Nonclinical Candidate Dossier (HTML and JSON)."""
    import json

    from tdc_studio.dossier.collector import DossierCollector
    from tdc_studio.dossier.renderer import DossierRenderer

    os.makedirs(output_dir, exist_ok=True)
    console.print(f"[bold cyan]🔍 Collecting nonclinical data for candidate:[/bold cyan] {smiles}")

    collector = DossierCollector()
    payload = collector.collect(smiles=smiles, target_seq=target_seq, target_name=target_name, dose_mg=dose_mg)

    renderer = DossierRenderer()
    html_content = renderer.render_html(payload)

    safe_name = "".join(c if c.isalnum() else "_" for c in target_name)[:20]
    html_path = os.path.join(output_dir, f"dossier_{safe_name}.html")
    json_path = os.path.join(output_dir, f"dossier_{safe_name}.json")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload.to_dict(), f, indent=2)

    console.print(f"[bold green]✅ Standalone HTML Report:[/bold green] {html_path}")
    console.print(f"[bold green]✅ Machine-readable JSON:[/bold green] {json_path}")


if __name__ == "__main__":
    app()
