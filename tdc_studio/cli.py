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

console = Console()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
):
    """Train a model with validation evaluation, checkpointing, and W&B tracking."""
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
    train_loader, val_loader, test_loader = data_module.setup_loaders(
        batch_size=data_cfg.get("batch_size", 32)
    )

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
            else ("mae" if task_type == "regression" else "roc_auc")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)))

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

    max_epochs = 1 if dry_run else (epochs or cfg.get("max_epochs", 5))
    best_metric = -float("inf") if higher_is_better else float("inf")
    best_epoch = 0
    patience_counter = 0

    console.print(f"Training for {max_epochs} epoch(s) (dry_run={dry_run})...")

    try:
        for epoch in range(max_epochs):
            # --- Train Epoch ---
            model.train()
            train_loss_sum = 0.0
            train_batches = 0

            for batch in train_loader:
                dev_batch = _batch_to_device(batch, device)
                optimizer.zero_grad()
                preds = model(dev_batch)
                mask = dev_batch.get("mask")
                if mask is not None:
                    loss = model.compute_loss(preds, dev_batch["labels"], mask=mask)
                else:
                    loss = model.compute_loss(preds, dev_batch["labels"])
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
                        if val_masks_cat.numel() > 0 and val_masks_cat.ndim >= 2 and val_masks_cat.size(1) > t_idx:
                            t_valid = val_masks_cat[:, t_idx]
                        elif val_labels_cat.numel() > 0 and val_labels_cat.ndim >= 2 and val_labels_cat.size(1) > t_idx:
                            t_valid = ~torch.isnan(val_labels_cat[:, t_idx])
                        else:
                            t_valid = torch.tensor([], dtype=torch.bool)
                        if int(t_valid.sum().item()) > 0 and val_preds_cat.ndim >= 2 and val_labels_cat.ndim >= 2:
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

            console.print(
                f"Epoch {epoch + 1}/{max_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val {target_metric.upper()}: [bold cyan]{current_val_metric:.4f}[/bold cyan]"
            )

            # Track metrics
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
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
                if early_stopping and patience_counter >= early_stopping:
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
):
    """Train an ensemble of models with multiple random seeds and compute consensus predictions."""
    import json
    import os

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
                    loss.backward()
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

            # Evaluate Model on Test Set
            model.eval()
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

            if t_masks_cat.numel() > 0:
                t_valid = t_masks_cat.bool() & ~torch.isnan(real_t_y) & ~torch.isnan(real_t_p)
            else:
                t_valid = ~torch.isnan(real_t_y) & ~torch.isnan(real_t_p)

            eval_t_p = real_t_p[t_valid]
            eval_t_y = real_t_y[t_valid]
            eval_v_p = real_v_p[val_mask]
            eval_v_y = real_v_y[val_mask]

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
        console.print(f"Configured MODEL_DIR: [yellow]{abs_model_dir}[/yellow]")

    console.print(f"[bold green]Starting TDC-Studio Serving API[/bold green] on {host}:{port}...")
    uvicorn.run("tdc_studio.serving.app:app", host=host, port=port, workers=workers)


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


if __name__ == "__main__":
    app()
