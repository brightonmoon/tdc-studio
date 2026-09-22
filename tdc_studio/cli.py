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
    data_module = data_cls(**data_params)

    data_module.prepare_data()
    train_loader, val_loader, test_loader = data_module.setup_loaders(
        batch_size=data_cfg.get("batch_size", 32)
    )

    task_type = data_module.task_type
    target_metric = (
        eval_metric
        or data_cfg.get("metric_name")
        or getattr(data_module, "metric_name", None)
        or ("val_loss" if task_type == "multi_task" else ("mae" if task_type == "regression" else "roc_auc"))
    )
    higher_is_better = is_metric_higher_better(target_metric) if task_type != "multi_task" else False
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
                        if val_masks_cat.numel() > 0:
                            t_valid = val_masks_cat[:, t_idx]
                        else:
                            t_valid = ~torch.isnan(val_labels_cat[:, t_idx])
                        if int(t_valid.sum().item()) > 0:
                            t_p = val_preds_cat[t_valid, t_idx]
                            t_y = val_labels_cat[t_valid, t_idx]
                            t_metric_name = "mae" if t_type == "regression" else "roc_auc"
                            all_val_metrics[f"{t_name}_{t_metric_name}"] = evaluator.compute(
                                t_p, t_y, t_metric_name
                            )
            else:
                current_val_metric = evaluator.compute(
                    val_preds_cat.squeeze(-1) if val_preds_cat.ndim > 1 else val_preds_cat,
                    val_labels_cat.squeeze(-1) if val_labels_cat.ndim > 1 else val_labels_cat,
                    target_metric,
                )
                all_val_metrics = evaluator.compute_all(
                    val_preds_cat.squeeze(-1) if val_preds_cat.ndim > 1 else val_preds_cat,
                    val_labels_cat.squeeze(-1) if val_labels_cat.ndim > 1 else val_labels_cat,
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
            with torch.no_grad():
                for batch in test_loader:
                    dev_batch = _batch_to_device(batch, device)
                    preds = eval_model(dev_batch)
                    test_preds_list.append(preds.squeeze(-1).detach().cpu())
                    test_labels_list.append(dev_batch["labels"].detach().cpu())

            test_preds_cat = (
                torch.cat(test_preds_list, dim=0) if test_preds_list else torch.tensor([])
            )
            test_labels_cat = (
                torch.cat(test_labels_list, dim=0) if test_labels_list else torch.tensor([])
            )

            test_metric_val = evaluator.compute(test_preds_cat, test_labels_cat, target_metric)
            all_test_metrics = evaluator.compute_all(test_preds_cat, test_labels_cat, task_type)

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
        console.print(f"[bold green]Successfully switched active Colab account to:[/bold green] [cyan]{account}[/cyan]")
    except Exception as e:
        console.print(f"[bold red]Failed to switch account:[/bold red] {e}")
        raise typer.Exit(1)


@switch_app.command("save")
def save_account(account: str = typer.Argument(..., help="Name to save current active credentials as")):
    """Save currently active credentials as a named account."""
    from tdc_studio.remote.colab_account import ColabAccountManager

    mgr = ColabAccountManager()
    try:
        path = mgr.save_account(account)
        console.print(f"[bold green]Saved current Colab credentials as:[/bold green] [cyan]{account}[/cyan] ({path})")
    except Exception as e:
        console.print(f"[bold red]Failed to save account:[/bold red] {e}")
        raise typer.Exit(1)


@switch_app.command("new")
def new_account(account: str = typer.Argument(..., help="Name for the newly authenticated account")):
    """Authenticate a new Google account via OAuth browser flow and save credentials."""
    from tdc_studio.remote.colab_account import ColabAccountManager

    mgr = ColabAccountManager()
    try:
        success = mgr.new_account(account)
        if success:
            console.print(f"[bold green]Successfully created and registered account:[/bold green] [cyan]{account}[/cyan]")
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
        console.print(f"[bold green]Deleted credentials for account:[/bold green] [cyan]{account}[/cyan]")
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
    script: Optional[str] = typer.Option(None, "-f", "--script", help="Local python script to execute"),
    script_args: Optional[List[str]] = typer.Option(
        None, "--arg", help="Arguments to inject and pass to the script"
    ),
    account: Optional[str] = typer.Option(None, help="Colab account to use"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command without executing"),
):
    """Execute code or a local script directly on an active Google Colab instance via `colab exec`."""
    from tdc_studio.remote.colab_runner import ColabRunner

    runner = ColabRunner(account=account)
    console.print("[bold cyan]Executing remote job on Colab instance via `colab exec`[/bold cyan]...")

    returncode = runner.run_remote_exec(
        command_to_run=command or "tdc-studio train --config configs/config.yaml",
        session=session,
        script_file=script,
        script_args=script_args,
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

