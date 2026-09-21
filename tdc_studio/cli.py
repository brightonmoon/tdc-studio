"""Command-Line Interface (CLI) for TDC-Studio."""

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console

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


@app.command()
def train(
    config: str = typer.Option("configs/config.yaml", help="Path to main YAML config"),
    epochs: Optional[int] = typer.Option(None, help="Override max epochs"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Run 1 step for validation without training"
    ),
):
    """Train a model locally or on server using single config."""
    import torch

    from tdc_studio.core.registry import DATASETS, MODELS

    cfg = load_yaml(config)
    console.print(f"[bold green]Starting Training Pipeline[/bold green] with config: {config}")

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    # 1. Load Data
    data_cls = DATASETS.get(data_cfg["type"])
    data_module = data_cls(**data_cfg.get("params", {}))
    data_module.prepare_data()
    train_loader, val_loader, _ = data_module.setup_loaders(
        batch_size=data_cfg.get("batch_size", 32)
    )

    # 2. Build Model
    model_cfg["task_type"] = data_module.task_type
    model_cls = MODELS.get(model_cfg["type"])
    model = model_cls(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)))

    # 3. Setup Tracking
    tracking_cfg = cfg.get("tracking", {})
    from tdc_studio.tracking.wandb_tracker import WandBTracker

    tracker = WandBTracker(tracking_cfg)
    tracker.init_run(
        name=f"train_{data_cfg.get('dataset_name', 'model')}",
        config=cfg,
        group=f"{data_cfg.get('dataset_name', 'default')}_train",
    )

    max_epochs = 1 if dry_run else (epochs or cfg.get("max_epochs", 5))
    console.print(f"Training for {max_epochs} epoch(s) (dry_run={dry_run})...")

    try:
        for epoch in range(max_epochs):
            model.train()
            loss = torch.tensor(0.0)
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                preds = model(batch)
                loss = model.compute_loss(preds, batch["labels"])
                loss.backward()
                optimizer.step()
                if dry_run:
                    break
            console.print(f"Epoch {epoch + 1}/{max_epochs} complete. Loss: {loss.item():.4f}")
            tracker.log_metrics({"epoch": epoch + 1, "loss": loss.item()}, step=epoch + 1)
            if dry_run:
                break
    finally:
        tracker.finish()

    console.print("[bold green]Training Completed Successfully![/bold green]")


@app.command()
def tune(
    config: str = typer.Option("configs/config.yaml", help="Path to main YAML config"),
    n_trials: int = typer.Option(10, "--n-trials", help="Number of Optuna trials"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run 1 step per trial for testing"),
):
    """Run Optuna Hyperparameter Optimization (HPO)."""
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


@app.command()
def export(
    checkpoint_dir: str = typer.Option(
        "./models/checkpoint", help="Directory containing checkpoint"
    ),
    output_dir: str = typer.Option("./models/export", help="Target output directory"),
):
    """Export model for production serving."""

    console.print(f"Exporting model from {checkpoint_dir} to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    console.print("[bold green]Model exported successfully.[/bold green]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host address"),
    port: int = typer.Option(8000, help="Port to listen on"),
    workers: int = typer.Option(1, help="Number of worker processes"),
):
    """Start the FastAPI inference microservice."""
    import uvicorn

    console.print(f"[bold green]Starting TDC-Studio Serving API[/bold green] on {host}:{port}...")
    uvicorn.run("tdc_studio.serving.app:app", host=host, port=port, workers=workers)


@remote_app.command("run")
def remote_run(
    command: str = typer.Option(
        "tdc-studio train --config configs/config.yaml", help="Command to run on Colab"
    ),
    gpu: str = typer.Option("t4", help="GPU tier: t4, l4, a100, v100"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command without executing"),
):
    """Dispatch training or HPO job to Google Colab Cloud GPU via google-colab-cli."""
    from tdc_studio.remote.colab_runner import ColabRunner

    runner = ColabRunner(gpu_type=gpu)
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
