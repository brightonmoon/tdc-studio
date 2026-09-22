"""Objective function builder for Optuna HPO with W&B integration."""

from typing import Any, Dict, Optional

import optuna
import torch

from tdc_studio.automl.sampler import sample_parameters
from tdc_studio.core.registry import DATASETS, MODELS
from tdc_studio.evaluation.evaluator import TherapeuticsEvaluator, is_metric_higher_better


def _batch_to_device(batch: dict, device: Any) -> dict:
    """Move tensor and PyG graph structures in batch to target device."""
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class TDCStudioObjective:
    """Optuna objective function for tuning TDC deep learning pipelines."""

    def __init__(
        self,
        data_cfg: Dict[str, Any],
        model_cfg: Dict[str, Any],
        hpo_cfg: Dict[str, Any],
        tracking_cfg: Optional[Dict[str, Any]] = None,
        data_module: Optional[Any] = None,
        dry_run: bool = False,
    ):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.hpo_cfg = hpo_cfg
        self.tracking_cfg = tracking_cfg or {"enabled": False}
        self.dry_run = dry_run
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Prepare data module once to prevent re-parsing overhead in trials
        if data_module is not None:
            self.data_module = data_module
        else:
            data_cls = DATASETS.get(data_cfg["type"])
            self.data_module = data_cls(**data_cfg.get("params", {}))

        if not getattr(self.data_module, "is_prepared", False):
            self.data_module.prepare_data()

        self.batch_size = data_cfg.get("batch_size", 32)
        self.train_loader, self.val_loader, _ = self.data_module.setup_loaders(
            batch_size=self.batch_size
        )

        self.task_type = getattr(self.data_module, "task_type", "regression")
        self.metric_name = (
            self.data_cfg.get("metric_name")
            or getattr(self.data_module, "metric_name", None)
            or ("mae" if self.task_type == "regression" else "roc_auc")
        )
        self.evaluator = TherapeuticsEvaluator(
            default_metric=self.metric_name, task_type=self.task_type
        )
        self.direction = "maximize" if is_metric_higher_better(self.metric_name) else "minimize"

    def __call__(self, trial: optuna.Trial) -> float:
        # Sample parameters
        search_space = self.hpo_cfg.get("search_space", {})
        sampled_params = sample_parameters(trial, search_space)

        # Setup tracking if enabled
        wandb_run = None
        if self.tracking_cfg.get("enabled", False):
            try:
                import wandb

                wandb_run = wandb.init(
                    project=self.tracking_cfg.get("project", "tdc-learning"),
                    entity=self.tracking_cfg.get("entity", None),
                    group=f"{self.data_cfg.get('dataset_name', 'default')}_optuna",
                    name=f"trial_{trial.number}",
                    config=sampled_params,
                    reinit="finish_previous",
                )
            except Exception:
                wandb_run = None

        try:
            # Build model with sampled hyperparams and move to GPU
            curr_model_cfg = {
                **self.model_cfg,
                **sampled_params,
                "task_type": self.task_type,
            }
            model_cls = MODELS.get(curr_model_cfg["type"])
            model = model_cls(curr_model_cfg).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=sampled_params.get("lr", 1e-3))

            max_epochs = 1 if self.dry_run else self.data_cfg.get("max_epochs", 5)
            best_metric = -float("inf") if self.direction == "maximize" else float("inf")

            for epoch in range(max_epochs):
                # Train single epoch (or single batch in dry_run)
                model.train()
                for i, batch in enumerate(self.train_loader):
                    dev_batch = _batch_to_device(batch, self.device)
                    optimizer.zero_grad()
                    preds = model(dev_batch)
                    mask = dev_batch.get("mask")
                    if mask is not None:
                        loss = model.compute_loss(preds, dev_batch["labels"], mask=mask)
                    else:
                        loss = model.compute_loss(preds, dev_batch["labels"])
                    loss.backward()
                    optimizer.step()
                    if self.dry_run:
                        break

                # Evaluation step
                val_metric = self.evaluate(model, self.val_loader)

                if wandb_run is not None:
                    wandb_run.log(
                        {"epoch": epoch, f"val_{self.metric_name}": val_metric}, step=epoch
                    )

                trial.report(val_metric, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                if self.direction == "maximize":
                    best_metric = max(best_metric, val_metric)
                else:
                    best_metric = min(best_metric, val_metric)

                if self.dry_run:
                    break

            return best_metric
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    def evaluate(self, model: torch.nn.Module, val_loader: Any) -> float:
        """Evaluate model on validation loader using domain metric."""
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                dev_batch = _batch_to_device(batch, self.device)
                preds = model(dev_batch)
                all_preds.append(preds.squeeze(-1).detach().cpu())
                all_labels.append(dev_batch["labels"].detach().cpu())
                if self.dry_run:
                    break

        if not all_preds:
            return 0.0

        preds_cat = torch.cat(all_preds, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        return self.evaluator.compute(preds_cat, labels_cat, self.metric_name)
