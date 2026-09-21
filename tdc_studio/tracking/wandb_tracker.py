"""Weights & Biases experiment tracking and artifact logger."""

import os
from typing import Any, Dict, Optional


class WandBTracker:
    """Helper for managing Weights & Biases runs and artifact uploads."""

    def __init__(self, config: Dict[str, Any]):
        self.project = config.get("project", "tdc-studio")
        self.entity = config.get("entity", None)
        self.enabled = config.get("enabled", True)
        self.run = None

    def init_run(self, name: str, config: Dict[str, Any], group: Optional[str] = None) -> Any:
        """Initialize a new W&B run."""
        if not self.enabled:
            return None
        try:
            import wandb

            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=name,
                group=group,
                config=config,
                reinit=True,
            )
            return self.run
        except Exception as e:
            # Fallback gracefully if wandb login or offline
            print(f"[Warning] Failed to initialize W&B run: {e}")
            return None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log training or validation metrics."""
        if self.run is not None:
            self.run.log(metrics, step=step)

    def log_artifact(self, artifact_path: str, name: str, artifact_type: str = "model") -> None:
        """Log a file or directory as a W&B Artifact."""
        if self.run is not None and os.path.exists(artifact_path):
            import wandb

            artifact = wandb.Artifact(name=name, type=artifact_type)
            if os.path.isdir(artifact_path):
                artifact.add_dir(artifact_path)
            else:
                artifact.add_file(artifact_path)
            self.run.log_artifact(artifact)

    def finish(self) -> None:
        """Finish the active run."""
        if self.run is not None:
            self.run.finish()
            self.run = None
