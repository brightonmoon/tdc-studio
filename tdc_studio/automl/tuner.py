"""StudioTuner orchestrates Optuna studies for TDC-Studio."""

from typing import Any, Dict

import optuna

from tdc_studio.automl.objective import TDCStudioObjective


class StudioTuner:
    """Orchestrator for hyperparameter search studies."""

    def __init__(
        self,
        data_cfg: Dict[str, Any],
        model_cfg: Dict[str, Any],
        hpo_cfg: Dict[str, Any],
        tracking_cfg: Dict[str, Any] | None = None,
        data_module: Any | None = None,
        dry_run: bool = False,
    ):
        self.objective = TDCStudioObjective(
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            hpo_cfg=hpo_cfg,
            tracking_cfg=tracking_cfg,
            data_module=data_module,
            dry_run=dry_run,
        )
        self.hpo_cfg = hpo_cfg
        self.direction = self.objective.direction

    def tune(self, n_trials: int = 10, timeout: int | None = None) -> Dict[str, Any]:
        """Run the optimization study."""
        sampler_name = self.hpo_cfg.get("sampler", "tpe").lower()
        if sampler_name == "random":
            sampler = optuna.samplers.RandomSampler()
        else:
            sampler = optuna.samplers.TPESampler()

        pruner_type = self.hpo_cfg.get("pruner", {}).get("type", "MedianPruner")
        if pruner_type == "MedianPruner":
            pruner = optuna.pruners.MedianPruner()
        else:
            pruner = optuna.pruners.NopPruner()

        study = optuna.create_study(direction=self.direction, sampler=sampler, pruner=pruner)
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)

        return {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial_number": study.best_trial.number,
        }
