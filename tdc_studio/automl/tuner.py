"""StudioTuner orchestrates Optuna studies for TDC-Studio."""

from typing import Any, Dict, Optional

import optuna

from tdc_studio.automl.objective import TDCStudioObjective


class StudioTuner:
    """Orchestrator for hyperparameter search studies."""

    def __init__(
        self,
        data_cfg: Dict[str, Any],
        model_cfg: Dict[str, Any],
        hpo_cfg: Dict[str, Any],
        tracking_cfg: Optional[Dict[str, Any]] = None,
        data_module: Optional[Any] = None,
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

    def tune(self, n_trials: int = 10, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Run the optimization study."""
        sampler_name = self.hpo_cfg.get("sampler", "tpe").lower()
        if sampler_name == "random":
            sampler = optuna.samplers.RandomSampler()
        else:
            sampler = optuna.samplers.TPESampler()

        pruner_cfg = self.hpo_cfg.get("pruner", {})
        pruner_type = pruner_cfg.get("type", "MedianPruner")
        if pruner_type == "MedianPruner":
            n_startup_trials = pruner_cfg.get("n_startup_trials", 3)
            n_warmup_steps = pruner_cfg.get("n_warmup_steps", 1)
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=n_startup_trials,
                n_warmup_steps=n_warmup_steps,
            )
        else:
            pruner = optuna.pruners.NopPruner()

        def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                val = f"{trial.value:.4f}" if trial.value is not None else "N/A"
                print(
                    f"  [Optuna] Trial #{trial.number} finished with {self.objective.metric_name.upper()}: {val} "
                    f"| Best so far: {study.best_value:.4f}",
                    flush=True,
                )
            elif trial.state == optuna.trial.TrialState.PRUNED:
                print(f"  [Optuna] Trial #{trial.number} was PRUNED early.", flush=True)

        study = optuna.create_study(direction=self.direction, sampler=sampler, pruner=pruner)
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[_trial_callback],
        )

        trials_summary = [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            }
            for t in study.trials
        ]

        return {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial_number": study.best_trial.number,
            "n_trials": len(study.trials),
            "trials": trials_summary,
        }
