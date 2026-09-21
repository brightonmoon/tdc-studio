"""Dynamic search space sampler for Optuna trials from YAML specifications."""

from typing import Any, Dict

import optuna


def sample_parameters(trial: optuna.Trial, search_space_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Sample hyperparameters for a trial based on declarative search space config.

    Supported formats in YAML:
      param_name:
        type: float
        low: 1e-5
        high: 1e-2
        log: true
      param_name:
        type: int
        low: 64
        high: 256
        step: 32
      param_name:
        type: categorical
        choices: [128, 256, 512]
    """
    params: Dict[str, Any] = {}

    for name, spec in search_space_cfg.items():
        stype = spec.get("type", "float").lower()
        if stype == "float":
            low = float(spec["low"])
            high = float(spec["high"])
            log = bool(spec.get("log", False))
            params[name] = trial.suggest_float(name, low, high, log=log)
        elif stype == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec.get("step", 1))
            params[name] = trial.suggest_int(name, low, high, step=step)
        elif stype == "categorical":
            choices = spec["choices"]
            params[name] = trial.suggest_categorical(name, choices)
        else:
            raise ValueError(f"Unsupported parameter type '{stype}' for '{name}'.")

    return params
