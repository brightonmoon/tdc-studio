"""Tests for AutoML search space sampling and dry-run objective execution."""

import optuna

from tdc_studio.automl.objective import TDCStudioObjective
from tdc_studio.automl.sampler import sample_parameters
from tdc_studio.data.single_pred import ADMETDataModule


def test_sample_parameters():
    study = optuna.create_study()
    trial = study.ask()

    spec = {
        "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "hidden_dim": {"type": "categorical", "choices": [32, 64]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
    }

    params = sample_parameters(trial, spec)
    assert "lr" in params
    assert 1e-4 <= params["lr"] <= 1e-2
    assert params["hidden_dim"] in [32, 64]
    assert 1 <= params["num_layers"] <= 3


def test_objective_dry_run(dummy_smiles_df):
    dm = ADMETDataModule(dataset_name="toy", synthetic_df=dummy_smiles_df)
    dm.prepare_data()

    data_cfg = {"batch_size": 2, "max_epochs": 1}
    model_cfg = {"type": "graph_transformer", "in_dim": 14, "num_layers": 1}
    hpo_cfg = {
        "search_space": {
            "hidden_dim": {"type": "categorical", "choices": [16, 32]},
            "lr": {"type": "float", "low": 1e-3, "high": 1e-2},
        }
    }

    objective = TDCStudioObjective(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        hpo_cfg=hpo_cfg,
        tracking_cfg={"enabled": False},
        data_module=dm,
        dry_run=True,  # No heavy training
    )

    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    val_loss = objective(trial)
    assert isinstance(val_loss, float)
    assert not float("nan") == val_loss
