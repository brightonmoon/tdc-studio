"""Unit tests for Clearance Cascading and Lipophilicity Stacker modules."""

import numpy as np

from tdc_studio.models.hybrid.clearance_cascading import CascadedClearancePredictor
from tdc_studio.models.hybrid.lipo_stacker import LipophilicityStacker


def test_clearance_cascaded_predictor_toy():
    smiles_train = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  # Caffeine
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "CCN(CC)C(=O)C1CN(C2CC3=CNC4=CC=CC(=C34)C2=C1)C",  # LSD
        "COc1ccc2[nH]c3c(c2c1)C(=O)c1ccccc1-3",
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
        "O=C(O)c1ccccc1O",
        "c1ccccc1",
    ]
    y_train = np.array([12.5, 3.2, 45.0, 18.2, 5.0, 60.1, 8.4, 2.1], dtype=np.float32)
    mic_preds = np.array([10.0, 4.0, 40.0, 20.0, 6.0, 55.0, 9.0, 2.0], dtype=np.float32)
    caco2_preds = np.array([-4.8, -4.5, -4.2, -5.1, -4.9, -4.1, -5.2, -4.3], dtype=np.float32)

    predictor = CascadedClearancePredictor(
        base_gbdt_params={"max_iter": 10, "min_samples_leaf": 1, "random_state": 42}
    )
    predictor.fit(smiles_train, y_train, mic_preds, caco2_preds)

    preds = predictor.predict(smiles_train[:2], mic_preds[:2], caco2_preds[:2])
    assert len(preds) == 2
    assert np.isfinite(preds).all()

    metrics = predictor.evaluate(smiles_train, y_train, mic_preds, caco2_preds)
    assert "spearman_rho" in metrics
    assert "mae" in metrics


def test_lipophilicity_stacker_toy():
    smiles_train = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "c1ccccc1",
        "c1ccccc1Cl",
        "c1ccccc1F",
        "c1ccccc1C(F)(F)F",
        "CCO",
    ]
    y_train = np.array([1.31, -0.07, 3.50, 2.13, 2.84, 2.27, 3.12, -0.31], dtype=np.float32)

    stacker = LipophilicityStacker(
        gbdt_params={"max_iter": 10, "min_samples_leaf": 1, "random_state": 42},
        use_chemberta=False,
    )
    stacker.fit(smiles_train, y_train)

    preds = stacker.predict(smiles_train[:3])
    assert len(preds) == 3
    assert np.isfinite(preds).all()

    metrics = stacker.evaluate(smiles_train, y_train)
    assert "r2" in metrics
    assert "pearson_r" in metrics
