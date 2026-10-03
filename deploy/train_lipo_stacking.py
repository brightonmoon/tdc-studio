"""Task E-4: Lipophilicity AstraZeneca 24-dim Motif + GBDT + ChemBERTa Stacking.

Aims for Test R² >= 0.85 (Pearson r >= 0.93) on the official Bemis-Murcko scaffold benchmark.
"""

import argparse
import json
import logging
import os

import numpy as np
from tdc.single_pred import ADME

from tdc_studio.models.hybrid.lipo_stacker import LipophilicityStacker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_lipo_stacking")


def main():
    parser = argparse.ArgumentParser(description="Lipophilicity Stacking Training")
    parser.add_argument("--dry-run", action="store_true", help="Execute on subset")
    parser.add_argument("--export-dir", type=str, default="models/export/lipophilicity_stacker")
    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)

    logger.info("Loading Lipophilicity AstraZeneca dataset (4,200 compounds)...")
    data = ADME(name="Lipophilicity_AstraZeneca")
    split = data.get_split(method="scaffold", seed=42)

    train_df = split["train"]
    val_df = split["valid"]
    test_df = split["test"]

    if args.dry_run:
        train_df = train_df.head(80)
        val_df = val_df.head(40)
        test_df = test_df.head(40)

    smiles_train = train_df["Drug"].tolist()
    y_train = np.asarray(train_df["Y"], dtype=np.float32)

    smiles_val = val_df["Drug"].tolist()
    y_val = np.asarray(val_df["Y"], dtype=np.float32)

    smiles_test = test_df["Drug"].tolist()
    y_test = np.asarray(test_df["Y"], dtype=np.float32)

    logger.info("Initializing LipophilicityStacker with 24-dim Biophysical Motifs...")
    stacker = LipophilicityStacker(
        gbdt_params={
            "max_iter": 400,
            "learning_rate": 0.03,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 15,
            "l2_regularization": 1.5,
            "random_state": 42,
        },
        use_chemberta=not args.dry_run,
    )

    logger.info("Fitting stacker on %d training samples with validation calibration...", len(smiles_train))
    stacker.fit(
        smiles_train=smiles_train,
        y_train=y_train,
        val_data=(smiles_val, y_val, None),
    )

    test_metrics = stacker.evaluate(
        smiles_test=smiles_test,
        y_test=y_test,
    )

    logger.info("=================================================================")
    logger.info("🏆 FINAL LIPOPHILICITY TEST METRICS (Scaffold Split):")
    logger.info("   R^2          : %.4f (Target: >= 0.85)", test_metrics["r2"])
    logger.info("   Pearson r    : %.4f (Target: >= 0.93)", test_metrics["pearson_r"])
    logger.info("   Spearman rho : %.4f", test_metrics["spearman_rho"])
    logger.info("   MAE          : %.4f", test_metrics["mae"])
    logger.info("   RMSE         : %.4f", test_metrics["rmse"])
    logger.info("   Stack Weights: %s", test_metrics["weights"])
    logger.info("=================================================================")

    with open(os.path.join(args.export_dir, "lipophilicity_stacking_summary.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    import joblib
    save_path = os.path.join(args.export_dir, "lipo_stacker_model.joblib")
    joblib.dump(stacker, save_path)
    logger.info("Successfully saved Lipophilicity Stacker model to: %s", save_path)


if __name__ == "__main__":
    main()
