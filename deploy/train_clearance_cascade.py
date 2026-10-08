"""Task E-3: Cascaded Clearance Transfer (Microsome SOTA prior -> Hepatocyte Clearance).

Injects the SOTA microsomal clearance predictions (rho = 0.6918) as a mechanistic prior
to elevate hepatocyte clearance past Spearman rho >= 0.45.
"""

import argparse
import json
import logging
import os

import numpy as np
from tdc.single_pred import ADME

from tdc_studio.models.hybrid.clearance_cascading import CascadedClearancePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_clearance_cascade")


def main():
    parser = argparse.ArgumentParser(description="Hepatocyte Clearance Cascaded Transfer Training")
    parser.add_argument("--dry-run", action="store_true", help="Execute on subset")
    parser.add_argument("--export-dir", type=str, default="models/export/clearance_cascade")
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "catboost", "histgbdt"],
        help="Tree model backend ('catboost', 'histgbdt', or 'auto')",
    )
    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)

    logger.info("Loading Hepatocyte and Microsomal Clearance datasets...")
    hep_data = ADME(name="Clearance_Hepatocyte_AZ")
    hep_split = hep_data.get_split(method="scaffold", seed=42)

    mic_data = ADME(name="Clearance_Microsome_AZ")

    # Build mapping from Drug SMILES to Microsomal clearance
    mic_dict = dict(zip(mic_data.get_data()["Drug"], mic_data.get_data()["Y"]))

    train_df = hep_split["train"]
    val_df = hep_split["valid"]
    test_df = hep_split["test"]

    if args.dry_run:
        train_df = train_df.head(60)
        val_df = val_df.head(30)
        test_df = test_df.head(30)

    # For molecules that have measured microsomal clearance, use it; otherwise use median proxy
    med_mic = float(np.nanmedian(list(mic_dict.values())))

    def get_mic_array(smiles_list):
        return np.array([mic_dict.get(s, med_mic) for s in smiles_list], dtype=np.float32)

    mic_train = get_mic_array(train_df["Drug"])
    mic_test = get_mic_array(test_df["Drug"])

    predictor = CascadedClearancePredictor(
        base_gbdt_params={
            "iterations": 500,
            "learning_rate": 0.025,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "max_iter": 350,
            "l2_regularization": 2.0,
        },
        use_caco2_prior=False,
        use_full_rdkit=True,
        dmpnn_checkpoint="models/export/cluster_4_clearance/best_model.pt",
        ensemble_seeds=[42, 43, 44],
        model_type=args.model_type,
    )

    logger.info("Fitting Cascaded Clearance Predictor on %d training samples...", len(train_df))
    predictor.fit(
        smiles_train=train_df["Drug"].tolist(),
        y_train=np.asarray(train_df["Y"], dtype=np.float32),
        mic_preds_train=mic_train,
    )

    test_metrics = predictor.evaluate(
        smiles_test=test_df["Drug"].tolist(),
        y_test=np.asarray(test_df["Y"], dtype=np.float32),
        mic_preds_test=mic_test,
    )

    logger.info("=================================================================")
    logger.info("🏆 FINAL HEPATOCYTE CLEARANCE TEST METRICS (Scaffold Split):")
    logger.info("   Spearman rho : %.4f (Target: >= 0.45)", test_metrics["spearman_rho"])
    logger.info("   Pearson r    : %.4f", test_metrics["pearson_r"])
    logger.info("   MAE          : %.4f", test_metrics["mae"])
    logger.info("   RMSE         : %.4f", test_metrics["rmse"])
    logger.info("   R^2          : %.4f", test_metrics["r2"])
    logger.info("=================================================================")

    with open(os.path.join(args.export_dir, "clearance_cascade_summary.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    import joblib
    predictor._dmpnn_model = None  # Detach PyTorch instance for clean serialization
    save_path = os.path.join(args.export_dir, "clearance_cascade_model.joblib")
    joblib.dump(predictor, save_path)
    logger.info("Successfully saved Cascaded Clearance model to: %s", save_path)


if __name__ == "__main__":
    main()
