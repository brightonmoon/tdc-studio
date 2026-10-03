"""Cascaded Clearance Transfer Modeling: Microsome SOTA Prior -> Hepatocyte Clearance.

Biochemical rationale:
    Intrinsic clearance in human liver microsomes (CYP Phase I oxidation) is a primary
    biochemical determinant of hepatic clearance. By cascading the SOTA microsomal clearance
    predictions (Spearman rho = 0.6918) alongside membrane permeability (Caco-2) and molecular
    descriptors, the hepatocyte clearance prediction directly benefits from the pre-learned
    CYP catalytic manifold without data leakage.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_clearance_features(smiles_list: List[str]) -> np.ndarray:
    """Extract standard physicochemical and RDKit descriptors for clearance modeling."""
    features = []
    for s in smiles_list:
        row = [0.0] * 12
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                logp = float(Descriptors.MolLogP(mol))
                mw = float(Descriptors.MolWt(mol))
                tpsa = float(rdMolDescriptors.CalcTPSA(mol))
                hbd = float(rdMolDescriptors.CalcNumHBD(mol))
                hba = float(rdMolDescriptors.CalcNumHBA(mol))
                rotb = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
                f_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
                rings = float(mol.GetRingInfo().NumRings())
                arom_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
                heavy = float(mol.GetNumHeavyAtoms())
                # Halogens
                halogens = float(
                    sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))
                )
                # Basic amine flag
                basic = float(
                    len(mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;!$(NC=O)][CX4]")))
                )

                row = [
                    logp,
                    mw,
                    tpsa,
                    hbd,
                    hba,
                    rotb,
                    f_csp3,
                    rings,
                    arom_rings,
                    heavy,
                    halogens,
                    basic,
                ]
        except Exception:
            pass
        features.append(row)
    return np.array(features, dtype=np.float32)


class CascadedClearancePredictor:
    """Cascaded Transfer Model injecting Microsomal Clearance prior into Hepatocyte Clearance."""

    def __init__(
        self,
        base_gbdt_params: Optional[Dict[str, Any]] = None,
        use_caco2_prior: bool = True,
    ):
        params = base_gbdt_params or {
            "max_iter": 300,
            "learning_rate": 0.03,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 15,
            "l2_regularization": 1.0,
            "random_state": 42,
        }
        self.gbdt = HistGradientBoostingRegressor(**params)
        self.use_caco2_prior = use_caco2_prior
        self.is_fitted = False
        self._mic_mean = 0.0
        self._mic_std = 1.0

    def fit(
        self,
        smiles_train: List[str],
        y_train: np.ndarray,
        mic_preds_train: np.ndarray,
        caco2_preds_train: Optional[np.ndarray] = None,
    ) -> "CascadedClearancePredictor":
        """Fit the cascaded hepatocyte model with prior features.

        Args:
            smiles_train: List of training SMILES.
            y_train: Target hepatocyte clearance values (log10 or standardized).
            mic_preds_train: Predicted microsomal clearance (out-of-fold for train to prevent leakage).
            caco2_preds_train: Optional predicted Caco-2 permeability.
        """
        physchem_feats = compute_clearance_features(smiles_train)
        mic_col = mic_preds_train.reshape(-1, 1).astype(np.float32)

        feat_list = [physchem_feats, mic_col]
        if self.use_caco2_prior and caco2_preds_train is not None:
            caco2_col = caco2_preds_train.reshape(-1, 1).astype(np.float32)
            # Permeability x Microsome interaction term
            interaction = (mic_col * caco2_col).astype(np.float32)
            feat_list.extend([caco2_col, interaction])

        X_train = np.hstack(feat_list)
        valid_mask = ~np.isnan(y_train) & ~np.isnan(X_train).any(axis=1)

        self.gbdt.fit(X_train[valid_mask], y_train[valid_mask])
        self.is_fitted = True
        return self

    def predict(
        self,
        smiles_list: List[str],
        mic_preds: np.ndarray,
        caco2_preds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict hepatocyte clearance given new SMILES and microsomal predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        physchem_feats = compute_clearance_features(smiles_list)
        mic_col = mic_preds.reshape(-1, 1).astype(np.float32)

        feat_list = [physchem_feats, mic_col]
        if self.use_caco2_prior and caco2_preds is not None:
            caco2_col = caco2_preds.reshape(-1, 1).astype(np.float32)
            interaction = (mic_col * caco2_col).astype(np.float32)
            feat_list.extend([caco2_col, interaction])

        X = np.hstack(feat_list)
        return self.gbdt.predict(X)

    def evaluate(
        self,
        smiles_test: List[str],
        y_test: np.ndarray,
        mic_preds_test: np.ndarray,
        caco2_preds_test: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute comprehensive benchmark metrics on test set."""
        preds = self.predict(smiles_test, mic_preds_test, caco2_preds_test)
        valid_mask = ~np.isnan(y_test) & ~np.isnan(preds)

        y_true = y_test[valid_mask]
        y_pred = preds[valid_mask]

        spearman_rho, _ = spearmanr(y_true, y_pred)
        pearson_r, _ = pearsonr(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = r2_score(y_true, y_pred)

        return {
            "spearman_rho": float(spearman_rho),
            "pearson_r": float(pearson_r),
            "mae": float(mae),
            "rmse": rmse,
            "r2": float(r2),
            "n_samples": int(len(y_true)),
        }
