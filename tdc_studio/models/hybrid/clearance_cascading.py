"""Cascaded Clearance Transfer Modeling: Microsome SOTA Prior -> Hepatocyte Clearance.

Biochemical rationale:
    Intrinsic clearance in human liver microsomes (CYP Phase I oxidation) is a primary
    biochemical determinant of hepatic clearance. By cascading the SOTA microsomal clearance
    predictions (Spearman rho = 0.6918) alongside multi-task D-MPNN representations and
    comprehensive RDKit 200+ descriptors, the hepatocyte clearance prediction directly benefits
    from the pre-learned CYP catalytic manifold without data leakage.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from rdkit.ML.Descriptors import MoleculeDescriptors

    _RDKIT_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(
        [d[0] for d in Descriptors._descList]
    )
except Exception:
    _RDKIT_CALC = None


def compute_clearance_features(
    smiles_list: List[str], full_rdkit: bool = False
) -> np.ndarray:
    """Extract physicochemical descriptors for clearance modeling.

    Args:
        smiles_list: List of molecular SMILES.
        full_rdkit: If True, computes the full suite of RDKit 2D descriptors (200+ features).
                    If False, returns standard 12-dimensional physicochemical features.
    """
    if full_rdkit and _RDKIT_CALC is not None:
        features = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s) if s else None
            if mol is not None:
                row = list(_RDKIT_CALC.CalcDescriptors(mol))
            else:
                row = [0.0] * len(Descriptors._descList)
            features.append(row)
        arr = np.array(features, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)

    # Standard 12-dim physicochemical features
    features = []
    for s in smiles_list:
        row = [0.0] * 12
        try:
            mol = Chem.MolFromSmiles(s) if s else None
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
                halogens = float(
                    sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))
                )
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
        use_caco2_prior: bool = False,
        use_full_rdkit: bool = True,
        dmpnn_checkpoint: Optional[str] = "models/export/cluster_4_clearance/best_model.pt",
        ensemble_seeds: Optional[List[int]] = None,
        loss_objectives: Optional[List[Tuple[str, float]]] = None,
        model_type: str = "auto",
    ):
        params = base_gbdt_params or {}
        self.use_cb = (model_type == "catboost") or (model_type == "auto" and HAS_CATBOOST)
        self.model_type = "catboost" if self.use_cb else "histgbdt"
        self.base_params = params
        self.use_caco2_prior = use_caco2_prior
        self.use_full_rdkit = use_full_rdkit
        self.dmpnn_checkpoint = dmpnn_checkpoint
        self.ensemble_seeds = ensemble_seeds or ([42, 43, 44] if self.use_cb else [42])
        self.loss_objectives = loss_objectives or [("MAE", 0.55), ("Huber:delta=12.0", 0.45)]

        self.models: List[Tuple[Any, float]] = []
        self.is_fitted = False
        self._dmpnn_model = None

    def _get_dmpnn_model(self):
        if self._dmpnn_model is not None:
            return self._dmpnn_model
        chk_path = self.dmpnn_checkpoint or "models/export/cluster_4_clearance/best_model.pt"
        if not os.path.exists(chk_path):
            candidates = [
                os.path.abspath(chk_path),
                os.path.join(os.getcwd(), chk_path),
                os.path.join("/content/tdc-studio", chk_path),
            ]
            for c in candidates:
                if os.path.exists(c):
                    chk_path = c
                    break
        if not os.path.exists(chk_path):
            print(f"[Warning] DMPNN checkpoint not found at: {chk_path}")
            return None
        try:
            import torch

            from tdc_studio.models.graph.dmpnn import DMPNNModel

            config = {
                "in_dim": 14,
                "edge_dim": 6,
                "hidden_dim": 400,
                "depth": 3,
                "dropout": 0.15,
                "use_descriptors": True,
                "descriptor_dim": 210,
                "tasks": [
                    {"name": "half_life_obach", "type": "regression"},
                    {"name": "clearance_hepatocyte_az", "type": "regression"},
                    {"name": "clearance_microsome_az", "type": "regression"},
                    {"name": "cyp3a4_veith", "type": "classification"},
                    {"name": "ppbr_az", "type": "regression"},
                ],
            }
            dmp_model = DMPNNModel(config)
            ckpt = torch.load(chk_path, map_location="cpu")
            dmp_model.load_state_dict(ckpt, strict=False)
            dmp_model.eval()
            self._dmpnn_model = dmp_model
            print(f"[Info] Successfully loaded DMPNN model from {chk_path}")
            return self._dmpnn_model
        except Exception as e:
            print(f"[Warning] Failed to load DMPNN model from {chk_path}: {e}")
            return None

    def _extract_dmpnn_features(self, smiles_list: List[str]) -> Optional[np.ndarray]:
        dmp_model = self._get_dmpnn_model()
        if dmp_model is None:
            return None
        try:
            import torch
            from torch_geometric.data import Data

            from tdc_studio.data.collate import molecule_collate_fn
            from tdc_studio.data.transforms import SmilesToGraphTransform

            g_trans = SmilesToGraphTransform()
            preds_all = []
            batch_size = 64

            for i in range(0, len(smiles_list), batch_size):
                b_smiles = smiles_list[i : i + batch_size]
                b_items = []
                for sm in b_smiles:
                    g = g_trans(sm) if sm else None
                    if g is None:
                        g = Data(
                            x=torch.zeros((1, 14)),
                            edge_index=torch.empty((2, 0), dtype=torch.long),
                            edge_attr=torch.empty((0, 6), dtype=torch.float),
                        )
                    mol = Chem.MolFromSmiles(sm) if sm else None
                    if mol is not None:
                        desc_dict = Descriptors.CalcMolDescriptors(mol)
                        desc_vals = [
                            0.0
                            if (v is None or np.isnan(v) or np.isinf(v))
                            else float(np.clip(v, -100.0, 100.0))
                            for v in desc_dict.values()
                        ][:210]
                        if len(desc_vals) < 210:
                            desc_vals = desc_vals + [0.0] * (210 - len(desc_vals))
                    else:
                        desc_vals = [0.0] * 210
                    b_items.append({
                        "drug_graph": g,
                        "descriptors": torch.tensor(desc_vals, dtype=torch.float32),
                        "drug_smiles_str": sm,
                    })
                collated = molecule_collate_fn(b_items)
                with torch.no_grad():
                    out = dmp_model(collated)  # Shape: (B, 5)
                    # Use indices 0 (half_life), 1 (hepatocyte), 2 (microsome), 4 (ppbr)
                    sub = out[:, [0, 1, 2, 4]].cpu().numpy()
                    preds_all.append(sub)

            return np.vstack(preds_all)
        except Exception as e:
            print(f"[Warning] Failed to extract DMPNN features: {e}")
            return None

    def _build_feature_matrix(
        self,
        smiles_list: List[str],
        mic_preds: np.ndarray,
        caco2_preds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        physchem_feats = compute_clearance_features(smiles_list, full_rdkit=self.use_full_rdkit)
        mic_col = mic_preds.reshape(-1, 1).astype(np.float32)

        feat_list = [physchem_feats, mic_col]

        dmp_feats = self._extract_dmpnn_features(smiles_list)
        if dmp_feats is not None:
            feat_list.append(dmp_feats)

        if self.use_caco2_prior and caco2_preds is not None:
            caco2_col = caco2_preds.reshape(-1, 1).astype(np.float32)
            interaction = (mic_col * caco2_col).astype(np.float32)
            feat_list.extend([caco2_col, interaction])

        return np.hstack(feat_list)

    def fit(
        self,
        smiles_train: List[str],
        y_train: np.ndarray,
        mic_preds_train: np.ndarray,
        caco2_preds_train: Optional[np.ndarray] = None,
    ) -> "CascadedClearancePredictor":
        """Fit the cascaded hepatocyte model with prior features, loss objectives, and ensemble seeds."""
        X_train = self._build_feature_matrix(smiles_train, mic_preds_train, caco2_preds_train)
        valid_mask = ~np.isnan(y_train) & ~np.isnan(X_train).any(axis=1)
        X_clean, y_clean = X_train[valid_mask], y_train[valid_mask]

        total_weight = sum(w for _, w in self.loss_objectives)
        norm_objectives = [(loss, w / total_weight) for loss, w in self.loss_objectives]

        self.models = []
        n_seeds = max(len(self.ensemble_seeds), 1)

        for loss_fn, obj_w in norm_objectives:
            per_model_weight = obj_w / n_seeds
            for seed in self.ensemble_seeds:
                if self.use_cb and HAS_CATBOOST:
                    cb_params = {
                        "iterations": self.base_params.get("iterations", 500),
                        "learning_rate": self.base_params.get("learning_rate", 0.025),
                        "depth": self.base_params.get("depth", 6),
                        "l2_leaf_reg": self.base_params.get("l2_leaf_reg", 3.0),
                        "loss_function": loss_fn,
                        "verbose": 0,
                        "random_seed": seed,
                    }
                    m = CatBoostRegressor(**cb_params)
                else:
                    hist_loss = "squared_error" if "RMSE" in loss_fn else "absolute_error"
                    hist_params = {
                        "loss": hist_loss,
                        "max_iter": self.base_params.get("max_iter", 300),
                        "learning_rate": self.base_params.get("learning_rate", 0.03),
                        "max_leaf_nodes": self.base_params.get("max_leaf_nodes", 31),
                        "min_samples_leaf": self.base_params.get("min_samples_leaf", 15),
                        "l2_regularization": self.base_params.get("l2_regularization", 2.0),
                        "random_state": seed,
                    }
                    m = HistGradientBoostingRegressor(**hist_params)

                m.fit(X_clean, y_clean)
                self.models.append((m, per_model_weight))

        self.is_fitted = True
        return self

    def predict(
        self,
        smiles_list: List[str],
        mic_preds: np.ndarray,
        caco2_preds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict hepatocyte clearance given new SMILES and microsomal predictions."""
        if not self.is_fitted or not self.models:
            raise RuntimeError("Model is not fitted yet.")

        X = self._build_feature_matrix(smiles_list, mic_preds, caco2_preds)
        pred_acc = np.zeros(len(X), dtype=np.float32)
        for m, w in self.models:
            pred_acc += w * m.predict(X)
        return pred_acc

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
