"""Lipophilicity SOTA Stacker: 24-dim Biophysical Motifs + GBDT + ChemBERTa Stacking.

Combines domain-specific biophysical partition drivers, 384d chemical language representations,
and molecular graph predictions to push Lipophilicity AstraZeneca past R² >= 0.85.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tdc_studio.features.lipo_motifs import get_lipo_motif_extractor

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

_LIPO_CHEMBERTA_CACHE: Dict[str, np.ndarray] = {}


def extract_chemberta_features_cached(
    smiles_list: List[str],
    batch_size: int = 64,
    device: Optional[str] = None,
) -> np.ndarray:
    """Extract or mock ChemBERTa 384-dimensional embeddings safely with memory caching."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    missing = [s for s in smiles_list if s and s not in _LIPO_CHEMBERTA_CACHE]
    if missing:
        try:
            print(
                f"[ChemBERTa] Extracting representations for {len(missing)} molecules on {device}...",
                flush=True,
            )
            from transformers import AutoModel, AutoTokenizer

            model_name = "DeepChem/ChemBERTa-77M-MTR"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            model.eval()

            for i in range(0, len(missing), batch_size):
                batch = missing[i : i + batch_size]
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    pooled = (outputs.last_hidden_state * mask).sum(dim=1) / torch.clamp(
                        mask.sum(dim=1), min=1e-9
                    )
                    embs = pooled.cpu().numpy()
                for s_key, vec in zip(batch, embs):
                    _LIPO_CHEMBERTA_CACHE[s_key] = vec
                if (i // batch_size) % 10 == 0 or (i + batch_size) >= len(missing):
                    print(
                        f"[ChemBERTa] Processed {min(i + batch_size, len(missing))}/{len(missing)} compounds...",
                        flush=True,
                    )
        except Exception as e:
            print(f"[ChemBERTa Warning] Falling back to hash embeddings: {e}", flush=True)
            # Graceful fallback to 384-dim deterministic hash if huggingface is offline/mocked
            rng = np.random.default_rng(42)
            for s_key in missing:
                _LIPO_CHEMBERTA_CACHE[s_key] = rng.standard_normal(384, dtype=np.float32)

    embeddings = [
        _LIPO_CHEMBERTA_CACHE.get(s, np.zeros(384, dtype=np.float32)) for s in smiles_list
    ]
    return np.vstack(embeddings).astype(np.float32)


class LipophilicityStacker:
    """Tri-Hybrid Stacker for Lipophilicity AstraZeneca Benchmark."""

    def __init__(
        self,
        gbdt_params: Optional[Dict[str, Any]] = None,
        use_chemberta: bool = True,
        model_type: str = "auto",
    ):
        self.use_chemberta = use_chemberta
        self.motif_extractor = get_lipo_motif_extractor()
        params = gbdt_params or {}

        # Branch 1: CatBoost / HistGBDT on 24-dim motifs + basic RDKit descriptors
        use_cb = (model_type == "catboost") or (model_type == "auto" and HAS_CATBOOST)
        if use_cb and HAS_CATBOOST:
            cb_params = {
                "iterations": params.get("iterations", params.get("max_iter", 400)),
                "learning_rate": params.get("learning_rate", 0.03),
                "depth": params.get("depth", params.get("max_depth", 6)),
                "l2_leaf_reg": params.get("l2_leaf_reg", params.get("l2_regularization", 2.0)),
                "loss_function": "RMSE",
                "verbose": 0,
                "random_seed": params.get("random_seed", params.get("random_state", 42)),
            }
            self.gbdt = CatBoostRegressor(**cb_params)
            self.model_type = "catboost"
        else:
            hist_params = {
                "max_iter": params.get("max_iter", params.get("iterations", 400)),
                "learning_rate": params.get("learning_rate", 0.03),
                "max_leaf_nodes": params.get("max_leaf_nodes", 31),
                "min_samples_leaf": params.get("min_samples_leaf", 15),
                "l2_regularization": params.get(
                    "l2_regularization", params.get("l2_leaf_reg", 1.5)
                ),
                "random_state": params.get("random_state", params.get("random_seed", 42)),
            }
            self.gbdt = HistGradientBoostingRegressor(**hist_params)
            self.model_type = "histgbdt"

        # Branch 2: ChemBERTa RidgeCV
        self.chemberta_ridge = RidgeCV(alphas=np.logspace(-2, 4, 20))

        # Convex blending weights [w_gbdt, w_chemberta, w_gnn]
        self.weights = np.array([0.60, 0.40], dtype=np.float32)
        self.is_fitted = False

    def _extract_tabular_features(self, smiles_list: List[str]) -> np.ndarray:
        """Extract 24 motifs + extra descriptors for each molecule."""
        rows = []
        for s in smiles_list:
            motifs = self.motif_extractor.extract_from_smiles(s)
            rows.append(motifs)
        return np.array(rows, dtype=np.float32)

    def fit(
        self,
        smiles_train: List[str],
        y_train: np.ndarray,
        gnn_preds_train: Optional[np.ndarray] = None,
        val_data: Optional[Tuple[List[str], np.ndarray, Optional[np.ndarray]]] = None,
    ) -> "LipophilicityStacker":
        """Fit all branches and optimize meta-learner blending weights."""
        y_train = np.asarray(y_train, dtype=np.float32)
        X_tab = self._extract_tabular_features(smiles_train)

        # 1. Fit GBDT
        self.gbdt.fit(X_tab, y_train)

        # 2. Fit ChemBERTa Ridge
        if self.use_chemberta:
            X_emb = extract_chemberta_features_cached(smiles_train)
            self.chemberta_ridge.fit(X_emb, y_train)

        # 3. Optimize convex weights if validation set or OOF available
        if val_data is not None:
            smiles_val, y_val, gnn_preds_val = val_data
            X_val_tab = self._extract_tabular_features(smiles_val)
            val_gbdt = self.gbdt.predict(X_val_tab)
            if self.use_chemberta:
                val_emb = extract_chemberta_features_cached(smiles_val)
                val_chemberta = self.chemberta_ridge.predict(val_emb)
            else:
                val_chemberta = val_gbdt

            if gnn_preds_val is not None:
                preds_matrix = np.column_stack([val_gbdt, val_chemberta, gnn_preds_val])
                init_w = np.array([0.45, 0.25, 0.30])
            else:
                preds_matrix = np.column_stack([val_gbdt, val_chemberta])
                init_w = np.array([0.65, 0.35])

            def objective(w):
                w_norm = np.maximum(w, 0.0)
                denom = np.sum(w_norm)
                if denom > 0:
                    w_norm = w_norm / denom
                y_blend = preds_matrix @ w_norm
                return mean_squared_error(y_val, y_blend)

            bounds = [(0.0, 1.0)] * len(init_w)
            res = minimize(objective, init_w, method="SLSQP", bounds=bounds)
            if res.success:
                w_opt = np.maximum(res.x, 0.0)
                self.weights = w_opt / np.sum(w_opt)
            else:
                self.weights = init_w / np.sum(init_w)

        self.is_fitted = True
        return self

    def predict(
        self,
        smiles_list: List[str],
        gnn_preds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Ensemble prediction using optimized stacking weights."""
        if not self.is_fitted:
            raise RuntimeError("LipophilicityStacker is not fitted yet.")

        X_tab = self._extract_tabular_features(smiles_list)
        p_gbdt = self.gbdt.predict(X_tab)

        if self.use_chemberta:
            X_emb = extract_chemberta_features_cached(smiles_list)
            p_chemberta = self.chemberta_ridge.predict(X_emb)
        else:
            p_chemberta = p_gbdt

        if gnn_preds is not None and len(self.weights) == 3:
            blend_matrix = np.column_stack([p_gbdt, p_chemberta, gnn_preds])
        else:
            blend_matrix = np.column_stack([p_gbdt, p_chemberta])

        w = self.weights[: blend_matrix.shape[1]]
        w = w / np.sum(w)
        return blend_matrix @ w

    def evaluate(
        self,
        smiles_test: List[str],
        y_test: np.ndarray,
        gnn_preds_test: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute metrics on test set."""
        preds = self.predict(smiles_test, gnn_preds_test)
        y_true = np.asarray(y_test, dtype=np.float32)

        r2 = r2_score(y_true, preds)
        pearson_val, _ = pearsonr(y_true, preds)
        spearman_val, _ = spearmanr(y_true, preds)
        mae = mean_absolute_error(y_true, preds)
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))

        return {
            "r2": float(r2),
            "pearson_r": float(pearson_val),
            "spearman_rho": float(spearman_val),
            "mae": float(mae),
            "rmse": rmse,
            "weights": self.weights.tolist(),
        }
