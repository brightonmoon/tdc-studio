"""Multi-Modal Hybrid Stacking (DMPNN Graph + GBDT Descriptors) & Parametric Calibration."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Precompiled SMARTS patterns for biophysical pH 7.4 & HSA/AAG binding motifs
_ACIDIC_SMARTS = {
    "cooh": Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]"),
    "sulfonamide": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#7X3H1,H2;!$(NC=O)])"),
    "tetrazole": Chem.MolFromSmarts("c1nnn[nH]1"),
    "phenol": Chem.MolFromSmarts("[OX2H][cX3]"),
    "phosphate": Chem.MolFromSmarts("[PX4](=O)([OX2H,OX1-])([OX2H,OX1-])"),
}
_BASIC_SMARTS = {
    "aliphatic_amine": Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(NS(=O)=O);!$(Nc)]"),
    "aromatic_amine": Chem.MolFromSmarts("[NX3;H2,H1;$(Nc)]"),
    "guanidine": Chem.MolFromSmarts("[NX3][CX3](=[NX2])"),
    "pyridine": Chem.MolFromSmarts("[nX2;$(n1ccccc1)]"),
    "piperazine": Chem.MolFromSmarts("[NX3]1CCNCC1"),
}
_CHEMBERTA_CACHE: Dict[str, np.ndarray] = {}


def compute_biophysical_motifs(mol: Optional[Chem.Mol]) -> List[float]:
    """Compute 14 physiological pH 7.4 ionization & HSA/AAG binding motifs."""
    if mol is None:
        return [0.0] * 14

    n_cooh = len(mol.GetSubstructMatches(_ACIDIC_SMARTS["cooh"]))
    n_sulfonamide = len(mol.GetSubstructMatches(_ACIDIC_SMARTS["sulfonamide"]))
    n_tetrazole = len(mol.GetSubstructMatches(_ACIDIC_SMARTS["tetrazole"]))
    n_phenol = len(mol.GetSubstructMatches(_ACIDIC_SMARTS["phenol"]))
    n_phosphate = len(mol.GetSubstructMatches(_ACIDIC_SMARTS["phosphate"]))

    f_anion = (
        n_cooh * 0.999
        + n_sulfonamide * 0.962
        + n_tetrazole * 0.997
        + n_phosphate * 1.99
        + n_phenol * 0.004
    )

    n_aliphatic_amine = len(mol.GetSubstructMatches(_BASIC_SMARTS["aliphatic_amine"]))
    n_aromatic_amine = len(mol.GetSubstructMatches(_BASIC_SMARTS["aromatic_amine"]))
    n_guanidine = len(mol.GetSubstructMatches(_BASIC_SMARTS["guanidine"]))
    n_pyridine = len(mol.GetSubstructMatches(_BASIC_SMARTS["pyridine"]))
    n_piperazine = len(mol.GetSubstructMatches(_BASIC_SMARTS["piperazine"]))

    f_cation = (
        n_aliphatic_amine * 0.996
        + n_guanidine * 0.9999
        + n_piperazine * 0.863
        + n_pyridine * 0.006
        + n_aromatic_amine * 0.002
    )

    q_net_74 = f_cation - f_anion
    is_anion_74 = 1.0 if f_anion >= 0.5 else 0.0
    is_cation_74 = 1.0 if f_cation >= 0.5 else 0.0
    is_neutral_74 = 1.0 if (f_anion < 0.5 and f_cation < 0.5) else 0.0
    is_zwitterion_74 = 1.0 if (f_anion >= 0.5 and f_cation >= 0.5) else 0.0

    n_aromatic_rings = Descriptors.NumAromaticRings(mol)
    from rdkit.Chem import Crippen
    logp = Crippen.MolLogP(mol)

    sudlow_site_1 = 1.0 if (n_aromatic_rings >= 2 and (n_cooh + n_sulfonamide + n_tetrazole >= 1 or logp >= 3.0)) else 0.0
    sudlow_site_2 = 1.0 if (n_cooh >= 1 and n_aromatic_rings >= 1 and logp >= 1.5) else 0.0
    aag_motif = 1.0 if (f_cation >= 0.5 and logp >= 2.0) else 0.0

    if is_anion_74:
        logd_74 = logp - np.log10(1.0 + 10 ** (7.4 - 4.2))
    elif is_cation_74:
        logd_74 = logp - np.log10(1.0 + 10 ** (9.5 - 7.4))
    else:
        logd_74 = logp

    return [
        float(f_anion), float(f_cation), float(q_net_74), float(is_anion_74),
        float(is_cation_74), float(is_neutral_74), float(is_zwitterion_74),
        float(sudlow_site_1), float(sudlow_site_2), float(aag_motif), float(logd_74),
        float(n_cooh), float(n_sulfonamide), float(n_aliphatic_amine)
    ]


def extract_chemberta_embeddings_batch(
    smiles_list: List[str],
    model_name: str = "DeepChem/ChemBERTa-77M-MTR",
    batch_size: int = 64,
) -> Optional[np.ndarray]:
    """Extract 384-dimensional ChemBERTa-77M-MTR mean-pooled contextual embeddings with caching."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return None

    missing = [s for s in smiles_list if s and s not in _CHEMBERTA_CACHE]
    if missing:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        for i in range(0, len(missing), batch_size):
            chunk = [s if (s and isinstance(s, str)) else "C" for s in missing[i : i + batch_size]]
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=mask)
                tok_emb = out.last_hidden_state
                mask_exp = mask.unsqueeze(-1).expand(tok_emb.size()).float()
                sum_emb = torch.sum(tok_emb * mask_exp, 1)
                sum_m = torch.clamp(mask_exp.sum(1), min=1e-9)
                pooled = (sum_emb / sum_m).detach().cpu().numpy()
            for s_key, vec in zip(missing[i : i + batch_size], pooled):
                _CHEMBERTA_CACHE[s_key] = vec

    vectors = [_CHEMBERTA_CACHE.get(s, np.zeros(384, dtype=np.float32)) for s in smiles_list]
    return np.vstack(vectors).astype(np.float32)


def extract_molecular_features(
    smiles_list: List[str],
    labels: Optional[List[float]] = None,
    n_bits: int = 1024,
    transform: Optional[str] = "logit",
    use_biophysics: bool = True,
    use_chemberta: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract Morgan Fingerprints (1024-bit) + 200+ RDKit Descriptors + 14 Biophysical Motifs + 384-dim ChemBERTa."""
    features = []
    y_logit = []
    y_real = []

    dummy_mol = Chem.MolFromSmiles("C")
    n_desc = len(Descriptors.CalcMolDescriptors(dummy_mol)) if dummy_mol is not None else 210

    for idx, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s) if (s and isinstance(s, str)) else None
        if mol is None:
            base_vec = [0.0] * (n_bits + n_desc)
            if use_biophysics:
                base_vec.extend([0.0] * 14)
            features.append(base_vec)
        else:
            # 1. Morgan Fingerprint (radius 2, bit vector)
            fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))

            # 2. RDKit 2D Physico-chemical Descriptors
            desc_dict = Descriptors.CalcMolDescriptors(mol)
            desc_vals = []
            for v in desc_dict.values():
                if v is None or np.isnan(v) or np.isinf(v):
                    desc_vals.append(0.0)
                else:
                    desc_vals.append(float(np.clip(v, -100.0, 100.0)))

            mol_features = fp + desc_vals

            # 3. Physiological pH 7.4 & HSA/AAG Binding Motifs (14 features)
            if use_biophysics:
                bio_vals = compute_biophysical_motifs(mol)
                mol_features.extend(bio_vals)

            features.append(mol_features)

        if labels is not None and idx < len(labels):
            raw_y = float(labels[idx])
            y_real.append(raw_y)
            if transform == "logit":
                fb = np.clip(raw_y / 100.0, 1e-4, 1.0 - 1e-4)
                y_logit.append(np.log(fb / (1.0 - fb)))
            else:
                y_logit.append(raw_y)

    X_base = np.array(features, dtype=np.float32)

    # 4. ChemBERTa-77M-MTR 384-dimensional Embeddings (if enabled)
    if use_chemberta:
        chemberta_embs = extract_chemberta_embeddings_batch(smiles_list)
        if chemberta_embs is not None and len(chemberta_embs) == len(X_base):
            X = np.hstack([X_base, chemberta_embs])
        else:
            X = X_base
    else:
        X = X_base

    y_l = np.array(y_logit, dtype=np.float32) if labels is not None else None
    y_r = np.array(y_real, dtype=np.float32) if labels is not None else None
    return X, y_l, y_r


class GBDTDMPNNBlender:
    """Hybrid Stacker that blends DMPNN continuous graph representations with GBDT orthogonal tree splits."""

    def __init__(
        self,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        l2_regularization: float = 2.0,
        random_state: int = 42,
    ):
        self.gbdt = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            min_samples_leaf=15,
            random_state=random_state,
        )
        self.optimal_w: float = 0.70
        self.optimal_alpha: float = 1.0
        self.optimal_beta: float = 0.0

    def fit_gbdt(self, X_train: np.ndarray, y_train_logit: np.ndarray) -> None:
        """Fit GBDT in thermodynamic Gibbs logit space."""
        self.gbdt.fit(X_train, y_train_logit)

    def predict_gbdt(self, X: np.ndarray) -> np.ndarray:
        """Predict logit representations from GBDT."""
        return self.gbdt.predict(X)

    def fit_calibration_and_blend(
        self,
        z_dmpnn_val: np.ndarray,
        z_gbdt_val: np.ndarray,
        y_val_real: np.ndarray,
    ) -> Dict[str, float]:
        """Jointly optimize blending weight (w) and parametric sigmoid calibration (alpha, beta) on validation set."""

        def objective(params: np.ndarray) -> float:
            w, alpha, beta = params
            w_c = np.clip(w, 0.0, 1.0)
            z_blend = w_c * z_dmpnn_val + (1.0 - w_c) * z_gbdt_val
            # Parametric sigmoid in [0, 100]% space
            y_pred = 100.0 / (1.0 + np.exp(-np.clip(alpha * z_blend + beta, -40.0, 40.0)))
            return float(np.mean((y_pred - y_val_real) ** 2))

        # Initial guess: 70% DMPNN, 30% GBDT, alpha=1.0, beta=0.0
        init_guess = [0.70, 1.0, 0.0]
        bounds = [(0.0, 1.0), (0.1, 3.0), (-2.0, 2.0)]
        res = minimize(objective, init_guess, bounds=bounds, method="L-BFGS-B")

        self.optimal_w = float(np.clip(res.x[0], 0.0, 1.0))
        self.optimal_alpha = float(res.x[1])
        self.optimal_beta = float(res.x[2])

        # Evaluate on validation
        z_b = self.optimal_w * z_dmpnn_val + (1.0 - self.optimal_w) * z_gbdt_val
        val_pred = 100.0 / (1.0 + np.exp(-np.clip(self.optimal_alpha * z_b + self.optimal_beta, -40.0, 40.0)))

        val_r2 = float(r2_score(y_val_real, val_pred))
        val_mae = float(mean_absolute_error(y_val_real, val_pred))
        val_rmse = float(np.sqrt(mean_squared_error(y_val_real, val_pred)))

        return {
            "optimal_w_dmpnn": self.optimal_w,
            "optimal_w_gbdt": 1.0 - self.optimal_w,
            "optimal_alpha": self.optimal_alpha,
            "optimal_beta": self.optimal_beta,
            "val_r2": val_r2,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
        }

    def fit_single_calibration(self, z_val: np.ndarray, y_val_real: np.ndarray) -> Tuple[float, float]:
        """Fit parametric calibration (alpha, beta) for a single model's logits."""
        def obj(params: np.ndarray) -> float:
            a, b = params
            p = 100.0 / (1.0 + np.exp(-np.clip(a * z_val + b, -40.0, 40.0)))
            return float(np.mean((p - y_val_real) ** 2))
        res = minimize(obj, [1.0, 0.0], bounds=[(0.1, 3.0), (-2.0, 2.0)], method="L-BFGS-B")
        return float(res.x[0]), float(res.x[1])

    def predict_hybrid(
        self,
        z_dmpnn: np.ndarray,
        z_gbdt: np.ndarray,
    ) -> np.ndarray:
        """Apply optimal blending and parametric calibration to generate final percentage predictions."""
        z_blend = self.optimal_w * z_dmpnn + (1.0 - self.optimal_w) * z_gbdt
        return 100.0 / (1.0 + np.exp(-np.clip(self.optimal_alpha * z_blend + self.optimal_beta, -40.0, 40.0)))

    def evaluate_test(
        self,
        z_dmpnn_test: np.ndarray,
        z_gbdt_test: np.ndarray,
        y_test_real: np.ndarray,
        z_dmpnn_val: Optional[np.ndarray] = None,
        z_gbdt_val: Optional[np.ndarray] = None,
        y_val_real: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute full benchmark metrics for standalone and hybrid models."""
        def _metrics(p: np.ndarray, y: np.ndarray) -> Dict[str, float]:
            pr, _ = pearsonr(p, y)
            sp, _ = spearmanr(p, y)
            return {
                "r2": float(r2_score(y, p)),
                "rmse": float(np.sqrt(mean_squared_error(y, p))),
                "mae": float(mean_absolute_error(y, p)),
                "pearson": float(pr),
                "spearman": float(sp),
            }

        # 1. Standalone DMPNN (Raw Sigmoid)
        p_dmpnn_raw = 100.0 / (1.0 + np.exp(-np.clip(z_dmpnn_test, -40.0, 40.0)))
        # 2. Standalone GBDT (Raw Sigmoid)
        p_gbdt_raw = 100.0 / (1.0 + np.exp(-np.clip(z_gbdt_test, -40.0, 40.0)))
        # 3. Hybrid Stacking + Calibrated
        p_hybrid = self.predict_hybrid(z_dmpnn_test, z_gbdt_test)

        res: Dict[str, Any] = {
            "dmpnn_raw_metrics": _metrics(p_dmpnn_raw, y_test_real),
            "gbdt_raw_metrics": _metrics(p_gbdt_raw, y_test_real),
            "hybrid_metrics": _metrics(p_hybrid, y_test_real),
            "optimal_parameters": {
                "w_dmpnn": self.optimal_w,
                "w_gbdt": 1.0 - self.optimal_w,
                "alpha": self.optimal_alpha,
                "beta": self.optimal_beta,
            },
        }

        # 4. Standalone Calibrated DMPNN & GBDT (if validation data provided)
        if z_dmpnn_val is not None and y_val_real is not None:
            a_dmpnn, b_dmpnn = self.fit_single_calibration(z_dmpnn_val, y_val_real)
            p_dmpnn_cal = 100.0 / (1.0 + np.exp(-np.clip(a_dmpnn * z_dmpnn_test + b_dmpnn, -40.0, 40.0)))
            res["dmpnn_calibrated_metrics"] = _metrics(p_dmpnn_cal, y_test_real)
            res["dmpnn_calibrated_params"] = {"alpha": a_dmpnn, "beta": b_dmpnn}

        if z_gbdt_val is not None and y_val_real is not None:
            a_gbdt, b_gbdt = self.fit_single_calibration(z_gbdt_val, y_val_real)
            p_gbdt_cal = 100.0 / (1.0 + np.exp(-np.clip(a_gbdt * z_gbdt_test + b_gbdt, -40.0, 40.0)))
            res["gbdt_calibrated_metrics"] = _metrics(p_gbdt_cal, y_test_real)
            res["gbdt_calibrated_params"] = {"alpha": a_gbdt, "beta": b_gbdt}

        return res
