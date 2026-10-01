"""Chemical reaction yield prediction model using Morgan fingerprints."""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem

from tdc_studio.core.registry import MODELS


@MODELS.register("yield_predictor")
class YieldPredictor(nn.Module):
    """Predicts expected chemical reaction yield percentage from reaction components."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.n_bits = int(self.config.get("n_bits", 2048))
        self.hidden_dim = int(self.config.get("hidden_dim", 256))
        self.dropout = float(self.config.get("dropout", 0.1))

        self.net = nn.Sequential(
            nn.Linear(self.n_bits, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),  # Yield output in [0.0, 1.0]
        )

    def _extract_reaction_fp(self, reactants_smiles: str, product_smiles: Optional[str] = None) -> torch.Tensor:
        """Compute average/combined fingerprint vector for reaction components."""
        all_smiles = [s.strip() for s in reactants_smiles.split(".") if s.strip()]
        if product_smiles:
            all_smiles.append(product_smiles.strip())

        fp_sum = np.zeros(self.n_bits, dtype=np.float32)
        valid = 0
        for s in all_smiles:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.n_bits)
                fp_sum += np.array(bv, dtype=np.float32)
                valid += 1

        if valid > 0:
            fp_sum /= valid
        return torch.tensor(fp_sum, dtype=torch.float32).unsqueeze(0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass taking batch of fingerprints [B, n_bits]."""
        return self.net(features).squeeze(-1)

    @torch.no_grad()
    def predict_yield(
        self,
        reactants_smiles: str,
        product_smiles: Optional[str] = None,
    ) -> float:
        """Predict expected reaction yield percentage (0.0% ~ 100.0%)."""
        self.eval()
        device = next(self.parameters()).device
        fp = self._extract_reaction_fp(reactants_smiles, product_smiles).to(device)

        # In case batch norm needs eval mode
        with torch.no_grad():
            pred_norm = self.net(fp).item()

        # Scale to percentage [0.0, 100.0]
        # Realistic heuristic floor: reactions usually fall in 20% ~ 95%
        yield_pct = round(pred_norm * 100.0, 1)
        return max(10.0, min(99.0, yield_pct))
