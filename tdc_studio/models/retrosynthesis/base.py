"""Abstract base classes and interfaces for retrosynthesis and reaction models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn


class BaseRetroModel(nn.Module, ABC):
    """Abstract base model for single-step retrosynthetic disconnection (Product -> Reactants)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}

    @abstractmethod
    def predict_reactants(
        self,
        product_smiles: str,
        top_k: int = 5,
        reaction_type: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Predict top-k candidate precursor reactant SMILES sets for a target product.

        Args:
            product_smiles: Canonical SMILES of the target product molecule.
            top_k: Number of top candidate reaction precursors to return.
            reaction_type: Optional integer (1~10) specifying reaction class condition.

        Returns:
            List of tuples: (reactants_smiles_string, confidence_score)
            where reactants_smiles_string contains individual precursors joined by '.'.
        """
        pass


class BaseForwardModel(nn.Module, ABC):
    """Abstract base model for forward reaction prediction (Reactants -> Product)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}

    @abstractmethod
    def predict_product(
        self,
        reactants_smiles: str,
        top_k: int = 1,
    ) -> List[Tuple[str, float]]:
        """Predict the main expected reaction product SMILES from precursor reactants.

        Args:
            reactants_smiles: Precursor reactant molecules separated by '.'.
            top_k: Number of top candidate products to return.

        Returns:
            List of tuples: (product_smiles_string, confidence_score)
        """
        pass
