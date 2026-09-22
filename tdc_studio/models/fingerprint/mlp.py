"""MLP Baseline Model for tabular and fingerprint-based representations."""

from typing import Any, Dict

import torch
import torch.nn as nn

from tdc_studio.core.registry import MODELS
from tdc_studio.models.base import BaseTherapeuticsModel


@MODELS.register("mlp_baseline")
class MLPBaselineModel(BaseTherapeuticsModel):
    """Multi-Layer Perceptron for ECFP / Morgan Fingerprint vectors."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        in_dim = config.get("in_dim", 2048)
        hidden_dim = config.get("hidden_dim", 512)
        dropout = config.get("dropout", 0.2)
        num_layers = config.get("num_layers", 2)

        layers = []
        curr_dim = in_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(curr_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            curr_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.head = nn.Linear(curr_dim, 1)

    def extract_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Extract fingerprint representation h in R^{hidden_dim}."""
        fp = batch.get("fingerprint")
        if fp is None:
            raise ValueError("Batch must contain 'fingerprint' tensor.")
        if fp.ndim == 1:
            fp = fp.unsqueeze(0)
        return self.feature_extractor(fp)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        h = self.extract_features(batch)
        return self.head(h)

