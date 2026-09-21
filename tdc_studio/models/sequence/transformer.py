"""Sequence Transformer model for SMILES or Protein Sequence property prediction."""

from typing import Any, Dict

import torch
import torch.nn as nn

from tdc_studio.core.registry import MODELS
from tdc_studio.models.base import BaseTherapeuticsModel


@MODELS.register("sequence_transformer")
class SequenceTransformerModel(BaseTherapeuticsModel):
    """Transformer Encoder for sequence-based representations."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        vocab_size = config.get("vocab_size", 100)
        hidden_dim = config.get("hidden_dim", 128)
        nhead = config.get("nhead", 4)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.1)

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Accepts 'target_seq' or 'smiles_seq'
        seq = batch.get("target_seq")
        if seq is None:
            seq = batch.get("smiles_seq")
        if seq is None:
            raise ValueError("Batch must contain 'target_seq' or 'smiles_seq'.")

        mask = seq == 0
        h = self.embedding(seq)
        encoded = self.encoder(h, src_key_padding_mask=mask)

        # Mean pooling over non-padded tokens
        expanded_mask = (~mask).unsqueeze(-1).float()
        pooled = (encoded * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)
