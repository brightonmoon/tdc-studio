"""Graph Transformer and GNN models for molecular properties."""

from typing import Any, Dict

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

from tdc_studio.core.registry import MODELS
from tdc_studio.models.base import BaseTherapeuticsModel


@MODELS.register("graph_transformer")
class GraphTransformerModel(BaseTherapeuticsModel):
    """Molecular Graph Convolutional Network / Transformer for single-molecule property prediction."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        in_dim = config.get("in_dim", 14)  # DEFAULT_ATOM_LIST size (10) + 4 extra features
        hidden_dim = config.get("hidden_dim", 128)
        num_layers = config.get("num_layers", 3)
        dropout = config.get("dropout", 0.1)

        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        self.act = nn.ReLU()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        graph = batch["drug_graph"]
        x, edge_index, batch_idx = graph.x, graph.edge_index, graph.batch

        h = self.embedding(x)
        for conv, drop in zip(self.convs, self.dropouts):
            h = conv(h, edge_index)
            h = self.act(h)
            h = drop(h)

        # Global readout pooling (mean across nodes in each graph)
        hg = global_mean_pool(h, batch_idx)
        out = self.head(hg)
        return out


@MODELS.register("graph_transformer_dta")
class GraphTransformerDTAModel(BaseTherapeuticsModel):
    """Multimodal interaction model for Drug-Target Affinity (Graph + Sequence)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        in_dim = config.get("in_dim", 14)
        hidden_dim = config.get("hidden_dim", 128)
        vocab_size = config.get("vocab_size", 30)
        dropout = config.get("dropout", 0.1)

        # Drug Graph Encoder
        self.drug_embed = nn.Linear(in_dim, hidden_dim)
        self.drug_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.drug_conv2 = GCNConv(hidden_dim, hidden_dim)

        # Target Sequence Encoder
        self.target_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.target_lstm = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Interaction / Joint Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        # 1. Encode drug graph
        graph = batch["drug_graph"]
        x, edge_index, batch_idx = graph.x, graph.edge_index, graph.batch
        dh = self.act(self.drug_conv1(self.drug_embed(x), edge_index))
        dh = self.act(self.drug_conv2(dh, edge_index))
        drug_rep = global_mean_pool(dh, batch_idx)  # [B, hidden_dim]

        # 2. Encode target sequence
        target_seq = batch["target_seq"]
        te = self.target_embed(target_seq)
        _, th = self.target_lstm(te)
        # Cat forward and backward hidden states -> [B, hidden_dim]
        target_rep = torch.cat([th[0], th[1]], dim=-1)

        # 3. Concatenate and predict
        joint = torch.cat([drug_rep, target_rep], dim=-1)
        return self.head(joint)
