"""Graph Transformer and GNN models for molecular properties."""

from typing import Any, Dict

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, TransformerConv, global_mean_pool

from tdc_studio.core.registry import MODELS
from tdc_studio.models.base import BaseTherapeuticsModel


@MODELS.register("graph_transformer")
class GraphTransformerModel(BaseTherapeuticsModel):
    """Molecular Graph Transformer with Multi-Head Attention and edge feature bias."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        in_dim = config.get("in_dim", 14)
        edge_dim = config.get("edge_dim", 6)
        hidden_dim = config.get("hidden_dim", 128)
        num_layers = config.get("num_layers", 3)
        heads = config.get("heads", 4)
        dropout = config.get("dropout", 0.1)
        self.use_transformer_conv = config.get("use_transformer_conv", True)

        self.embedding = nn.Linear(in_dim, hidden_dim)

        if self.use_transformer_conv:
            out_per_head = max(1, hidden_dim // heads)
            self.convs = nn.ModuleList(
                [
                    TransformerConv(
                        in_channels=hidden_dim,
                        out_channels=out_per_head,
                        heads=heads,
                        edge_dim=edge_dim,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            # Projection in case hidden_dim is not evenly divisible by heads
            self.proj = (
                nn.Linear(out_per_head * heads, hidden_dim)
                if out_per_head * heads != hidden_dim
                else nn.Identity()
            )
        else:
            self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.proj = nn.Identity()

        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        self.act = nn.ReLU()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def extract_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Extract graph representation h in R^{hidden_dim} via global mean pooling."""
        graph = batch["drug_graph"]
        x, edge_index, batch_idx = graph.x, graph.edge_index, graph.batch

        h = self.embedding(x)

        has_edge_attr = (
            hasattr(graph, "edge_attr")
            and graph.edge_attr is not None
            and graph.edge_attr.numel() > 0
            and graph.edge_attr.size(0) == edge_index.size(1)
        )
        edge_attr = graph.edge_attr if has_edge_attr else None

        for conv, drop in zip(self.convs, self.dropouts):
            if self.use_transformer_conv:
                if edge_attr is not None:
                    h = conv(h, edge_index, edge_attr=edge_attr)
                else:
                    h = conv(h, edge_index)
                h = self.proj(h)
            else:
                h = conv(h, edge_index)
            h = self.act(h)
            h = drop(h)

        # Global readout pooling (mean across nodes in each graph)
        return global_mean_pool(h, batch_idx)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        hg = self.extract_features(batch)
        return self.head(hg)



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
