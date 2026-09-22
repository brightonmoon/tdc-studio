"""GINE (Graph Isomorphism Network with Edge attributes) model for molecular property prediction."""

from typing import Any, Dict

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

from tdc_studio.core.registry import MODELS
from tdc_studio.models.base import BaseTherapeuticsModel


@MODELS.register("gine")
class GINEModel(BaseTherapeuticsModel):
    """Graph Isomorphism Network incorporating chemical bond features (GINEConv)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        in_dim = config.get("in_dim", 14)  # Node features dim
        edge_dim = config.get("edge_dim", 6)  # Bond features dim (6 dims)
        hidden_dim = config.get("hidden_dim", 128)
        num_layers = config.get("num_layers", 4)
        dropout = config.get("dropout", 0.1)

        self.node_embed = nn.Linear(in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.act = nn.ReLU()
        # Head combines mean and add pooling (hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def extract_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Extract pooled molecular graph representation h in R^{hidden_dim * 2}."""
        graph = batch["drug_graph"]
        x, edge_index, batch_idx = graph.x, graph.edge_index, graph.batch

        # Embed nodes
        h = self.node_embed(x)

        # Handle edge attributes
        if (
            hasattr(graph, "edge_attr")
            and graph.edge_attr is not None
            and graph.edge_attr.numel() > 0
        ):
            edge_attr = self.edge_embed(graph.edge_attr)
        else:
            # Fallback zero edge attributes if missing
            edge_attr = torch.zeros((edge_index.size(1), h.size(1)), device=x.device)

        # Message passing layers
        for conv, bn, drop in zip(self.convs, self.batch_norms, self.dropouts):
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = bn(h)
            h = self.act(h)
            h = drop(h)

        # Dual readout pooling (mean + sum)
        h_mean = global_mean_pool(h, batch_idx)
        h_sum = global_add_pool(h, batch_idx)
        return torch.cat([h_mean, h_sum], dim=-1)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        h_pool = self.extract_features(batch)
        return self.head(h_pool)

