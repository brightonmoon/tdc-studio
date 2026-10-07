"""Directed Message Passing Neural Network with Internalized Two-Stage Hurdle Multi-Task Architecture.

Couples a shared D-MPNN molecular representation with:
1. High-binding Gating Classifier Head (extreme binding detection, AUC ~0.87)
2. High-binding Specialist Regressor Head
3. Low-binding Specialist Regressor Head
4. End-to-End Gated Mixture Prediction
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import scatter

from tdc_studio.core.registry import MODELS
from tdc_studio.models.base import BaseTherapeuticsModel
from tdc_studio.models.loss.hurdle_loss import HurdleMultiTaskLoss


@MODELS.register("dmpnn_hurdle")
class DMPNNHurdleModel(BaseTherapeuticsModel):
    """Internalized Two-Stage Hurdle D-MPNN Multi-Task Architecture."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        in_dim = config.get("in_dim", 14)  # Atom features
        edge_dim = config.get("edge_dim", 6)  # Bond features
        hidden_dim = config.get("hidden_dim", 300)
        depth = config.get("depth", config.get("num_layers", 3))
        dropout = config.get("dropout", 0.15)
        self.depth = depth

        # 1. Directed bond message passing
        self.w_i = nn.Linear(in_dim + edge_dim, hidden_dim, bias=False)
        self.w_m = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_a = nn.Linear(in_dim + hidden_dim, hidden_dim)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 2. Descriptors branch (2D + Biophysical + Boltzmann 3D)
        self.use_descriptors = config.get("use_descriptors", True)
        self.descriptor_dim = config.get("descriptor_dim", 210)

        pooled_dim = hidden_dim * 2

        if self.use_descriptors:
            self.desc_encoder = nn.Sequential(
                nn.BatchNorm1d(self.descriptor_dim),
                nn.Linear(self.descriptor_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            head_in_dim = pooled_dim + hidden_dim
        else:
            head_in_dim = pooled_dim

        self.head_in_dim = head_in_dim
        self.hidden_dim = hidden_dim

        # 3. Internalized Hurdle Multi-Task Heads
        # Head A: High-binding gating classification (Is binding >= 90%?)
        self.gate_head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Head B: High-binding specialist regressor
        self.high_head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Head C: Low/Mid-binding specialist regressor
        self.low_head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # 4. Hurdle Loss Function
        high_threshold = config.get("high_threshold", 90.0)
        low_threshold = config.get("low_threshold", 70.0)
        self.loss_fn = HurdleMultiTaskLoss(
            high_threshold=high_threshold,
            high_subthreshold=config.get("high_subthreshold", 85.0),
            low_threshold=low_threshold,
            weight_gate=config.get("weight_gate", 1.0),
            weight_high=config.get("weight_high", 0.5),
            weight_low=config.get("weight_low", 0.8),
            weight_mixture=config.get("weight_mixture", 1.0),
            low_sample_weight=config.get("low_sample_weight", 3.5),
        )

    def _extract_graph_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Extract graph molecular representation via directed message passing."""
        graph = batch["drug_graph"]
        x, edge_index, batch_idx = graph.x, graph.edge_index, graph.batch
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            h_v = self.act(
                self.w_a(
                    torch.cat(
                        [x, torch.zeros(num_nodes, self.w_m.out_features, device=x.device)], dim=-1
                    )
                )
            )
            h_mean = global_mean_pool(h_v, batch_idx)
            h_sum = global_add_pool(h_v, batch_idx)
            return torch.cat([h_mean, h_sum], dim=-1)

        src, dst = edge_index[0], edge_index[1]
        edge_attr = (
            graph.edge_attr
            if (
                hasattr(graph, "edge_attr")
                and graph.edge_attr is not None
                and graph.edge_attr.numel() > 0
            )
            else torch.zeros((num_edges, 6), device=x.device)
        )

        init_input = torch.cat([x[src], edge_attr], dim=-1)
        h0 = self.act(self.w_i(init_input))
        h = h0

        rev = torch.arange(num_edges, device=edge_index.device) ^ 1

        for _ in range(self.depth):
            incoming_sum = scatter(h, dst, dim=0, dim_size=num_nodes, reduce="sum")
            msg = incoming_sum[src] - h[rev]
            h = self.act(h0 + self.w_m(msg))
            h = self.dropout(h)

        node_incoming = scatter(h, dst, dim=0, dim_size=num_nodes, reduce="sum")
        atom_rep = self.act(self.w_a(torch.cat([x, node_incoming], dim=-1)))
        atom_rep = self.dropout(atom_rep)

        h_mean = global_mean_pool(atom_rep, batch_idx)
        h_sum = global_add_pool(atom_rep, batch_idx)
        return torch.cat([h_mean, h_sum], dim=-1)

    def extract_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Extract shared molecular embedding (Graph + optional Descriptors)."""
        h_mol = self._extract_graph_features(batch)

        if (
            self.use_descriptors
            and hasattr(self, "desc_encoder")
            and "descriptors" in batch
            and batch["descriptors"] is not None
        ):
            desc = batch["descriptors"]
            desc = torch.nan_to_num(desc, nan=0.0, posinf=50.0, neginf=-50.0)
            desc = torch.clamp(desc, min=-100.0, max=100.0)
            if desc.size(0) == 1 and self.desc_encoder[0].training:
                h_desc = self.desc_encoder[1:](desc)
            else:
                h_desc = self.desc_encoder(desc)
            return torch.cat([h_mol, h_desc], dim=-1)

        return h_mol

    def forward(
        self, batch: Dict[str, Any], return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass generating internalized Hurdle gating and regression outputs."""
        h = self.extract_features(batch)

        gate_logits = self.gate_head(h)  # Shape: (B, 1)
        p_gate = torch.sigmoid(gate_logits)

        pred_high = self.high_head(h)  # Shape: (B, 1)
        pred_low = self.low_head(h)  # Shape: (B, 1)

        # Smooth mixture prediction
        pred_mixture = p_gate * pred_high + (1.0 - p_gate) * pred_low

        out_dict = {
            "gate_logits": gate_logits,
            "gate_prob": p_gate,
            "pred_high": pred_high,
            "pred_low": pred_low,
            "pred_mixture": pred_mixture,
        }

        if return_dict:
            return out_dict

        # If in training mode and called by standard PyTorch loop, return tensor of shape (B, 4)
        if self.training:
            return torch.cat([gate_logits, pred_high, pred_low, pred_mixture], dim=-1)

        # In evaluation / inference mode, default to returning mixture prediction
        return pred_mixture

    def compute_loss(
        self,
        preds: Union[Dict[str, torch.Tensor], torch.Tensor],
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_per_task: bool = False,
    ):
        """Compute internalized Hurdle multi-task loss."""
        loss, metrics = self.loss_fn(preds, targets, mask=mask)
        if return_per_task:
            return metrics
        return loss
