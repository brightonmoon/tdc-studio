"""Few-Shot LoRA and Residual Adapter for Drug-Target Affinity (DTA) fine-tuning.

Designed for real-world pharma/biotech settings where proprietary target data consists of
only 10 to 50 in-house measured assay data points:
1. Freezes massive pre-trained backbones (ChemBERTa, Uni-Mol, ESM-2, GINE) to prevent catastrophic forgetting.
2. Injects lightweight Bottleneck Residual Adapters and Low-Rank Adaptation (LoRA) layers (<1% parameter footprint).
3. Employs zero-initialised residual output projections so base model predictions are strictly preserved at step 0.
4. Supports independent serialisation of adapter checkpoints (<500 KB) for lightweight sharing and deployment.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("tdc_studio.models.dti.adapter")


class ResidualBottleneckAdapter(nn.Module):
    """Bottleneck Residual Adapter layer.

    Transforms representation x as:
        x_adapt = x + scale * Linear_up(Act(Dropout(Linear_down(x))))

    Initialized such that output at initialization is exactly zero or identity.

    Args:
        in_dim: Input and output representation dimension.
        bottleneck_dim: Compressed inner bottleneck dimension (typically 8 to 32).
        dropout: Dropout probability.
        scale: Scaling multiplier for adapter perturbation.
    """

    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int = 16,
        dropout: float = 0.1,
        scale: float = 0.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale

        self.down = nn.Linear(in_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, in_dim)

        # Zero-initialize the up projection so adapter starts as exact identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.kaiming_uniform_(self.down.weight, a=0.1)
        nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.dropout(self.act(self.down(x))))
        return x + self.scale * delta


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) wrapper for an existing nn.Linear layer.

    Computes:
        y = W_0 x + (alpha / r) * B(A(x))
    where W_0 is frozen, A has Gaussian init, and B is zero-initialized.

    Args:
        linear_layer: The base nn.Linear layer to wrap and freeze.
        rank: Low-rank dimension r (typically 4, 8, 16).
        alpha: LoRA scaling factor.
        dropout: LoRA dropout probability.
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(1, rank)

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


class FewShotDTAAdapter(nn.Module):
    """Wraps a pre-trained GraphDTAModel with lightweight, freeze-protected adapters.

    All pre-trained backbone parameters are strictly frozen (requires_grad=False).
    Only the residual adapter and delta prediction head receive gradients,
    preventing catastrophic forgetting on small datasets (N = 10 ~ 50).

    Args:
        base_model: Pre-trained GraphDTAModel instance.
        bottleneck_dim: Inner dimension for residual adapters (default 16).
        use_output_delta: If True, adds a zero-initialized delta MLP to predicted affinity.
    """

    def __init__(
        self,
        base_model: nn.Module,
        bottleneck_dim: int = 16,
        use_output_delta: bool = True,
    ):
        super().__init__()
        self.base_model = base_model

        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Determine drug_dim and target_dim dynamically from base model fusion layer
        if hasattr(self.base_model.fusion, "proj_drug"):
            drug_dim = self.base_model.fusion.proj_drug.in_features
            target_dim = self.base_model.fusion.proj_target.in_features
        elif hasattr(self.base_model.fusion, "bilinear"):
            drug_dim = self.base_model.fusion.bilinear.in1_features
            target_dim = self.base_model.fusion.bilinear.in2_features
        else:
            drug_dim = getattr(self.base_model, "drug_out_dim", getattr(self.base_model.fusion, "drug_dim", 256))
            target_dim = getattr(self.base_model, "target_out_dim", getattr(self.base_model.fusion, "target_dim", 256))

        self.drug_adapter = ResidualBottleneckAdapter(drug_dim, bottleneck_dim=bottleneck_dim)
        self.target_adapter = ResidualBottleneckAdapter(target_dim, bottleneck_dim=bottleneck_dim)

        self.use_output_delta = use_output_delta
        if use_output_delta:
            fusion_hidden = getattr(self.base_model.fusion, "hidden_dim", 512)
            # Small delta head mapping adapted representations to delta affinity
            self.delta_head = nn.Sequential(
                nn.Linear(drug_dim + target_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, 1),
            )
            # Initialize delta head to zeros
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)
        else:
            self.delta_head = None

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return only the parameters belonging to the adapter."""
        return [p for p in self.parameters() if p.requires_grad]

    def forward(
        self,
        batch: Dict[str, Any],
        return_attention: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Forward pass with adapter applied to extracted representations."""
        # 1. Extract base representations from frozen backbones
        with torch.no_grad():
            h_drug, h_target = self.base_model.extract_features(batch)

        # 2. Apply trainable residual adapters
        if h_drug.dim() == 2:
            h_drug_adapt = self.drug_adapter(h_drug)
        else:
            # 3D tensor [B, L, D]
            h_drug_adapt = self.drug_adapter(h_drug)

        if h_target.dim() == 2:
            h_target_adapt = self.target_adapter(h_target)
        else:
            h_target_adapt = self.target_adapter(h_target)

        # 3. Base fusion prediction with adapted representations
        extra_kwargs = {
            k: batch[k]
            for k in ("pocket_coords", "residue_importance", "target_padding_mask", "drug_padding_mask")
            if k in batch
        }

        if return_attention:
            try:
                base_pred, attn_dict = self.base_model.fusion(
                    h_drug_adapt, h_target_adapt, return_attention=True, **extra_kwargs
                )
            except TypeError:
                base_pred = self.base_model.fusion(h_drug_adapt, h_target_adapt, **extra_kwargs)
                attn_dict = {}
        else:
            base_pred = self.base_model.fusion(h_drug_adapt, h_target_adapt, **extra_kwargs)
            attn_dict = None

        # 4. Optional zero-init delta residual adjustment
        if self.use_output_delta and self.delta_head is not None:
            # Pool if sequence
            d_p = h_drug_adapt.mean(dim=1) if h_drug_adapt.dim() == 3 else h_drug_adapt
            t_p = h_target_adapt.mean(dim=1) if h_target_adapt.dim() == 3 else h_target_adapt
            delta = self.delta_head(torch.cat([d_p, t_p], dim=-1))
            final_pred = base_pred + delta
        else:
            final_pred = base_pred

        if return_attention:
            return final_pred, attn_dict
        return final_pred

    def save_adapter_weights(self, path: Union[str, Path]) -> None:
        """Save only the lightweight adapter weights (<500 KB)."""
        adapter_state = {
            "drug_adapter": self.drug_adapter.state_dict(),
            "target_adapter": self.target_adapter.state_dict(),
        }
        if self.delta_head is not None:
            adapter_state["delta_head"] = self.delta_head.state_dict()
        torch.save(adapter_state, str(path))
        logger.info("Saved few-shot adapter weights to %s", path)

    def load_adapter_weights(self, path: Union[str, Path]) -> None:
        """Load adapter weights from checkpoint."""
        state = torch.load(str(path), map_location="cpu")
        self.drug_adapter.load_state_dict(state["drug_adapter"])
        self.target_adapter.load_state_dict(state["target_adapter"])
        if "delta_head" in state and self.delta_head is not None:
            self.delta_head.load_state_dict(state["delta_head"])
        logger.info("Loaded few-shot adapter weights from %s", path)


class FewShotTrainer:
    """Trainer specializing in extreme sample efficiency (10 to 50 samples) fine-tuning."""

    def __init__(
        self,
        adapter_model: FewShotDTAAdapter,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = adapter_model
        self.optimizer = torch.optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.loss_fn = nn.MSELoss()

    def train_few_shot(
        self,
        dataloader: DataLoader,
        epochs: int = 25,
        early_stopping_patience: int = 5,
    ) -> Dict[str, Any]:
        """Fit adapter to small assay dataset.

        Args:
            dataloader: DataLoader delivering batches with batch["affinity"] or batch["labels"].
            epochs: Maximum training epochs.
            early_stopping_patience: Patience for early loss convergence.

        Returns:
            Dict containing training history and convergence metrics.
        """
        self.model.train()
        history: List[float] = []
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for batch in dataloader:
                self.optimizer.zero_grad()
                preds = self.model(batch)

                targets = batch.get("affinity", batch.get("labels"))
                if targets is None:
                    continue
                targets = targets.float().view_as(preds)

                loss = self.loss_fn(preds, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            history.append(avg_loss)

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.debug("Early stopping triggered at epoch %d", epoch)
                break

        return {
            "initial_loss": history[0] if history else 0.0,
            "final_loss": history[-1] if history else 0.0,
            "best_loss": best_loss,
            "epochs_run": len(history),
            "loss_history": history,
        }
