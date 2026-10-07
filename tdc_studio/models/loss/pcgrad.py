"""PCGrad (Projecting Conflicting Gradients) Optimizer Wrapper.

Reference:
    Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
    https://arxiv.org/abs/2001.06782
"""

import random
from typing import Any, List, Optional, Sequence

import torch
from torch.optim import Optimizer


class PCGrad(Optimizer):
    """PCGrad (Projecting Conflicting Gradients) optimizer wrapper.

    When multi-task learning exhibits conflicting gradients (i.e. g_i . g_j < 0),
    PCGrad projects g_i onto the normal plane of g_j:
        g_i = g_i - (g_i . g_j / ||g_j||^2) * g_j
    This eliminates destructive interference between disparate ADMET endpoints
    while preserving non-conflicting directional progress.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        reduction: str = "sum",
        eps: float = 1e-12,
    ):
        """Initialize PCGrad.

        Args:
            optimizer: Underlying PyTorch optimizer (e.g., AdamW, SGD).
            reduction: Reduction across tasks ('sum' or 'mean'). Default: 'sum'.
            eps: Epsilon for numerical stability when dividing by gradient norms.
        """
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"Expected torch.optim.Optimizer, got {type(optimizer)}")
        if reduction not in ("sum", "mean"):
            raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction}")

        self.optimizer = optimizer
        self.reduction = reduction
        self.eps = eps
        # Share optimizer internals for lr_scheduler and inspect compatibility
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self.defaults = optimizer.defaults
        self._step_count = getattr(optimizer, "_step_count", 0)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients of all optimized parameters."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> Any:
        """Perform a single optimization step using underlying optimizer."""
        return self.optimizer.step(closure=closure)

    def _get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Collect all unique trainable parameters across all parameter groups."""
        params = []
        seen = set()
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        return params

    def _project_conflicting(
        self,
        task_grads: List[torch.Tensor],
        rng: Optional[random.Random] = None,
    ) -> List[torch.Tensor]:
        """Perform gradient surgery by projecting conflicting gradients.

        Args:
            task_grads: List of 1D tensors, each representing the flattened
                        gradient vector for a task.
            rng: Optional random generator for shuffling order.

        Returns:
            List of projected 1D gradient tensors.
        """
        num_tasks = len(task_grads)
        projected_grads = [g.clone() for g in task_grads]

        order = list(range(num_tasks))
        if rng is not None:
            rng.shuffle(order)
        else:
            random.shuffle(order)

        for i in order:
            # Randomize the projection check order of other tasks for task i
            other_tasks = [j for j in range(num_tasks) if j != i]
            if rng is not None:
                rng.shuffle(other_tasks)
            else:
                random.shuffle(other_tasks)

            for j in other_tasks:
                g_j = task_grads[j]
                g_i_proj = projected_grads[i]

                # Dot product between task i's current projected grad and task j's original grad
                dot = torch.dot(g_i_proj, g_j)
                if dot < 0.0:
                    norm_sq = torch.dot(g_j, g_j)
                    if norm_sq > self.eps:
                        # Subtract the parallel projection component
                        projected_grads[i] = g_i_proj - (dot / (norm_sq + self.eps)) * g_j

        return projected_grads

    def pc_backward(
        self,
        objectives: Sequence[torch.Tensor],
        rng: Optional[random.Random] = None,
    ) -> None:
        """Compute PCGrad backward pass for multiple task loss objectives.

        Args:
            objectives: A list or tuple of scalar loss tensors for each task.
            rng: Optional random generator for reproducible order shuffling.
        """
        # Filter valid scalar objectives that require grad
        valid_objs = [
            obj for obj in objectives if obj is not None and getattr(obj, "requires_grad", False)
        ]
        if not valid_objs:
            return

        params = self._get_trainable_params()
        if not params:
            return

        # 1. Compute flattened gradients for each task
        task_grads: List[torch.Tensor] = []
        for idx, obj in enumerate(valid_objs):
            # Retain computation graph for all except possibly the last objective
            retain = idx < len(valid_objs) - 1
            grads = torch.autograd.grad(
                obj,
                params,
                retain_graph=retain,
                allow_unused=True,
            )

            # Flatten into a single contiguous 1D vector
            flat_grad_chunks = []
            for g, p in zip(grads, params):
                if g is not None:
                    flat_grad_chunks.append(g.contiguous().view(-1))
                else:
                    flat_grad_chunks.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))

            task_grads.append(torch.cat(flat_grad_chunks, dim=0))

        # 2. Apply PCGrad projection surgery
        projected = self._project_conflicting(task_grads, rng=rng)

        # 3. Aggregate across tasks (sum or mean)
        stacked = torch.stack(projected, dim=0)
        if self.reduction == "mean":
            final_grad = stacked.mean(dim=0)
        else:
            final_grad = stacked.sum(dim=0)

        # 4. Unflatten and assign back to parameter .grad
        offset = 0
        for p in params:
            numel = p.numel()
            p.grad = final_grad[offset : offset + numel].view_as(p).clone()
            offset += numel

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute lookups to the underlying optimizer."""
        return getattr(self.optimizer, name)
