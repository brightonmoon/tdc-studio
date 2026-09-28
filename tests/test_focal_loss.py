"""Unit tests for Binary Focal Loss."""

import torch
import torch.nn.functional as F

from tdc_studio.models.loss.focal_loss import BinaryFocalLossWithLogits


def test_focal_loss_reduces_to_bce_when_gamma_zero_and_alpha_none():
    loss_fn = BinaryFocalLossWithLogits(gamma=0.0, alpha=None)
    logits = torch.randn(20, requires_grad=True)
    targets = torch.randint(0, 2, (20,)).float()

    focal_val = loss_fn(logits, targets)
    bce_val = F.binary_cross_entropy_with_logits(logits, targets)

    assert torch.allclose(focal_val, bce_val, atol=1e-5)


def test_focal_loss_downweights_easy_examples():
    loss_fn = BinaryFocalLossWithLogits(gamma=2.0, alpha=None, reduction="none")

    # Well-classified positive (logit = 5.0, target = 1.0) -> easy
    # Misclassified positive (logit = -5.0, target = 1.0) -> hard
    logits = torch.tensor([5.0, -5.0])
    targets = torch.tensor([1.0, 1.0])

    losses = loss_fn(logits, targets)
    # The hard example should have vastly higher focal loss than the easy example
    assert losses[1] > losses[0] * 100.0


def test_focal_loss_with_mask_and_nans():
    loss_fn = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.7)
    logits = torch.tensor([1.0, -1.0, 2.0, 0.5])
    targets = torch.tensor([1.0, 0.0, float("nan"), 1.0])
    mask = torch.tensor([True, True, True, False])  # 4th item masked out

    val = loss_fn(logits, targets, mask=mask)
    assert torch.isfinite(val)
    assert val > 0.0


def test_focal_loss_backward():
    loss_fn = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.7)
    logits = torch.randn(10, requires_grad=True)
    targets = torch.randint(0, 2, (10,)).float()

    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
