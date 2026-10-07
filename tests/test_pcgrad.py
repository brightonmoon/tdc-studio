"""Unit tests for PCGrad (Projecting Conflicting Gradients) optimizer wrapper."""

import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tdc_studio.models.loss.multitask_loss import MaskedMultiTaskLoss
from tdc_studio.models.loss.pcgrad import PCGrad


def test_pcgrad_orthogonal_projection_math():
    """Verify that conflicting gradients (dot < 0) are strictly projected onto normal planes."""
    # Shared 2D weight parameter w
    w = nn.Parameter(torch.tensor([0.0, 0.0], requires_grad=True))
    base_opt = SGD([w], lr=0.1)
    pc_opt = PCGrad(base_opt, reduction="sum")

    # Objective 1: L1 = w[0] -> g1 = [1.0, 0.0]
    # Objective 2: L2 = -w[0] + w[1] -> g2 = [-1.0, 1.0]
    # Dot product: g1 . g2 = -1.0 (conflict!)
    # Projection of g1 onto normal plane of g2:
    # g1' = g1 - (g1.g2 / ||g2||^2) * g2 = [1, 0] - (-1 / 2) * [-1, 1] = [0.5, 0.5]
    # Notice: g1' . g2 = 0.5 * (-1) + 0.5 * 1 = 0.0 (perfectly orthogonal!)
    # Projection of g2 onto normal plane of g1:
    # g2' = g2 - (g2.g1 / ||g1||^2) * g1 = [-1, 1] - (-1 / 1) * [1, 0] = [0.0, 1.0]
    # Final gradient (sum): g1' + g2' = [0.5, 1.5]
    l1 = w[0]
    l2 = -w[0] + w[1]

    pc_opt.zero_grad()
    pc_opt.pc_backward([l1, l2])

    assert w.grad is not None
    expected_grad = torch.tensor([0.5, 1.5])
    assert torch.allclose(w.grad, expected_grad, atol=1e-5), (
        f"Expected {expected_grad}, got {w.grad}"
    )


def test_pcgrad_non_conflicting_gradients():
    """Verify that non-conflicting gradients (dot >= 0) are preserved without modification."""
    w = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
    base_opt = SGD([w], lr=0.1)
    pc_opt = PCGrad(base_opt, reduction="sum")

    # g1 = [1.0, 2.0], g2 = [2.0, 3.0] -> g1 . g2 = 2 + 6 = 8 > 0 (No conflict)
    l1 = w[0] + 2.0 * w[1]
    l2 = 2.0 * w[0] + 3.0 * w[1]

    pc_opt.zero_grad()
    pc_opt.pc_backward([l1, l2])

    expected_sum = torch.tensor([3.0, 5.0])
    assert torch.allclose(w.grad, expected_sum, atol=1e-5)


def test_pcgrad_multi_task_and_reduction_mean():
    """Verify 3-task setup with mean reduction."""
    w = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], requires_grad=True))
    base_opt = AdamW([w], lr=0.01)
    pc_opt = PCGrad(base_opt, reduction="mean")

    l1 = w[0] + w[1]
    l2 = -w[0] + w[2]
    l3 = w[1] + w[2]

    pc_opt.zero_grad()
    pc_opt.pc_backward([l1, l2, l3])

    assert w.grad is not None
    assert torch.isfinite(w.grad).all()
    assert w.grad.shape == w.shape


def test_pcgrad_step_updates_weights():
    """Verify that optimizer.step() updates parameters."""
    w = nn.Parameter(torch.tensor([5.0, -5.0], requires_grad=True))
    init_val = w.clone().detach()

    base_opt = SGD([w], lr=0.1)
    pc_opt = PCGrad(base_opt)

    l1 = w[0] ** 2
    l2 = w[1] ** 2

    pc_opt.zero_grad()
    pc_opt.pc_backward([l1, l2])
    pc_opt.step()

    assert not torch.equal(w, init_val)


def test_pcgrad_with_cosine_scheduler():
    """Verify PyTorch LR scheduler compatibility."""
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    base_opt = AdamW(model.parameters(), lr=1e-3)
    pc_opt = PCGrad(base_opt)

    scheduler = CosineAnnealingLR(pc_opt, T_max=10, eta_min=1e-5)

    x = torch.randn(4, 8)
    out = model(x)
    l1 = out[:, 0].sum()
    l2 = out[:, 1].sum()

    pc_opt.zero_grad()
    pc_opt.pc_backward([l1, l2])
    pc_opt.step()
    scheduler.step()

    # Learning rate should still be accessible and updated
    current_lr = scheduler.get_last_lr()[0]
    assert current_lr <= 1e-3


def test_pcgrad_integration_with_masked_multitask_loss():
    """Verify end-to-end integration between MaskedMultiTaskLoss and PCGrad."""
    criterion = MaskedMultiTaskLoss(
        task_names=["tox_a", "tox_b", "tox_c"],
        task_types=["binary_classification", "regression", "binary_classification"],
        use_uncertainty=True,
    )

    linear = nn.Linear(10, 3)
    opt = PCGrad(AdamW(list(linear.parameters()) + list(criterion.parameters()), lr=1e-3))

    features = torch.randn(8, 10)
    preds = linear(features)
    targets = torch.tensor(
        [
            [1.0, 2.5, float("nan")],
            [0.0, 1.2, 1.0],
            [float("nan"), 3.1, 0.0],
            [1.0, float("nan"), 1.0],
            [0.0, 0.5, 0.0],
            [1.0, 4.2, float("nan")],
            [0.0, float("nan"), 0.0],
            [1.0, 1.8, 1.0],
        ]
    )

    opt.zero_grad()
    total_loss, loss_dict, task_loss_list = criterion(preds, targets, return_per_task=True)

    assert len(task_loss_list) == 3
    for l_t in task_loss_list:
        assert torch.isfinite(l_t)

    opt.pc_backward(task_loss_list)
    opt.step()

    for p in linear.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_pcgrad_empty_or_zero_objectives():
    """Verify graceful handling when objectives are empty or zero."""
    w = nn.Parameter(torch.tensor([1.0], requires_grad=True))
    pc_opt = PCGrad(SGD([w], lr=0.1))

    # Empty list should not raise error
    pc_opt.pc_backward([])
    assert w.grad is None

    # Loss without requires_grad
    pc_opt.pc_backward([torch.tensor(0.0)])
    assert w.grad is None
