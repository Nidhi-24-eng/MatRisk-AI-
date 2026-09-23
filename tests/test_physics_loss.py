"""
test_physics_loss.py - Unit tests for Phase 2 Day 4 Physics-Constrained Born Stability Loss
"""

import torch
import pytest

from src.atomistic.physics_loss import BornStabilityLoss, PhysicsConstrainedCGCNNLoss


def test_born_stability_loss_penalties():
    """Verify Born loss penalizes unphysical values (negative moduli, violation of Pugh ratio)."""
    born_fn = BornStabilityLoss()

    # Physically unstable test inputs:
    # 1. Negative bulk modulus (-50 GPa)
    # 2. Violation of Pugh ratio: K = 20, G = 50 -> 4/3 * G = 66.7 > K
    k_unstable = torch.tensor([[-50.0], [20.0]])
    g_unstable = torch.tensor([[80.0], [50.0]])

    stiffness_unstable = torch.tensor([
        [-10.0, 50.0, 30.0],   # C11 < 0, |C12| > C11
        [100.0, 150.0, -20.0]  # |C12| > C11, C44 < 0
    ])

    loss, breakdown = born_fn(k_unstable, g_unstable, stiffness_unstable)

    assert loss.item() > 0.0
    assert breakdown["loss_k_nonneg"] > 0.0
    assert breakdown["loss_pugh"] > 0.0
    assert breakdown["loss_c11"] > 0.0
    assert breakdown["loss_c44"] > 0.0
    assert breakdown["loss_shear"] > 0.0


def test_born_stability_loss_zero_on_stable_crystal():
    """Verify Born loss produces negligible penalty for a stable physical material (e.g., steel/titanium)."""
    born_fn = BornStabilityLoss()

    # Stable steel matrix: K=160, G=75, C11=220, C12=130, C44=80
    k_stable = torch.tensor([[160.0]])
    g_stable = torch.tensor([[75.0]])
    stiffness_stable = torch.tensor([[220.0, 130.0, 80.0]])

    loss, breakdown = born_fn(k_stable, g_stable, stiffness_stable)
    assert loss.item() < 1e-2


def test_physics_constrained_loss_gradient_flow():
    """Verify gradients propagate through the combined physics loss to model parameters."""
    loss_fn = PhysicsConstrainedCGCNNLoss(beta_born=0.2)

    pred_k = torch.tensor([[150.0], [60.0]], requires_grad=True)
    pred_g = torch.tensor([[70.0], [25.0]], requires_grad=True)
    pred_e = torch.tensor([[-0.1], [-0.05]], requires_grad=True)
    pred_k_ic = torch.tensor([[50.0], [30.0]], requires_grad=True)
    stiffness = torch.tensor([[200.0, 120.0, 70.0], [80.0, 45.0, 25.0]], requires_grad=True)

    preds = {
        "bulk_modulus": pred_k,
        "shear_modulus": pred_g,
        "formation_energy": pred_e,
        "fracture_toughness": pred_k_ic,
        "stiffness_components": stiffness
    }

    targets = torch.tensor([
        [160.0, 75.0, -0.12, 55.0],
        [70.0, 30.0, -0.04, 32.0]
    ])

    loss, metrics = loss_fn(preds, targets)
    loss.backward()

    assert pred_k.grad is not None
    assert pred_g.grad is not None
    assert stiffness.grad is not None
    assert metrics["mae_k"] >= 0.0
    assert metrics["mae_g"] >= 0.0
