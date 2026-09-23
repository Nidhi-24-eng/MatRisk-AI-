"""
test_cgcnn.py - Unit tests for Phase 2 Day 3 DA-CGCNN Architecture
"""

import math
import torch
import pytest

from src.atomistic.cgcnn_da import (
    GaussianRBF,
    ChannelAttentionModule,
    SpatialSelfAttention,
    DACGCNNConv,
    DualAttentionCGCNN,
    build_da_cgcnn_model
)


def test_gaussian_rbf_layer():
    """Verify Gaussian RBF produces 64 bins bounded in [0, 1]."""
    rbf = GaussianRBF(dmin=0.0, dmax=8.0, num_bins=64)
    distances = torch.tensor([1.2, 2.5, 5.0, 7.8])
    expanded = rbf(distances)

    assert expanded.shape == (4, 64)
    assert torch.all(expanded >= 0.0)
    assert torch.all(expanded <= 1.0)


def test_channel_attention_module():
    """Verify CAM module scales message channels dynamically."""
    cam = ChannelAttentionModule(channels=64, reduction_ratio=4)
    msg = torch.randn(12, 64)
    attended = cam(msg)

    assert attended.shape == (12, 64)


def test_spatial_self_attention():
    """Verify spatial attention weights sum to 1 over neighborhood segments."""
    num_nodes = 3
    # Directed edges: (0->1), (0->2), (1->2), (2->0)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]], dtype=torch.long)
    edge_attr = torch.randn(4, 32)
    x = torch.randn(num_nodes, 32)

    spatial_attn = SpatialSelfAttention(atom_dim=32, edge_dim=32, num_heads=2)
    alpha = spatial_attn(x, edge_index, edge_attr, num_nodes=num_nodes)

    assert alpha.shape == (4, 1)
    # Incoming to node 0 has 2 edges (edges 0 and 1)
    sum_node_0 = alpha[0].item() + alpha[1].item()
    assert math.isclose(sum_node_0, 1.0, rel_tol=1e-3, abs_tol=1e-3)


def test_da_cgcnn_conv_layer():
    """Verify DACGCNNConv layer preserves node embedding dimensions."""
    conv = DACGCNNConv(atom_dim=32, edge_dim=64)
    num_nodes = 4
    x = torch.randn(num_nodes, 32)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.randn(4, 64)

    out = conv(x, edge_index, edge_attr, num_nodes=num_nodes)
    assert out.shape == (num_nodes, 32)


def test_dual_attention_cgcnn_full_model():
    """Verify full DA-CGCNN model predictions and physical positivity constraints."""
    model = build_da_cgcnn_model(node_dim=4, edge_dim=64, hidden_dim=32, num_layers=2)

    # 5 atoms in crystal, 10 edges
    x = torch.randn(5, 4)
    edge_index = torch.randint(0, 5, (2, 10))
    edge_attr = torch.rand(10, 64)

    outputs = model(x, edge_index, edge_attr)

    # Assert all target outputs are present
    assert "bulk_modulus" in outputs
    assert "shear_modulus" in outputs
    assert "fracture_toughness" in outputs
    assert "formation_energy" in outputs
    assert "stiffness_components" in outputs

    # Assert physical positivity constraints
    k = outputs["bulk_modulus"].item()
    g = outputs["shear_modulus"].item()
    k_ic = outputs["fracture_toughness"].item()
    stiffness = outputs["stiffness_components"]

    assert k > 0.0, f"Bulk modulus must be positive, got {k}"
    assert g > 0.0, f"Shear modulus must be positive, got {g}"
    assert k_ic > 0.0, f"Fracture toughness must be positive, got {k_ic}"
    assert stiffness.shape == (1, 3)  # [C11, C12, C44]
