"""
cgcnn_da.py - Dual-Attention Crystal Graph Convolutional Neural Network (DA-CGCNN)
Part of MatRisk AI (Phase 2, Day 3)

Architecture:
- 64-center Gaussian Radial Basis Function (RBF) expansion of interatomic edge vectors.
- Channel Attention Module (CAM) for dynamic inter-channel message scaling.
- Spatial Multi-Head Self-Attention over periodic neighbor interactions.
- Dual-gated convolution message passing with residual connections.
- Global crystal attention pooling with multi-property prediction heads:
  * Bulk Modulus (K), Shear Modulus (G)
  * Fracture Toughness (K_Ic)
  * Formation energy per atom (E_form)
  * Elastic stiffness components (C11, C12, C44) for Born stability constraints
"""

import math
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    Data = None
    Batch = None


class GaussianRBF(nn.Module):
    """
    Gaussian Radial Basis Function (RBF) Layer.
    Expands scalar interatomic distances d_ij into 64 Gaussian bins:
    phi_k(d_ij) = exp( - gamma * (d_ij - mu_k)^2 )
    """

    def __init__(self, dmin: float = 0.0, dmax: float = 8.0, num_bins: int = 64):
        super().__init__()
        self.num_bins = num_bins
        self.dmin = dmin
        self.dmax = dmax

        centers = torch.linspace(dmin, dmax, num_bins)
        self.register_buffer("centers", centers)
        delta = (dmax - dmin) / (num_bins - 1)
        self.gamma = 1.0 / (delta ** 2)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: [num_edges] or [num_edges, 1] tensor of interatomic distances.
        Returns:
            [num_edges, num_bins] RBF expansion.
        """
        if distances.dim() == 1:
            distances = distances.unsqueeze(-1)
        return torch.exp(-self.gamma * (distances - self.centers) ** 2)


class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module (CAM) for message representations.
    Dynamically recalibrates channel-wise feature responses using a bottleneck MLP.
    m'_c = m_c * sigmoid( W2 * ReLU( W1 * m ) )
    """

    def __init__(self, channels: int, reduction_ratio: int = 4):
        super().__init__()
        reduced_dim = max(4, channels // reduction_ratio)
        self.fc1 = nn.Linear(channels, reduced_dim, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_dim, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_edges, channels] message tensor
        Returns:
            [num_edges, channels] channel-attended message
        """
        weights = self.fc2(self.act(self.fc1(x)))
        return x * self.sigmoid(weights)


class SpatialSelfAttention(nn.Module):
    """
    Spatial Attention mechanism over atomic neighborhoods.
    Computes normalized attention coefficients alpha_ij for neighbor j around atom i.
    alpha_ij = Softmax_j ( (Q(v_i) * K(v_j) + W_edge * e_ij) / sqrt(d_k) )
    """

    def __init__(self, atom_dim: int, edge_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = atom_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(atom_dim, atom_dim, bias=False)
        self.k_proj = nn.Linear(atom_dim, atom_dim, bias=False)
        self.edge_proj = nn.Linear(edge_dim, atom_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, atom_dim] node embeddings
            edge_index: [2, num_edges] source and target indices
            edge_attr: [num_edges, edge_dim] edge embeddings
            num_nodes: Total number of nodes in graph/batch
        Returns:
            alpha: [num_edges, 1] normalized spatial attention weights
        """
        src, dst = edge_index[0], edge_index[1]

        q = self.q_proj(x[src])       # [num_edges, atom_dim]
        k = self.k_proj(x[dst])       # [num_edges, atom_dim]
        e = self.edge_proj(edge_attr) # [num_edges, atom_dim]

        # Dot product attention + edge bias
        raw_scores = torch.sum((q * k + e), dim=-1, keepdim=True) * self.scale  # [num_edges, 1]

        # Segment softmax over incoming edges to src (node i)
        # Numerator exp(score - max)
        score_exp = torch.exp(raw_scores - raw_scores.max())
        sum_exp = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
        sum_exp.scatter_add_(0, src.unsqueeze(-1), score_exp)
        denom = sum_exp[src] + 1e-8
        alpha = score_exp / denom

        return alpha


class DACGCNNConv(nn.Module):
    """
    Dual-Attention Crystal Graph Convolution Layer.
    Combines:
    1. CGCNN gated convolution: z_ij = [v_i, v_j, e_ij], m_ij = sig(z * W_f) * softplus(z * W_s)
    2. Channel Attention Module (CAM)
    3. Spatial Self-Attention weights alpha_ij
    4. Residual connection: v_i^(t+1) = v_i^(t) + LayerNorm( sum_j alpha_ij * CAM(m_ij) )
    """

    def __init__(self, atom_dim: int = 64, edge_dim: int = 64, reduction_ratio: int = 4):
        super().__init__()
        self.atom_dim = atom_dim
        self.edge_dim = edge_dim

        # Input to convolution is [v_i, v_j, e_ij] -> 2 * atom_dim + edge_dim
        total_in_dim = 2 * atom_dim + edge_dim

        # Filter and core convolutions
        self.fc_filter = nn.Linear(total_in_dim, atom_dim)
        self.fc_core = nn.Linear(total_in_dim, atom_dim)

        # Dual attention modules
        self.channel_attention = ChannelAttentionModule(atom_dim, reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialSelfAttention(atom_dim, edge_dim)

        # Normalization
        self.norm = nn.LayerNorm(atom_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, atom_dim] atom embeddings
            edge_index: [2, num_edges] directed edge connectivity
            edge_attr: [num_edges, edge_dim] edge attributes
        Returns:
            [num_nodes, atom_dim] updated atom embeddings
        """
        if num_nodes is None:
            num_nodes = x.size(0)

        src, dst = edge_index[0], edge_index[1]

        # 1. Form pairwise interaction vectors z_ij = [v_src, v_dst, e_ij]
        z_ij = torch.cat([x[src], x[dst], edge_attr], dim=-1)

        # 2. CGCNN gated message formulation
        msg_filter = torch.sigmoid(self.fc_filter(z_ij))
        msg_core = F.softplus(self.fc_core(z_ij))
        msg = msg_filter * msg_core  # [num_edges, atom_dim]

        # 3. Channel Attention on messages
        msg_cam = self.channel_attention(msg)  # [num_edges, atom_dim]

        # 4. Spatial Attention weights across neighbors
        alpha = self.spatial_attention(x, edge_index, edge_attr, num_nodes=num_nodes)  # [num_edges, 1]

        # 5. Weighted aggregation: sum_j alpha_ij * msg_cam
        weighted_msg = alpha * msg_cam  # [num_edges, atom_dim]

        # Scatter add into target node src
        agg_msg = torch.zeros(num_nodes, self.atom_dim, device=x.device, dtype=x.dtype)
        agg_msg.scatter_add_(0, src.unsqueeze(-1).expand(-1, self.atom_dim), weighted_msg)

        # 6. Residual connection + normalization
        out = self.norm(x + agg_msg)
        return out


class CrystalGlobalAttentionPool(nn.Module):
    """Global attention pooling over atomic nodes to produce crystal-level embedding."""

    def __init__(self, atom_dim: int):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(atom_dim, atom_dim // 2),
            nn.SiLU(),
            nn.Linear(atom_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, atom_dim]
            batch: [num_nodes] batch index tensor indicating crystal membership
        Returns:
            [batch_size, atom_dim] crystal-level representations
        """
        if batch is None:
            # Single crystal case
            weights = F.softmax(self.gate_nn(x), dim=0)
            return torch.sum(weights * x, dim=0, keepdim=True)

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        scores = self.gate_nn(x)  # [num_nodes, 1]

        # Segment softmax
        scores_exp = torch.exp(scores - scores.max())
        sum_exp = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        sum_exp.scatter_add_(0, batch.unsqueeze(-1), scores_exp)
        denom = sum_exp[batch] + 1e-8
        weights = scores_exp / denom

        out = torch.zeros(batch_size, x.size(-1), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, batch.unsqueeze(-1).expand(-1, x.size(-1)), weights * x)
        return out


class DualAttentionCGCNN(nn.Module):
    """
    Dual-Attention Crystal Graph Convolutional Neural Network (DA-CGCNN).

    Predicts:
    1. Bulk Modulus K (GPa)
    2. Shear Modulus G (GPa)
    3. Fracture Toughness K_Ic (MPa*m^0.5)
    4. Formation energy per atom (eV/atom)
    5. Cubic elastic stiffness parameters C11, C12, C44 (GPa) for Born mechanical stability
    """

    def __init__(
        self,
        node_in_dim: int = 4,         # [atomic_number, electronegativity, radius, valence]
        edge_in_dim: int = 64,        # 64 Gaussian RBF bins
        atom_hidden_dim: int = 64,
        edge_hidden_dim: int = 64,
        num_conv_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.node_embedding = nn.Sequential(
            nn.Linear(node_in_dim, atom_hidden_dim),
            nn.LayerNorm(atom_hidden_dim),
            nn.SiLU()
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_in_dim, edge_hidden_dim),
            nn.LayerNorm(edge_hidden_dim),
            nn.SiLU()
        )

        # Stack of Dual-Attention CGCNN convolution layers
        self.conv_layers = nn.ModuleList([
            DACGCNNConv(atom_dim=atom_hidden_dim, edge_dim=edge_hidden_dim)
            for _ in range(num_conv_layers)
        ])

        # Global crystal readout
        self.pool = CrystalGlobalAttentionPool(atom_hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Multi-task Prediction Heads
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(atom_hidden_dim, 128),
            nn.SiLU(),
            self.dropout,
            nn.Linear(128, 64),
            nn.SiLU()
        )

        # 1. Moduli head: [Bulk Modulus K, Shear Modulus G] (constrained positive via Softplus)
        self.head_moduli = nn.Linear(64, 2)

        # 2. Fracture toughness head: K_Ic (positive via Softplus)
        self.head_fracture = nn.Linear(64, 1)

        # 3. Thermodynamic head: Formation energy per atom (unbounded eV/atom)
        self.head_thermo = nn.Linear(64, 1)

        # 4. Elastic stiffness components for Born stability: [C11, C12, C44]
        self.head_stiffness = nn.Linear(64, 3)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of DA-CGCNN.

        Args:
            x: [num_nodes, node_in_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_in_dim] (64-bin Gaussian RBF features)
            batch: [num_nodes] batch index tensor

        Returns:
            Dictionary with predicted properties:
            - 'bulk_modulus': [batch_size, 1] (GPa)
            - 'shear_modulus': [batch_size, 1] (GPa)
            - 'fracture_toughness': [batch_size, 1] (MPa*m^0.5)
            - 'formation_energy': [batch_size, 1] (eV/atom)
            - 'stiffness_components': [batch_size, 3] (C11, C12, C44 in GPa)
        """
        num_nodes = x.size(0)

        # Embed nodes and edges
        h_node = self.node_embedding(x)
        h_edge = self.edge_embedding(edge_attr)

        # Dual-Attention message passing
        for conv in self.conv_layers:
            h_node = conv(h_node, edge_index, h_edge, num_nodes=num_nodes)

        # Global pooling
        c_repr = self.pool(h_node, batch=batch)  # [batch_size, atom_hidden_dim]
        features = self.trunk(c_repr)           # [batch_size, 64]

        # Prediction heads
        raw_moduli = self.head_moduli(features)
        # Softplus ensures non-negative physical moduli
        k_pred = F.softplus(raw_moduli[:, 0:1]) + 1.0   # K > 0 (GPa)
        g_pred = F.softplus(raw_moduli[:, 1:2]) + 0.5   # G > 0 (GPa)

        k_ic_pred = F.softplus(self.head_fracture(features)) + 1.0  # K_Ic > 0
        e_form_pred = self.head_thermo(features)                    # eV/atom

        # Stiffness tensor components (C11, C12, C44)
        raw_c = self.head_stiffness(features)
        c11 = F.softplus(raw_c[:, 0:1]) + 5.0
        c12 = raw_c[:, 1:2]
        c44 = F.softplus(raw_c[:, 2:3]) + 1.0
        stiffness = torch.cat([c11, c12, c44], dim=-1)

        return {
            "bulk_modulus": k_pred,
            "shear_modulus": g_pred,
            "fracture_toughness": k_ic_pred,
            "formation_energy": e_form_pred,
            "stiffness_components": stiffness,
            "crystal_representation": c_repr
        }


def build_da_cgcnn_model(
    node_dim: int = 4,
    edge_dim: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 4
) -> DualAttentionCGCNN:
    """Helper factory function to instantiate DA-CGCNN."""
    return DualAttentionCGCNN(
        node_in_dim=node_dim,
        edge_in_dim=edge_dim,
        atom_hidden_dim=hidden_dim,
        edge_hidden_dim=hidden_dim,
        num_conv_layers=num_layers
    )


if __name__ == "__main__":
    print("[*] Testing DA-CGCNN architecture...")

    # Dummy test graph: 4 atoms, 8 edges, 64-bin RBF features
    num_atoms = 4
    num_edges = 8
    x_dummy = torch.randn(num_atoms, 4)
    edge_index_dummy = torch.randint(0, num_atoms, (2, num_edges))
    edge_attr_dummy = torch.rand(num_edges, 64)

    model = build_da_cgcnn_model(node_dim=4, edge_dim=64, hidden_dim=64, num_layers=3)
    outputs = model(x_dummy, edge_index_dummy, edge_attr_dummy)

    print("[+] DA-CGCNN forward pass successful!")
    print(f"    - Bulk Modulus K shape: {outputs['bulk_modulus'].shape}, value: {outputs['bulk_modulus'].item():.2f} GPa")
    print(f"    - Shear Modulus G shape: {outputs['shear_modulus'].shape}, value: {outputs['shear_modulus'].item():.2f} GPa")
    print(f"    - Fracture Toughness K_Ic: {outputs['fracture_toughness'].item():.2f} MPa*m^0.5")
    print(f"    - Stiffness [C11, C12, C44]: {outputs['stiffness_components'].detach().numpy()}")
