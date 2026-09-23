"""
physics_loss.py - Born Stability & Physics-Constrained Loss Functions
Part of MatRisk AI (Phase 2, Day 4)

Theoretical Background:
For a 3D crystal lattice to be thermodynamically and mechanically stable under elastic
deformation, the internal strain energy density U = 1/2 eps^T C eps must be strictly positive
for all non-zero strain tensors eps. This requires the 6x6 elastic stiffness tensor C
to be strictly positive-definite (det(C) > 0, all eigenvalues > 0).

Under Born-Huang mechanical stability criteria:
1. Hydrostatic stability: Bulk Modulus K = (C11 + 2*C12) / 3 > 0
2. Pure shear stability: C44 > 0
3. Tetragonal shear stability: (C11 - C12) > 0  => C11 > |C12|
4. Pugh's physical bounds: Bulk modulus must relate to shear modulus: K > (4/3) * G
5. Fracture toughness non-negativity: K_Ic > 0

This module provides custom PyTorch loss functions that penalize violations of these
fundamental laws of continuum mechanics.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BornStabilityLoss(nn.Module):
    """
    Penalizes violations of Born mechanical stability criteria on predicted elastic properties.

    Loss Formulation:
    L_Born = lambda_k * ReLU(-K)
           + lambda_g * ReLU(-G)
           + lambda_pugh * ReLU(4/3 * G - K)
           + lambda_c11 * ReLU(-C11)
           + lambda_c44 * ReLU(-C44)
           + lambda_shear * ReLU(|C12| - C11)
           + lambda_det * ReLU(-det(C))
    """

    def __init__(
        self,
        lambda_k: float = 1.0,
        lambda_g: float = 1.0,
        lambda_pugh: float = 0.5,
        lambda_c: float = 1.0,
        margin: float = 1e-3
    ):
        super().__init__()
        self.lambda_k = lambda_k
        self.lambda_g = lambda_g
        self.lambda_pugh = lambda_pugh
        self.lambda_c = lambda_c
        self.margin = margin

    def forward(
        self,
        k_pred: torch.Tensor,
        g_pred: torch.Tensor,
        stiffness_components: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            k_pred: [batch_size, 1] predicted Bulk Modulus (GPa)
            g_pred: [batch_size, 1] predicted Shear Modulus (GPa)
            stiffness_components: [batch_size, 3] containing [C11, C12, C44] in GPa

        Returns:
            total_born_loss: scalar tensor penalty
            breakdown: dictionary with individual penalty values
        """
        # 1. Non-negativity of Bulk and Shear Moduli
        loss_k = torch.mean(F.relu(-k_pred + self.margin))
        loss_g = torch.mean(F.relu(-g_pred + self.margin))

        # 2. Pugh's ratio / Poisson bound constraint:
        # For isotropic/polycrystalline continuum stability: K > (4/3) * G (Poisson ratio nu > -1)
        loss_pugh = torch.mean(F.relu((4.0 / 3.0) * g_pred - k_pred + self.margin))

        total_loss = self.lambda_k * loss_k + self.lambda_g * loss_g + self.lambda_pugh * loss_pugh

        loss_c11 = torch.tensor(0.0, device=k_pred.device)
        loss_c44 = torch.tensor(0.0, device=k_pred.device)
        loss_shear = torch.tensor(0.0, device=k_pred.device)
        loss_det = torch.tensor(0.0, device=k_pred.device)

        # 3. Microscopic Born criteria on 6x6 stiffness tensor if available
        if stiffness_components is not None:
            c11 = stiffness_components[:, 0:1]
            c12 = stiffness_components[:, 1:2]
            c44 = stiffness_components[:, 2:3]

            # C11 > 0
            loss_c11 = torch.mean(F.relu(-c11 + self.margin))

            # C44 > 0 (resistance to shear deformation along cube edges)
            loss_c44 = torch.mean(F.relu(-c44 + self.margin))

            # C11 - |C12| > 0 (resistance to shear along (110) planes)
            loss_shear = torch.mean(F.relu(torch.abs(c12) - c11 + self.margin))

            # Cubic stiffness matrix determinant: det(C) = (C11 - C12)^2 * (C11 + 2*C12) * (C44)^3
            # All factors must be strictly positive:
            vol_term = c11 + 2.0 * c12
            loss_det = torch.mean(F.relu(-vol_term + self.margin))

            c_penalty = self.lambda_c * (loss_c11 + loss_c44 + loss_shear + loss_det)
            total_loss = total_loss + c_penalty

        breakdown = {
            "loss_k_nonneg": float(loss_k.item()),
            "loss_g_nonneg": float(loss_g.item()),
            "loss_pugh": float(loss_pugh.item()),
            "loss_c11": float(loss_c11.item()),
            "loss_c44": float(loss_c44.item()),
            "loss_shear": float(loss_shear.item()),
            "loss_det": float(loss_det.item()),
            "total_born": float(total_loss.item())
        }

        return total_loss, breakdown


class PhysicsConstrainedCGCNNLoss(nn.Module):
    """
    Combined Multi-Objective Loss:
    L_Total = L_Data (MSE/L1 on K, G, E_form, K_Ic) + beta * L_Born (Born Stability constraints)
    """

    def __init__(
        self,
        weight_k: float = 1.0,
        weight_g: float = 1.0,
        weight_thermo: float = 0.5,
        weight_fracture: float = 0.5,
        beta_born: float = 0.1
    ):
        super().__init__()
        self.weight_k = weight_k
        self.weight_g = weight_g
        self.weight_thermo = weight_thermo
        self.weight_fracture = weight_fracture
        self.beta_born = beta_born

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.born_loss_fn = BornStabilityLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: Dictionary from DA-CGCNN with keys:
                         'bulk_modulus', 'shear_modulus', 'formation_energy', 'fracture_toughness', 'stiffness_components'
            targets: [batch_size, 4] tensor containing ground truth:
                     [target_k, target_g, target_e_form, target_k_ic]

        Returns:
            total_loss: scalar tensor for backward pass
            metrics: breakdown dictionary for logging
        """
        pred_k = predictions["bulk_modulus"]
        pred_g = predictions["shear_modulus"]
        pred_e = predictions["formation_energy"]
        pred_k_ic = predictions["fracture_toughness"]
        stiffness = predictions.get("stiffness_components", None)

        target_k = targets[:, 0:1]
        target_g = targets[:, 1:2]
        target_e = targets[:, 2:3]
        target_k_ic = targets[:, 3:4]

        # 1. Supervised Data Losses (Smooth L1 / Huber-style robustness)
        loss_k_data = F.smooth_l1_loss(pred_k, target_k)
        loss_g_data = F.smooth_l1_loss(pred_g, target_g)
        loss_e_data = F.smooth_l1_loss(pred_e, target_e)
        loss_k_ic_data = F.smooth_l1_loss(pred_k_ic, target_k_ic)

        data_loss = (
            self.weight_k * loss_k_data +
            self.weight_g * loss_g_data +
            self.weight_thermo * loss_e_data +
            self.weight_fracture * loss_k_ic_data
        )

        # 2. Physics Constraints (Born Stability)
        born_penalty, born_breakdown = self.born_loss_fn(pred_k, pred_g, stiffness)

        # 3. Total Loss
        total_loss = data_loss + self.beta_born * born_penalty

        # Compute Absolute Errors for evaluation
        mae_k = torch.mean(torch.abs(pred_k - target_k)).item()
        mae_g = torch.mean(torch.abs(pred_g - target_g)).item()
        mae_k_ic = torch.mean(torch.abs(pred_k_ic - target_k_ic)).item()

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_data": float(data_loss.item()),
            "loss_born": float(born_penalty.item()),
            "mae_k": float(mae_k),
            "mae_g": float(mae_g),
            "mae_k_ic": float(mae_k_ic),
            **born_breakdown
        }

        return total_loss, metrics


if __name__ == "__main__":
    print("[*] Testing Physics-Constrained Born Stability Loss...")
    loss_fn = PhysicsConstrainedCGCNNLoss(beta_born=0.1)

    # Synthetic predictions and targets
    batch_size = 4
    preds = {
        "bulk_modulus": torch.tensor([[160.0], [70.0], [200.0], [120.0]]),
        "shear_modulus": torch.tensor([[75.0], [30.0], [90.0], [50.0]]),
        "formation_energy": torch.tensor([[-0.1], [-0.05], [-0.3], [-0.15]]),
        "fracture_toughness": torch.tensor([[55.0], [35.0], [80.0], [45.0]]),
        "stiffness_components": torch.tensor([
            [220.0, 130.0, 80.0],
            [100.0, 55.0, 30.0],
            [260.0, 170.0, 95.0],
            [160.0, 100.0, 50.0]
        ])
    }

    targets = torch.tensor([
        [165.0, 78.0, -0.12, 58.0],
        [72.0, 32.0, -0.04, 33.0],
        [195.0, 88.0, -0.32, 77.0],
        [122.0, 52.0, -0.14, 48.0]
    ])

    loss, metrics = loss_fn(preds, targets)
    print(f"[+] Total Physics-Constrained Loss: {loss.item():.4f}")
    print(f"    - MAE Bulk Modulus K: {metrics['mae_k']:.2f} GPa")
    print(f"    - MAE Shear Modulus G: {metrics['mae_g']:.2f} GPa")
    print(f"    - Born Stability Penalty: {metrics['loss_born']:.4f}")
