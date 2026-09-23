"""
train.py - Training & Evaluation Engine for DA-CGCNN with Physics Constraints
Part of MatRisk AI (Phase 2, Day 4)

Features:
- Batches crystal multigraphs into disjoint block-diagonal mini-batches.
- Enforces Born mechanical stability constraints during backpropagation.
- Evaluates Mean Absolute Error (MAE) for Bulk Modulus (K) and Shear Modulus (G).
- Checkpoints best model to models/da_cgcnn_best.pt.
"""

import os
import sys
import time
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Local project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.atomistic.cgcnn_da import DualAttentionCGCNN, build_da_cgcnn_model
from src.atomistic.physics_loss import PhysicsConstrainedCGCNNLoss
from src.atomistic.cif_parser import process_all_cifs, PROCESSED_DIR, RAW_CIF_DIR

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class CrystalGraphDataset(Dataset):
    """Dataset wrapper for pre-processed crystal multigraph .pt tensors."""

    def __init__(self, data_list: List[Any]):
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Any:
        return self.data_list[idx]


def collate_crystal_graphs(batch: List[Any]) -> Dict[str, torch.Tensor]:
    """
    Collates a list of PyG-style crystal graph objects into a single disjoint batch.
    Adjusts edge indices with node offsets to enable parallel message passing on a CPU or GPU.
    """
    batch_x = []
    batch_edge_index = []
    batch_edge_attr = []
    batch_y = []
    batch_indices = []

    node_offset = 0
    for graph_idx, data in enumerate(batch):
        num_nodes = data.x.size(0)
        batch_x.append(data.x)

        if data.edge_index.numel() > 0:
            offset_edges = data.edge_index + node_offset
            batch_edge_index.append(offset_edges)
            batch_edge_attr.append(data.edge_attr)

        if hasattr(data, "y") and data.y is not None:
            batch_y.append(data.y)

        batch_indices.append(torch.full((num_nodes,), graph_idx, dtype=torch.long))
        node_offset += num_nodes

    x = torch.cat(batch_x, dim=0)
    batch_tensor = torch.cat(batch_indices, dim=0)

    if len(batch_edge_index) > 0:
        edge_index = torch.cat(batch_edge_index, dim=1)
        edge_attr = torch.cat(batch_edge_attr, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_dim_fallback = batch[0].edge_attr.size(1) if len(batch) > 0 and hasattr(batch[0], "edge_attr") and batch[0].edge_attr is not None and batch[0].edge_attr.ndim > 1 else 64
        edge_attr = torch.zeros((0, edge_dim_fallback), dtype=torch.float32)

    if len(batch_y) > 0:
        y = torch.cat(batch_y, dim=0)
    else:
        y = torch.zeros((len(batch), 4), dtype=torch.float32)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch_tensor,
        "y": y
    }


def load_dataset(processed_dir: Path = PROCESSED_DIR) -> List[Any]:
    """Loads all cached crystal graph tensors from disk."""
    pt_files = sorted(list(processed_dir.glob("*.pt")))
    if len(pt_files) == 0:
        print("[*] No processed .pt files found. Parsing CIFs from data/raw_cifs...")
        process_all_cifs(cutoff=8.0)
        pt_files = sorted(list(processed_dir.glob("*.pt")))

    graphs = []
    for f in pt_files:
        try:
            g = torch.load(f, weights_only=False)
            if hasattr(g, "y") and g.y is not None:
                graphs.append(g)
        except Exception as e:
            continue

    return graphs


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: PhysicsConstrainedCGCNNLoss,
    device: torch.device
) -> Tuple[float, float, float]:
    """Runs one training epoch over the dataset."""
    model.train()
    total_loss = 0.0
    total_mae_k = 0.0
    total_mae_g = 0.0
    count = 0

    for batch in loader:
        x = batch["x"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_idx = batch["batch"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        predictions = model(x, edge_index, edge_attr, batch=batch_idx)
        loss, metrics = loss_fn(predictions, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_mae_k += metrics["mae_k"] * batch_size
        total_mae_g += metrics["mae_g"] * batch_size
        count += batch_size

    return total_loss / count, total_mae_k / count, total_mae_g / count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: PhysicsConstrainedCGCNNLoss,
    device: torch.device
) -> Tuple[float, float, float, float]:
    """Evaluates validation/test set performance."""
    model.eval()
    total_loss = 0.0
    total_mae_k = 0.0
    total_mae_g = 0.0
    total_mae_k_ic = 0.0
    count = 0

    for batch in loader:
        x = batch["x"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_idx = batch["batch"].to(device)
        y = batch["y"].to(device)

        predictions = model(x, edge_index, edge_attr, batch=batch_idx)
        loss, metrics = loss_fn(predictions, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_mae_k += metrics["mae_k"] * batch_size
        total_mae_g += metrics["mae_g"] * batch_size
        total_mae_k_ic += metrics["mae_k_ic"] * batch_size
        count += batch_size

    return (
        total_loss / count,
        total_mae_k / count,
        total_mae_g / count,
        total_mae_k_ic / count
    )


def train_da_cgcnn(
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    beta_born: float = 0.1,
    device_name: Optional[str] = None
) -> Dict[str, Any]:
    """Full training pipeline for Phase 2 Day 4."""
    # Determine compute device
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[*] Initializing DA-CGCNN Training on device: {device}")

    # Load dataset
    graphs = load_dataset()
    if len(graphs) < 4:
        raise ValueError(f"Insufficient graphs found ({len(graphs)}). Run data generation first.")

    random.seed(42)
    random.shuffle(graphs)

    n_total = len(graphs)
    n_train = max(2, int(n_total * 0.8))
    n_val = max(1, int(n_total * 0.1))
    n_test = n_total - n_train - n_val

    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:n_train + n_val]
    test_graphs = graphs[n_train + n_val:]

    print(f"[*] Dataset split: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")

    train_loader = DataLoader(
        CrystalGraphDataset(train_graphs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_crystal_graphs
    )
    val_loader = DataLoader(
        CrystalGraphDataset(val_graphs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_crystal_graphs
    )
    test_loader = DataLoader(
        CrystalGraphDataset(test_graphs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_crystal_graphs
    )

    # Detect input edge dimension dynamically from the dataset
    detected_edge_dim = 64
    for g in graphs:
        if hasattr(g, "edge_attr") and g.edge_attr is not None and g.edge_attr.numel() > 0:
            detected_edge_dim = g.edge_attr.size(1)
            break

    print(f"[*] Configured DA-CGCNN edge feature dimension: {detected_edge_dim}")

    # Initialize model, loss, optimizer
    model = build_da_cgcnn_model(node_dim=4, edge_dim=detected_edge_dim, hidden_dim=64, num_layers=4).to(device)
    loss_fn = PhysicsConstrainedCGCNNLoss(beta_born=beta_born)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_mae = float("inf")
    best_checkpoint = MODELS_DIR / "da_cgcnn_best.pt"

    start_time = time.time()
    print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val MAE(K)':<14} {'Val MAE(G)':<14} {'Time':<8}")
    print("-" * 72)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_k, tr_g = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_k, val_g, val_k_ic = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"{epoch:<8} {tr_loss:<12.4f} {val_loss:<12.4f} {val_k:<14.2f} {val_g:<14.2f} {elapsed:<6.2f}s")

        # Save checkpoint if composite MAE improved
        avg_val_mae = (val_k + val_g) / 2.0
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae_k": val_k,
                "val_mae_g": val_g,
                "val_mae_k_ic": val_k_ic,
            }, best_checkpoint)

    total_time = time.time() - start_time
    print("-" * 72)
    print(f"[+] Training completed in {total_time:.2f}s (~{total_time/60:.1f} min)")
    print(f"[+] Best checkpoint saved to {best_checkpoint} with Val MAE(K): {val_k:.2f} GPa, MAE(G): {val_g:.2f} GPa")

    # Final evaluation on Test Set using best weights
    checkpoint = torch.load(best_checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_k, test_g, test_k_ic = evaluate(model, test_loader, loss_fn, device)

    print("\n" + "=" * 50)
    print(f"      FINAL TEST EVALUATION METRICS")
    print("=" * 50)
    print(f"  * Bulk Modulus (K) MAE:      {test_k:.2f} GPa")
    print(f"  * Shear Modulus (G) MAE:     {test_g:.2f} GPa")
    print(f"  * Fracture Toughness MAE:    {test_k_ic:.2f} MPa*m^0.5")
    print("=" * 50)

    return {
        "test_mae_k": test_k,
        "test_mae_g": test_g,
        "test_mae_k_ic": test_k_ic,
        "checkpoint_path": str(best_checkpoint)
    }


def main():
    parser = argparse.ArgumentParser(description="DA-CGCNN Physics-Constrained Training Engine")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--beta-born", type=float, default=0.1, help="Physics constraint loss weight (default: 0.1)")
    parser.add_argument("--device", type=str, default=None, help="Device to train on ('cpu' or 'cuda')")

    args = parser.parse_args()
    train_da_cgcnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta_born=args.beta_born,
        device_name=args.device
    )


if __name__ == "__main__":
    main()
