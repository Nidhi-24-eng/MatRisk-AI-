"""
cif_parser.py - Crystallographic CIF Parser & Periodic Graph Featurizer
Part of MatRisk AI (Phase 1, Day 2)

Converts raw Crystallographic Information Files (CIF) into PyTorch Geometric
crystal multigraphs with:
- Periodic neighbor detection within an 8.0 Å cutoff radius
- Gaussian Radial Basis Function (RBF) edge distance featurization
- Atomic node featurization (atomic number, electronegativity, radius)
- Mechanical & thermodynamic target vector assignment (K, G, E_form, Eg, K_Ic)
"""

import os
import sys
import math
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    # Fallback container mimicking PyG Data for testing without PyG installed
    class Data:
        def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, **kwargs):
            self.x = x
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.y = y
            for k, v in kwargs.items():
                setattr(self, k, v)


# Default directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CIF_DIR = DATA_DIR / "raw_cifs"
PROCESSED_DIR = DATA_DIR / "processed"
ID_PROP_CSV = RAW_CIF_DIR / "id_prop.csv"

# Periodic table elemental properties: [Atomic Number, Electronegativity (Pauling), Covalent Radius (pm), Valence Electrons]
ELEMENT_PROPERTIES = {
    "H":  [1, 2.20, 31, 1],
    "C":  [6, 2.55, 76, 4],
    "N":  [7, 3.04, 71, 5],
    "O":  [8, 3.44, 66, 6],
    "Al": [13, 1.61, 121, 3],
    "Si": [14, 1.90, 111, 4],
    "Ti": [22, 1.54, 160, 4],
    "V":  [23, 1.63, 153, 5],
    "Cr": [24, 1.66, 139, 6],
    "Mn": [25, 1.55, 139, 7],
    "Fe": [26, 1.83, 132, 8],
    "Co": [27, 1.88, 126, 9],
    "Ni": [28, 1.91, 124, 10],
    "Cu": [29, 1.90, 132, 11],
    "Zn": [30, 1.65, 122, 12],
    "Zr": [40, 1.33, 175, 4],
    "Nb": [41, 1.60, 164, 5],
    "Mo": [42, 2.16, 154, 6],
    "Ta": [73, 1.50, 170, 5],
    "W":  [74, 2.36, 162, 6],
}
DEFAULT_ELEM = [0, 1.50, 140, 4]


class GaussianRBFExpansion:
    """Expands interatomic scalar distances into Gaussian Radial Basis Functions across 64 center bins."""

    def __init__(self, dmin: float = 0.0, dmax: float = 8.0, num_steps: int = 64, var: Optional[float] = None):
        self.centers = np.linspace(dmin, dmax, num_steps)
        if var is None:
            self.gamma = 1.0 / ((self.centers[1] - self.centers[0]) ** 2)
        else:
            self.gamma = 1.0 / var

    def expand(self, distances: np.ndarray) -> np.ndarray:
        """Expand [E] distances to [E, num_steps] Gaussian features."""
        # distances shape: (E, 1), centers: (num_steps,)
        return np.exp(-self.gamma * ((distances[:, np.newaxis] - self.centers) ** 2))


def lattice_params_to_matrix(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Converts lattice lengths and angles (in degrees) to 3x3 Cartesian lattice matrix."""
    alpha_r = math.radians(alpha)
    beta_r = math.radians(beta)
    gamma_r = math.radians(gamma)

    v_x = a
    v_y = 0.0
    v_z = 0.0

    b_x = b * math.cos(gamma_r)
    b_y = b * math.sin(gamma_r)
    b_z = 0.0

    c_x = c * math.cos(beta_r)
    c_y = c * (math.cos(alpha_r) - math.cos(beta_r) * math.cos(gamma_r)) / math.sin(gamma_r)
    c_z = math.sqrt(max(0.0, c ** 2 - c_x ** 2 - c_y ** 2))

    return np.array([
        [v_x, v_y, v_z],
        [b_x, b_y, b_z],
        [c_x, c_y, c_z]
    ], dtype=np.float64)


def parse_cif_text(cif_text: str) -> Dict[str, Any]:
    """Pure Python CIF parser for cell parameters and atomic fractional coordinates."""
    a = b = c = 1.0
    alpha = beta = gamma = 90.0
    species = []
    coords = []

    lines = cif_text.splitlines()
    in_atom_loop = False
    atom_col_indices = {}
    col_counter = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("_cell_length_a"):
            a = float(line.split()[1].split("(")[0])
        elif line.startswith("_cell_length_b"):
            b = float(line.split()[1].split("(")[0])
        elif line.startswith("_cell_length_c"):
            c = float(line.split()[1].split("(")[0])
        elif line.startswith("_cell_angle_alpha"):
            alpha = float(line.split()[1].split("(")[0])
        elif line.startswith("_cell_angle_beta"):
            beta = float(line.split()[1].split("(")[0])
        elif line.startswith("_cell_angle_gamma"):
            gamma = float(line.split()[1].split("(")[0])

        elif line.startswith("loop_"):
            in_atom_loop = False
            atom_col_indices = {}
            col_counter = 0

        elif line.startswith("_atom_site_"):
            tag = line.split()[0]
            atom_col_indices[tag] = col_counter
            col_counter += 1
            in_atom_loop = True

        elif in_atom_loop:
            parts = line.split()
            if len(parts) >= len(atom_col_indices) and len(atom_col_indices) > 0:
                # Extract element symbol
                type_col = atom_col_indices.get("_atom_site_type_symbol", atom_col_indices.get("_atom_site_label", 0))
                sym = parts[type_col]
                # Filter out numbers from label e.g., 'Fe1' -> 'Fe'
                clean_sym = "".join([char for char in sym if char.isalpha()])

                x_col = atom_col_indices.get("_atom_site_fract_x", 1)
                y_col = atom_col_indices.get("_atom_site_fract_y", 2)
                z_col = atom_col_indices.get("_atom_site_fract_z", 3)

                x = float(parts[x_col].split("(")[0])
                y = float(parts[y_col].split("(")[0])
                z = float(parts[z_col].split("(")[0])

                species.append(clean_sym)
                coords.append([x, y, z])
            else:
                in_atom_loop = False

    return {
        "a": a, "b": b, "c": c,
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "species": species,
        "coords": np.array(coords, dtype=np.float64)
    }


def build_crystal_graph(cif_path: Path, cutoff: float = 8.0, rbf_steps: int = 64) -> Optional[Data]:
    """
    Constructs a PyG graph from CIF with periodic boundary conditions.
    Edges represent interatomic pairs within cutoff radius (default 8.0 Å).
    """
    try:
        content = cif_path.read_text(encoding="utf-8")
        parsed = parse_cif_text(content)
        species = parsed["species"]
        fract_coords = parsed["coords"]
        num_atoms = len(species)

        if num_atoms == 0:
            return None

        lattice_matrix = lattice_params_to_matrix(
            parsed["a"], parsed["b"], parsed["c"],
            parsed["alpha"], parsed["beta"], parsed["gamma"]
        )

        # Cartesian coordinates: r = f * L
        cart_coords = fract_coords @ lattice_matrix

        # Build Node Features [N, 4]: [atomic_num, electronegativity, radius, valence]
        node_feats = []
        for s in species:
            props = ELEMENT_PROPERTIES.get(s, DEFAULT_ELEM)
            node_feats.append(props)
        x = torch.tensor(node_feats, dtype=torch.float32)

        # Periodic neighbor search across supercells
        # Determine supercell bounds required by cutoff
        max_dist = cutoff
        recip_norms = np.linalg.norm(np.linalg.inv(lattice_matrix), axis=0)
        n_images = np.ceil(max_dist * recip_norms).astype(int)
        n_images = np.clip(n_images, 1, 3)

        i_range = range(-n_images[0], n_images[0] + 1)
        j_range = range(-n_images[1], n_images[1] + 1)
        k_range = range(-n_images[2], n_images[2] + 1)

        src_list = []
        dst_list = []
        distances = []

        rbf = GaussianRBFExpansion(dmin=0.0, dmax=cutoff, num_steps=rbf_steps)

        for di in i_range:
            for dj in j_range:
                for dk in k_range:
                    offset = np.array([di, dj, dk], dtype=np.float64) @ lattice_matrix
                    is_zero_offset = (di == 0 and dj == 0 and dk == 0)

                    for src_idx in range(num_atoms):
                        pos_src = cart_coords[src_idx]
                        for dst_idx in range(num_atoms):
                            if is_zero_offset and src_idx == dst_idx:
                                continue  # No self loops in central cell

                            pos_dst = cart_coords[dst_idx] + offset
                            dist = np.linalg.norm(pos_dst - pos_src)
                            if dist <= cutoff:
                                src_list.append(src_idx)
                                dst_list.append(dst_idx)
                                distances.append(dist)

        if len(distances) == 0:
            # Fallback if cutoff is too small: connect to self or closest
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, rbf_steps), dtype=torch.float32)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            dist_array = np.array(distances, dtype=np.float32)
            edge_rbf = rbf.expand(dist_array)
            edge_attr = torch.tensor(edge_rbf, dtype=torch.float32)

        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_atoms
        )
        return graph_data

    except Exception as e:
        print(f"[!] Error processing {cif_path.name}: {e}")
        return None


def process_all_cifs(cutoff: float = 8.0, max_files: Optional[int] = None) -> int:
    """Processes all raw CIF files and caches processed graph objects as PyTorch .pt files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Read targets from id_prop.csv
    targets_map: Dict[str, Dict[str, float]] = {}
    if ID_PROP_CSV.exists():
        with open(ID_PROP_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                targets_map[row["material_id"]] = {
                    "bulk_modulus": float(row.get("bulk_modulus", 0.0)),
                    "shear_modulus": float(row.get("shear_modulus", 0.0)),
                    "formation_energy": float(row.get("formation_energy_per_atom", 0.0)),
                    "band_gap": float(row.get("band_gap", 0.0)),
                    "fracture_toughness": float(row.get("fracture_toughness", 50.0)),
                }

    cif_files = sorted(list(RAW_CIF_DIR.glob("*.cif")))
    if max_files:
        cif_files = cif_files[:max_files]

    print(f"[*] Processing {len(cif_files)} CIF files into crystal multigraphs (cutoff: {cutoff} Å)...")
    processed_count = 0

    for cif_path in cif_files:
        mat_id = cif_path.stem
        graph = build_crystal_graph(cif_path, cutoff=cutoff)
        if graph is None:
            continue

        # Attach target labels if available
        if mat_id in targets_map:
            t = targets_map[mat_id]
            # y tensor: [bulk_modulus (GPa), shear_modulus (GPa), formation_energy (eV), fracture_toughness (MPa*m^0.5)]
            graph.y = torch.tensor([
                [t["bulk_modulus"], t["shear_modulus"], t["formation_energy"], t["fracture_toughness"]]
            ], dtype=torch.float32)
            graph.material_id = mat_id

        out_path = PROCESSED_DIR / f"{mat_id}.pt"
        torch.save(graph, out_path)
        processed_count += 1

    print(f"[+] Successfully saved {processed_count} crystal graph tensors to {PROCESSED_DIR}")
    return processed_count


def main():
    parser = argparse.ArgumentParser(description="MatRisk AI Crystallographic Graph Featurizer")
    parser.add_argument("--cutoff", type=float, default=8.0, help="Radial cutoff distance in Angstroms (default 8.0)")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum CIF files to process")

    args = parser.parse_args()
    process_all_cifs(cutoff=args.cutoff, max_files=args.max_files)


if __name__ == "__main__":
    main()
