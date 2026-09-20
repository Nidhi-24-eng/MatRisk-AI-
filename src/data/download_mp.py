"""
download_mp.py - Materials Project Ingestion & Calibrated Benchmark Generator
Part of MatRisk AI (Phase 1, Day 1)

This module handles:
1. Ingesting structural alloy CIFs and mechanical targets (K, G, E_form, E_g) from the Materials Project API (mp-api).
2. Generating calibrated structural alloy benchmark CIFs offline if no API key is provided, enabling instant local development.
"""

import os
import sys
import argparse
import random
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any

# Target paths
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_CIF_DIR = DATA_DIR / "raw_cifs"
ID_PROP_CSV = RAW_CIF_DIR / "id_prop.csv"

# Pre-defined benchmark structural material archetypes
STRUCTURAL_ARCHETYPES = [
    {
        "id": "mp-iron-bcc",
        "formula": "Fe",
        "name": "Ferritic Steel Matrix (alpha-Fe BCC)",
        "a": 2.866, "b": 2.866, "c": 2.866,
        "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        "elements": ["Fe", "Fe"],
        "coords": [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        "bulk_modulus": 170.0,       # GPa
        "shear_modulus": 82.0,        # GPa
        "formation_energy": -0.05,    # eV/atom
        "band_gap": 0.0,              # Metallic
        "fracture_toughness": 50.0    # MPa*m^0.5
    },
    {
        "id": "mp-austenitic-ss",
        "formula": "Fe11Cr4Ni1",
        "name": "Austenitic Stainless Steel 316L (FCC matrix)",
        "a": 3.590, "b": 3.590, "c": 3.590,
        "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        "elements": ["Fe", "Cr", "Ni", "Fe"],
        "coords": [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
        "bulk_modulus": 160.0,
        "shear_modulus": 77.0,
        "formation_energy": -0.12,
        "band_gap": 0.0,
        "fracture_toughness": 75.0
    },
    {
        "id": "mp-inconel-718",
        "formula": "Ni3Al",
        "name": "Nickel Superalloy gamma-prime precipitate",
        "a": 3.568, "b": 3.568, "c": 3.568,
        "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        "elements": ["Al", "Ni", "Ni", "Ni"],
        "coords": [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
        "bulk_modulus": 182.0,
        "shear_modulus": 88.0,
        "formation_energy": -0.42,
        "band_gap": 0.0,
        "fracture_toughness": 85.0
    },
    {
        "id": "mp-titanium-hcp",
        "formula": "Ti",
        "name": "Titanium Grade 5 (alpha-phase HCP)",
        "a": 2.950, "b": 2.950, "c": 4.686,
        "alpha": 90.0, "beta": 90.0, "gamma": 120.0,
        "elements": ["Ti", "Ti"],
        "coords": [(0.333333, 0.666667, 0.25), (0.666667, 0.333333, 0.75)],
        "bulk_modulus": 110.0,
        "shear_modulus": 44.0,
        "formation_energy": -0.08,
        "band_gap": 0.0,
        "fracture_toughness": 55.0
    },
    {
        "id": "mp-aluminum-7075",
        "formula": "Al",
        "name": "High-Strength Structural Aluminum 7075 (FCC)",
        "a": 4.049, "b": 4.049, "c": 4.049,
        "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        "elements": ["Al", "Al", "Al", "Al"],
        "coords": [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
        "bulk_modulus": 76.0,
        "shear_modulus": 26.0,
        "formation_energy": -0.02,
        "band_gap": 0.0,
        "fracture_toughness": 29.0
    },
    {
        "id": "mp-copper-structural",
        "formula": "Cu",
        "name": "Structural Copper Alloy (FCC)",
        "a": 3.615, "b": 3.615, "c": 3.615,
        "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        "elements": ["Cu", "Cu", "Cu", "Cu"],
        "coords": [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
        "bulk_modulus": 137.0,
        "shear_modulus": 48.0,
        "formation_energy": 0.0,
        "band_gap": 0.0,
        "fracture_toughness": 60.0
    },
    {
        "id": "mp-tungsten-carbide",
        "formula": "WC",
        "name": "Hardened Wear Resistant Tungsten Carbide",
        "a": 2.906, "b": 2.906, "c": 2.837,
        "alpha": 90.0, "beta": 90.0, "gamma": 120.0,
        "elements": ["W", "C"],
        "coords": [(0.0, 0.0, 0.0), (0.333333, 0.666667, 0.5)],
        "bulk_modulus": 390.0,
        "shear_modulus": 280.0,
        "formation_energy": -0.41,
        "band_gap": 0.0,
        "fracture_toughness": 15.0
    }
]


def write_cif_file(file_path: Path, arch: Dict[str, Any], strain_factor: float = 1.0) -> None:
    """Generate standard Crystallographic Information File (CIF) format."""
    a = arch["a"] * strain_factor
    b = arch["b"] * strain_factor
    c = arch["c"] * strain_factor
    alpha = arch["alpha"]
    beta = arch["beta"]
    gamma = arch["gamma"]

    lines = [
        f"data_{arch['id']}",
        f"_chemical_formula_sum '{arch['formula']}'",
        f"_cell_length_a {a:.5f}",
        f"_cell_length_b {b:.5f}",
        f"_cell_length_c {c:.5f}",
        f"_cell_angle_alpha {alpha:.3f}",
        f"_cell_angle_beta {beta:.3f}",
        f"_cell_angle_gamma {gamma:.3f}",
        "_symmetry_space_group_name_H-M 'P 1'",
        "loop_",
        " _atom_site_label",
        " _atom_site_type_symbol",
        " _atom_site_fract_x",
        " _atom_site_fract_y",
        " _atom_site_fract_z"
    ]

    for idx, (elem, coord) in enumerate(zip(arch["elements"], arch["coords"])):
        lines.append(f"  {elem}{idx+1} {elem} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")

    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_offline_benchmark(num_samples: int = 500) -> None:
    """
    Generate calibrated structural alloy CIF files and metadata CSV
    for high-speed local development without cloud or API dependencies.
    """
    RAW_CIF_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    rows = []
    print(f"[*] Generating {num_samples} calibrated structural CIF benchmark structures in {RAW_CIF_DIR}...")

    # First write base archetypes
    for arch in STRUCTURAL_ARCHETYPES:
        cif_path = RAW_CIF_DIR / f"{arch['id']}.cif"
        write_cif_file(cif_path, arch, strain_factor=1.0)
        rows.append({
            "material_id": arch["id"],
            "formula": arch["formula"],
            "bulk_modulus": arch["bulk_modulus"],
            "shear_modulus": arch["shear_modulus"],
            "formation_energy_per_atom": arch["formation_energy"],
            "band_gap": arch["band_gap"],
            "fracture_toughness": arch["fracture_toughness"]
        })

    # Generate synthetic perturbations (compositional/strain variations)
    remaining = num_samples - len(STRUCTURAL_ARCHETYPES)
    for i in range(remaining):
        base_arch = random.choice(STRUCTURAL_ARCHETYPES)
        strain = random.uniform(0.96, 1.04)
        sample_id = f"matrisk-bench-{i+1:04d}"

        # Physical scaling: elastic modulus typically inversely scales with volume expansion
        k_mod = base_arch["bulk_modulus"] * (1.0 / (strain ** 3)) * random.uniform(0.95, 1.05)
        g_mod = base_arch["shear_modulus"] * (1.0 / (strain ** 3)) * random.uniform(0.95, 1.05)
        e_form = base_arch["formation_energy"] + random.uniform(-0.05, 0.05)
        bg = base_arch["band_gap"]
        k_ic = base_arch["fracture_toughness"] * random.uniform(0.92, 1.08)

        sample_arch = dict(base_arch)
        sample_arch["id"] = sample_id

        cif_path = RAW_CIF_DIR / f"{sample_id}.cif"
        write_cif_file(cif_path, sample_arch, strain_factor=strain)

        rows.append({
            "material_id": sample_id,
            "formula": base_arch["formula"],
            "bulk_modulus": round(k_mod, 2),
            "shear_modulus": round(g_mod, 2),
            "formation_energy_per_atom": round(e_form, 4),
            "band_gap": round(bg, 3),
            "fracture_toughness": round(k_ic, 2)
        })

    # Write id_prop.csv
    with open(ID_PROP_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["material_id", "formula", "bulk_modulus", "shear_modulus", "formation_energy_per_atom", "band_gap", "fracture_toughness"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Successfully saved {len(rows)} structural CIFs and targets to {ID_PROP_CSV}")


def download_from_materials_project(api_key: str, max_materials: int = 3000) -> None:
    """Download structural material CIFs and elasticity targets using mp-api."""
    try:
        from mp_api.client import MPRester
    except ImportError:
        print("[!] mp-api is not installed. Please run: pip install mp-api")
        sys.exit(1)

    RAW_CIF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Querying Materials Project API for structural materials (target: {max_materials})...")

    with MPRester(api_key) as mpr:
        # Search for transition-metal structural systems with elasticity data
        docs = mpr.materials.summary.search(
            has_props=["elasticity"],
            fields=["material_id", "formula_pretty", "structure", "k_vrh", "g_vrh", "formation_energy_per_atom", "band_gap"],
            num_chunks=max_materials // 100 + 1
        )

        rows = []
        count = 0
        for doc in docs:
            if doc.k_vrh is None or doc.g_vrh is None:
                continue

            mat_id = str(doc.material_id)
            cif_path = RAW_CIF_DIR / f"{mat_id}.cif"
            doc.structure.to(filename=str(cif_path), fmt="cif")

            rows.append({
                "material_id": mat_id,
                "formula": doc.formula_pretty,
                "bulk_modulus": round(doc.k_vrh, 2),
                "shear_modulus": round(doc.g_vrh, 2),
                "formation_energy_per_atom": round(doc.formation_energy_per_atom, 4) if doc.formation_energy_per_atom else 0.0,
                "band_gap": round(doc.band_gap, 3) if doc.band_gap else 0.0,
                "fracture_toughness": round(0.5 * (doc.k_vrh * doc.g_vrh) ** 0.5, 2)  # Ashby rule-of-thumb estimate
            })
            count += 1
            if count >= max_materials:
                break

        with open(ID_PROP_CSV, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["material_id", "formula", "bulk_modulus", "shear_modulus", "formation_energy_per_atom", "band_gap", "fracture_toughness"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[+] Successfully fetched {count} Materials Project CIFs to {RAW_CIF_DIR}")


def main():
    parser = argparse.ArgumentParser(description="MatRisk AI Materials Ingestion Engine")
    parser.add_argument("--api-key", type=str, default=os.getenv("MP_API_KEY", ""), help="Materials Project API Key")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of benchmark samples to generate/fetch")
    parser.add_argument("--offline", action="store_true", help="Force offline calibrated benchmark generation")

    args = parser.parse_args()

    if args.offline or not args.api_key:
        print("[*] Running in Offline Calibrated Benchmark Mode (no MP_API_KEY required).")
        generate_offline_benchmark(num_samples=args.num_samples)
    else:
        download_from_materials_project(api_key=args.api_key, max_materials=args.num_samples)


if __name__ == "__main__":
    main()
