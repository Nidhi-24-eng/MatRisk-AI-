"""
test_phase1.py - Unit Tests for Phase 1 (Data Ingestion & CIF Parsing)
"""

import os
from pathlib import Path
import pytest

from src.data.download_mp import generate_offline_benchmark, RAW_CIF_DIR, ID_PROP_CSV
from src.atomistic.cif_parser import parse_cif_text, build_crystal_graph, GaussianRBFExpansion


def test_offline_benchmark_generation(tmp_path):
    """Test generating calibrated benchmark CIFs."""
    generate_offline_benchmark(num_samples=10)
    assert RAW_CIF_DIR.exists()
    assert ID_PROP_CSV.exists()
    cif_files = list(RAW_CIF_DIR.glob("*.cif"))
    assert len(cif_files) >= 10


def test_gaussian_rbf_expansion():
    """Test Gaussian RBF expansion dimensions and properties."""
    import numpy as np
    rbf = GaussianRBFExpansion(dmin=0.0, dmax=8.0, num_steps=50)
    distances = np.array([1.5, 2.5, 4.0], dtype=np.float32)
    feats = rbf.expand(distances)
    assert feats.shape == (3, 50)
    # Check values are between 0 and 1
    assert np.all(feats >= 0.0) and np.all(feats <= 1.0)


def test_cif_parser_and_graph_construction():
    """Test parsing a generated CIF into a periodic crystal graph."""
    cif_files = list(RAW_CIF_DIR.glob("*.cif"))
    assert len(cif_files) > 0, "Run download_mp.py first to produce CIF files."

    test_cif = cif_files[0]
    graph = build_crystal_graph(test_cif, cutoff=8.0, rbf_steps=50)

    assert graph is not None
    assert graph.x.shape[1] == 4  # [atomic_num, electronegativity, radius, valence]
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == 50
    assert graph.edge_index.shape[1] == graph.edge_attr.shape[0]
