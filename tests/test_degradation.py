"""
test_degradation.py - Unit tests for Phase 3 (Paris' Law Mechanics & Stochastic SDE Solver)
"""

import time
import pytest
import numpy as np

from src.degradation.paris_law import MaterialMechanics, analyze_alloy_mechanics
from src.degradation.sde_engine import (
    StochasticDegradationEngine,
    simulate_sde_vectorized_numpy,
    quick_degradation_simulation
)


def test_material_mechanics_elastic_conversions():
    """Verify Young's modulus and Poisson's ratio relations."""
    # Steel benchmark: K=160 GPa, G=77 GPa
    mech = MaterialMechanics(bulk_modulus=160.0, shear_modulus=77.0, fracture_toughness=75.0)

    # Expected E ~ 198.8 GPa, nu ~ 0.29
    assert 190.0 < mech.e_gpa < 210.0
    assert 0.25 < mech.nu < 0.32
    assert 2.5 <= mech.paris_m <= 4.2
    assert mech.paris_c > 0.0


def test_critical_crack_depth_scaling():
    """Verify critical crack depth increases with fracture toughness and decreases with stress."""
    mech_tough = MaterialMechanics(160.0, 77.0, fracture_toughness=100.0)
    mech_brittle = MaterialMechanics(160.0, 77.0, fracture_toughness=40.0)

    a_crit_tough = mech_tough.compute_critical_crack_depth(sigma_max_mpa=250.0)
    a_crit_brittle = mech_brittle.compute_critical_crack_depth(sigma_max_mpa=250.0)

    assert a_crit_tough > a_crit_brittle


def test_sde_engine_execution_speed():
    """Verify 10,000 Monte Carlo paths simulate in under 2.0 seconds on laptop CPU."""
    mech = MaterialMechanics(160.0, 77.0, 75.0)
    engine = StochasticDegradationEngine(mech)

    t0 = time.time()
    res = engine.run_monte_carlo(
        delta_sigma_mpa=120.0,
        sigma_max_mpa=240.0,
        horizon_years=10.0,
        steps_per_year=52,
        num_trajectories=10000
    )
    elapsed = time.time() - t0

    assert elapsed < 2.0, f"SDE simulation exceeded laptop budget: {elapsed:.2f}s"
    assert len(res["failure_probability"]) == 521
    assert 0.0 <= res["final_failure_prob_10yr"] <= 1.0


def test_sde_failure_probability_monotonicity():
    """Verify cumulative failure probability P_f(t) is monotonically non-decreasing."""
    res = quick_degradation_simulation(
        bulk_modulus=160.0,
        shear_modulus=77.0,
        fracture_toughness=75.0,
        delta_sigma_mpa=140.0,
        sigma_max_mpa=280.0,
        num_trajectories=5000
    )
    pf = np.array(res["failure_probability"])

    # np.diff(pf) must be non-negative
    diffs = np.diff(pf)
    assert np.all(diffs >= -1e-6), "Cumulative failure probability decreased over time!"


def test_stress_shock_increases_failure_risk():
    """Verify higher cyclic stress delta_sigma increases 10-year failure probability."""
    res_low_stress = quick_degradation_simulation(delta_sigma_mpa=100.0, num_trajectories=3000)
    res_high_stress = quick_degradation_simulation(delta_sigma_mpa=150.0, num_trajectories=3000)

    p_low = res_low_stress["final_failure_prob_10yr"]
    p_high = res_high_stress["final_failure_prob_10yr"]

    assert p_high >= p_low, f"Higher stress should produce higher failure probability: {p_high} vs {p_low}"
