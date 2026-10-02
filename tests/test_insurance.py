"""
test_insurance.py - Unit tests for Phase 5 (EVT Catastrophe Insurance & PARS ESG Resilience)
"""

import pytest
import numpy as np

from src.insurance.evt_pricing import CatastropheLossEVT, price_catastrophe_risk
from src.insurance.esg_resilience import PARSEngine, quick_pars_evaluation


def test_catastrophe_evt_loss_generation_and_tail_fit():
    """Verify EVT GPD tail fitting and actuarial solvency inequalities."""
    np.random.seed(42)
    n_sims = 2000
    n_steps = 100
    a_crit = 0.030  # 30 mm

    mock_traj = np.zeros((n_sims, n_steps))
    mock_traj[:1900, -1] = np.random.uniform(0.002, 0.010, size=1900)
    mock_traj[1900:, -1] = np.random.uniform(0.030, 0.045, size=100)  # 5% catastrophic

    evt = CatastropheLossEVT(asset_value_usd=100_000_000.0)
    losses = evt.generate_structural_loss_distribution(mock_traj, a_crit)
    pricing = evt.fit_gpd_tail(losses, alpha=0.99)

    # Fundamental Actuarial Inequalities:
    # Expected Shortfall >= Value at Risk >= Pure Expected Loss
    assert pricing["es_99_usd"] >= pricing["var_99_usd"]
    assert pricing["var_99_usd"] >= pricing["pure_premium_usd"]
    assert pricing["gross_actuarial_premium_usd"] > pricing["pure_premium_usd"]
    assert pricing["rate_on_line_pct"] > 0.0


def test_pars_resilience_bounds_and_ratings():
    """Verify PARS score remains bounded in [0, 100] and ratings match thresholds."""
    engine = PARSEngine()

    # Highly resilient asset
    res_high = engine.compute_asset_score(
        fracture_toughness_k_ic=95.0,
        a_crit_mm=50.0,
        mean_crack_10yr_mm=2.0,
        failure_prob_10yr=0.005,
        coastal_salinity_factor=1.0,
        thermal_cycling_range_c=10.0,
        min_dscr=1.45
    )

    assert 85.0 <= res_high["pars_score"] <= 100.0
    assert res_high["rating"] in ["AAA", "AA"]

    # Highly degraded asset
    res_low = engine.compute_asset_score(
        fracture_toughness_k_ic=30.0,
        a_crit_mm=10.0,
        mean_crack_10yr_mm=9.5,
        failure_prob_10yr=0.45,
        coastal_salinity_factor=2.0,
        thermal_cycling_range_c=55.0,
        min_dscr=0.95
    )

    assert res_low["pars_score"] < 50.0
    assert res_low["rating"] in ["B", "CCC/D"]


def test_environmental_stress_reduces_resilience():
    """Verify that shifting from inland to severe marine coastal environment lowers PARS score."""
    inland = quick_pars_evaluation(salinity=1.0, thermal_delta=15.0)
    coastal_severe = quick_pars_evaluation(salinity=2.0, thermal_delta=45.0)

    assert inland["pars_score"] > coastal_severe["pars_score"]
    assert inland["sub_scores"]["environmental_fragility"] > coastal_severe["sub_scores"]["environmental_fragility"]


def test_portfolio_evaluation():
    """Verify multi-asset portfolio rollup evaluation."""
    engine = PARSEngine()
    portfolio = [
        {"name": "Wind-A", "k_ic": 80.0, "a_crit_mm": 40.0, "mean_crack_mm": 5.0, "pf_10yr": 0.02, "salinity": 1.5, "thermal_cycle": 20.0, "min_dscr": 1.35},
        {"name": "Wind-B", "k_ic": 70.0, "a_crit_mm": 30.0, "mean_crack_mm": 8.0, "pf_10yr": 0.08, "salinity": 1.8, "thermal_cycle": 25.0, "min_dscr": 1.25},
    ]

    res = engine.evaluate_portfolio(portfolio)
    assert res["total_assets"] == 2
    assert 0.0 <= res["portfolio_average_pars"] <= 100.0
    assert res["portfolio_rating"] in ["AAA", "AA", "A", "BBB", "BB"]
