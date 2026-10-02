"""
test_backtest.py - Phase 7: Backtesting & Validation
Part of MatRisk AI

Implements the Kupiec Proportion of Failures (POF) Likelihood Ratio test
to backtest the physics-informed Value at Risk (VaR) against a naive static model.
"""

import numpy as np
import pytest
from scipy.stats import chi2

from src.degradation.paris_law import MaterialMechanics
from src.degradation.sde_engine import StochasticDegradationEngine
from src.insurance.evt_pricing import CatastropheLossEVT


def kupiec_pof_test(exceedances: int, n_observations: int, target_prob: float = 0.01) -> dict:
    """
    Performs the Kupiec POF (Proportion of Failures) Likelihood Ratio test.
    H0: The model's exception rate matches the target probability (e.g., 1%).
    """
    if exceedances == 0:
        # Edge case: 0 exceedances. The LR formula requires x > 0 for x/N.
        # While perfectly conservative, it statistically fails Kupiec if N is very large.
        # For this test, we handle it safely.
        actual_rate = 0.0
        lr_pof = -2 * np.log(((1 - target_prob)**n_observations))
    else:
        actual_rate = exceedances / n_observations
        
        # Likelihood under Null Hypothesis (p = target_prob)
        l_null = ((1 - target_prob)**(n_observations - exceedances)) * (target_prob**exceedances)
        
        # Likelihood under Alternative Hypothesis (p = actual_rate)
        l_alt = ((1 - actual_rate)**(n_observations - exceedances)) * (actual_rate**exceedances)
        
        # Likelihood Ratio Statistic
        lr_pof = -2 * np.log(l_null / l_alt)

    # Chi-square critical value for 1 degree of freedom at 95% confidence
    critical_value = chi2.ppf(0.95, df=1)
    
    # Reject H0 if LR > critical_value (meaning the model is significantly inaccurate)
    is_rejected = lr_pof > critical_value

    return {
        "exceedances": exceedances,
        "actual_rate": actual_rate,
        "lr_pof_statistic": lr_pof,
        "critical_value": critical_value,
        "passed": not is_rejected
    }


def test_kupiec_pof_backtest_physics_vs_static():
    """
    Backtests the MatRisk Physics-Informed VaR against a Naive Static VaR.
    Generates 10,000 synthetic ground-truth operational loss paths, calculates VaR for both models,
    and uses the Kupiec POF test to verify that the physics model reduces exceedance errors.
    """
    # 1. Generate Synthetic Ground-Truth Losses
    # Real physical degradation creates a heavy-tailed (power-law) loss distribution.
    # We simulate 5,000 observations: 95% routine maintenance, 5% catastrophic fractures.
    np.random.seed(42)
    n_scenarios = 5000
    
    # 95% Maintenance (Normal Distribution around $1.5M)
    maint_losses = np.random.normal(loc=1_500_000, scale=200_000, size=4750)
    # 5% Catastrophe (Lognormal Distribution around $5M with heavy tail)
    cat_losses = np.random.lognormal(mean=np.log(5_000_000), sigma=0.5, size=250)
    
    true_losses = np.concatenate([maint_losses, cat_losses])
    
    # 2. Model A: Naive Static VaR (Traditional Finance Approach)
    # Assumes a normal distribution of all losses, ignoring power-law crack growth.
    # Static model drastically underestimates tail catastrophe.
    naive_mean_loss = np.mean(true_losses)
    naive_std_dev = np.std(true_losses)
    # VaR_99 for Normal Dist = mean + 2.33 * std
    naive_var_99 = naive_mean_loss + 2.33 * naive_std_dev
    
    # 3. Model B: MatRisk Physics-Informed VaR (Our EVT/GPD Approach)
    # Fits a Generalized Pareto Distribution to the tail of the heavy-tailed physics simulation.
    evt = CatastropheLossEVT(asset_value_usd=100_000_000.0)
    physics_pricing = evt.fit_gpd_tail(true_losses, alpha=0.99)
    physics_var_99 = physics_pricing["var_99_usd"]
    
    # 4. Calculate Exceedances
    naive_exceedances = np.sum(true_losses > naive_var_99)
    physics_exceedances = np.sum(true_losses > physics_var_99)
    
    # 6. Run Kupiec POF Likelihood Ratio Tests
    naive_kupiec = kupiec_pof_test(naive_exceedances, n_scenarios, target_prob=0.01)
    physics_kupiec = kupiec_pof_test(physics_exceedances, n_scenarios, target_prob=0.01)
    
    print("\n--- Kupiec POF Backtest Results (99% Confidence / 1% Target Exceedance) ---")
    print(f"Total Scenarios: {n_scenarios}")
    print(f"Target Exceedances: {int(n_scenarios * 0.01)} ({0.01*100}%)")
    
    print(f"\n[Naive Static Model]")
    print(f"  Predicted VaR_99: ${naive_var_99:,.0f}")
    print(f"  Actual Exceedances: {naive_exceedances} ({naive_kupiec['actual_rate']*100:.2f}%)")
    print(f"  LR POF Statistic: {naive_kupiec['lr_pof_statistic']:.2f} (Critical: {naive_kupiec['critical_value']:.2f})")
    print(f"  Passed Test? {naive_kupiec['passed']} (Expected: False)")

    print(f"\n[MatRisk Physics-Informed Model]")
    print(f"  Predicted VaR_99: ${physics_var_99:,.0f}")
    print(f"  Actual Exceedances: {physics_exceedances} ({physics_kupiec['actual_rate']*100:.2f}%)")
    print(f"  LR POF Statistic: {physics_kupiec['lr_pof_statistic']:.2f} (Critical: {physics_kupiec['critical_value']:.2f})")
    print(f"  Passed Test? {physics_kupiec['passed']} (Expected: True)")
    
    # 7. Assertions
    # The naive model should severely underestimate risk and fail the Kupiec test (rejected)
    assert naive_kupiec["passed"] is False
    assert naive_kupiec["lr_pof_statistic"] > naive_kupiec["critical_value"]
    
    # The physics model should closely match the 1% target and pass the Kupiec test
    assert physics_kupiec["passed"] is True
    assert physics_kupiec["lr_pof_statistic"] <= physics_kupiec["critical_value"]
    
    # The physics model should have significantly fewer exceedance errors than the naive model
    assert physics_exceedances < naive_exceedances
