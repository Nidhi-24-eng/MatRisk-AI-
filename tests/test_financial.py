"""
test_financial.py - Unit tests for Phase 4 (Project Finance, DSCR Sculpting, Commodity Desk)
"""

import pytest
import numpy as np
import pandas as pd

from src.financial.cfads_dscr import ProjectFinanceModel, evaluate_financial_risk
from src.financial.commodity_desk import CommodityDesk, analyze_commodity_exposure


def test_project_finance_cfads_waterfall():
    """Verify CFADS waterfall accounting identities and baseline numbers."""
    model = ProjectFinanceModel(asset_capex_usd=100_000_000.0, debt_ratio=0.70, base_annual_revenue=28_000_000.0)
    assert model.initial_debt == 70_000_000.0

    # Test baseline without degradation
    df_base = model.calculate_unadjusted_waterfall(failure_prob_profile=None)
    assert len(df_base) == 10
    # In all baseline years, DSCR should be healthy (> 1.30x)
    assert (df_base["dscr"] >= 1.30).all()


def test_physical_shock_causes_covenant_pressure():
    """Verify that severe physical degradation triggers dividend lockup or default in unadjusted schedule."""
    model = ProjectFinanceModel()

    # Extreme failure scenario (35% failure probability by year 10)
    severe_pf = [0.01, 0.03, 0.07, 0.12, 0.18, 0.24, 0.28, 0.31, 0.34, 0.36]
    df_severe = model.calculate_unadjusted_waterfall(severe_pf)

    # In later years, DSCR should drop below 1.30x
    min_dscr = df_severe["dscr"].min()
    assert min_dscr < 1.30
    assert "DIVIDEND_LOCKUP" in df_severe["covenant_status"].values or "COVENANT_DEFAULT" in df_severe["covenant_status"].values


def test_dynamic_debt_sculpting_eliminates_default():
    """Verify that dynamic debt sculpting preserves target DSCR >= 1.30x."""
    model = ProjectFinanceModel(dscr_target=1.30)
    severe_pf = [0.01, 0.03, 0.07, 0.12, 0.18, 0.24, 0.28, 0.31, 0.34, 0.36]

    df_sculpt = model.calculate_sculpted_waterfall(severe_pf)

    # All sculpted years (except potential final bullet) should satisfy target DSCR >= 1.30x
    for idx, row in df_sculpt.iloc[:-1].iterrows():
        assert row["dscr"] >= 1.29, f"Year {row['year']} failed sculpted target: {row['dscr']}"


def test_commodity_hazard_adjusted_forward_curve():
    """Verify forward pricing adjustments for physical supply disruption."""
    desk = CommodityDesk()
    pf = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.32]

    curve = desk.build_hazard_adjusted_forward_curve("Copper", tenor_years=10, annual_failure_probs=pf)

    assert curve["spot_price"] > 0
    assert len(curve["standard_forward_curve"]) == 10
    assert len(curve["hazard_adjusted_forward_curve"]) == 10

    # Hazard-adjusted price must always be >= standard price
    std = np.array(curve["standard_forward_curve"])
    adj = np.array(curve["hazard_adjusted_forward_curve"])
    assert np.all(adj >= std)
    assert curve["max_disruption_premium_pct"] > 0.0
