"""
commodity_desk.py - Physical Hazard-Adjusted Commodity Forward Curves
Part of MatRisk AI (Phase 4, Day 8)

Theoretical Foundation:
1. Standard Cost-of-Carry Forward Pricing:
   F(0, T) = S_0 * exp( (r + u - y) * T )
   where:
   - S_0: Spot commodity price
   - r: Risk-free interest rate
   - u: Marginal storage/warehousing cost
   - y: Convenience yield

2. Physical Disruption Hazard Spread:
   When extraction, transportation, or refining infrastructure experiences physical fatigue
   degradation with instantaneous hazard rate lambda(t) and cumulative failure probability P_f(t),
   the probability of an unscheduled physical supply bottleneck increases:
   Cumulative Hazard: Lambda(T) = int_0^T lambda(t) dt = -ln( 1 - P_f(T) )

   The hazard-adjusted forward price prices in physical supply scarcity:
   F_adj(0, T) = F(0, T) * exp( gamma_outage * Lambda(T) )
               = F(0, T) * ( 1 - P_f(T) )^( -gamma_outage )
   where gamma_outage is the supply shock elasticity factor.
"""

from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


# Benchmark commodity profiles
DEFAULT_COMMODITIES = {
    "Copper": {
        "ticker": "HG=F",
        "name": "COMEX Copper Futures",
        "default_spot": 4.25,        # USD / lb
        "unit": "USD/lb",
        "storage_cost": 0.015,       # 1.5% storage
        "convenience_yield": 0.025,  # 2.5% convenience yield
        "shock_elasticity": 0.35     # Highly sensitive to smelter/mine outages
    },
    "Crude_Oil": {
        "ticker": "CL=F",
        "name": "WTI Crude Oil Futures",
        "default_spot": 78.50,       # USD / barrel
        "unit": "USD/bbl",
        "storage_cost": 0.020,
        "convenience_yield": 0.030,
        "shock_elasticity": 0.25
    },
    "Structural_Steel": {
        "ticker": "SLX",
        "name": "Steel Benchmark Index",
        "default_spot": 850.0,       # USD / metric ton
        "unit": "USD/ton",
        "storage_cost": 0.010,
        "convenience_yield": 0.015,
        "shock_elasticity": 0.30
    }
}


class CommodityDesk:
    """Calculates forward curves adjusted for physical infrastructure disruption hazard."""

    def __init__(self, risk_free_rate: float = 0.045):
        self.r = risk_free_rate

    def fetch_spot_price(self, commodity_key: str = "Copper") -> Tuple[float, float, str]:
        """
        Fetches live spot price and annualized volatility via yfinance.
        Falls back to calibrated historical benchmarks if offline.
        """
        config = DEFAULT_COMMODITIES.get(commodity_key, DEFAULT_COMMODITIES["Copper"])
        ticker_sym = config["ticker"]
        default_spot = config["default_spot"]
        unit = config["unit"]

        if HAS_YFINANCE:
            try:
                tk = yf.Ticker(ticker_sym)
                hist = tk.history(period="6mo")
                if len(hist) > 10:
                    latest_close = float(hist["Close"].iloc[-1])
                    returns = np.diff(np.log(hist["Close"].values))
                    ann_vol = float(np.std(returns) * np.sqrt(252))
                    return round(latest_close, 2), round(ann_vol, 3), unit
            except Exception:
                pass

        # Offline fallback
        return default_spot, 0.22, unit

    def build_hazard_adjusted_forward_curve(
        self,
        commodity_key: str,
        tenor_years: float = 10.0,
        annual_failure_probs: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Builds both standard cost-of-carry and physical-hazard-adjusted forward curves.
        """
        config = DEFAULT_COMMODITIES.get(commodity_key, DEFAULT_COMMODITIES["Copper"])
        spot, vol, unit = self.fetch_spot_price(commodity_key)

        u = config["storage_cost"]
        y = config["convenience_yield"]
        gamma = config["shock_elasticity"]

        years = np.arange(1, int(tenor_years) + 1)
        n = len(years)

        if annual_failure_probs is None or len(annual_failure_probs) < n:
            pf = np.linspace(0.01, 0.20, n)
        else:
            pf = np.array(annual_failure_probs[:n])

        std_forward = []
        adj_forward = []
        hazard_spread = []

        for t_idx, t in enumerate(years):
            # Standard carry forward: F = S0 * exp((r + u - y) * T)
            carry_rate = self.r + u - y
            f_std = spot * math.exp(carry_rate * t)
            std_forward.append(round(f_std, 2))

            # Cumulative failure probability at time t
            p_fail = float(pf[t_idx])
            p_fail_clamped = min(0.85, max(0.0, p_fail))

            # Hazard multiplier: (1 - P_f)^(-gamma)
            outage_mult = (1.0 - p_fail_clamped) ** (-gamma)
            f_adj = f_std * outage_mult
            adj_forward.append(round(f_adj, 2))

            spread = f_adj - f_std
            hazard_spread.append(round(spread, 2))

        return {
            "commodity": commodity_key,
            "unit": unit,
            "spot_price": spot,
            "annualized_volatility": vol,
            "years": years.tolist(),
            "standard_forward_curve": std_forward,
            "hazard_adjusted_forward_curve": adj_forward,
            "hazard_spread": hazard_spread,
            "max_disruption_premium_pct": round(((adj_forward[-1] / std_forward[-1]) - 1.0) * 100, 2)
        }


def analyze_commodity_exposure(
    commodity_key: str = "Copper",
    failure_probs: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Helper diagnostic function."""
    desk = CommodityDesk()
    return desk.build_hazard_adjusted_forward_curve(commodity_key, annual_failure_probs=failure_probs)


if __name__ == "__main__":
    print("[*] Testing Commodity Trading Desk Physical Hazard Forward Curves...")

    sample_pf = [0.005, 0.012, 0.025, 0.045, 0.075, 0.110, 0.155, 0.195, 0.230, 0.260]
    cu_curve = analyze_commodity_exposure("Copper", sample_pf)

    print(f"[+] Commodity: {cu_curve['commodity']} ({cu_curve['unit']})")
    print(f"    - Current Spot: ${cu_curve['spot_price']}")
    print(f"    - Standard 10Y Forward: ${cu_curve['standard_forward_curve'][-1]}")
    print(f"    - Hazard-Adjusted 10Y Forward: ${cu_curve['hazard_adjusted_forward_curve'][-1]}")
    print(f"    - Max Supply Outage Spread: +${cu_curve['hazard_spread'][-1]} (+{cu_curve['max_disruption_premium_pct']}%)")
