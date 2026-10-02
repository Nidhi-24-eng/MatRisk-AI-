"""
esg_resilience.py - Physical Asset Resilience Score (PARS) & ESG Risk Engine
Part of MatRisk AI (Phase 5, Day 10)

Theoretical Framework:
PARS is a quantitative composite index (0 to 100 scale) providing a credit-style rating
for physical infrastructure assets subject to climate, mechanical, and financial stress.

Component Pillars:
1. Mechanical Margin (Weight: 30%): Distance to critical crack depth a_crit and fracture toughness K_Ic.
2. Degradation Velocity (Weight: 25%): Subcritical crack acceleration rate and 10-year failure probability P_f(10).
3. Environmental Fragility (Weight: 25%): Vulnerability to marine salinity, thermal cycling, and humidity.
4. Financial Covenant Buffer (Weight: 20%): Minimum DSCR cushion above 1.05x default threshold.

Rating Bands:
- 90 - 100 : AAA (Benchmark Resilience)
- 80 - 89  : AA  (High Resilience)
- 70 - 79  : A   (Robust Investment Grade)
- 60 - 69  : BBB (Moderate Resilience / Monitoring Required)
- 50 - 59  : BB  (Elevated Degradation / Covenant Pressure)
- 40 - 49  : B   (Vulnerable / High Insurance Loading)
- < 40     : CCC / D (Critical Structural Distress)
"""

from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np


class PARSEngine:
    """Computes Physical Asset Resilience Scores (PARS) for single assets and portfolios."""

    def __init__(
        self,
        weight_mech: float = 0.30,
        weight_deg: float = 0.25,
        weight_env: float = 0.25,
        weight_fin: float = 0.20
    ):
        self.w_mech = weight_mech
        self.w_deg = weight_deg
        self.w_env = weight_env
        self.w_fin = weight_fin

    def compute_asset_score(
        self,
        fracture_toughness_k_ic: float,     # MPa*m^0.5
        a_crit_mm: float,                    # Critical crack depth in mm
        mean_crack_10yr_mm: float,           # Mean crack depth at year 10 in mm
        failure_prob_10yr: float,            # Cumulative P_f(10)
        coastal_salinity_factor: float,      # 1.0 = Inland, 1.5 = Moderate Coastal, 2.0 = Severe Marine Splash
        thermal_cycling_range_c: float,      # Daily temperature oscillation delta T (deg C)
        min_dscr: float                      # Minimum project DSCR across loan tenor
    ) -> Dict[str, Any]:
        """Calculates granular component scores and final composite PARS rating."""

        # 1. Mechanical Margin Score (0 to 100)
        crack_margin_ratio = max(0.0, min(1.0, (a_crit_mm - mean_crack_10yr_mm) / max(1e-3, a_crit_mm)))
        toughness_ratio = min(1.0, max(0.0, fracture_toughness_k_ic / 100.0))
        s_mech = 100.0 * (0.5 * crack_margin_ratio + 0.5 * toughness_ratio)

        # 2. Degradation Velocity Score (0 to 100)
        # Exponential penalty as failure probability increases
        s_deg = 100.0 * math.exp(-3.5 * min(1.0, max(0.0, failure_prob_10yr)))

        # 3. Environmental Fragility Score (0 to 100)
        # Normalized environmental penalties
        salinity_penalty = min(1.0, max(0.0, (coastal_salinity_factor - 1.0) / 1.0))  # 0 to 1
        thermal_penalty = min(1.0, max(0.0, thermal_cycling_range_c / 60.0))           # 0 to 1
        s_env = 100.0 * max(0.0, 1.0 - (0.55 * salinity_penalty + 0.45 * thermal_penalty))

        # 4. Financial Covenant Buffer Score (0 to 100)
        # Target: DSCR >= 1.40 gets 100, DSCR <= 0.90 gets 0
        s_fin = 100.0 * min(1.0, max(0.0, (min_dscr - 0.90) / (1.40 - 0.90)))

        # Composite PARS
        pars = (
            self.w_mech * s_mech +
            self.w_deg * s_deg +
            self.w_env * s_env +
            self.w_fin * s_fin
        )
        pars = round(float(np.clip(pars, 0.0, 100.0)), 1)

        # Rating assignment
        if pars >= 90.0:
            rating = "AAA"
            outlook = "Benchmark Resilience"
        elif pars >= 80.0:
            rating = "AA"
            outlook = "High Resilience"
        elif pars >= 70.0:
            rating = "A"
            outlook = "Robust Investment Grade"
        elif pars >= 60.0:
            rating = "BBB"
            outlook = "Moderate Resilience / Monitoring"
        elif pars >= 50.0:
            rating = "BB"
            outlook = "Elevated Stress / Surcharges"
        elif pars >= 40.0:
            rating = "B"
            outlook = "Vulnerable / High Failure Risk"
        else:
            rating = "CCC/D"
            outlook = "Critical Structural Distress"

        return {
            "pars_score": pars,
            "rating": rating,
            "outlook": outlook,
            "sub_scores": {
                "mechanical_margin": round(s_mech, 1),
                "degradation_velocity": round(s_deg, 1),
                "environmental_fragility": round(s_env, 1),
                "financial_buffer": round(s_fin, 1)
            },
            "weights": {
                "mechanical": self.w_mech,
                "degradation": self.w_deg,
                "environmental": self.w_env,
                "financial": self.w_fin
            }
        }

    def evaluate_portfolio(self, asset_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates a multi-asset infrastructure portfolio."""
        results = []
        scores = []

        for asset in asset_list:
            res = self.compute_asset_score(
                fracture_toughness_k_ic=asset.get("k_ic", 75.0),
                a_crit_mm=asset.get("a_crit_mm", 35.0),
                mean_crack_10yr_mm=asset.get("mean_crack_mm", 5.0),
                failure_prob_10yr=asset.get("pf_10yr", 0.05),
                coastal_salinity_factor=asset.get("salinity", 1.0),
                thermal_cycling_range_c=asset.get("thermal_cycle", 20.0),
                min_dscr=asset.get("min_dscr", 1.30)
            )
            res["asset_name"] = asset.get("name", "Asset")
            results.append(res)
            scores.append(res["pars_score"])

        avg_score = round(float(np.mean(scores)), 1) if scores else 0.0

        if avg_score >= 80.0:
            port_rating = "AA"
        elif avg_score >= 70.0:
            port_rating = "A"
        elif avg_score >= 60.0:
            port_rating = "BBB"
        else:
            port_rating = "BB"

        return {
            "portfolio_average_pars": avg_score,
            "portfolio_rating": port_rating,
            "total_assets": len(asset_list),
            "assets": results
        }


def quick_pars_evaluation(
    k_ic: float = 75.0,
    a_crit_mm: float = 35.0,
    mean_crack_mm: float = 6.0,
    failure_prob: float = 0.04,
    salinity: float = 1.2,
    thermal_delta: float = 25.0,
    min_dscr: float = 1.30
) -> Dict[str, Any]:
    """Helper convenience function."""
    engine = PARSEngine()
    return engine.compute_asset_score(
        fracture_toughness_k_ic=k_ic,
        a_crit_mm=a_crit_mm,
        mean_crack_10yr_mm=mean_crack_mm,
        failure_prob_10yr=failure_prob,
        coastal_salinity_factor=salinity,
        thermal_cycling_range_c=thermal_delta,
        min_dscr=min_dscr
    )


if __name__ == "__main__":
    print("[*] Testing Physical Asset Resilience Score (PARS) Engine...")

    # Case 1: Offshore Wind Turbine (Marine Splash Zone, Steel Jacket)
    offshore_wind = quick_pars_evaluation(
        k_ic=75.0,
        a_crit_mm=32.0,
        mean_crack_mm=8.5,
        failure_prob=0.08,
        salinity=1.8,
        thermal_delta=30.0,
        min_dscr=1.32
    )

    print(f"[+] Offshore Wind Turbine Assessment:")
    print(f"    - PARS Score: {offshore_wind['pars_score']}/100 (Rating: {offshore_wind['rating']} - {offshore_wind['outlook']})")
    print(f"    - Sub-Scores: {offshore_wind['sub_scores']}")
