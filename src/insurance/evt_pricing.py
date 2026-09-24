"""
evt_pricing.py - Extreme Value Theory (EVT) & Catastrophe Insurance Pricing
Part of MatRisk AI (Phase 5, Day 9)

Theoretical Foundation:
1. Peak-over-Threshold (PoT) Approach:
   By the Pickands-Balkema-de Haan Theorem, for a sufficiently high threshold u,
   excess losses Y = L - u | L > u asymptotically follow a Generalized Pareto Distribution (GPD):
   F_u(y) = 1 - (1 + xi * y / beta)^(-1 / xi)
   where:
   - xi: Shape parameter (tail heavy-tailedness / Pareto index)
   - beta: Scale parameter (severity)

2. Tail Risk Metrics:
   - Value at Risk (VaR_alpha):
     VaR_alpha(L) = u + (beta / xi) * [ ((N / N_u) * (1 - alpha))^(-xi) - 1 ]
   - Expected Shortfall (ES_alpha / Conditional VaR):
     ES_alpha(L) = (VaR_alpha + beta - xi * u) / (1 - xi)    for xi < 1

3. Actuarial Pricing Structure:
   - Pure Premium = E[Loss] (expected baseline annual losses)
   - Solvency II / Capital Reserve Buffer = lambda_cap * (ES_0.99 - Pure Premium)
   - Expense & Profit Loading = theta_load * Pure Premium
   - Gross Annual Actuarial Premium = Pure Premium + Capital Reserve Buffer + Expense Loading
"""

from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np
import scipy.stats as stats


class CatastropheLossEVT:
    """Extreme Value Theory tail loss engine for structural catastrophe insurance."""

    def __init__(
        self,
        asset_value_usd: float = 100_000_000.0,
        threshold_quantile: float = 0.90,
        solvency_capital_factor: float = 0.20,  # 20% cost of capital on 99% Expected Shortfall
        expense_loading_factor: float = 0.15    # 15% administrative & underwriting margin
    ):
        self.asset_value = asset_value_usd
        self.threshold_quantile = threshold_quantile
        self.solvency_capital_factor = solvency_capital_factor
        self.expense_loading_factor = expense_loading_factor

    def generate_structural_loss_distribution(
        self,
        trajectories: np.ndarray,      # [num_sims, num_steps] in meters
        a_crit: float,                 # Critical crack depth in meters
        base_repair_cost: float = 250_000.0,
        catastrophic_fracture_cost: float = 35_000_000.0, # Complete structural collapse/replacement
        business_interruption_daily_loss: float = 85_000.0
    ) -> np.ndarray:
        """
        Converts simulated Monte Carlo crack trajectories into annual structural monetary losses.
        Loss = Unplanned Repair Cost + Business Interruption + Catastrophic Fracture Penalty.
        """
        num_sims, num_steps = trajectories.shape
        losses = np.zeros(num_sims, dtype=np.float64)

        final_crack = trajectories[:, -1]
        max_crack = np.max(trajectories, axis=1)

        # 1. Catastrophic brittle fracture paths (crack >= a_crit)
        fractured = max_crack >= a_crit
        # Downtime for catastrophic rebuild: 120-270 days
        rebuild_downtime_days = np.random.uniform(120, 270, size=num_sims)
        fracture_loss = (
            catastrophic_fracture_cost +
            rebuild_downtime_days * business_interruption_daily_loss
        )
        losses = np.where(fractured, fracture_loss, 0.0)

        # 2. Sub-critical damage paths (crack growing, requires intervention)
        sub_crit = ~fractured
        # Repair severity scales quadratically with crack depth ratio (a / a_crit)
        crack_ratio = np.clip(final_crack / a_crit, 0.0, 1.0)
        unplanned_repairs = base_repair_cost * (1.0 + 15.0 * (crack_ratio ** 2))

        # Maintenance downtime: 5 to 30 days depending on severity
        downtime_days = 5.0 + 25.0 * crack_ratio
        interruption_loss = downtime_days * business_interruption_daily_loss

        # Random micro-damage shock (Poisson arrival of secondary inspection findings)
        noise = np.random.exponential(scale=50_000.0, size=num_sims)

        sub_loss = unplanned_repairs + interruption_loss + noise
        losses = np.where(sub_crit, sub_loss, losses)

        return losses

    def fit_gpd_tail(
        self,
        losses: np.ndarray,
        alpha: float = 0.99
    ) -> Dict[str, Any]:
        """
        Fits Generalized Pareto Distribution (GPD) to losses exceeding threshold u.
        Computes VaR_0.99 and ES_0.99 using EVT.
        """
        n_total = len(losses)
        # Select threshold u at high quantile (e.g. 90th percentile)
        u = float(np.quantile(losses, self.threshold_quantile))
        exceedances = losses[losses > u] - u
        n_u = len(exceedances)

        if n_u < 10:
            # Fallback empirical if too few exceedances
            var_alpha = float(np.quantile(losses, alpha))
            tail_losses = losses[losses >= var_alpha]
            es_alpha = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_alpha
            xi, beta = 0.25, float(np.std(losses))
        else:
            # Fit GPD using SciPy genpareto: pdf(y) = (1 + c * y / scale)^(-1 - 1/c) / scale
            # In scipy notation: c = xi, scale = beta, floc=0
            fit_res = stats.genpareto.fit(exceedances, floc=0)
            xi = float(fit_res[0])
            beta = float(fit_res[2])

            # Clamp xi < 0.95 to maintain finite Expected Shortfall
            xi = float(np.clip(xi, -0.2, 0.95))
            beta = max(1000.0, beta)

            # EVT Tail Quantile: VaR_alpha
            prob_tail = (n_total / n_u) * (1.0 - alpha)
            if prob_tail > 0:
                if abs(xi) < 1e-4:
                    var_alpha = u - beta * math.log(prob_tail)
                else:
                    var_alpha = u + (beta / xi) * ((prob_tail ** (-xi)) - 1.0)
            else:
                var_alpha = float(np.max(losses))

            # EVT Expected Shortfall: ES_alpha
            if xi < 1.0:
                es_alpha = (var_alpha + beta - xi * u) / (1.0 - xi)
            else:
                es_alpha = var_alpha * 1.25

        pure_premium = float(np.mean(losses))
        empirical_var = float(np.quantile(losses, alpha))

        # Fundamental Actuarial Consistency:
        # VaR_alpha must not fall below empirical quantile or expected loss (pure premium)
        var_alpha = max(var_alpha, empirical_var, pure_premium)

        # Expected Shortfall must be >= VaR and >= empirical tail average
        tail_losses = losses[losses >= var_alpha]
        empirical_es = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_alpha
        es_alpha = max(es_alpha, var_alpha, empirical_es)

        # Actuarial pricing
        solvency_buffer = self.solvency_capital_factor * max(0.0, es_alpha - pure_premium)
        expense_load = self.expense_loading_factor * pure_premium
        gross_premium = pure_premium + solvency_buffer + expense_load

        # Loss rate as % of total asset value
        rate_on_line_pct = (gross_premium / self.asset_value) * 100.0

        return {
            "num_loss_samples": n_total,
            "threshold_u": round(u, 2),
            "num_exceedances": n_u,
            "gpd_xi_shape": round(xi, 4),
            "gpd_beta_scale": round(beta, 2),
            "pure_premium_usd": round(pure_premium, 2),
            "var_99_usd": round(var_alpha, 2),
            "es_99_usd": round(es_alpha, 2),
            "solvency_capital_buffer_usd": round(solvency_buffer, 2),
            "expense_loading_usd": round(expense_load, 2),
            "gross_actuarial_premium_usd": round(gross_premium, 2),
            "rate_on_line_pct": round(rate_on_line_pct, 3)
        }


def price_catastrophe_risk(
    trajectories_mm: np.ndarray,
    a_crit_mm: float,
    asset_value_usd: float = 100_000_000.0
) -> Dict[str, Any]:
    """Top-level convenience interface for EVT insurance pricing."""
    # Convert mm to meters
    traj_m = trajectories_mm / 1000.0
    a_crit_m = a_crit_mm / 1000.0

    evt_engine = CatastropheLossEVT(asset_value_usd=asset_value_usd)
    losses = evt_engine.generate_structural_loss_distribution(traj_m, a_crit_m)
    pricing = evt_engine.fit_gpd_tail(losses, alpha=0.99)
    return pricing


if __name__ == "__main__":
    print("[*] Testing EVT Generalized Pareto Distribution Insurance Pricing...")

    # Mock trajectories: 10,000 simulations over 520 time steps
    np.random.seed(42)
    n_sims = 10000
    n_steps = 520
    a_crit = 0.035  # 35 mm critical crack

    # Create synthetic crack trajectories with fat tail
    mock_traj = np.zeros((n_sims, n_steps))
    # 97% survive with small subcritical cracks (1mm -> 12mm)
    mock_traj[:9700, -1] = np.random.uniform(0.001, 0.015, size=9700)
    # 3% experience catastrophic failure
    mock_traj[9700:, -1] = np.random.uniform(0.035, 0.050, size=300)

    engine = CatastropheLossEVT(asset_value_usd=100_000_000.0)
    loss_dist = engine.generate_structural_loss_distribution(mock_traj, a_crit)
    pricing_res = engine.fit_gpd_tail(loss_dist)

    print(f"[+] EVT Fitting Results:")
    print(f"    - GPD Shape parameter (xi): {pricing_res['gpd_xi_shape']} (heavy-tailed)")
    print(f"    - Pure Expected Loss:       ${pricing_res['pure_premium_usd']:,.2f}")
    print(f"    - Value at Risk (VaR 99%):  ${pricing_res['var_99_usd']:,.2f}")
    print(f"    - Expected Shortfall (ES):  ${pricing_res['es_99_usd']:,.2f}")
    print(f"    - Solvency II Buffer:       ${pricing_res['solvency_capital_buffer_usd']:,.2f}")
    print(f"    - Gross Actuarial Premium:  ${pricing_res['gross_actuarial_premium_usd']:,.2f}")
    print(f"    - Rate on Line:             {pricing_res['rate_on_line_pct']}% of asset value")
