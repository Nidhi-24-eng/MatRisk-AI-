"""
cfads_dscr.py - Cash Flow Available for Debt Service (CFADS) & DSCR Debt Sculpting Engine
Part of MatRisk AI (Phase 4, Day 7 & 8)

Theoretical Framework:
1. Cash Flow Waterfall:
   EBITDA_t = Revenue_t * (1 - Downtime(P_f(t))) - OpEx_t
   CapEx_maint(t) = CapEx_base + CapEx_unplanned_max * P_f(t)
   CFADS_t = EBITDA_t - CapEx_maint(t) - Delta NWC_t - Taxes_t

2. Debt Service Coverage Ratio (DSCR):
   DSCR_t = CFADS_t / (Principal_t + Interest_t)
   - DSCR >= 1.30x : Safe operating / target cushion
   - 1.05x <= DSCR < 1.20x : Dividend Lock-up (Cash sweep activated, no equity distributions)
   - DSCR < 1.05x : Technical / Covenant Default (lender acceleration)

3. Dynamic Debt Sculpting:
   Solves for periodic Principal_t such that DSCR_t == DSCR_target:
   Target Debt Service_t = CFADS_t / DSCR_target
   Principal_t = max(0, Target Debt Service_t - Interest_t)
   where Interest_t = Debt_Balance_{t-1} * interest_rate
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


class ProjectFinanceModel:
    """Models infrastructure asset cash flows, degradation shocks, and debt covenants."""

    def __init__(
        self,
        asset_capex_usd: float = 100_000_000.0,   # $100M total initial asset capital investment
        debt_ratio: float = 0.75,                 # 75% Debt / 25% Equity gearing ($75M debt)
        interest_rate: float = 0.065,             # 6.50% senior loan coupon rate
        tenor_years: int = 10,                    # 10-year loan amortization
        base_annual_revenue: float = 22_000_000.0,# $22M/yr baseline revenue
        base_annual_opex: float = 7_000_000.0,    # $7M/yr operational expenses
        base_annual_capex: float = 1_500_000.0,   # $1.5M/yr routine maintenance CapEx
        tax_rate: float = 0.22,                   # 22% corporate income tax
        dscr_target: float = 1.30,                # 1.30x target coverage cushion
        dscr_lockup: float = 1.20,                # 1.20x dividend lockup threshold
        dscr_default: float = 1.05                # 1.05x covenant default threshold
    ):
        self.asset_capex = asset_capex_usd
        self.initial_debt = asset_capex_usd * debt_ratio
        self.equity = asset_capex_usd * (1.0 - debt_ratio)
        self.interest_rate = interest_rate
        self.tenor_years = tenor_years

        self.base_revenue = base_annual_revenue
        self.base_opex = base_annual_opex
        self.base_capex = base_annual_capex
        self.tax_rate = tax_rate

        self.dscr_target = dscr_target
        self.dscr_lockup = dscr_lockup
        self.dscr_default = dscr_default

    def calculate_unadjusted_waterfall(
        self,
        failure_prob_profile: Optional[List[float]] = None,
        max_unplanned_capex_spike: float = 6_000_000.0,  # $6M catastrophic replacement spike
        max_downtime_fraction: float = 0.18              # Up to 18% operational downtime on severe failure
    ) -> pd.DataFrame:
        """
        Builds standard annuity/straight-line debt amortization and applies physical failure shocks.
        """
        years = list(range(1, self.tenor_years + 1))
        n = len(years)

        # Baseline linear principal amortization without dynamic sculpting
        annual_principal = self.initial_debt / n

        # If no physical failure curve passed, assume zero degradation
        if failure_prob_profile is None or len(failure_prob_profile) < n:
            pf = np.zeros(n)
        else:
            # Resample or take annual points
            pf = np.array(failure_prob_profile[:n])

        balance = self.initial_debt
        records = []

        for yr_idx, yr in enumerate(years):
            p_fail = float(pf[yr_idx]) if yr_idx < len(pf) else float(pf[-1])

            # Physical degradation impacts
            downtime = max_downtime_fraction * p_fail
            eff_revenue = self.base_revenue * (1.0 - downtime)
            ebitda = eff_revenue - self.base_opex

            unplanned_capex = max_unplanned_capex_spike * p_fail
            total_capex = self.base_capex + unplanned_capex

            # Taxes on EBIT (after depreciation allowance)
            depreciation = self.asset_capex / 20.0  # 20-year tax depreciation
            ebit = max(0.0, ebitda - depreciation)
            taxes = ebit * self.tax_rate

            # CFADS = EBITDA - CapEx - Taxes
            cfads = max(0.0, ebitda - total_capex - taxes)

            # Straight-line debt service
            interest = balance * self.interest_rate
            principal = min(balance, annual_principal)
            debt_service = principal + interest

            dscr = cfads / max(1.0, debt_service)

            # Covenant Status
            if dscr >= self.dscr_target:
                status = "COMPLIANT"
            elif dscr >= self.dscr_lockup:
                status = "COMPLIANT_NO_DISTRIBUTION"
            elif dscr >= self.dscr_default:
                status = "DIVIDEND_LOCKUP"
            else:
                status = "COVENANT_DEFAULT"

            records.append({
                "year": yr,
                "failure_probability": round(p_fail, 4),
                "downtime_pct": round(downtime * 100, 2),
                "revenue_usd": round(eff_revenue, 2),
                "ebitda_usd": round(ebitda, 2),
                "capex_maint_usd": round(total_capex, 2),
                "cfads_usd": round(cfads, 2),
                "beg_debt_balance_usd": round(balance, 2),
                "interest_usd": round(interest, 2),
                "principal_usd": round(principal, 2),
                "debt_service_usd": round(debt_service, 2),
                "end_debt_balance_usd": round(max(0.0, balance - principal), 2),
                "dscr": round(dscr, 3),
                "covenant_status": status
            })

            balance = max(0.0, balance - principal)

        return pd.DataFrame(records)

    def calculate_sculpted_waterfall(
        self,
        failure_prob_profile: Optional[List[float]] = None,
        max_unplanned_capex_spike: float = 6_000_000.0,
        max_downtime_fraction: float = 0.18
    ) -> pd.DataFrame:
        """
        Dynamically sculpts principal repayments to strictly enforce DSCR >= dscr_target.
        Target Debt Service_t = CFADS_t / DSCR_target
        Principal_t = max(0, Target Debt Service_t - Interest_t)
        """
        years = list(range(1, self.tenor_years + 1))
        n = len(years)

        if failure_prob_profile is None or len(failure_prob_profile) < n:
            pf = np.zeros(n)
        else:
            pf = np.array(failure_prob_profile[:n])

        balance = self.initial_debt
        records = []

        for yr_idx, yr in enumerate(years):
            p_fail = float(pf[yr_idx]) if yr_idx < len(pf) else float(pf[-1])

            # Financial impacts of failure
            downtime = max_downtime_fraction * p_fail
            eff_revenue = self.base_revenue * (1.0 - downtime)
            ebitda = eff_revenue - self.base_opex

            unplanned_capex = max_unplanned_capex_spike * p_fail
            total_capex = self.base_capex + unplanned_capex

            depreciation = self.asset_capex / 20.0
            ebit = max(0.0, ebitda - depreciation)
            taxes = ebit * self.tax_rate

            cfads = max(0.0, ebitda - total_capex - taxes)

            # SCULPTING LOGIC:
            # 1. Determine maximum affordable debt service that guarantees DSCR >= 1.30x
            target_ds = cfads / self.dscr_target

            # 2. Interest due this period
            interest = balance * self.interest_rate

            # 3. Principal sculpted
            if yr_idx == n - 1:
                # Final bullet balloon or full payoff
                principal = min(balance, max(0.0, balance))
            else:
                principal = min(balance, max(0.0, target_ds - interest))

            actual_ds = principal + interest
            actual_dscr = cfads / max(1.0, actual_ds)

            # Status check
            if actual_dscr >= self.dscr_target:
                status = "SCULPTED_OPTIMAL"
            elif actual_dscr >= self.dscr_default:
                status = "TIGHT_COVERAGE"
            else:
                status = "DEFICIT"

            records.append({
                "year": yr,
                "failure_probability": round(p_fail, 4),
                "revenue_usd": round(eff_revenue, 2),
                "cfads_usd": round(cfads, 2),
                "beg_debt_balance_usd": round(balance, 2),
                "interest_usd": round(interest, 2),
                "principal_usd": round(principal, 2),
                "debt_service_usd": round(actual_ds, 2),
                "end_debt_balance_usd": round(max(0.0, balance - principal), 2),
                "dscr": round(actual_dscr, 3),
                "covenant_status": status
            })

            balance = max(0.0, balance - principal)

        return pd.DataFrame(records)


def evaluate_financial_risk(
    annual_failure_probs: List[float],
    asset_capex_usd: float = 100_000_000.0,
    debt_ratio: float = 0.75,
    interest_rate: float = 0.065
) -> Dict[str, Any]:
    """Helper function to compare unadjusted vs sculpted financial metrics."""
    fin_model = ProjectFinanceModel(
        asset_capex_usd=asset_capex_usd,
        debt_ratio=debt_ratio,
        interest_rate=interest_rate
    )

    unadj_df = fin_model.calculate_unadjusted_waterfall(annual_failure_probs)
    sculpt_df = fin_model.calculate_sculpted_waterfall(annual_failure_probs)

    min_unadj_dscr = float(unadj_df["dscr"].min())
    min_sculpt_dscr = float(sculpt_df["dscr"].min())

    default_years = unadj_df[unadj_df["covenant_status"] == "COVENANT_DEFAULT"]["year"].tolist()
    lockup_years = unadj_df[unadj_df["covenant_status"] == "DIVIDEND_LOCKUP"]["year"].tolist()

    return {
        "initial_debt_usd": fin_model.initial_debt,
        "min_unadjusted_dscr": round(min_unadj_dscr, 3),
        "min_sculpted_dscr": round(min_sculpt_dscr, 3),
        "default_years_unadjusted": default_years,
        "lockup_years_unadjusted": lockup_years,
        "has_covenant_default": len(default_years) > 0,
        "unadjusted_schedule": unadj_df.to_dict(orient="records"),
        "sculpted_schedule": sculpt_df.to_dict(orient="records")
    }


if __name__ == "__main__":
    print("[*] Testing Project Finance CFADS & DSCR Debt Sculpting Engine...")

    # Simulated rising 10-year failure probability curve from Engine 2 (0% -> 22%)
    sample_pf = [0.001, 0.005, 0.015, 0.035, 0.065, 0.105, 0.145, 0.180, 0.210, 0.235]

    eval_res = evaluate_financial_risk(sample_pf)
    print(f"[+] Initial Debt: ${eval_res['initial_debt_usd']:,.2f}")
    print(f"[+] Minimum Unadjusted DSCR: {eval_res['min_unadjusted_dscr']}x")
    print(f"[+] Minimum Sculpted DSCR:   {eval_res['min_sculpted_dscr']}x")
    print(f"[!] Covenant Default Years (Unadjusted): {eval_res['default_years_unadjusted']}")
    print(f"[!] Dividend Lockup Years (Unadjusted):  {eval_res['lockup_years_unadjusted']}")
