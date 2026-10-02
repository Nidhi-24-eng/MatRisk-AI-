"""
main.py - MatRisk AI FastAPI Microservice
Part of MatRisk AI (Phase 8, Day 15)

Provides a REST API for programmatic access to the MatRisk engine pipeline.
Enables integration with external risk management systems and trading desks.
"""

import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.degradation.paris_law import MaterialMechanics
from src.degradation.sde_engine import StochasticDegradationEngine
from src.financial.cfads_dscr import ProjectFinanceModel
from src.insurance.evt_pricing import CatastropheLossEVT
from src.insurance.esg_resilience import PARSEngine

app = FastAPI(
    title="MatRisk AI API",
    description="Physical-to-Financial Risk Engine for Structural Assets",
    version="1.0.0"
)

# ─── Pydantic Request Models ───

class RiskEvaluationRequest(BaseModel):
    # Material Properties
    bulk_modulus_gpa: float = Field(default=160.0, description="Bulk modulus K (GPa)")
    shear_modulus_gpa: float = Field(default=77.0, description="Shear modulus G (GPa)")
    fracture_toughness_k_ic: float = Field(default=55.0, description="Fracture toughness K_Ic (MPa√m)")
    
    # Stress & Environment
    delta_sigma_mpa: float = Field(default=180.0, description="Cyclic stress range Δσ (MPa)")
    cycles_per_year: float = Field(default=1_000_000.0, description="Operational cycles per year")
    environmental_multiplier: float = Field(default=1.0, description="Corrosion/Thermal acceleration factor")
    
    # Financial Parameters
    asset_capex_usd: float = Field(default=100_000_000.0, description="Total capital expenditure ($)")
    debt_ratio: float = Field(default=0.70, description="Senior debt percentage (0.0 to 1.0)")
    interest_rate: float = Field(default=0.065, description="Annual interest rate (e.g. 0.065)")
    base_ebitda_usd: float = Field(default=18_000_000.0, description="Annual EBITDA ($)")
    target_dscr: float = Field(default=1.30, description="Target Debt Service Coverage Ratio")

# ─── API Endpoints ───

@app.get("/")
def root():
    return {
        "title": "MatRisk AI API",
        "description": "Cross-domain Physical-to-Financial Asset Resilience & Risk Engine",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
        "evaluate_risk_endpoint": "/evaluate_risk"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "engine": "MatRisk AI"}

@app.post("/evaluate_risk")
def evaluate_risk(req: RiskEvaluationRequest):
    try:
        # 1. Physics Mechanics
        mechanics = MaterialMechanics(
            bulk_modulus=req.bulk_modulus_gpa,
            shear_modulus=req.shear_modulus_gpa,
            fracture_toughness=req.fracture_toughness_k_ic
        )
        
        # 2. Stochastic Degradation (SDE)
        sde = StochasticDegradationEngine(
            mechanics=mechanics,
            initial_crack_m=0.001,
            cycles_per_year=req.cycles_per_year
        )
        sde_result = sde.run_monte_carlo(
            delta_sigma_mpa=req.delta_sigma_mpa,
            sigma_max_mpa=req.delta_sigma_mpa * 1.8,
            horizon_years=10.0,
            num_trajectories=2000,  # Fast inference mode for API
            environmental_salinity_factor=req.environmental_multiplier,
            seed=42
        )
        
        # 3. Project Finance (CFADS & DSCR)
        pf = ProjectFinanceModel(
            asset_capex_usd=req.asset_capex_usd,
            debt_ratio=req.debt_ratio,
            interest_rate=req.interest_rate,
            base_annual_revenue=req.base_ebitda_usd + 7_000_000.0,
            base_annual_opex=7_000_000.0,
            base_annual_capex=1_500_000.0,
            dscr_target=req.target_dscr
        )
        # Downsample P_f for annual waterfall
        time_grid = sde_result["time_grid"]
        pf_curve = sde_result["failure_probability"]
        annual_pf = [float(pf_curve[min(i * 52, len(pf_curve)-1)]) for i in range(1, 11)]
        
        sculpt_df = pf.calculate_sculpted_waterfall(annual_pf)
        min_dscr = float(sculpt_df["dscr"].min())
        
        # 4. PARS ESG Score
        pars = PARSEngine()
        pars_score = pars.compute_asset_score(
            fracture_toughness_k_ic=req.fracture_toughness_k_ic,
            a_crit_mm=sde_result["a_crit_mm"],
            mean_crack_10yr_mm=float(sde_result["percentiles"]["p50"][-1]),
            failure_prob_10yr=sde_result["final_failure_prob_10yr"],
            coastal_salinity_factor=req.environmental_multiplier,
            thermal_cycling_range_c=25.0,
            min_dscr=min_dscr
        )
        
        return {
            "physics": {
                "paris_constant_c": mechanics.paris_c,
                "critical_crack_mm": sde_result["a_crit_mm"],
                "mttf_years": sde_result["mttf_years"],
                "failure_probability_10yr": sde_result["final_failure_prob_10yr"]
            },
            "finance": {
                "min_dscr_sculpted": min_dscr,
                "covenant_breached": min_dscr < 1.05,
                "dividend_lockup": min_dscr < 1.20
            },
            "esg_resilience": {
                "pars_score": pars_score["pars_score"],
                "rating": pars_score["rating"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
