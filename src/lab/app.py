"""
app.py - MatRisk Lab Interactive Dashboard
Part of MatRisk AI (Phase 6, Days 11–12)

Enterprise-grade real-time command center connecting all 5 MatRisk AI engines.
Dragging a single physical slider (e.g. marine salinity) triggers an instant visual cascade
across atomistic physics, stochastic mechanics, corporate finance, and actuarial insurance.

Launch: streamlit run src/lab/app.py
"""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st

# Engine imports
from src.degradation.paris_law import MaterialMechanics
from src.degradation.sde_engine import StochasticDegradationEngine
from src.financial.cfads_dscr import ProjectFinanceModel
from src.insurance.evt_pricing import CatastropheLossEVT
from src.insurance.esg_resilience import PARSEngine

# UI imports
from src.lab.shock_generator import ALLOY_PROFILES, BASELINE_PARAMS, get_shock_preset, compute_environmental_multiplier
from src.lab.plotly_charts import (
    render_monte_carlo_crack_chart,
    render_failure_probability_chart,
    render_cfads_waterfall_chart,
    render_dscr_covenant_chart,
    render_sculpted_vs_flat_principal,
    render_evt_tail_loss_chart,
    render_pars_radar_chart,
)

# ─────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MatRisk Lab — Atomistic-to-Financial Risk Command Center",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS injection
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    section[data-testid="stSidebar"] { background-color: #0D1117; border-right: 1px solid #1E2A3A; }
    .stMetric { background-color: #111922; border-radius: 8px; padding: 12px; border: 1px solid #1E2A3A; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .stTabs [data-baseweb="tab"] { color: #E0E0E0; font-weight: 600; }
    h1, h2, h3 { color: #E0E0E0 !important; }
    .alert-lockup { background-color: rgba(255,179,0,0.15); border-left: 4px solid #FFB300;
                    padding: 10px 15px; border-radius: 4px; color: #FFB300; font-weight: bold; margin: 8px 0; }
    .alert-default { background-color: rgba(255,23,68,0.15); border-left: 4px solid #FF1744;
                     padding: 10px 15px; border-radius: 4px; color: #FF1744; font-weight: bold; margin: 8px 0; }
    .alert-safe { background-color: rgba(0,230,118,0.10); border-left: 4px solid #00E676;
                  padding: 10px 15px; border-radius: 4px; color: #00E676; font-weight: bold; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Sidebar: Input Controls
# ─────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚛ MatRisk Lab Controls")

# Section A: Structural Material Selection
st.sidebar.markdown("### 🔬 A. Material Selection")
alloy_name = st.sidebar.selectbox("Structural Alloy", list(ALLOY_PROFILES.keys()), index=0)
initial_flaw_mm = st.sidebar.slider("Initial Flaw Size a₀ (mm)", 0.1, 5.0, 1.0, 0.1)

# Section B: Environmental & Mechanical Stress
st.sidebar.markdown("### 🌊 B. Stress & Environment")
delta_sigma = st.sidebar.slider("Cyclic Stress Range Δσ (MPa)", 50, 500, 180, 5)
cycles_per_year = st.sidebar.select_slider(
    "Operational Cycles / Year",
    options=[10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
    value=1_000_000,
    format_func=lambda x: f"{x:,.0f}"
)
salinity_pct = st.sidebar.slider("Marine Salinity Cl⁻ (%)", 0.0, 7.0, 1.5, 0.1)
temperature_c = st.sidebar.slider("Operating Temperature (°C)", -20, 600, 25, 5)

# Section C: Project Finance
st.sidebar.markdown("### 🏦 C. Project Finance")
capex_m = st.sidebar.number_input("Asset CapEx ($M)", value=100.0, min_value=10.0, max_value=1000.0, step=10.0)
debt_pct = st.sidebar.slider("Senior Debt Share (%)", 50, 85, 70, 1)
base_ebitda_m = st.sidebar.number_input("Base EBITDA ($M/yr)", value=18.0, min_value=5.0, max_value=100.0, step=1.0)
interest_rate = st.sidebar.slider("Interest Rate (%)", 3.0, 15.0, 6.5, 0.25)
target_dscr = st.sidebar.slider("Target DSCR Cushion (x)", 1.20, 1.50, 1.30, 0.05)
sculpting_enabled = st.sidebar.toggle("🔧 Apply Physics-Informed Debt Sculpting", value=False)

# Section D: Quick-Run Scenario Shock Presets
st.sidebar.markdown("### ⚡ D. Shock Presets")
col_s1, col_s2 = st.sidebar.columns(2)
shock_marine = col_s1.button("🔴 Marine Extreme", use_container_width=True)
shock_overcap = col_s2.button("🟠 Over-Capacity", use_container_width=True)
col_s3, col_s4 = st.sidebar.columns(2)
shock_stagflation = col_s3.button("🟡 Stagflation", use_container_width=True)
shock_reset = col_s4.button("🟢 Reset", use_container_width=True)

# Apply shock presets via session state
if shock_marine or shock_overcap or shock_stagflation or shock_reset:
    if shock_marine:
        preset = get_shock_preset("extreme_marine")
    elif shock_overcap:
        preset = get_shock_preset("over_capacity")
    elif shock_stagflation:
        preset = get_shock_preset("stagflation")
    else:
        preset = get_shock_preset("reset")
    st.sidebar.info(f"**{preset['label']}**\n\n{preset['description']}")


# ─────────────────────────────────────────────────────────────────────
# Engine Computation Pipeline
# ─────────────────────────────────────────────────────────────────────
alloy = ALLOY_PROFILES[alloy_name]
env_mult = compute_environmental_multiplier(salinity_pct, temperature_c)

# Engine 1/2: Atomistic Mechanics -> Stochastic SDE
mechanics = MaterialMechanics(
    bulk_modulus=alloy["bulk_modulus"],
    shear_modulus=alloy["shear_modulus"],
    fracture_toughness=alloy["fracture_toughness"]
)

sde_engine = StochasticDegradationEngine(
    mechanics=mechanics,
    initial_crack_m=initial_flaw_mm / 1000.0,
    cycles_per_year=float(cycles_per_year),
    noise_intensity=0.002
)

t_sim_start = time.time()
sde_result = sde_engine.run_monte_carlo(
    delta_sigma_mpa=float(delta_sigma),
    sigma_max_mpa=float(delta_sigma) * 1.8,
    horizon_years=10.0,
    steps_per_year=52,
    num_trajectories=10000,
    environmental_salinity_factor=env_mult,
    seed=42
)
sim_time = time.time() - t_sim_start

# Extract annual failure probabilities for finance engine (10 annual values)
time_grid = np.array(sde_result["time_grid"])
pf_curve = np.array(sde_result["failure_probability"])
annual_indices = [np.argmin(np.abs(time_grid - yr)) for yr in range(1, 11)]
annual_pf = [float(pf_curve[idx]) for idx in annual_indices]

# Engine 3: Project Finance
fin_model = ProjectFinanceModel(
    asset_capex_usd=capex_m * 1e6,
    debt_ratio=debt_pct / 100.0,
    interest_rate=interest_rate / 100.0,
    base_annual_revenue=(base_ebitda_m + 7.0) * 1e6,
    base_annual_opex=7_000_000.0,
    base_annual_capex=1_500_000.0,
    dscr_target=target_dscr
)

unadj_df = fin_model.calculate_unadjusted_waterfall(annual_pf)
sculpt_df = fin_model.calculate_sculpted_waterfall(annual_pf)

unadj_schedule = unadj_df.to_dict(orient="records")
sculpt_schedule = sculpt_df.to_dict(orient="records")
min_unadj_dscr = float(unadj_df["dscr"].min())
min_sculpt_dscr = float(sculpt_df["dscr"].min())

# Engine 4: EVT Insurance
evt_engine = CatastropheLossEVT(asset_value_usd=capex_m * 1e6)
# Build synthetic trajectories for loss generation from SDE percentiles
np.random.seed(42)
n_loss_sims = 5000
p50_arr = np.array(sde_result["percentiles"]["p50"]) / 1000.0
p95_arr = np.array(sde_result["percentiles"]["p95"]) / 1000.0
a_crit_m = sde_result["a_crit_mm"] / 1000.0
traj_for_loss = np.zeros((n_loss_sims, len(p50_arr)))

for i in range(n_loss_sims):
    # Mix between median and extreme trajectories based on failure rate
    mix = np.random.beta(2, 5)
    traj_for_loss[i, :] = p50_arr * (1 - mix) + p95_arr * mix
    # 3% catastrophic paths
    if np.random.random() < max(0.01, sde_result["final_failure_prob_10yr"]):
        traj_for_loss[i, -1] = a_crit_m * np.random.uniform(1.0, 1.3)

loss_dist = evt_engine.generate_structural_loss_distribution(traj_for_loss, a_crit_m)
evt_pricing = evt_engine.fit_gpd_tail(loss_dist, alpha=0.99)

# Engine 4b: PARS Resilience Score
pars_engine = PARSEngine()
pars_result = pars_engine.compute_asset_score(
    fracture_toughness_k_ic=alloy["fracture_toughness"],
    a_crit_mm=sde_result["a_crit_mm"],
    mean_crack_10yr_mm=float(np.array(sde_result["percentiles"]["p50"])[-1]),
    failure_prob_10yr=sde_result["final_failure_prob_10yr"],
    coastal_salinity_factor=1.0 + salinity_pct / 5.0,
    thermal_cycling_range_c=min(60.0, abs(temperature_c - 25.0)),
    min_dscr=min_sculpt_dscr if sculpting_enabled else min_unadj_dscr
)

# Pick active schedule for display
active_dscr = min_sculpt_dscr if sculpting_enabled else min_unadj_dscr
gross_premium_k = evt_pricing["gross_actuarial_premium_usd"] / 1000.0


# ─────────────────────────────────────────────────────────────────────
# Top Banner: Executive Summary KPI Cards
# ─────────────────────────────────────────────────────────────────────
st.markdown("# ⚛ MatRisk Lab — Physical-to-Financial Risk Command Center")
st.caption(f"Alloy: **{alloy_name}** | Env Multiplier: **{env_mult}x** | SDE: **10,000 paths in {sim_time:.2f}s** | Engine Pipeline: Atomistic → SDE → Finance → EVT → PARS")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

# KPI 1: Fracture Toughness
k_ic = alloy["fracture_toughness"]
kpi1.metric(
    "Fracture Toughness K_Ic",
    f"{k_ic:.1f} MPa√m",
    delta="Brittle ⚠" if k_ic < 20 else "Ductile ✓",
    delta_color="inverse" if k_ic < 20 else "normal"
)

# KPI 2: 10-Year Failure Probability
pf_10 = sde_result["final_failure_prob_10yr"] * 100
pf_delta = "HIGH RISK 🔴" if pf_10 > 15 else ("Elevated ⚠" if pf_10 > 5 else "Low ✅")
kpi2.metric("10-Yr Failure P_f", f"{pf_10:.1f}%", delta=pf_delta, delta_color="inverse" if pf_10 > 5 else "normal")

# KPI 3: Minimum DSCR
dscr_display = active_dscr
dscr_label = "Sculpted" if sculpting_enabled else "Unadjusted"
if dscr_display < 1.05:
    dscr_delta = "DEFAULT 🛑"
elif dscr_display < 1.20:
    dscr_delta = "LOCK-UP ⚠"
else:
    dscr_delta = "Healthy ✅"
kpi3.metric(f"Min DSCR ({dscr_label})", f"{dscr_display:.2f}x", delta=dscr_delta,
            delta_color="inverse" if dscr_display < 1.20 else "normal")

# KPI 4: Gross Insurance Premium
kpi4.metric("Gross Premium", f"${gross_premium_k:.0f}K/yr", delta=f"RoL {evt_pricing['rate_on_line_pct']:.2f}%")

# KPI 5: PARS Rating
pars_s = pars_result["pars_score"]
pars_r = pars_result["rating"]
kpi5.metric("PARS Resilience", f"{pars_s}/100 ({pars_r})", delta=pars_result["outlook"],
            delta_color="inverse" if pars_s < 60 else "normal")


# ─────────────────────────────────────────────────────────────────────
# Covenant Alert Banners
# ─────────────────────────────────────────────────────────────────────
if not sculpting_enabled:
    if min_unadj_dscr < 1.05:
        st.markdown('<div class="alert-default">🛑 SENIOR DEBT COVENANT DEFAULT — DSCR has breached the 1.05x threshold. Lender acceleration rights triggered. Toggle debt sculpting to restore coverage.</div>', unsafe_allow_html=True)
    elif min_unadj_dscr < 1.20:
        st.markdown('<div class="alert-lockup">⚠️ EQUITY DIVIDEND LOCK-UP ACTIVATED — DSCR below 1.20x. Cash swept into reserve accounts. No distributions to equity sponsors.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-safe">✅ All bank covenants satisfied. DSCR above target cushion. Cash flow stable.</div>', unsafe_allow_html=True)
else:
    if min_sculpt_dscr >= target_dscr - 0.05:
        st.markdown('<div class="alert-safe">✅ Physics-Informed Debt Sculpting Active — Principal repayments dynamically adjusted. All covenant thresholds cleared.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-lockup">⚠️ Sculpted DSCR remains under pressure. Consider reducing debt gearing or extending loan tenor.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Main Dashboard: 3-Tab Telemetry Layout
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "⚛ Microstructural Physics & Failure Mechanics",
    "🏦 Project Finance & Debt Covenants",
    "🔥 Catastrophe Insurance & ESG Analytics"
])

# ── TAB 1: Physics ──────────────────────────────────────────────────
with tab1:
    # Material Property Summary Card
    mcol1, mcol2 = st.columns([1, 2])
    with mcol1:
        st.markdown(f"#### 🔬 {alloy_name}")
        st.caption(alloy["description"])
        st.markdown(f"""
| Property | Value |
|:---|:---|
| Crystal Lattice | `{alloy['lattice']}` |
| Atoms in Unit Cell | `{alloy['atoms']}` |
| Bulk Modulus (K) | `{alloy['bulk_modulus']:.0f} GPa` |
| Shear Modulus (G) | `{alloy['shear_modulus']:.0f} GPa` |
| Young's Modulus (E) | `{mechanics.e_gpa:.1f} GPa` |
| Poisson's Ratio (ν) | `{mechanics.nu:.3f}` |
| Fracture Toughness (K_Ic) | `{alloy['fracture_toughness']:.1f} MPa√m` |
| Paris Exponent (m) | `{mechanics.paris_m:.3f}` |
| Paris Constant (C) | `{mechanics.paris_c:.2e} m/cycle` |
| Critical Crack (a_crit) | `{sde_result['a_crit_mm']:.2f} mm` |
| Env. Acceleration | `{env_mult}x` |
""")

    with mcol2:
        st.plotly_chart(
            render_monte_carlo_crack_chart(
                sde_result["time_grid"],
                sde_result["sample_trajectories"],
                sde_result["percentiles"],
                sde_result["a_crit_mm"]
            ),
            use_container_width=True
        )

    st.plotly_chart(
        render_failure_probability_chart(sde_result["time_grid"], sde_result["failure_probability"]),
        use_container_width=True
    )

    st.caption(f"Mean Time to Failure (MTTF): **{sde_result['mttf_years']:.1f} years** | "
               f"10-Year P_f: **{sde_result['final_failure_prob_10yr']*100:.2f}%** | "
               f"Median Crack at 10yr: **{np.array(sde_result['percentiles']['p50'])[-1]:.2f} mm**")


# ── TAB 2: Finance ──────────────────────────────────────────────────
with tab2:
    st.plotly_chart(render_cfads_waterfall_chart(unadj_schedule), use_container_width=True)

    st.plotly_chart(
        render_dscr_covenant_chart(
            unadj_schedule,
            sculpt_schedule,
            dscr_target=target_dscr,
            show_sculpted=sculpting_enabled
        ),
        use_container_width=True
    )

    if sculpting_enabled:
        st.plotly_chart(
            render_sculpted_vs_flat_principal(unadj_schedule, sculpt_schedule),
            use_container_width=True
        )
        st.caption(f"Flat Min DSCR: **{min_unadj_dscr:.2f}x** → Sculpted Min DSCR: **{min_sculpt_dscr:.2f}x** | "
                   f"Sculpting preserved covenant compliance across {sum(1 for r in sculpt_schedule if r['dscr'] >= target_dscr - 0.05)}/10 years.")

    # Tabular schedule
    with st.expander("📊 Full Debt Service Schedule (Click to Expand)"):
        display_df = sculpt_df if sculpting_enabled else unadj_df
        format_cols = {
            "revenue_usd": "${:,.0f}",
            "cfads_usd": "${:,.0f}",
            "beg_debt_balance_usd": "${:,.0f}",
            "interest_usd": "${:,.0f}",
            "principal_usd": "${:,.0f}",
            "debt_service_usd": "${:,.0f}",
            "end_debt_balance_usd": "${:,.0f}",
        }
        st.dataframe(display_df.style.format(format_cols), use_container_width=True)


# ── TAB 3: Insurance & ESG ──────────────────────────────────────────
with tab3:
    ins_col1, ins_col2 = st.columns(2)

    with ins_col1:
        st.plotly_chart(
            render_evt_tail_loss_chart(
                loss_dist,
                evt_pricing["var_99_usd"],
                evt_pricing["es_99_usd"],
                evt_pricing["threshold_u"]
            ),
            use_container_width=True
        )

        st.markdown(f"""
| Metric | Value |
|:---|:---|
| GPD Shape ξ (Tail Index) | `{evt_pricing['gpd_xi_shape']}` |
| GPD Scale β | `${evt_pricing['gpd_beta_scale']:,.0f}` |
| Pure Expected Loss | `${evt_pricing['pure_premium_usd']:,.0f}` |
| VaR₉₉ | `${evt_pricing['var_99_usd']:,.0f}` |
| Expected Shortfall ES₉₉ | `${evt_pricing['es_99_usd']:,.0f}` |
| Solvency II Capital Buffer | `${evt_pricing['solvency_capital_buffer_usd']:,.0f}` |
| **Gross Actuarial Premium** | **`${evt_pricing['gross_actuarial_premium_usd']:,.0f}/yr`** |
| Rate on Line (RoL) | `{evt_pricing['rate_on_line_pct']:.3f}%` |
""")

    with ins_col2:
        st.plotly_chart(
            render_pars_radar_chart(
                pars_result["sub_scores"],
                pars_result["pars_score"],
                pars_result["rating"]
            ),
            use_container_width=True
        )

        st.markdown(f"""
| PARS Pillar | Score |
|:---|:---|
| Mechanical Safety Margin (30%) | `{pars_result['sub_scores']['mechanical_margin']:.1f}/100` |
| Degradation Velocity (25%) | `{pars_result['sub_scores']['degradation_velocity']:.1f}/100` |
| Environmental Fragility (25%) | `{pars_result['sub_scores']['environmental_fragility']:.1f}/100` |
| Financial Covenant Buffer (20%) | `{pars_result['sub_scores']['financial_buffer']:.1f}/100` |
| **Composite PARS** | **`{pars_result['pars_score']}/100 ({pars_result['rating']})`** |
""")


# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("MatRisk AI v1.0 — Atomistic-to-Financial Asset Resilience & Risk Engine | "
           "Engines: DA-CGCNN → Itô SDE → CFADS/DSCR → EVT/GPD → PARS | "
           f"Computed in {sim_time:.2f}s on CPU")
