"""
plotly_charts.py - Enterprise-Grade Plotly Chart Rendering Library
Part of MatRisk AI (Phase 6, Day 12)

Provides 7 publication-quality interactive chart builders for the MatRisk Lab dashboard,
following a Bloomberg Terminal-inspired dark theme with physics/finance color hierarchy:
  - Electric Cyan (#00E5FF): Atomistic physics & GNN outputs
  - Neon Amber (#FFB300): Corporate finance, debt schedules, CFADS
  - Crimson Red (#FF1744): Risk breaches, critical thresholds, covenant defaults
  - Emerald Green (#00E676): Healthy coverage, strong ESG resilience
"""

from typing import Dict, List, Any, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Theme Constants
BG_COLOR = "#0E1117"
PAPER_COLOR = "#0E1117"
GRID_COLOR = "#1E2A3A"
TEXT_COLOR = "#E0E0E0"
CYAN = "#00E5FF"
AMBER = "#FFB300"
RED = "#FF1744"
GREEN = "#00E676"
CYAN_TRANS = "rgba(0,229,255,0.08)"
RED_TRANS = "rgba(255,23,68,0.15)"
AMBER_TRANS = "rgba(255,179,0,0.10)"


def _base_layout(title: str, xaxis_title: str = "", yaxis_title: str = "", height: int = 420) -> dict:
    """Returns the shared dark-theme layout configuration."""
    return dict(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=16), x=0.01),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, size=12),
        xaxis=dict(title=xaxis_title, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(title=yaxis_title, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        height=height,
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hovermode="x unified",
    )


def render_monte_carlo_crack_chart(
    time_grid: List[float],
    sample_trajectories: List[List[float]],
    percentiles: Dict[str, List[float]],
    a_crit_mm: float
) -> go.Figure:
    """
    Chart 1B: Monte Carlo Sub-Critical Crack Trajectories (a_t vs Time).
    Shows 50 sample paths, percentile bands (p05-p95), and red critical threshold line.
    """
    fig = go.Figure()
    t = np.array(time_grid)

    # Percentile confidence bands (p05 to p95 shaded region)
    p05 = np.array(percentiles["p05"])
    p25 = np.array(percentiles["p25"])
    p75 = np.array(percentiles["p75"])
    p95 = np.array(percentiles["p95"])
    p50 = np.array(percentiles["p50"])

    # 5th-95th percentile band
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]).tolist(),
        y=np.concatenate([p95, p05[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(0,229,255,0.08)",
        line=dict(width=0), name="5th–95th Percentile", showlegend=True, hoverinfo="skip"
    ))

    # 25th-75th percentile band
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]).tolist(),
        y=np.concatenate([p75, p25[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(0,229,255,0.15)",
        line=dict(width=0), name="25th–75th Percentile", showlegend=True, hoverinfo="skip"
    ))

    # 50 sample trajectories
    trajectories = np.array(sample_trajectories)
    for i in range(min(50, len(trajectories))):
        path = trajectories[i]
        breached = np.any(path >= a_crit_mm)
        color = RED if breached else "rgba(0,229,255,0.25)"
        width = 1.5 if breached else 0.6
        fig.add_trace(go.Scatter(
            x=t.tolist(), y=path.tolist(),
            mode="lines", line=dict(color=color, width=width),
            showlegend=False, hoverinfo="skip"
        ))

    # Median line
    fig.add_trace(go.Scatter(
        x=t.tolist(), y=p50.tolist(),
        mode="lines", line=dict(color=CYAN, width=2.5, dash="dot"),
        name="Median Crack Depth"
    ))

    # Critical threshold line
    fig.add_hline(
        y=a_crit_mm, line_dash="dash", line_color=RED, line_width=2,
        annotation_text=f"a_crit = {a_crit_mm:.1f} mm (Catastrophic Fracture)",
        annotation_position="top left",
        annotation_font=dict(color=RED, size=11)
    )

    fig.update_layout(**_base_layout(
        "⚛ Monte Carlo Crack Growth Trajectories (10,000 Itô SDE Paths)",
        xaxis_title="Time (Years)",
        yaxis_title="Crack Depth a(t) [mm]",
        height=440
    ))
    return fig


def render_failure_probability_chart(
    time_grid: List[float],
    failure_probability: List[float]
) -> go.Figure:
    """Chart 1C: Cumulative Failure Probability P_f(t) area chart."""
    fig = go.Figure()
    t = np.array(time_grid)
    pf = np.array(failure_probability) * 100.0  # Convert to percentage

    fig.add_trace(go.Scatter(
        x=t.tolist(), y=pf.tolist(),
        fill="tozeroy", fillcolor="rgba(0,229,255,0.12)",
        line=dict(color=CYAN, width=2.5),
        name="Cumulative P_f(t)"
    ))

    # 5% operational tolerance warning
    fig.add_hline(y=5.0, line_dash="dot", line_color=AMBER, line_width=1.5,
                  annotation_text="5% Operational Tolerance", annotation_position="top left",
                  annotation_font=dict(color=AMBER, size=10))

    # 15% high structural risk
    fig.add_hline(y=15.0, line_dash="dash", line_color=RED, line_width=1.5,
                  annotation_text="15% High Structural Risk", annotation_position="top left",
                  annotation_font=dict(color=RED, size=10))

    fig.update_layout(**_base_layout(
        "📈 Cumulative Failure Probability P_f(t)",
        xaxis_title="Time (Years)",
        yaxis_title="Failure Probability [%]",
        height=350
    ))
    return fig


def render_cfads_waterfall_chart(schedule: List[Dict[str, Any]]) -> go.Figure:
    """Chart 2A: CFADS Cash Flow Decay Waterfall."""
    fig = go.Figure()
    years = [r["year"] for r in schedule]

    # Stacked bars: Revenue base, then reductions
    revenue = [r["revenue_usd"] / 1e6 for r in schedule]
    cfads = [r["cfads_usd"] / 1e6 for r in schedule]
    capex = [r.get("capex_maint_usd", r.get("capex_maint_usd", 1_500_000)) / 1e6 for r in schedule]

    fig.add_trace(go.Bar(
        x=years, y=revenue,
        name="Effective Revenue", marker_color="rgba(255,179,0,0.6)",
        text=[f"${v:.1f}M" for v in revenue], textposition="outside"
    ))
    fig.add_trace(go.Bar(
        x=years, y=cfads,
        name="CFADS (Net Cash for Debt)", marker_color=AMBER,
        text=[f"${v:.1f}M" for v in cfads], textposition="outside"
    ))

    fig.update_layout(**_base_layout(
        "💰 CFADS Cash Flow Available for Debt Service",
        xaxis_title="Year",
        yaxis_title="USD (Millions)",
        height=400
    ))
    fig.update_layout(barmode="group")
    return fig


def render_dscr_covenant_chart(
    unadjusted_schedule: List[Dict[str, Any]],
    sculpted_schedule: Optional[List[Dict[str, Any]]] = None,
    dscr_target: float = 1.30,
    show_sculpted: bool = False
) -> go.Figure:
    """Chart 2B: DSCR Covenant Monitor with threshold zones."""
    fig = go.Figure()
    years = [r["year"] for r in unadjusted_schedule]
    dscr_unadj = [r["dscr"] for r in unadjusted_schedule]

    fig.add_trace(go.Scatter(
        x=years, y=dscr_unadj,
        mode="lines+markers",
        line=dict(color=AMBER, width=2.5),
        marker=dict(size=8, color=[GREEN if d >= dscr_target else (AMBER if d >= 1.05 else RED) for d in dscr_unadj]),
        name="Unadjusted DSCR"
    ))

    if show_sculpted and sculpted_schedule:
        dscr_sculpt = [r["dscr"] for r in sculpted_schedule]
        fig.add_trace(go.Scatter(
            x=years, y=dscr_sculpt,
            mode="lines+markers",
            line=dict(color=GREEN, width=2.5),
            marker=dict(size=8, color=GREEN),
            name="Sculpted DSCR (Physics-Informed)"
        ))

    # Covenant threshold zones
    fig.add_hrect(y0=dscr_target, y1=3.0, fillcolor="rgba(0,230,118,0.06)", line_width=0,
                  annotation_text=f"Safe Zone (≥ {dscr_target}x)", annotation_position="top left",
                  annotation_font=dict(color=GREEN, size=10))
    fig.add_hrect(y0=1.05, y1=1.20, fillcolor="rgba(255,179,0,0.08)", line_width=0)
    fig.add_hrect(y0=0.0, y1=1.05, fillcolor="rgba(255,23,68,0.08)", line_width=0)

    fig.add_hline(y=dscr_target, line_dash="dash", line_color=GREEN, line_width=1.5,
                  annotation_text=f"Target {dscr_target}x", annotation_position="bottom right",
                  annotation_font=dict(color=GREEN, size=10))
    fig.add_hline(y=1.20, line_dash="dot", line_color=AMBER, line_width=1.5,
                  annotation_text="1.20x Dividend Lock-Up", annotation_position="bottom right",
                  annotation_font=dict(color=AMBER, size=10))
    fig.add_hline(y=1.05, line_dash="dash", line_color=RED, line_width=2,
                  annotation_text="1.05x Covenant Default", annotation_position="bottom right",
                  annotation_font=dict(color=RED, size=10))

    fig.update_layout(**_base_layout(
        "🏦 Debt Service Coverage Ratio (DSCR) Covenant Monitor",
        xaxis_title="Year",
        yaxis_title="DSCR (x)",
        height=420
    ))
    fig.update_yaxes(range=[0, max(3.0, max(dscr_unadj) * 1.15)])
    return fig


def render_sculpted_vs_flat_principal(
    unadjusted_schedule: List[Dict[str, Any]],
    sculpted_schedule: List[Dict[str, Any]]
) -> go.Figure:
    """Chart 2C: Flat vs Sculpted Principal Repayment comparison."""
    fig = go.Figure()
    years = [r["year"] for r in unadjusted_schedule]
    flat_principal = [r["principal_usd"] / 1e6 for r in unadjusted_schedule]
    sculpt_principal = [r["principal_usd"] / 1e6 for r in sculpted_schedule]

    fig.add_trace(go.Bar(
        x=years, y=flat_principal,
        name="Flat Amortization", marker_color="rgba(255,179,0,0.5)",
        text=[f"${v:.1f}M" for v in flat_principal], textposition="outside"
    ))
    fig.add_trace(go.Bar(
        x=years, y=sculpt_principal,
        name="Physics-Sculpted Principal", marker_color=GREEN,
        text=[f"${v:.1f}M" for v in sculpt_principal], textposition="outside"
    ))

    fig.update_layout(**_base_layout(
        "🔧 Flat vs Physics-Informed Sculpted Principal Repayment",
        xaxis_title="Year",
        yaxis_title="Principal Repayment (USD Millions)",
        height=380
    ))
    fig.update_layout(barmode="group")
    return fig


def render_evt_tail_loss_chart(
    losses: np.ndarray,
    var_99: float,
    es_99: float,
    threshold_u: float
) -> go.Figure:
    """Chart 3A: EVT GPD Tail Loss Distribution with VaR/ES markers."""
    fig = go.Figure()

    # Histogram of all losses
    fig.add_trace(go.Histogram(
        x=losses / 1e6,
        nbinsx=80,
        marker_color="rgba(0,229,255,0.35)",
        name="Loss Distribution"
    ))

    # VaR line
    fig.add_vline(x=var_99 / 1e6, line_dash="dash", line_color=AMBER, line_width=2,
                  annotation_text=f"VaR₉₉ = ${var_99/1e6:.1f}M",
                  annotation_position="top right",
                  annotation_font=dict(color=AMBER, size=12))

    # ES line
    fig.add_vline(x=es_99 / 1e6, line_dash="dash", line_color=RED, line_width=2.5,
                  annotation_text=f"ES₉₉ = ${es_99/1e6:.1f}M",
                  annotation_position="top right",
                  annotation_font=dict(color=RED, size=12))

    # Tail shading beyond VaR
    fig.add_vrect(x0=var_99 / 1e6, x1=max(losses) / 1e6 * 1.05,
                  fillcolor="rgba(255,23,68,0.12)", line_width=0,
                  annotation_text="Catastrophe Tail Region", annotation_position="top left",
                  annotation_font=dict(color=RED, size=10))

    fig.update_layout(**_base_layout(
        "🔥 Extreme Value Theory (GPD) Structural Loss Distribution",
        xaxis_title="Annual Structural Loss (USD Millions)",
        yaxis_title="Frequency",
        height=400
    ))
    return fig


def render_pars_radar_chart(sub_scores: Dict[str, float], pars_score: float, rating: str) -> go.Figure:
    """Chart 3B: PARS Physical Asset Resilience Radar (Spider) Chart."""
    categories = [
        "Mechanical\nSafety Margin",
        "Degradation\nVelocity",
        "Environmental\nFragility",
        "Financial\nCovenant Buffer",
        "Overall\nPARS Score"
    ]
    values = [
        sub_scores.get("mechanical_margin", 50.0),
        sub_scores.get("degradation_velocity", 50.0),
        sub_scores.get("environmental_fragility", 50.0),
        sub_scores.get("financial_buffer", 50.0),
        pars_score
    ]
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    # Determine fill color based on rating
    if pars_score >= 80:
        fill_color = "rgba(0,230,118,0.20)"
        line_color = GREEN
    elif pars_score >= 60:
        fill_color = "rgba(255,179,0,0.20)"
        line_color = AMBER
    else:
        fill_color = "rgba(255,23,68,0.20)"
        line_color = RED

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2.5),
        marker=dict(size=7, color=line_color),
        name=f"PARS {pars_score} ({rating})"
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=BG_COLOR,
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR, size=9)),
            angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR, size=10))
        ),
        paper_bgcolor=PAPER_COLOR,
        font=dict(color=TEXT_COLOR),
        title=dict(text=f"🛡 Physical Asset Resilience Score: {pars_score}/100 ({rating})", font=dict(size=16, color=TEXT_COLOR), x=0.01),
        height=420,
        margin=dict(l=80, r=80, t=60, b=60),
        showlegend=False
    )
    return fig
