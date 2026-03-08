"""
Reusable UI components for the EPA-AfCFTA platform.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# ─── Color Palette ──────────────────────────────────────────────────────

COLORS = {
    "primary": "#20808D",
    "secondary": "#A84B2F",
    "dark_teal": "#1B474D",
    "light_cyan": "#BCE2E7",
    "mauve": "#944454",
    "gold": "#FFC553",
    "olive": "#848456",
    "brown": "#6E522B",
    "bg": "#FCFAF6",
    "bg_alt": "#F3F3EE",
    "text": "#13343B",
    "text_muted": "#2E565D",
    "critical": "#DC2626",
    "high": "#F59E0B",
    "moderate": "#3B82F6",
    "low": "#10B981",
}

RISK_COLORS = {
    "Critical": COLORS["critical"],
    "High": COLORS["high"],
    "Moderate": COLORS["moderate"],
    "Low": COLORS["low"],
}

CHART_COLORS = [
    COLORS["primary"], COLORS["secondary"], COLORS["dark_teal"],
    COLORS["light_cyan"], COLORS["mauve"], COLORS["gold"],
    COLORS["olive"], COLORS["brown"],
]


def render_kpi_card(label: str, value: str, delta: str = "", delta_color: str = "normal"):
    """Render a KPI metric card."""
    st.metric(label=label, value=value, delta=delta if delta else None,
              delta_color=delta_color)


def render_kpi_row(metrics: List[Dict]):
    """
    Render a row of KPI cards.
    
    Each dict: {label, value, delta (optional), delta_color (optional)}
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            render_kpi_card(
                m["label"], m["value"],
                m.get("delta", ""),
                m.get("delta_color", "normal")
            )


def render_risk_badge(rating: str):
    """Render a colored risk badge."""
    color = RISK_COLORS.get(rating, "#666")
    st.markdown(
        f'<span style="background-color:{color}; color:white; '
        f'padding:4px 12px; border-radius:4px; font-weight:600; font-size:14px;">'
        f'{rating}</span>',
        unsafe_allow_html=True
    )


def plot_risk_heatmap(
    heatmap_df: pd.DataFrame,
    title: str = "Country × HS Category Risk Heatmap"
) -> go.Figure:
    """Create an interactive risk heatmap using Plotly."""
    if heatmap_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=[c[:25] for c in heatmap_df.columns],
        y=heatmap_df.index,
        colorscale=[
            [0, "#E8F5E9"],
            [0.25, "#FFF9C4"],
            [0.5, "#FFE0B2"],
            [0.75, "#FFAB91"],
            [1, "#EF5350"]
        ],
        colorbar=dict(title="Risk Score"),
        hovertemplate="Country: %{y}<br>Category: %{x}<br>Score: %{z:.1f}<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=11)),
        height=max(400, len(heatmap_df) * 35 + 150),
        margin=dict(l=120, r=40, t=60, b=120),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
    )
    
    return fig


def plot_country_risk_bars(
    risk_scores_df: pd.DataFrame,
    title: str = "Country Risk Ranking"
) -> go.Figure:
    """Create horizontal bar chart of country risk scores."""
    df = risk_scores_df.sort_values("overall_score", ascending=True)
    
    colors = [RISK_COLORS.get(r, "#666") for r in df["rating"]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df["country"],
        x=df["overall_score"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{s:.1f}" for s in df["overall_score"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Risk Score: %{x:.1f}<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        xaxis=dict(title="Risk Score (0-100)", range=[0, 105]),
        yaxis=dict(tickfont=dict(size=11)),
        height=max(400, len(df) * 32 + 100),
        margin=dict(l=120, r=60, t=60, b=40),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg_alt"],
        showlegend=False,
    )
    
    return fig


def plot_monte_carlo_fan(
    mc_results: Dict,
    title: str = "Leakage Rate Forecast"
) -> go.Figure:
    """Create fan chart showing Monte Carlo simulation results with confidence intervals."""
    periods = list(range(1, len(mc_results["leakage_mean"]) + 1))
    labels = [f"Year {p}" for p in periods]
    
    mean_vals = [v * 100 for v in mc_results["leakage_mean"]]
    p5 = [v * 100 for v in mc_results["leakage_p5"]]
    p25 = [v * 100 for v in mc_results["leakage_p25"]]
    p75 = [v * 100 for v in mc_results["leakage_p75"]]
    p95 = [v * 100 for v in mc_results["leakage_p95"]]
    
    fig = go.Figure()
    
    # 90% CI band
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1],
        y=p95 + p5[::-1],
        fill="toself",
        fillcolor="rgba(32, 128, 141, 0.1)",
        line=dict(color="rgba(32, 128, 141, 0)"),
        name="90% CI",
        showlegend=True,
    ))
    
    # 50% CI band
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1],
        y=p75 + p25[::-1],
        fill="toself",
        fillcolor="rgba(32, 128, 141, 0.25)",
        line=dict(color="rgba(32, 128, 141, 0)"),
        name="50% CI",
        showlegend=True,
    ))
    
    # Mean line
    fig.add_trace(go.Scatter(
        x=labels,
        y=mean_vals,
        mode="lines+markers",
        name="Mean",
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=8),
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        xaxis=dict(title="Forecast Horizon"),
        yaxis=dict(title="Leakage Rate (%)", rangemode="tozero"),
        height=400,
        margin=dict(l=60, r=40, t=60, b=40),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg_alt"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def plot_scenario_comparison_bars(
    scenario_df: pd.DataFrame,
    metric: str = "leakage_mean_pct",
    title: str = "Scenario Comparison: Expected Leakage Rate"
) -> go.Figure:
    """Create grouped bar chart comparing scenarios."""
    df = scenario_df.sort_values(metric, ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df["scenario"],
        x=df[metric],
        orientation="h",
        marker=dict(
            color=[COLORS["primary"] if v < df[metric].median()
                   else COLORS["secondary"] for v in df[metric]]
        ),
        error_x=dict(
            type="data",
            symmetric=False,
            array=(df["leakage_ci_high_pct"] - df[metric]).tolist(),
            arrayminus=(df[metric] - df["leakage_ci_low_pct"]).tolist(),
            color="#666",
        ),
        text=[f"{v:.1f}%" for v in df[metric]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Leakage: %{x:.1f}%<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        xaxis=dict(title="Leakage Rate (%)", rangemode="tozero"),
        yaxis=dict(tickfont=dict(size=10)),
        height=max(350, len(df) * 55 + 100),
        margin=dict(l=250, r=80, t=60, b=40),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg_alt"],
        showlegend=False,
    )
    
    return fig


def plot_trade_flow_timeseries(
    trade_df: pd.DataFrame,
    country: str,
    hs_category: str = None,
    partner: str = "EU27"
) -> go.Figure:
    """Plot trade flow time series for a specific country."""
    mask = (trade_df["reporter"] == country) & (trade_df["partner"] == partner)
    if hs_category:
        mask = mask & (trade_df["hs_category"] == hs_category)
    
    df = trade_df[mask].copy()
    
    if hs_category:
        agg = df.groupby("year").agg(
            export=("export_value_usd", "sum"),
            import_val=("import_value_usd", "sum"),
        ).reset_index()
        title = f"{country} — {partner} Trade Flows: {hs_category}"
    else:
        agg = df.groupby("year").agg(
            export=("export_value_usd", "sum"),
            import_val=("import_value_usd", "sum"),
        ).reset_index()
        title = f"{country} — {partner} Total Trade Flows"
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=agg["year"], y=agg["export"],
        name="Exports", mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2.5),
    ))
    
    fig.add_trace(go.Scatter(
        x=agg["year"], y=agg["import_val"],
        name="Imports", mode="lines+markers",
        line=dict(color=COLORS["secondary"], width=2.5, dash="dash"),
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
        xaxis=dict(title="Year", dtick=1),
        yaxis=dict(title="Value (USD)"),
        height=380,
        margin=dict(l=60, r=40, t=50, b=40),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg_alt"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def plot_regional_radar(
    regional_df: pd.DataFrame,
    title: str = "Regional Risk Profile Comparison"
) -> go.Figure:
    """Create radar chart comparing regional risk profiles."""
    fig = go.Figure()
    
    categories = ["Avg Risk Score", "Structural", "Anomaly", "Governance Gap"]
    
    for i, (_, row) in enumerate(regional_df.iterrows()):
        values = [
            row["avg_risk_score"],
            row["avg_structural"],
            row["avg_anomaly"],
            row["avg_governance"],
        ]
        values.append(values[0])  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["region"],
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)]),
            fillcolor=f"rgba({int(CHART_COLORS[i % len(CHART_COLORS)][1:3], 16)}, "
                      f"{int(CHART_COLORS[i % len(CHART_COLORS)][3:5], 16)}, "
                      f"{int(CHART_COLORS[i % len(CHART_COLORS)][5:7], 16)}, 0.15)",
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        height=450,
        margin=dict(l=80, r=80, t=60, b=40),
        paper_bgcolor=COLORS["bg"],
    )
    
    return fig


def format_usd(value: float) -> str:
    """Format USD values with appropriate suffixes."""
    if abs(value) >= 1e9:
        return f"${value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.1f}K"
    else:
        return f"${value:,.0f}"
