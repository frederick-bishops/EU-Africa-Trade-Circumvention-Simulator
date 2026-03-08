"""
African EPA-AfCFTA Overlap Integrity Platform
==============================================
Modular Multi-Country Rules-of-Origin Circumvention Simulator
with Behavioral Forecasting

A production-ready Streamlit application for analyzing circumvention risks
at the intersection of EU Economic Partnership Agreements (EPAs) and the
African Continental Free Trade Area (AfCFTA).

Data sources: UN Comtrade (trade flows), World Bank WGI (governance),
EU Access2Markets (tariff schedules), AfCFTA e-Tariff Book (concessions).
Calibrated to public EU customs enforcement statistics (EPPO/OLAF).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List

# ─── App Configuration ──────────────────────────────────────────────────

st.set_page_config(
    page_title="EPA-AfCFTA Integrity Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main styling */
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    h1 { color: #13343B; font-weight: 700; }
    h2 { color: #1B474D; font-weight: 600; border-bottom: 2px solid #20808D; padding-bottom: 0.3rem; }
    h3 { color: #2E565D; font-weight: 600; }
    
    /* Metric cards */
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #13343B; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #2E565D; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 16px;
        background-color: #F3F3EE;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
        color: #2E565D;
    }
    .stTabs [aria-selected="true"] {
        background-color: #20808D;
        color: white;
    }
    
    /* Info boxes */
    .risk-critical { background: #FEF2F2; border-left: 4px solid #DC2626; padding: 12px; border-radius: 4px; margin: 8px 0; }
    .risk-high { background: #FFFBEB; border-left: 4px solid #F59E0B; padding: 12px; border-radius: 4px; margin: 8px 0; }
    .risk-moderate { background: #EFF6FF; border-left: 4px solid #3B82F6; padding: 12px; border-radius: 4px; margin: 8px 0; }
    .risk-low { background: #ECFDF5; border-left: 4px solid #10B981; padding: 12px; border-radius: 4px; margin: 8px 0; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #F3F3EE; }
    [data-testid="stSidebar"] h1 { font-size: 1.2rem; }
    
    /* Footer */
    .footer-text { text-align: center; color: #2E565D; font-size: 0.75rem; padding: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ─── Imports (project modules) ──────────────────────────────────────────

from config.countries import (
    COUNTRIES, get_country_names, get_epa_groups, get_regional_blocs,
    REGION_CLUSTERS, EPA_GROUPS
)
from config.hs_codes import (
    HS_RISK_CATEGORIES, COUNTRY_HS_HOTSPOTS, get_circumvention_types,
    CIRCUMVENTION_DESCRIPTIONS
)
from data.synthetic_data import (
    generate_trade_flows, generate_governance_data,
    generate_eu_customs_cases, generate_rerouting_network
)
from engine.anomaly_detection import run_full_anomaly_pipeline
from engine.monte_carlo import (
    run_multi_country_simulation, PREDEFINED_SCENARIOS, ScenarioParams
)
from engine.risk_scoring import compute_all_country_scores
from engine.behavioral_forecast import (
    forecast_scenario_comparison, forecast_leakage_timeseries,
    generate_adaptation_narrative
)
from analysis.comparative import (
    build_risk_heatmap_data, build_regional_comparison,
    build_epa_group_comparison, identify_spillover_corridors,
    build_hs_vulnerability_ranking
)
from analysis.policy_menu import (
    generate_country_policy_menu, generate_regional_policy_brief,
    generate_stakeholder_summary
)
from ui.components import (
    COLORS, RISK_COLORS, CHART_COLORS,
    render_kpi_row, render_risk_badge,
    plot_risk_heatmap, plot_country_risk_bars,
    plot_monte_carlo_fan, plot_scenario_comparison_bars,
    plot_trade_flow_timeseries, plot_regional_radar, format_usd
)
from utils.export import (
    export_risk_scores_csv, export_anomaly_data_csv,
    export_simulation_summary_csv, export_policy_recommendations_csv,
    generate_executive_summary_text
)


# ─── Sidebar ────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with controls and documentation."""
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Flag_of_the_African_Union.svg/320px-Flag_of_the_African_Union.svg.png", width=60)
        st.title("EPA-AfCFTA Integrity Platform")
        
        st.markdown("---")
        
        # Country selection
        st.subheader("Country Selection")
        
        selection_mode = st.radio(
            "Select by:",
            ["Individual Countries", "EPA Group", "Regional Bloc", "All Countries"],
            index=0,
            key="selection_mode"
        )
        
        if selection_mode == "Individual Countries":
            selected_countries = st.multiselect(
                "Choose countries:",
                options=get_country_names(),
                default=["Ghana", "Côte d'Ivoire", "Kenya", "Cameroon", "Mauritius",
                         "South Africa", "Nigeria", "Tanzania"],
                key="country_select"
            )
        elif selection_mode == "EPA Group":
            selected_group = st.selectbox(
                "Choose EPA group:",
                options=list(EPA_GROUPS.keys()),
                key="epa_group_select"
            )
            selected_countries = EPA_GROUPS[selected_group]
            st.info(f"Countries: {', '.join(selected_countries)}")
        elif selection_mode == "Regional Bloc":
            selected_region = st.selectbox(
                "Choose region:",
                options=list(REGION_CLUSTERS.keys()),
                key="region_select"
            )
            selected_countries = REGION_CLUSTERS[selected_region]
            st.info(f"Countries: {', '.join(selected_countries)}")
        else:
            selected_countries = get_country_names()
        
        st.markdown("---")
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        
        z_threshold = st.slider(
            "Export spike Z-score threshold",
            min_value=1.0, max_value=4.0, value=2.0, step=0.25,
            help="Standard deviations above mean to flag as anomalous"
        )
        
        capacity_threshold = st.slider(
            "Capacity mismatch threshold",
            min_value=1.0, max_value=3.0, value=1.5, step=0.1,
            help="Ratio of exports to estimated production capacity"
        )
        
        n_simulations = st.select_slider(
            "Monte Carlo iterations",
            options=[1000, 2000, 3000, 5000, 10000],
            value=3000,
            help="More iterations = more precise estimates, slower computation"
        )
        
        st.markdown("---")
        
        # Scenario selection
        st.subheader("Scenario")
        selected_scenario_name = st.selectbox(
            "Choose scenario:",
            options=list(PREDEFINED_SCENARIOS.keys()),
            index=0,
            key="scenario_select"
        )
        
        scenario = PREDEFINED_SCENARIOS[selected_scenario_name]
        
        # Show scenario parameters
        with st.expander("Scenario Parameters", expanded=False):
            st.write(f"**Rerouting pressure**: {scenario.rerouting_pressure:.1f}")
            st.write(f"**AfCFTA liberalization**: {scenario.afcfta_liberalization:.1f}")
            st.write(f"**EPA tightening**: {scenario.epa_tightening:.1f}")
            st.write(f"**Digital traceability**: {scenario.digital_traceability:.1f}")
            st.write(f"**Regional harmonization**: {scenario.regional_harmonization:.1f}")
        
        st.markdown("---")
        
        # Documentation
        with st.expander("About This Platform", expanded=False):
            st.markdown("""
**African EPA-AfCFTA Overlap Integrity Platform**

This platform models the overlap between EU Economic Partnership 
Agreements (EPAs) and AfCFTA liberalization as a dynamic strategic 
arbitrage game, identifying circumvention risks across 20 African countries.

**Core capabilities:**
- Multi-country anomaly detection (export spikes, capacity mismatches, origin shifts)
- Monte Carlo simulation of firm-level and state-level adaptive behavior
- Comparative risk heatmaps and regional cluster analysis
- Scenario forecasting with confidence intervals
- Actionable policy recommendation engine

**Data sources:**
- Trade flows: Calibrated synthetic data based on UN Comtrade patterns
- Governance: World Bank WGI indicators
- Tariff schedules: EU Access2Markets EPA data
- Enforcement: EU customs infringement statistics (EPPO/OLAF)

**Methodology:**
- Z-score anomaly detection for export spike identification
- Production capacity ratio analysis for mismatch detection
- Pearson correlation for origin shift analysis
- Monte Carlo simulation with behavioral game-theoretic framework
- Composite weighted scoring (structural 30%, anomaly 30%, MC 25%, governance 15%)
            """)
        
        with st.expander("Disclaimer", expanded=False):
            st.warning("""
This analysis uses simulation-based modeling with synthetic data 
calibrated to public sources. Results should be interpreted as 
indicative risk assessments for capacity-building purposes, not 
precise predictions. Country comparisons are framed as neutral 
capacity diagnostics.
            """)
        
        return selected_countries, z_threshold, capacity_threshold, n_simulations, scenario, selected_scenario_name


# ─── Data Loading & Caching ────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_process_data(
    country_names: tuple,
    z_threshold: float,
    capacity_threshold: float,
    n_simulations: int,
    scenario_name: str,
):
    """Load data, run anomaly detection, simulations, and scoring."""
    country_list = list(country_names)
    
    # Generate synthetic trade flow data
    trade_df = generate_trade_flows(country_list, include_anomalies=True)
    
    # Generate governance data
    governance_df = generate_governance_data(country_list)
    
    # Run anomaly detection pipeline
    anomaly_df = run_full_anomaly_pipeline(
        trade_df,
        z_threshold=z_threshold,
        capacity_threshold=capacity_threshold,
    )
    
    # Get scenario
    scenario = PREDEFINED_SCENARIOS[scenario_name]
    
    # Run Monte Carlo simulations
    mc_results = run_multi_country_simulation(
        country_list,
        scenario=scenario,
        n_simulations=n_simulations,
    )
    
    # Remove raw arrays from mc_results for caching (not serializable)
    mc_results_clean = {}
    for country, results in mc_results.items():
        mc_results_clean[country] = {
            k: v for k, v in results.items()
            if k not in ["raw_circumvention", "raw_leakage", "raw_audit",
                         "firm_params", "state_params"]
        }
    
    # Compute risk scores
    risk_scores_df = compute_all_country_scores(
        country_list, anomaly_df, mc_results_clean, governance_df
    )
    
    # EU customs cases
    eu_cases_df = generate_eu_customs_cases()
    
    # Rerouting network
    rerouting_df = generate_rerouting_network(country_list)
    
    return trade_df, governance_df, anomaly_df, mc_results_clean, risk_scores_df, eu_cases_df, rerouting_df


# ─── Main App ───────────────────────────────────────────────────────────

def main():
    # Sidebar
    selected_countries, z_threshold, capacity_threshold, n_simulations, scenario, scenario_name = render_sidebar()
    
    if not selected_countries:
        st.warning("Please select at least one country from the sidebar.")
        return
    
    # Load & process data
    with st.spinner("Running analysis pipeline (anomaly detection + Monte Carlo simulation)..."):
        (trade_df, governance_df, anomaly_df, mc_results,
         risk_scores_df, eu_cases_df, rerouting_df) = load_and_process_data(
            tuple(sorted(selected_countries)),
            z_threshold,
            capacity_threshold,
            n_simulations,
            scenario_name,
        )
    
    # ─── Header ─────────────────────────────────────────────────────────
    st.title("African EPA-AfCFTA Overlap Integrity Platform")
    st.markdown(f"**Scenario**: {scenario.name}  ·  **Countries**: {len(selected_countries)}  ·  **MC Iterations**: {n_simulations:,}")
    
    # ─── Tab Navigation ─────────────────────────────────────────────────
    tab_overview, tab_country, tab_compare, tab_simulate, tab_policy, tab_data = st.tabs([
        "Overview", "Country Deep-Dive", "Comparative Analysis",
        "Simulation Lab", "Policy Menu", "Data Explorer"
    ])
    
    # ═══════════════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW
    # ═══════════════════════════════════════════════════════════════════
    with tab_overview:
        render_overview_tab(risk_scores_df, anomaly_df, mc_results, selected_countries, scenario)
    
    # ═══════════════════════════════════════════════════════════════════
    # TAB 2: COUNTRY DEEP-DIVE
    # ═══════════════════════════════════════════════════════════════════
    with tab_country:
        render_country_tab(
            risk_scores_df, anomaly_df, trade_df, mc_results,
            governance_df, selected_countries, scenario, n_simulations
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # TAB 3: COMPARATIVE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    with tab_compare:
        render_comparative_tab(risk_scores_df, anomaly_df, selected_countries)
    
    # ═══════════════════════════════════════════════════════════════════
    # TAB 4: SIMULATION LAB
    # ═══════════════════════════════════════════════════════════════════
    with tab_simulate:
        render_simulation_tab(selected_countries, mc_results, n_simulations)
    
    # ═══════════════════════════════════════════════════════════════════
    # TAB 5: POLICY MENU
    # ═══════════════════════════════════════════════════════════════════
    with tab_policy:
        render_policy_tab(risk_scores_df, anomaly_df, mc_results, selected_countries, scenario_name)
    
    # ═══════════════════════════════════════════════════════════════════
    # TAB 6: DATA EXPLORER
    # ═══════════════════════════════════════════════════════════════════
    with tab_data:
        render_data_tab(trade_df, anomaly_df, risk_scores_df, mc_results, eu_cases_df, rerouting_df, scenario_name)
    
    # ─── Footer ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="footer-text">'
        'EPA-AfCFTA Overlap Integrity Platform · Open-Source Simulation Tool · '
        'Data: UN Comtrade, World Bank WGI, EU Access2Markets, AfCFTA e-Tariff Book · '
        'Methodology: Monte Carlo behavioral simulation calibrated to EPPO/OLAF enforcement statistics'
        '</div>',
        unsafe_allow_html=True
    )


# ─── Tab Renderers ──────────────────────────────────────────────────────

def render_overview_tab(risk_scores_df, anomaly_df, mc_results, selected_countries, scenario):
    """Render the Overview dashboard tab."""
    
    # KPI row
    n_critical = len(risk_scores_df[risk_scores_df["rating"] == "Critical"])
    n_high = len(risk_scores_df[risk_scores_df["rating"] == "High"])
    avg_risk = risk_scores_df["overall_score"].mean()
    avg_leakage = np.mean([r["final_leakage_mean"] for r in mc_results.values()]) * 100
    
    eu_exports = anomaly_df[
        (anomaly_df["partner"] == "EU27") &
        (anomaly_df["year"] == anomaly_df["year"].max())
    ]["export_value_usd"].sum()
    
    render_kpi_row([
        {"label": "Countries Assessed", "value": str(len(selected_countries))},
        {"label": "Critical + High Risk", "value": str(n_critical + n_high),
         "delta": f"{n_critical} critical", "delta_color": "inverse"},
        {"label": "Avg Risk Score", "value": f"{avg_risk:.1f}/100"},
        {"label": "Avg Leakage Rate", "value": f"{avg_leakage:.1f}%"},
        {"label": "Total EU Exports (latest)", "value": format_usd(eu_exports)},
    ])
    
    st.markdown("---")
    
    # Two columns: risk ranking + distribution
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Country Risk Ranking")
        fig = plot_country_risk_bars(risk_scores_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Distribution")
        # Pie chart of risk categories
        dist = risk_scores_df["rating"].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=dist.index,
            values=dist.values,
            marker=dict(colors=[RISK_COLORS.get(r, "#666") for r in dist.index]),
            hole=0.45,
            textinfo="label+value",
            textfont=dict(size=13),
        )])
        fig_pie.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor=COLORS["bg"],
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Top risk alerts
        st.subheader("Risk Alerts")
        for _, row in risk_scores_df.head(5).iterrows():
            risk_class = f"risk-{row['rating'].lower()}"
            st.markdown(
                f'<div class="{risk_class}">'
                f'<strong>{row["country"]}</strong> — Score: {row["overall_score"]:.1f} '
                f'({row["rating"]})<br>'
                f'<small>Structural: {row["structural_score"]:.0f} · '
                f'Anomaly: {row["anomaly_score"]:.0f} · '
                f'MC Leakage: {row["mc_leakage_score"]:.0f} · '
                f'Governance: {row["governance_score"]:.0f}</small>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    # Risk heatmap
    st.markdown("---")
    st.subheader("Risk Heatmap: Countries x HS Categories")
    heatmap_df = build_risk_heatmap_data(anomaly_df, selected_countries)
    if not heatmap_df.empty:
        fig_heatmap = plot_risk_heatmap(heatmap_df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No heatmap data available for the selected countries.")


def render_country_tab(risk_scores_df, anomaly_df, trade_df, mc_results,
                       governance_df, selected_countries, scenario, n_simulations):
    """Render the Country Deep-Dive tab."""
    
    selected_country = st.selectbox(
        "Select country for deep-dive:",
        options=selected_countries,
        key="country_deepdive"
    )
    
    if selected_country not in COUNTRIES:
        st.error(f"Country '{selected_country}' not found in configuration.")
        return
    
    cc = COUNTRIES[selected_country]
    country_risk = risk_scores_df[risk_scores_df["country"] == selected_country]
    
    if len(country_risk) == 0:
        st.warning("No risk score data available for this country.")
        return
    
    cr = country_risk.iloc[0]
    
    # Country header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.subheader(f"{selected_country} — Risk Profile")
        st.markdown(
            f"**EPA Group**: {cc.epa_group} · **Region**: {cc.regional_bloc} · "
            f"**EPA Status**: {cc.epa_status} · **EU Access**: {cc.eu_market_access}"
        )
    with col_h2:
        render_risk_badge(cr["rating"])
        st.metric("Overall Score", f"{cr['overall_score']:.1f}/100")
    
    # KPI row
    render_kpi_row([
        {"label": "Structural Risk", "value": f"{cr['structural_score']:.1f}"},
        {"label": "Anomaly Score", "value": f"{cr['anomaly_score']:.1f}"},
        {"label": "MC Leakage Score", "value": f"{cr['mc_leakage_score']:.1f}"},
        {"label": "Governance Gap", "value": f"{cr['governance_score']:.1f}"},
    ])
    
    st.markdown("---")
    
    # Two columns: trade flows + MC forecast
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trade Flow Trends (EU27)")
        fig_trade = plot_trade_flow_timeseries(trade_df, selected_country, partner="EU27")
        st.plotly_chart(fig_trade, use_container_width=True)
    
    with col2:
        st.subheader("Leakage Forecast")
        if selected_country in mc_results:
            fig_mc = plot_monte_carlo_fan(mc_results[selected_country],
                                          title=f"{selected_country} — Leakage Forecast ({scenario.name})")
            st.plotly_chart(fig_mc, use_container_width=True)
    
    # Anomaly details
    st.markdown("---")
    st.subheader("Anomaly Detection Results")
    
    country_anomalies = anomaly_df[
        (anomaly_df["reporter"] == selected_country) &
        (anomaly_df["partner"] == "EU27")
    ].sort_values(["year", "composite_anomaly_score"], ascending=[False, False])
    
    if len(country_anomalies) > 0:
        # Summary by HS category
        latest_year = country_anomalies["year"].max()
        latest = country_anomalies[country_anomalies["year"] == latest_year]
        
        st.markdown(f"**Latest year: {latest_year}** — Showing EU27 export anomalies")
        
        display_cols = ["hs_description", "export_value_usd", "composite_anomaly_score",
                       "risk_level", "spike_flag", "capacity_mismatch_flag", "origin_shift_flag"]
        available = [c for c in display_cols if c in latest.columns]
        
        display_df = latest[available].copy()
        display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )
    
    # Key exports and hotspots
    st.markdown("---")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Key EPA Exports")
        for exp in cc.key_exports_to_eu:
            st.markdown(f"- {exp.replace('_', ' ').title()}")
    
    with col_b:
        st.subheader("High-Risk HS Categories")
        hotspots = COUNTRY_HS_HOTSPOTS.get(selected_country, [])
        for hs_key in hotspots:
            if hs_key in HS_RISK_CATEGORIES:
                hs = HS_RISK_CATEGORIES[hs_key]
                st.markdown(
                    f"- **HS {hs.chapter}** — {hs.description} "
                    f"(Tier {hs.risk_tier}, ~{hs.avg_tariff_arbitrage_pp:.0f}pp arbitrage)"
                )
    
    # Behavioral narrative
    if selected_country in mc_results:
        st.markdown("---")
        st.subheader("Behavioral Forecast Narrative")
        narrative = generate_adaptation_narrative(
            selected_country, scenario, mc_results[selected_country]
        )
        st.markdown(narrative)


def render_comparative_tab(risk_scores_df, anomaly_df, selected_countries):
    """Render the Comparative Analysis tab."""
    
    st.subheader("Cross-Country Comparison")
    
    # Regional comparison
    regional_df = build_regional_comparison(risk_scores_df)
    
    if len(regional_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Regional Risk Profiles")
            fig_radar = plot_regional_radar(regional_df)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            st.markdown("### Regional Summary Table")
            display_cols = ["region", "n_countries", "avg_risk_score", "max_risk_score",
                           "highest_risk_country"]
            st.dataframe(
                regional_df[display_cols],
                use_container_width=True,
                hide_index=True,
            )
    
    # EPA group comparison
    st.markdown("---")
    st.subheader("EPA Group Comparison")
    epa_df = build_epa_group_comparison(risk_scores_df)
    if len(epa_df) > 0:
        st.dataframe(epa_df, use_container_width=True, hide_index=True)
    
    # Spillover corridors
    st.markdown("---")
    st.subheader("Potential Spillover Corridors")
    st.markdown("Countries with correlated anomaly patterns that may indicate coordinated or cascading circumvention.")
    
    spillover_df = identify_spillover_corridors(anomaly_df, selected_countries, min_correlation=0.3)
    if len(spillover_df) > 0:
        st.dataframe(spillover_df, use_container_width=True, hide_index=True)
    else:
        st.info("No significant spillover corridors detected at current threshold.")
    
    # HS vulnerability ranking
    st.markdown("---")
    st.subheader("HS Category Vulnerability Ranking")
    hs_ranking = build_hs_vulnerability_ranking(anomaly_df, selected_countries)
    if len(hs_ranking) > 0:
        display_cols = ["rank", "hs_description", "avg_anomaly_score", "max_anomaly_score",
                       "n_countries_flagged", "avg_tariff_arbitrage"]
        available = [c for c in display_cols if c in hs_ranking.columns]
        st.dataframe(hs_ranking[available], use_container_width=True, hide_index=True)
    
    # Risk heatmap (repeated here for comparison focus)
    st.markdown("---")
    st.subheader("Detailed Risk Heatmap")
    heatmap_year = st.slider(
        "Select year:",
        min_value=2016, max_value=2025, value=2025,
        key="heatmap_year_slider"
    )
    heatmap_df = build_risk_heatmap_data(anomaly_df, selected_countries, year=heatmap_year)
    if not heatmap_df.empty:
        fig = plot_risk_heatmap(heatmap_df, title=f"Risk Heatmap — {heatmap_year}")
        st.plotly_chart(fig, use_container_width=True)


def render_simulation_tab(selected_countries, mc_results, n_simulations):
    """Render the Simulation Lab tab."""
    
    st.subheader("Monte Carlo Simulation Lab")
    st.markdown("Compare scenarios and explore how different policy interventions affect leakage rates.")
    
    # Country selector for detailed simulation
    sim_country = st.selectbox(
        "Select country for scenario analysis:",
        options=selected_countries,
        key="sim_country"
    )
    
    # Run scenario comparison
    st.markdown("### Scenario Comparison")
    
    with st.spinner("Running multi-scenario comparison..."):
        scenario_df = forecast_scenario_comparison(
            sim_country,
            n_simulations=min(n_simulations, 2000),  # Cap for speed
            n_periods=5
        )
    
    if len(scenario_df) > 0:
        # Scenario comparison bar chart
        fig_scenario = plot_scenario_comparison_bars(
            scenario_df,
            title=f"{sim_country} — Scenario Comparison (5-Year Leakage Rate)"
        )
        st.plotly_chart(fig_scenario, use_container_width=True)
        
        # Detailed table
        st.markdown("### Detailed Scenario Metrics")
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)
    
    # Time-series forecasts
    st.markdown("---")
    st.subheader("Time-Series Forecasts by Scenario")
    
    selected_scenarios = st.multiselect(
        "Select scenarios to compare:",
        options=list(PREDEFINED_SCENARIOS.keys()),
        default=["Baseline", "China Rerouting Shock", "Digital Traceability Rollout"],
        key="scenario_multiselect"
    )
    
    if selected_scenarios:
        cols = st.columns(min(3, len(selected_scenarios)))
        for i, scen_name in enumerate(selected_scenarios):
            with cols[i % len(cols)]:
                scen = PREDEFINED_SCENARIOS[scen_name]
                ts_df = forecast_leakage_timeseries(
                    sim_country, scen,
                    n_simulations=min(n_simulations, 1500),
                    n_periods=5
                )
                if len(ts_df) > 0:
                    # Simple line chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ts_df["year_label"], y=ts_df["leakage_mean"],
                        mode="lines+markers",
                        name="Mean Leakage",
                        line=dict(color=COLORS["primary"], width=2),
                        fill="tonexty" if i > 0 else None,
                    ))
                    fig.add_trace(go.Scatter(
                        x=ts_df["year_label"], y=ts_df["leakage_p95"],
                        mode="lines",
                        name="95th pctl",
                        line=dict(color=COLORS["secondary"], width=1, dash="dash"),
                    ))
                    fig.update_layout(
                        title=dict(text=scen_name, font=dict(size=12)),
                        yaxis=dict(title="Leakage (%)", rangemode="tozero"),
                        height=300,
                        margin=dict(l=40, r=20, t=40, b=30),
                        paper_bgcolor=COLORS["bg"],
                        plot_bgcolor=COLORS["bg_alt"],
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Custom scenario builder
    st.markdown("---")
    st.subheader("Custom Scenario Builder")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        custom_rerouting = st.slider("Rerouting Pressure", 0.0, 1.0, 0.0, 0.05, key="custom_rerouting")
        custom_afcfta = st.slider("AfCFTA Liberalization", 0.0, 1.0, 0.3, 0.05, key="custom_afcfta")
    with col2:
        custom_epa = st.slider("EPA Tightening", 0.0, 1.0, 0.0, 0.05, key="custom_epa")
        custom_digital = st.slider("Digital Traceability", 0.0, 1.0, 0.0, 0.05, key="custom_digital")
    with col3:
        custom_harmonization = st.slider("Regional Harmonization", 0.0, 1.0, 0.0, 0.05, key="custom_harmonization")
    
    if st.button("Run Custom Scenario", type="primary", key="run_custom"):
        custom_scenario = ScenarioParams(
            name="Custom Scenario",
            rerouting_pressure=custom_rerouting,
            afcfta_liberalization=custom_afcfta,
            epa_tightening=custom_epa,
            digital_traceability=custom_digital,
            regional_harmonization=custom_harmonization,
        )
        
        with st.spinner("Running custom scenario simulation..."):
            ts_df = forecast_leakage_timeseries(
                sim_country, custom_scenario,
                n_simulations=min(n_simulations, 2000),
                n_periods=5
            )
        
        if len(ts_df) > 0:
            # Create MC results dict for fan chart
            mc_custom = {
                "leakage_mean": (ts_df["leakage_mean"] / 100).tolist(),
                "leakage_p5": (ts_df["leakage_p5"] / 100).tolist(),
                "leakage_p25": (ts_df["leakage_p25"] / 100).tolist(),
                "leakage_p75": (ts_df["leakage_p75"] / 100).tolist(),
                "leakage_p95": (ts_df["leakage_p95"] / 100).tolist(),
            }
            fig_custom = plot_monte_carlo_fan(
                mc_custom,
                title=f"{sim_country} — Custom Scenario Leakage Forecast"
            )
            st.plotly_chart(fig_custom, use_container_width=True)
            
            st.markdown(f"""
**Custom Scenario Results for {sim_country}:**
- Year 5 mean leakage: **{ts_df.iloc[-1]['leakage_mean']:.1f}%** (range: {ts_df.iloc[-1]['leakage_p5']:.1f}%–{ts_df.iloc[-1]['leakage_p95']:.1f}%)
- Year 5 audit rate: **{ts_df.iloc[-1]['audit_mean']:.1f}%**
            """)


def render_policy_tab(risk_scores_df, anomaly_df, mc_results, selected_countries, scenario_name):
    """Render the Policy Menu tab."""
    
    st.subheader("Policy Recommendations")
    
    # Stakeholder selector
    stakeholder = st.selectbox(
        "Generate briefing for:",
        options=["AfCFTA Secretariat", "EU DG Trade"],
        key="stakeholder_select"
    )
    
    # Executive summary
    st.markdown("### Executive Briefing")
    summary = generate_stakeholder_summary(risk_scores_df, stakeholder)
    st.markdown(summary)
    
    st.markdown("---")
    
    # Country-specific recommendations
    st.subheader("Country-Specific Policy Menu")
    
    policy_country = st.selectbox(
        "Select country:",
        options=selected_countries,
        key="policy_country"
    )
    
    country_risk = risk_scores_df[risk_scores_df["country"] == policy_country]
    if len(country_risk) > 0:
        risk_dict = country_risk.iloc[0].to_dict()
        recommendations = generate_country_policy_menu(
            policy_country, risk_dict, anomaly_df
        )
        
        if recommendations:
            # Group by priority
            for priority in [1, 2, 3]:
                priority_recs = [r for r in recommendations if r["priority"] == priority]
                if priority_recs:
                    priority_labels = {1: "Priority 1: Urgent", 2: "Priority 2: Structural", 3: "Priority 3: Medium-Term"}
                    st.markdown(f"#### {priority_labels[priority]}")
                    
                    for rec in priority_recs:
                        with st.expander(f"{rec['category']}: {rec['recommendation'][:80]}...", expanded=priority == 1):
                            st.markdown(f"**Recommendation**: {rec['recommendation']}")
                            st.markdown(f"**Stakeholder**: {rec['stakeholder']}")
                            st.markdown(f"**Impact estimate**: {rec['impact_estimate']}")
            
            # Download recommendations
            st.markdown("---")
            csv_data = export_policy_recommendations_csv(recommendations)
            st.download_button(
                "Download Policy Recommendations (CSV)",
                data=csv_data,
                file_name=f"policy_recommendations_{policy_country.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )
    
    # Regional brief
    st.markdown("---")
    st.subheader("Regional Policy Briefs")
    
    region = st.selectbox(
        "Select region:",
        options=list(REGION_CLUSTERS.keys()),
        key="policy_region"
    )
    
    brief = generate_regional_policy_brief(region, risk_scores_df)
    st.markdown(brief)


def render_data_tab(trade_df, anomaly_df, risk_scores_df, mc_results, eu_cases_df, rerouting_df, scenario_name):
    """Render the Data Explorer tab."""
    
    st.subheader("Data Explorer & Downloads")
    
    data_view = st.selectbox(
        "Select dataset:",
        options=[
            "Risk Scores Summary",
            "Anomaly Detection Results",
            "Trade Flow Data",
            "EU Customs Cases (Calibration)",
            "Rerouting Network",
        ],
        key="data_view"
    )
    
    if data_view == "Risk Scores Summary":
        display_cols = ["rank", "country", "overall_score", "rating",
                       "structural_score", "anomaly_score", "mc_leakage_score", "governance_score"]
        available = [c for c in display_cols if c in risk_scores_df.columns]
        st.dataframe(risk_scores_df[available], use_container_width=True, hide_index=True)
        
        csv = export_risk_scores_csv(risk_scores_df)
        st.download_button("Download Risk Scores (CSV)", csv,
                          "risk_scores.csv", "text/csv")
    
    elif data_view == "Anomaly Detection Results":
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_country = st.multiselect("Country:", anomaly_df["reporter"].unique(),
                                            key="filter_country_data")
        with col2:
            filter_partner = st.selectbox("Partner:", ["All", "EU27", "China", "World"],
                                          key="filter_partner_data")
        with col3:
            filter_risk = st.selectbox("Risk Level:", ["All", "Critical", "High", "Moderate", "Low"],
                                       key="filter_risk_data")
        
        filtered = anomaly_df.copy()
        if filter_country:
            filtered = filtered[filtered["reporter"].isin(filter_country)]
        if filter_partner != "All":
            filtered = filtered[filtered["partner"] == filter_partner]
        if filter_risk != "All":
            filtered = filtered[filtered["risk_level"] == filter_risk]
        
        st.markdown(f"**{len(filtered):,} records**")
        st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
        
        csv = export_anomaly_data_csv(filtered)
        st.download_button("Download Anomaly Data (CSV)", csv,
                          "anomaly_data.csv", "text/csv")
    
    elif data_view == "Trade Flow Data":
        st.dataframe(trade_df.head(500), use_container_width=True, hide_index=True)
        st.download_button("Download Trade Data (CSV)",
                          trade_df.to_csv(index=False).encode("utf-8"),
                          "trade_flows.csv", "text/csv")
    
    elif data_view == "EU Customs Cases (Calibration)":
        st.markdown("Synthetic EU customs infringement cases calibrated to EPPO/OLAF statistics. Used for Monte Carlo parameter calibration.")
        st.dataframe(eu_cases_df, use_container_width=True, hide_index=True)
        
        # Summary chart
        cases_by_year = eu_cases_df.groupby("year").size().reset_index(name="n_cases")
        fig = px.bar(cases_by_year, x="year", y="n_cases",
                    title="EU Customs Cases by Year (Calibration Data)",
                    color_discrete_sequence=[COLORS["primary"]])
        fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg_alt"])
        st.plotly_chart(fig, use_container_width=True)
    
    elif data_view == "Rerouting Network":
        st.markdown("Potential rerouting pathways: Source → Transit (EPA country) → EU destination")
        st.dataframe(rerouting_df, use_container_width=True, hide_index=True)
        
        # Summary by transit country
        transit_summary = rerouting_df.groupby("transit_country").agg(
            n_pathways=("source", "count"),
            avg_probability=("rerouting_probability", "mean"),
            total_estimated_value=("estimated_annual_value_usd", "sum"),
        ).sort_values("avg_probability", ascending=False).reset_index()
        
        st.markdown("### Rerouting Risk by Transit Country")
        st.dataframe(transit_summary, use_container_width=True, hide_index=True)
    
    # Executive summary download
    st.markdown("---")
    st.subheader("Executive Summary Report")
    
    summary_text = generate_executive_summary_text(risk_scores_df, mc_results, scenario_name)
    st.download_button(
        "Download Executive Summary (TXT)",
        data=summary_text.encode("utf-8"),
        file_name=f"executive_summary_{scenario_name.lower().replace(' ', '_').replace(':', '')}.txt",
        mime="text/plain",
    )
    
    # MC simulation summary
    mc_csv = export_simulation_summary_csv(mc_results)
    st.download_button(
        "Download Simulation Results (CSV)",
        data=mc_csv,
        file_name="simulation_results.csv",
        mime="text/csv",
    )


# ─── Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
