"""
Behavioral forecasting module.

Forecasts adaptive behaviors of firms and states in response to
policy changes, enforcement shifts, and external shocks.

Uses game-theoretic framework where:
- Firms optimize: max(profit from circumvention) subject to detection risk
- States optimize: max(enforcement effectiveness) subject to capacity constraints
- Both adapt over time based on observed outcomes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from engine.monte_carlo import (
    FirmParams, StateParams, ScenarioParams,
    run_monte_carlo_simulation, PREDEFINED_SCENARIOS
)


def forecast_scenario_comparison(
    country_name: str,
    scenarios: List[ScenarioParams] = None,
    n_simulations: int = 3000,
    n_periods: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run multiple scenarios for a single country and return comparison table.
    
    Returns DataFrame with scenario names, key metrics, and confidence intervals.
    """
    from config.countries import COUNTRIES
    from config.hs_codes import HS_RISK_CATEGORIES, COUNTRY_HS_HOTSPOTS
    
    if scenarios is None:
        scenarios = list(PREDEFINED_SCENARIOS.values())
    
    if country_name not in COUNTRIES:
        return pd.DataFrame()
    
    cc = COUNTRIES[country_name]
    hotspots = COUNTRY_HS_HOTSPOTS.get(country_name, [])
    
    avg_arbitrage = np.mean([
        HS_RISK_CATEGORIES[hs].avg_tariff_arbitrage_pp
        for hs in hotspots if hs in HS_RISK_CATEGORIES
    ]) if hotspots else 6.0
    
    fp = FirmParams(
        tariff_arbitrage_pp=avg_arbitrage,
        rerouting_cost_pct=max(1.5, 5.0 - cc.port_volume_rank * 0.15),
        detection_probability=0.10 + cc.customs_capacity * 0.15,
        penalty_multiplier=2.0 + cc.governance_proxy * 2.0,
        risk_aversion=1.0 + cc.governance_proxy * 1.0,
        compliance_baseline=0.4 + cc.governance_proxy * 0.3,
    )
    
    sp = StateParams(
        governance_score=cc.governance_proxy,
        customs_capacity=cc.customs_capacity,
        enforcement_budget=0.2 + cc.governance_proxy * 0.3,
        political_will=0.3 + cc.governance_proxy * 0.4,
        tech_adoption=cc.customs_capacity * 0.6,
        max_audit_rate=0.05 + cc.customs_capacity * 0.2,
    )
    
    rows = []
    for i, scenario in enumerate(scenarios):
        results = run_monte_carlo_simulation(
            fp, sp, scenario,
            n_simulations=n_simulations,
            n_periods=n_periods,
            seed=seed + i * 1000,
        )
        
        rows.append({
            "scenario": scenario.name,
            "leakage_mean_pct": round(results["final_leakage_mean"] * 100, 1),
            "leakage_ci_low_pct": round(results["final_leakage_ci_90"][0] * 100, 1),
            "leakage_ci_high_pct": round(results["final_leakage_ci_90"][1] * 100, 1),
            "circumvention_mean_pct": round(results["final_circumvention_mean"] * 100, 1),
            "audit_rate_final_pct": round(results["audit_mean"][-1] * 100, 1),
            "detection_rate_final_pct": round(results["detected_mean"][-1] * 100, 1),
        })
    
    return pd.DataFrame(rows)


def forecast_leakage_timeseries(
    country_name: str,
    scenario: ScenarioParams,
    n_simulations: int = 3000,
    n_periods: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate time-series forecast of leakage with confidence intervals.
    
    Returns DataFrame with columns: period, mean, p5, p25, p50, p75, p95
    """
    from config.countries import COUNTRIES
    from config.hs_codes import HS_RISK_CATEGORIES, COUNTRY_HS_HOTSPOTS
    
    if country_name not in COUNTRIES:
        return pd.DataFrame()
    
    cc = COUNTRIES[country_name]
    hotspots = COUNTRY_HS_HOTSPOTS.get(country_name, [])
    
    avg_arbitrage = np.mean([
        HS_RISK_CATEGORIES[hs].avg_tariff_arbitrage_pp
        for hs in hotspots if hs in HS_RISK_CATEGORIES
    ]) if hotspots else 6.0
    
    fp = FirmParams(
        tariff_arbitrage_pp=avg_arbitrage,
        rerouting_cost_pct=max(1.5, 5.0 - cc.port_volume_rank * 0.15),
        detection_probability=0.10 + cc.customs_capacity * 0.15,
        penalty_multiplier=2.0 + cc.governance_proxy * 2.0,
        risk_aversion=1.0 + cc.governance_proxy * 1.0,
        compliance_baseline=0.4 + cc.governance_proxy * 0.3,
    )
    
    sp = StateParams(
        governance_score=cc.governance_proxy,
        customs_capacity=cc.customs_capacity,
        enforcement_budget=0.2 + cc.governance_proxy * 0.3,
        political_will=0.3 + cc.governance_proxy * 0.4,
        tech_adoption=cc.customs_capacity * 0.6,
        max_audit_rate=0.05 + cc.customs_capacity * 0.2,
    )
    
    results = run_monte_carlo_simulation(
        fp, sp, scenario,
        n_simulations=n_simulations,
        n_periods=n_periods,
        seed=seed,
    )
    
    rows = []
    for t in range(n_periods):
        rows.append({
            "period": t + 1,
            "year_label": f"Year {t + 1}",
            "leakage_mean": round(results["leakage_mean"][t] * 100, 2),
            "leakage_p5": round(results["leakage_p5"][t] * 100, 2),
            "leakage_p25": round(results["leakage_p25"][t] * 100, 2),
            "leakage_p50": round(results["leakage_p50"][t] * 100, 2),
            "leakage_p75": round(results["leakage_p75"][t] * 100, 2),
            "leakage_p95": round(results["leakage_p95"][t] * 100, 2),
            "circumvention_mean": round(results["circumvention_mean"][t] * 100, 2),
            "audit_mean": round(results["audit_mean"][t] * 100, 2),
        })
    
    return pd.DataFrame(rows)


def generate_adaptation_narrative(
    country_name: str,
    scenario: ScenarioParams,
    mc_results: Dict,
) -> str:
    """
    Generate a human-readable narrative describing the behavioral forecast.
    """
    leakage_mean = mc_results["final_leakage_mean"] * 100
    circ_mean = mc_results["final_circumvention_mean"] * 100
    ci_low = mc_results["final_leakage_ci_90"][0] * 100
    ci_high = mc_results["final_leakage_ci_90"][1] * 100
    
    # Trend direction
    leakage_series = mc_results["leakage_mean"]
    if leakage_series[-1] > leakage_series[0]:
        trend = "increasing"
        trend_desc = "worsening"
    elif leakage_series[-1] < leakage_series[0] * 0.9:
        trend = "decreasing"
        trend_desc = "improving"
    else:
        trend = "stable"
        trend_desc = "stable"
    
    narrative = f"""**{country_name} — {scenario.name}**

Under this scenario, the model forecasts a **{trend}** leakage trajectory over the 5-year horizon.

- **Expected leakage rate**: {leakage_mean:.1f}% of EPA-eligible trade (90% CI: {ci_low:.1f}%–{ci_high:.1f}%)
- **Circumvention attempt rate**: {circ_mean:.1f}% of firms
- **Trend**: {trend_desc.capitalize()} — leakage moved from {leakage_series[0]*100:.1f}% (Year 1) to {leakage_series[-1]*100:.1f}% (Year 5)

"""
    
    if scenario.rerouting_pressure > 0.3:
        narrative += "⚠️ **External rerouting pressure** (e.g., Chinese supply chain shifts) significantly elevates circumvention incentives, particularly for electronics, apparel, and steel products routed through port hubs.\n\n"
    
    if scenario.digital_traceability > 0.5:
        narrative += "✅ **Digital traceability** deployment substantially reduces undetected leakage by improving real-time origin verification and cross-border data sharing.\n\n"
    
    if scenario.epa_tightening > 0.4:
        narrative += "🔒 **EU enforcement tightening** raises effective detection rates but may reduce EPA utilization by compliant firms due to increased compliance costs.\n\n"
    
    if scenario.regional_harmonization > 0.5:
        narrative += "🤝 **Regional RoO harmonization** under AfCFTA reduces arbitrage opportunities by closing origin-shopping gaps between EPA groups.\n\n"
    
    return narrative
