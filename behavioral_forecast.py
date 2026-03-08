"""
Composite risk scoring module.

Combines anomaly detection outputs, Monte Carlo simulation results,
governance indicators, and structural vulnerability factors into
per-country and per-HS risk scores.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from config.countries import COUNTRIES, CountryConfig
from config.hs_codes import HS_RISK_CATEGORIES, COUNTRY_HS_HOTSPOTS


def compute_structural_vulnerability(country_name: str) -> Dict[str, float]:
    """
    Compute structural vulnerability factors for a country.
    
    Returns dict with individual factor scores (0-100) and composite.
    """
    if country_name not in COUNTRIES:
        return {"composite": 50.0}
    
    cc = COUNTRIES[country_name]
    
    # Port exposure: higher port volume = more rerouting opportunity
    port_exposure = max(0, (21 - cc.port_volume_rank) / 20) * 100
    
    # Governance gap: lower governance = higher vulnerability
    governance_gap = (1 - cc.governance_proxy) * 100
    
    # Customs weakness
    customs_weakness = (1 - cc.customs_capacity) * 100
    
    # EPA access value: DFQF access = higher target value for circumvention
    epa_value = {"DFQF": 90, "partial": 50, "EBA": 30}.get(cc.eu_market_access, 40)
    
    # Manufacturing complexity: higher manufacturing = more plausible cover for rerouting
    manufacturing_cover = min(100, cc.manufacturing_share_gdp * 5)
    
    # Number of high-risk HS categories
    n_hotspots = len(COUNTRY_HS_HOTSPOTS.get(country_name, []))
    hs_exposure = min(100, n_hotspots * 15)
    
    # Composite (weighted)
    composite = (
        0.20 * port_exposure +
        0.25 * governance_gap +
        0.20 * customs_weakness +
        0.15 * epa_value +
        0.10 * manufacturing_cover +
        0.10 * hs_exposure
    )
    
    return {
        "port_exposure": round(port_exposure, 1),
        "governance_gap": round(governance_gap, 1),
        "customs_weakness": round(customs_weakness, 1),
        "epa_access_value": round(epa_value, 1),
        "manufacturing_cover": round(manufacturing_cover, 1),
        "hs_exposure": round(hs_exposure, 1),
        "composite": round(composite, 1),
    }


def compute_country_risk_score(
    country_name: str,
    anomaly_df: Optional[pd.DataFrame] = None,
    mc_results: Optional[Dict] = None,
    governance_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Compute comprehensive risk score for a country.
    
    Combines:
    - Structural vulnerability (30%)
    - Anomaly detection signals (30%)
    - Monte Carlo leakage estimates (25%)
    - Governance trajectory (15%)
    
    Returns dict with component scores and overall risk rating.
    """
    # Structural vulnerability
    structural = compute_structural_vulnerability(country_name)
    structural_score = structural["composite"]
    
    # Anomaly score (from detection pipeline)
    if anomaly_df is not None and len(anomaly_df) > 0:
        country_anomalies = anomaly_df[anomaly_df["reporter"] == country_name]
        if len(country_anomalies) > 0:
            # Use latest year's average composite anomaly score
            latest_year = country_anomalies["year"].max()
            latest = country_anomalies[country_anomalies["year"] == latest_year]
            anomaly_score = latest["composite_anomaly_score"].mean()
        else:
            anomaly_score = 25.0  # Default moderate
    else:
        anomaly_score = 25.0
    
    # Monte Carlo leakage score
    if mc_results is not None:
        leakage_mean = mc_results.get("final_leakage_mean", 0.1)
        mc_score = min(100, leakage_mean * 400)  # Scale 0-25% leakage to 0-100
    else:
        mc_score = 30.0
    
    # Governance trajectory
    if governance_df is not None and len(governance_df) > 0:
        country_gov = governance_df[governance_df["country"] == country_name]
        if len(country_gov) >= 2:
            latest_ce = country_gov.sort_values("year").iloc[-1]["customs_effectiveness"]
            earliest_ce = country_gov.sort_values("year").iloc[0]["customs_effectiveness"]
            trajectory = latest_ce - earliest_ce  # Positive = improving
            gov_score = max(0, 100 - latest_ce - trajectory * 2)
        else:
            gov_score = 50.0
    else:
        gov_score = (1 - COUNTRIES.get(country_name, CountryConfig(
            name="", iso3="", iso2="", comtrade_code=0, epa_group="", regional_bloc="",
            epa_status="", eu_market_access="", afcfta_status="", port_volume_rank=10,
            governance_proxy=0.5, manufacturing_share_gdp=10.0
        )).governance_proxy) * 100
    
    # Weighted composite
    overall = (
        0.30 * structural_score +
        0.30 * anomaly_score +
        0.25 * mc_score +
        0.15 * gov_score
    )
    
    # Classification
    if overall >= 70:
        rating = "Critical"
        color = "#DC2626"
    elif overall >= 50:
        rating = "High"
        color = "#F59E0B"
    elif overall >= 30:
        rating = "Moderate"
        color = "#3B82F6"
    else:
        rating = "Low"
        color = "#10B981"
    
    return {
        "country": country_name,
        "overall_score": round(overall, 1),
        "rating": rating,
        "color": color,
        "structural_score": round(structural_score, 1),
        "anomaly_score": round(anomaly_score, 1),
        "mc_leakage_score": round(mc_score, 1),
        "governance_score": round(gov_score, 1),
        "structural_detail": structural,
    }


def compute_all_country_scores(
    country_names: List[str],
    anomaly_df: Optional[pd.DataFrame] = None,
    mc_results: Optional[Dict[str, Dict]] = None,
    governance_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute risk scores for all selected countries.
    
    Returns DataFrame sorted by overall risk score (descending).
    """
    scores = []
    for name in country_names:
        mc_country = mc_results.get(name) if mc_results else None
        score = compute_country_risk_score(
            name, anomaly_df, mc_country, governance_df
        )
        scores.append(score)
    
    df = pd.DataFrame(scores)
    df = df.sort_values("overall_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    
    return df
