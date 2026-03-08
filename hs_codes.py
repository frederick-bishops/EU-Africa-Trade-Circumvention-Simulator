"""
Comparative analysis module for cross-country and cross-regional analysis.

Provides:
- Country ranking tables
- Regional cluster comparisons
- HS-category risk heatmaps
- Spillover risk assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from config.countries import COUNTRIES, REGION_CLUSTERS, EPA_GROUPS
from config.hs_codes import HS_RISK_CATEGORIES, COUNTRY_HS_HOTSPOTS


def build_risk_heatmap_data(
    anomaly_df: pd.DataFrame,
    country_names: List[str],
    year: Optional[int] = None
) -> pd.DataFrame:
    """
    Build a country × HS category risk heatmap matrix.
    
    Returns pivot table with countries as rows, HS categories as columns,
    and composite anomaly scores as values.
    """
    if year is None:
        year = anomaly_df["year"].max()
    
    filtered = anomaly_df[
        (anomaly_df["reporter"].isin(country_names)) &
        (anomaly_df["year"] == year) &
        (anomaly_df["partner"] == "EU27")
    ]
    
    if len(filtered) == 0:
        return pd.DataFrame()
    
    heatmap = filtered.pivot_table(
        index="reporter",
        columns="hs_description",
        values="composite_anomaly_score",
        aggfunc="mean"
    ).fillna(0).round(1)
    
    # Sort by mean risk
    heatmap["_mean"] = heatmap.mean(axis=1)
    heatmap = heatmap.sort_values("_mean", ascending=False)
    heatmap = heatmap.drop(columns=["_mean"])
    
    return heatmap


def build_regional_comparison(
    risk_scores_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate country risk scores into regional cluster comparisons.
    
    Returns DataFrame with one row per region.
    """
    rows = []
    for region, countries in REGION_CLUSTERS.items():
        region_data = risk_scores_df[risk_scores_df["country"].isin(countries)]
        
        if len(region_data) == 0:
            continue
        
        rows.append({
            "region": region,
            "n_countries": len(region_data),
            "avg_risk_score": round(region_data["overall_score"].mean(), 1),
            "max_risk_score": round(region_data["overall_score"].max(), 1),
            "min_risk_score": round(region_data["overall_score"].min(), 1),
            "highest_risk_country": region_data.loc[region_data["overall_score"].idxmax(), "country"],
            "avg_structural": round(region_data["structural_score"].mean(), 1),
            "avg_anomaly": round(region_data["anomaly_score"].mean(), 1),
            "avg_governance": round(region_data["governance_score"].mean(), 1),
        })
    
    df = pd.DataFrame(rows).sort_values("avg_risk_score", ascending=False).reset_index(drop=True)
    return df


def build_epa_group_comparison(
    risk_scores_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare risk profiles across EPA agreement groups.
    """
    rows = []
    for group, countries in EPA_GROUPS.items():
        group_data = risk_scores_df[risk_scores_df["country"].isin(countries)]
        
        if len(group_data) == 0:
            continue
        
        rows.append({
            "epa_group": group,
            "n_countries": len(group_data),
            "avg_risk_score": round(group_data["overall_score"].mean(), 1),
            "max_risk_score": round(group_data["overall_score"].max(), 1),
            "risk_spread": round(group_data["overall_score"].max() - group_data["overall_score"].min(), 1),
            "avg_leakage_score": round(group_data["mc_leakage_score"].mean(), 1),
            "countries": ", ".join(group_data.sort_values("overall_score", ascending=False)["country"].tolist()),
        })
    
    return pd.DataFrame(rows).sort_values("avg_risk_score", ascending=False).reset_index(drop=True)


def identify_spillover_corridors(
    anomaly_df: pd.DataFrame,
    country_names: List[str],
    min_correlation: float = 0.5
) -> pd.DataFrame:
    """
    Identify potential spillover corridors between countries.
    
    Looks for correlated anomaly patterns between neighboring countries
    that suggest coordinated or cascading circumvention.
    """
    # Get per-country-year anomaly scores (EU exports only)
    eu_data = anomaly_df[
        (anomaly_df["partner"] == "EU27") &
        (anomaly_df["reporter"].isin(country_names))
    ]
    
    pivot = eu_data.pivot_table(
        index="year",
        columns="reporter",
        values="composite_anomaly_score",
        aggfunc="mean"
    ).fillna(0)
    
    if pivot.shape[1] < 2:
        return pd.DataFrame()
    
    # Compute correlation matrix
    corr_matrix = pivot.corr()
    
    # Extract significant correlations
    rows = []
    seen = set()
    for c1 in corr_matrix.columns:
        for c2 in corr_matrix.columns:
            if c1 >= c2:
                continue
            pair = tuple(sorted([c1, c2]))
            if pair in seen:
                continue
            seen.add(pair)
            
            corr_val = corr_matrix.loc[c1, c2]
            if abs(corr_val) >= min_correlation:
                # Check if they're in the same region
                same_region = any(
                    c1 in countries and c2 in countries
                    for countries in REGION_CLUSTERS.values()
                )
                
                rows.append({
                    "country_1": c1,
                    "country_2": c2,
                    "correlation": round(corr_val, 3),
                    "same_region": same_region,
                    "spillover_risk": "High" if corr_val > 0.7 else "Moderate",
                    "interpretation": (
                        f"Anomaly patterns in {c1} and {c2} are "
                        f"{'strongly' if corr_val > 0.7 else 'moderately'} correlated, "
                        f"suggesting {'regional contagion' if same_region else 'coordinated circumvention'} risk"
                    ),
                })
    
    return pd.DataFrame(rows).sort_values("correlation", ascending=False).reset_index(drop=True)


def build_hs_vulnerability_ranking(
    anomaly_df: pd.DataFrame,
    country_names: List[str]
) -> pd.DataFrame:
    """
    Rank HS categories by aggregate vulnerability across selected countries.
    """
    eu_data = anomaly_df[
        (anomaly_df["partner"] == "EU27") &
        (anomaly_df["reporter"].isin(country_names))
    ]
    
    latest_year = eu_data["year"].max()
    latest = eu_data[eu_data["year"] == latest_year]
    
    ranking = latest.groupby(["hs_category", "hs_description"]).agg(
        avg_anomaly_score=("composite_anomaly_score", "mean"),
        max_anomaly_score=("composite_anomaly_score", "max"),
        n_countries_flagged=("spike_flag", "sum"),
        total_export_value=("export_value_usd", "sum"),
        avg_tariff_arbitrage=("tariff_arbitrage_pp", "first"),
    ).reset_index()
    
    ranking = ranking.sort_values("avg_anomaly_score", ascending=False).reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking) + 1)
    
    return ranking
