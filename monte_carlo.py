"""
Anomaly detection module for identifying circumvention indicators.

Implements multiple detection methods:
1. Z-score based export spike detection
2. Capacity mismatch detection (exports vs. production capacity)
3. Origin shift detection (partner trade pattern changes)
4. Import-export ratio anomalies (transshipment indicators)

Each method returns anomaly scores and flagged observations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional


def detect_export_spikes(
    df: pd.DataFrame,
    z_threshold: float = 2.0,
    min_history_years: int = 3
) -> pd.DataFrame:
    """
    Detect abnormal export spikes using rolling z-score method.
    
    For each country-HS-partner combination, computes rolling mean/std
    and flags years where exports exceed z_threshold standard deviations.
    
    Args:
        df: Trade flow DataFrame with columns [reporter, partner, hs_category, year, export_value_usd]
        z_threshold: Number of standard deviations for flagging
        min_history_years: Minimum years of history required
    
    Returns:
        DataFrame with added columns: spike_zscore, spike_flag, spike_magnitude
    """
    result = df.copy()
    result["spike_zscore"] = 0.0
    result["spike_flag"] = False
    result["spike_magnitude"] = 0.0
    
    groups = result.groupby(["reporter", "hs_category", "partner"])
    
    for name, group in groups:
        if len(group) < min_history_years:
            continue
        
        idx = group.index
        values = group["export_value_usd"].values
        
        # Rolling statistics (expanding window for robustness)
        for i in range(min_history_years, len(values)):
            window = values[:i]
            mean_val = np.mean(window)
            std_val = np.std(window)
            
            if std_val > 0 and mean_val > 0:
                z = (values[i] - mean_val) / std_val
                result.loc[idx[i], "spike_zscore"] = round(z, 2)
                result.loc[idx[i], "spike_flag"] = z > z_threshold
                result.loc[idx[i], "spike_magnitude"] = round(
                    (values[i] - mean_val) / mean_val * 100, 1
                ) if mean_val > 0 else 0.0
    
    return result


def detect_capacity_mismatch(
    df: pd.DataFrame,
    mismatch_threshold: float = 1.5
) -> pd.DataFrame:
    """
    Detect when export volumes exceed plausible domestic production capacity.
    
    Uses the production_capacity_index (0-1) as a proxy for maximum
    sustainable export volume relative to historical baseline.
    
    Args:
        df: Trade flow DataFrame
        mismatch_threshold: Ratio above which exports are flagged as exceeding capacity
    
    Returns:
        DataFrame with added columns: capacity_ratio, capacity_mismatch_flag
    """
    result = df.copy()
    
    # Compute baseline export level per country-HS (median of first 3 years)
    baseline = (
        df[df["year"] <= df["year"].min() + 2]
        .groupby(["reporter", "hs_category"])["export_value_usd"]
        .median()
        .reset_index()
        .rename(columns={"export_value_usd": "baseline_export"})
    )
    
    result = result.merge(baseline, on=["reporter", "hs_category"], how="left")
    result["baseline_export"] = result["baseline_export"].fillna(result["export_value_usd"])
    
    # Capacity-adjusted maximum = baseline * (1 + capacity_index * growth_factor)
    # Higher capacity index → more plausible growth
    growth_factor = 3.0  # Maximum plausible growth multiplier
    result["max_plausible_export"] = result["baseline_export"] * (
        1 + result["production_capacity_index"] * growth_factor
    )
    
    result["capacity_ratio"] = np.where(
        result["max_plausible_export"] > 0,
        result["export_value_usd"] / result["max_plausible_export"],
        0
    )
    result["capacity_ratio"] = result["capacity_ratio"].round(2)
    result["capacity_mismatch_flag"] = result["capacity_ratio"] > mismatch_threshold
    
    # Clean up
    result.drop(columns=["baseline_export", "max_plausible_export"], inplace=True)
    
    return result


def detect_origin_shifts(
    df: pd.DataFrame,
    shift_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Detect suspicious origin shifts: when imports from China surge
    while EPA exports to EU also surge (potential transshipment).
    
    Computes year-over-year correlation between China imports and EU exports
    at the country-HS level.
    
    Args:
        df: Trade flow DataFrame
        shift_threshold: Correlation threshold for flagging
    
    Returns:
        DataFrame with origin_shift_score and origin_shift_flag per country-HS-year
    """
    result = df.copy()
    result["origin_shift_score"] = 0.0
    result["origin_shift_flag"] = False
    
    # Pivot to get China imports and EU exports side by side
    for country in df["reporter"].unique():
        for hs in df["hs_category"].unique():
            mask = (df["reporter"] == country) & (df["hs_category"] == hs)
            
            china_imports = (
                df[mask & (df["partner"] == "China")]
                .set_index("year")["import_value_usd"]
            )
            eu_exports = (
                df[mask & (df["partner"] == "EU27")]
                .set_index("year")["export_value_usd"]
            )
            
            if len(china_imports) < 3 or len(eu_exports) < 3:
                continue
            
            # Align on years
            common_years = china_imports.index.intersection(eu_exports.index)
            if len(common_years) < 3:
                continue
            
            ci = china_imports.loc[common_years]
            ee = eu_exports.loc[common_years]
            
            # YoY growth rates
            ci_growth = ci.pct_change().dropna()
            ee_growth = ee.pct_change().dropna()
            
            common = ci_growth.index.intersection(ee_growth.index)
            if len(common) < 2:
                continue
            
            # Correlation between import growth and export growth
            corr, _ = stats.pearsonr(ci_growth.loc[common], ee_growth.loc[common])
            
            if not np.isnan(corr):
                score = max(0, corr)  # Only positive correlation is suspicious
                
                # Flag years where both grew significantly
                for year in common:
                    if ci_growth.loc[year] > 0.2 and ee_growth.loc[year] > 0.2:
                        idx_mask = (
                            (result["reporter"] == country) &
                            (result["hs_category"] == hs) &
                            (result["year"] == year)
                        )
                        result.loc[idx_mask, "origin_shift_score"] = round(score, 3)
                        result.loc[idx_mask, "origin_shift_flag"] = score > shift_threshold
    
    return result


def detect_import_export_ratio_anomalies(
    df: pd.DataFrame,
    ratio_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Detect transshipment indicators: when imports from outside EPA
    are closely followed by similar-volume exports to EU.
    
    High import-to-export ratio with minimal value addition suggests
    pass-through/transshipment.
    
    Returns:
        DataFrame with ie_ratio, ie_anomaly_flag columns
    """
    result = df.copy()
    
    # Get China imports and EU exports per country-HS-year
    pivot = df.pivot_table(
        index=["reporter", "hs_category", "year"],
        columns="partner",
        values=["import_value_usd", "export_value_usd"],
        aggfunc="sum"
    ).fillna(0)
    
    # Flatten column names
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    
    # Import-to-export ratio: China imports / EU exports
    china_imp_col = "import_value_usd_China"
    eu_exp_col = "export_value_usd_EU27"
    
    if china_imp_col in pivot.columns and eu_exp_col in pivot.columns:
        pivot["ie_ratio"] = np.where(
            pivot[eu_exp_col] > 0,
            pivot[china_imp_col] / pivot[eu_exp_col],
            0
        )
        pivot["ie_ratio"] = pivot["ie_ratio"].round(3)
        pivot["ie_anomaly_flag"] = pivot["ie_ratio"] > ratio_threshold
        
        # Merge back
        merge_cols = ["reporter", "hs_category", "year", "ie_ratio", "ie_anomaly_flag"]
        result = result.merge(
            pivot[merge_cols],
            on=["reporter", "hs_category", "year"],
            how="left"
        )
    else:
        result["ie_ratio"] = 0.0
        result["ie_anomaly_flag"] = False
    
    result["ie_ratio"] = result["ie_ratio"].fillna(0.0)
    result["ie_anomaly_flag"] = result["ie_anomaly_flag"].fillna(False)
    
    return result


def compute_composite_anomaly_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite anomaly score (0-100) from all detection methods.
    
    Weights:
    - Export spike: 30%
    - Capacity mismatch: 25%
    - Origin shift: 25%
    - Import-export ratio: 20%
    """
    result = df.copy()
    
    # Normalize each indicator to 0-1
    def safe_normalize(series):
        s = series.fillna(0).clip(lower=0)
        max_val = s.quantile(0.99) if s.max() > 0 else 1.0
        return (s / max_val).clip(0, 1) if max_val > 0 else s * 0
    
    spike_score = safe_normalize(result.get("spike_zscore", pd.Series(0, index=result.index)).abs())
    capacity_score = safe_normalize(result.get("capacity_ratio", pd.Series(0, index=result.index)))
    origin_score = result.get("origin_shift_score", pd.Series(0, index=result.index)).fillna(0).clip(0, 1)
    ie_score = safe_normalize(result.get("ie_ratio", pd.Series(0, index=result.index)))
    
    # Weighted composite
    result["composite_anomaly_score"] = (
        0.30 * spike_score +
        0.25 * capacity_score +
        0.25 * origin_score +
        0.20 * ie_score
    ) * 100
    
    result["composite_anomaly_score"] = result["composite_anomaly_score"].round(1).clip(0, 100)
    
    # Risk classification
    result["risk_level"] = pd.cut(
        result["composite_anomaly_score"],
        bins=[0, 25, 50, 75, 100],
        labels=["Low", "Moderate", "High", "Critical"],
        include_lowest=True
    )
    
    return result


def run_full_anomaly_pipeline(
    df: pd.DataFrame,
    z_threshold: float = 2.0,
    capacity_threshold: float = 1.5,
    origin_shift_threshold: float = 0.5,
    ie_ratio_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Run the complete anomaly detection pipeline.
    
    Args:
        df: Raw trade flow DataFrame
        z_threshold: Z-score threshold for export spikes
        capacity_threshold: Threshold for capacity mismatch
        origin_shift_threshold: Correlation threshold for origin shifts
        ie_ratio_threshold: Import/export ratio threshold
    
    Returns:
        DataFrame with all anomaly indicators and composite score
    """
    # Step 1: Export spike detection
    df = detect_export_spikes(df, z_threshold=z_threshold)
    
    # Step 2: Capacity mismatch
    df = detect_capacity_mismatch(df, mismatch_threshold=capacity_threshold)
    
    # Step 3: Origin shifts
    df = detect_origin_shifts(df, shift_threshold=origin_shift_threshold)
    
    # Step 4: Import-export ratio anomalies
    df = detect_import_export_ratio_anomalies(df, ratio_threshold=ie_ratio_threshold)
    
    # Step 5: Composite score
    df = compute_composite_anomaly_score(df)
    
    return df
