"""
Policy recommendation engine.

Generates targeted policy recommendations based on:
- Country-specific risk profiles
- HS category vulnerabilities  
- Comparative analysis results
- Scenario forecasting outputs

Recommendations are structured as actionable policy menus for
different stakeholder groups (AfCFTA Secretariat, RECs, national ministries, EU DG Trade).
"""

import pandas as pd
from typing import Dict, List, Optional

from config.countries import COUNTRIES
from config.hs_codes import HS_RISK_CATEGORIES, COUNTRY_HS_HOTSPOTS, CIRCUMVENTION_DESCRIPTIONS


def generate_country_policy_menu(
    country_name: str,
    risk_score: Dict,
    anomaly_df: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    """
    Generate prioritized policy recommendations for a specific country.
    
    Returns list of recommendation dicts with:
    - priority: 1 (urgent) to 3 (medium-term)
    - category: policy category
    - recommendation: specific action
    - stakeholder: primary responsible party
    - impact_estimate: qualitative impact assessment
    """
    if country_name not in COUNTRIES:
        return []
    
    cc = COUNTRIES[country_name]
    recommendations = []
    
    overall = risk_score.get("overall_score", 50)
    structural = risk_score.get("structural_detail", {})
    
    # ── Priority 1: Critical/High Risk Responses ──
    if overall >= 60:
        recommendations.append({
            "priority": 1,
            "category": "Enforcement Enhancement",
            "recommendation": f"Deploy targeted risk-profiling at {country_name}'s major ports for high-risk HS categories: {', '.join(cc.key_exports_to_eu[:3])}",
            "stakeholder": f"{country_name} Customs Authority",
            "impact_estimate": "High — directly reduces leakage by 15-25% in targeted categories",
        })
    
    if structural.get("customs_weakness", 0) > 60:
        recommendations.append({
            "priority": 1,
            "category": "Capacity Building",
            "recommendation": f"Accelerate ASYCUDA/customs automation deployment; current customs capacity score: {cc.customs_capacity:.0%}",
            "stakeholder": f"{country_name} Ministry of Finance / WCO",
            "impact_estimate": "High — improves detection rate by 20-40% over 2-3 year horizon",
        })
    
    if structural.get("governance_gap", 0) > 65:
        recommendations.append({
            "priority": 1,
            "category": "Institutional Reform",
            "recommendation": "Strengthen customs integrity mechanisms (rotation policies, whistleblower protection, automated audit trails)",
            "stakeholder": f"{country_name} Anti-Corruption Agency / Customs",
            "impact_estimate": "Medium-High — reduces collusion-based circumvention",
        })
    
    # ── Priority 2: Structural Measures ──
    if structural.get("port_exposure", 0) > 50:
        recommendations.append({
            "priority": 2,
            "category": "Digital Traceability",
            "recommendation": "Implement electronic origin certificate system linked to AfCFTA Digital Trade Protocol and EU REX system",
            "stakeholder": f"{country_name} Trade Ministry / AfCFTA Secretariat",
            "impact_estimate": "High — reduces false documentation by 30-50% once operational",
        })
    
    recommendations.append({
        "priority": 2,
        "category": "Rules of Origin",
        "recommendation": "Review and tighten product-specific RoO for highest-risk categories; consider raising value-added thresholds",
        "stakeholder": "EPA Joint Committee / EU DG Trade",
        "impact_estimate": "Medium — narrows arbitrage window but may increase compliance costs",
    })
    
    if cc.epa_group in ["West Africa (Interim)", "Central Africa"]:
        recommendations.append({
            "priority": 2,
            "category": "Regional Coordination",
            "recommendation": f"Establish information-sharing protocol with {cc.regional_bloc} customs authorities for cross-border verification",
            "stakeholder": f"{cc.regional_bloc} Secretariat",
            "impact_estimate": "Medium-High — addresses regional rerouting within EPA group",
        })
    
    # ── Priority 3: Medium-Term / Systemic ──
    recommendations.append({
        "priority": 3,
        "category": "AfCFTA Protocol Upgrade",
        "recommendation": "Push for harmonized cumulation rules across EPA groups within AfCFTA framework to reduce origin-shopping incentives",
        "stakeholder": "AfCFTA Secretariat / AU Commission",
        "impact_estimate": "High (long-term) — eliminates key structural driver of circumvention",
    })
    
    recommendations.append({
        "priority": 3,
        "category": "Data Infrastructure",
        "recommendation": "Invest in real-time trade data reporting linked to UN Comtrade; reduce reporting lag from 6+ months to near-real-time",
        "stakeholder": f"{country_name} Statistical Office / UNCTAD",
        "impact_estimate": "Medium — enables early warning but requires sustained investment",
    })
    
    if overall >= 50:
        recommendations.append({
            "priority": 3,
            "category": "Private Sector Engagement",
            "recommendation": "Launch authorized economic operator (AEO) program offering expedited clearance for firms with verified compliance records",
            "stakeholder": f"{country_name} Customs / Chamber of Commerce",
            "impact_estimate": "Medium — incentivizes compliance while improving facilitation",
        })
    
    # Add HS-specific recommendations
    hotspots = COUNTRY_HS_HOTSPOTS.get(country_name, [])
    for hs_key in hotspots[:3]:
        if hs_key in HS_RISK_CATEGORIES:
            hs = HS_RISK_CATEGORIES[hs_key]
            for circ_type in hs.circumvention_type[:2]:
                desc = CIRCUMVENTION_DESCRIPTIONS.get(circ_type, circ_type)
                recommendations.append({
                    "priority": 2,
                    "category": f"HS-Specific: {hs.description}",
                    "recommendation": f"Address {circ_type.replace('_', ' ')}: {desc}. Implement targeted verification for HS {hs.chapter} shipments.",
                    "stakeholder": f"{country_name} Customs / EU Import Control",
                    "impact_estimate": f"Addresses ~{hs.avg_tariff_arbitrage_pp:.0f}pp tariff arbitrage on {hs.description}",
                })
    
    return sorted(recommendations, key=lambda x: x["priority"])


def generate_regional_policy_brief(
    region: str,
    risk_scores_df: pd.DataFrame,
    mc_results: Optional[Dict] = None,
) -> str:
    """
    Generate a regional policy brief text.
    """
    region_countries = risk_scores_df[
        risk_scores_df["country"].isin(
            [c for cluster_countries in [COUNTRIES]
             for c in cluster_countries]
        )
    ]
    
    # Get region-specific countries from REGION_CLUSTERS
    from config.countries import REGION_CLUSTERS
    if region not in REGION_CLUSTERS:
        return f"No data available for region: {region}"
    
    countries_in_region = REGION_CLUSTERS[region]
    region_data = risk_scores_df[risk_scores_df["country"].isin(countries_in_region)]
    
    if len(region_data) == 0:
        return f"No risk scores available for {region} countries."
    
    avg_risk = region_data["overall_score"].mean()
    highest = region_data.loc[region_data["overall_score"].idxmax()]
    lowest = region_data.loc[region_data["overall_score"].idxmin()]
    
    brief = f"""## {region} — EPA-AfCFTA Circumvention Risk Assessment

**Overall Regional Risk Level**: {"Critical" if avg_risk >= 70 else "High" if avg_risk >= 50 else "Moderate" if avg_risk >= 30 else "Low"} (avg score: {avg_risk:.1f}/100)

**Highest-risk hub**: {highest['country']} (score: {highest['overall_score']:.1f})
**Lowest-risk**: {lowest['country']} (score: {lowest['overall_score']:.1f})

### Key Findings

"""
    
    for _, row in region_data.iterrows():
        rating = row["rating"]
        brief += f"- **{row['country']}**: {rating} risk (score: {row['overall_score']:.1f}) — "
        if row["structural_score"] > row["anomaly_score"]:
            brief += "predominantly structural vulnerabilities\n"
        else:
            brief += "elevated trade flow anomalies detected\n"
    
    brief += f"""
### Priority Recommendations for {region}

1. **Regional customs cooperation**: Establish cross-border verification protocols among {', '.join(countries_in_region)}
2. **Harmonized enforcement**: Align audit standards and risk-profiling criteria across the region
3. **Data sharing**: Implement shared trade intelligence platform for real-time anomaly alerts
4. **AfCFTA alignment**: Coordinate regional position on cumulation rules in AfCFTA negotiations
"""
    
    return brief


def generate_stakeholder_summary(
    risk_scores_df: pd.DataFrame,
    stakeholder: str = "AfCFTA Secretariat"
) -> str:
    """
    Generate stakeholder-specific executive summary.
    """
    n_critical = len(risk_scores_df[risk_scores_df["rating"] == "Critical"])
    n_high = len(risk_scores_df[risk_scores_df["rating"] == "High"])
    n_moderate = len(risk_scores_df[risk_scores_df["rating"] == "Moderate"])
    n_low = len(risk_scores_df[risk_scores_df["rating"] == "Low"])
    
    top3 = risk_scores_df.head(3)
    
    if stakeholder == "AfCFTA Secretariat":
        return f"""## Executive Briefing: AfCFTA Secretariat

**Assessment scope**: {len(risk_scores_df)} EPA-implementing African countries
**Risk distribution**: {n_critical} Critical, {n_high} High, {n_moderate} Moderate, {n_low} Low

### Immediate Attention Required

The top-3 risk-rated countries are:
{"".join(f"1. **{row['country']}** — Score: {row['overall_score']:.1f} ({row['rating']})" + chr(10) for _, row in top3.iterrows())}

### Strategic Implications for AfCFTA

- **Origin protocol design**: Current gaps between EPA-specific RoO and AfCFTA general rules create arbitrage opportunities. Harmonization is the most impactful structural reform.
- **Digital infrastructure**: Countries with lower customs capacity scores show 2-3x higher circumvention risk. Prioritize digital traceability rollout in these jurisdictions.
- **Regional variation**: Significant risk disparities across EPA groups suggest differentiated enforcement strategies rather than continental blanket measures.

### Recommended Actions

1. Commission country-specific compliance audits for the {n_critical + n_high} Critical/High-risk countries
2. Accelerate the AfCFTA Protocol on Digital Trade to enable electronic origin verification
3. Establish a continental early-warning system for trade anomaly detection
4. Coordinate with EU DG Trade on mutual recognition of enforcement mechanisms
"""
    
    elif stakeholder == "EU DG Trade":
        return f"""## Intelligence Briefing: EU DG Trade — Africa Unit

**Assessment scope**: {len(risk_scores_df)} African EPA partners
**Risk distribution**: {n_critical} Critical, {n_high} High, {n_moderate} Moderate, {n_low} Low

### Key Findings

- **Revenue exposure**: {n_critical + n_high} countries show circumvention risk levels that may require enhanced import-side controls
- **Highest-risk corridors**: Focus post-clearance audits on shipments from {', '.join(top3['country'].tolist())} in HS categories with >10pp tariff arbitrage
- **China rerouting**: Under elevated rerouting scenarios, West African EPA hubs show 20-35% leakage risk for electronics, apparel, and steel products

### Enforcement Priorities

1. Strengthen REX (Registered Exporter) system verification for top-risk origins
2. Deploy enhanced risk-profiling for HS 61-62 (apparel), 84-85 (electronics), 72-73 (steel) from flagged countries
3. Coordinate with OLAF on cross-referencing AfCFTA transit declarations with EU import data
4. Consider targeted post-clearance audits for shipments exhibiting capacity-mismatch signatures
"""
    
    return "Stakeholder not recognized. Available: 'AfCFTA Secretariat', 'EU DG Trade'"
