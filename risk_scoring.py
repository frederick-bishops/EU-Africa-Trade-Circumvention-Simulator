"""
Monte Carlo simulation engine for firm-level and state-level behavioral modeling.

Models two interacting agents:
1. Firms: Make cost-driven rerouting/circumvention decisions based on tariff arbitrage,
   detection probability, and penalty costs. Calibrated to EU customs fraud case statistics.
2. States: Adjust audit intensity based on governance capacity, political will,
   and resource constraints.

The simulation generates probability distributions for leakage rates under
different scenarios, enabling confidence-interval-based risk assessment.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class FirmParams:
    """Parameters for firm-level behavioral model."""
    # Tariff arbitrage: percentage point difference EPA vs MFN
    tariff_arbitrage_pp: float = 8.0
    # Cost of rerouting as % of shipment value
    rerouting_cost_pct: float = 3.0
    # Probability of detection per shipment (0-1)
    detection_probability: float = 0.15
    # Penalty if caught: multiple of evaded duty
    penalty_multiplier: float = 3.0
    # Firm risk aversion (higher = more conservative)
    risk_aversion: float = 1.5
    # Fraction of firms that are "compliant by default"
    compliance_baseline: float = 0.6
    # Learning rate: how fast firms adapt to enforcement changes
    adaptation_rate: float = 0.1


@dataclass
class StateParams:
    """Parameters for state-level enforcement model."""
    # Governance effectiveness score (0-1)
    governance_score: float = 0.5
    # Customs capacity score (0-1)
    customs_capacity: float = 0.5
    # Budget allocation for trade enforcement (relative 0-1)
    enforcement_budget: float = 0.3
    # Political will factor (0-1): willingness to enforce against economic interests
    political_will: float = 0.5
    # Technology adoption (digital traceability, ASYCUDA etc.)
    tech_adoption: float = 0.3
    # Maximum feasible audit rate (% of shipments)
    max_audit_rate: float = 0.25


@dataclass
class ScenarioParams:
    """Parameters defining a specific scenario."""
    name: str = "Baseline"
    # External shock: sudden increase in rerouting pressure (0-1)
    rerouting_pressure: float = 0.0
    # AfCFTA liberalization impact: additional intra-African flow (0-1)
    afcfta_liberalization: float = 0.3
    # EPA tightening: stricter RoO enforcement by EU (0-1)
    epa_tightening: float = 0.0
    # Digital traceability boost (0-1)
    digital_traceability: float = 0.0
    # Regional harmonization of RoO (0-1)
    regional_harmonization: float = 0.0


def compute_firm_circumvention_probability(
    firm_params: FirmParams,
    state_params: StateParams,
    scenario: ScenarioParams,
    rng: np.random.RandomState
) -> float:
    """
    Compute probability that a representative firm attempts circumvention.
    
    Based on expected utility framework:
    - E[gain from circumvention] = tariff_saving - rerouting_cost
    - E[loss if caught] = detection_prob * penalty
    - Firm circumvents if E[gain] > risk_aversion * E[loss]
    
    Returns probability (0-1) of circumvention attempt.
    """
    # Effective tariff saving
    tariff_saving = firm_params.tariff_arbitrage_pp / 100
    rerouting_cost = firm_params.rerouting_cost_pct / 100
    net_gain = tariff_saving - rerouting_cost
    
    if net_gain <= 0:
        return firm_params.compliance_baseline * 0.02  # Minimal residual fraud
    
    # Effective detection probability (modified by state capacity and scenario)
    base_detection = firm_params.detection_probability
    
    # State enforcement modifies detection
    enforcement_factor = (
        state_params.governance_score * 0.3 +
        state_params.customs_capacity * 0.3 +
        state_params.enforcement_budget * 0.2 +
        state_params.tech_adoption * 0.2
    )
    
    # Scenario adjustments
    scenario_boost = (
        scenario.epa_tightening * 0.15 +
        scenario.digital_traceability * 0.20 +
        scenario.regional_harmonization * 0.10
    )
    
    effective_detection = np.clip(
        base_detection * enforcement_factor * 2 + scenario_boost,
        0.02, 0.80
    )
    
    # Expected loss if caught
    expected_loss = effective_detection * firm_params.penalty_multiplier * tariff_saving
    
    # Circumvention probability via logistic function
    decision_score = (net_gain - firm_params.risk_aversion * expected_loss) / tariff_saving
    
    # Add rerouting pressure from external shocks (e.g., China supply chain shifts)
    decision_score += scenario.rerouting_pressure * 0.3
    
    # AfCFTA effect: more intra-African trade creates more circumvention opportunities
    decision_score += scenario.afcfta_liberalization * 0.15
    
    # Logistic transformation to probability
    circumvention_prob = 1 / (1 + np.exp(-5 * decision_score))
    
    # Adjust for baseline compliance
    circumvention_prob = (1 - firm_params.compliance_baseline) * circumvention_prob
    
    # Add noise
    noise = rng.normal(0, 0.03)
    circumvention_prob = np.clip(circumvention_prob + noise, 0.01, 0.95)
    
    return circumvention_prob


def compute_state_audit_intensity(
    state_params: StateParams,
    scenario: ScenarioParams,
    observed_circumvention: float,
    rng: np.random.RandomState
) -> float:
    """
    Compute state's audit intensity response.
    
    States adjust audit rates based on:
    - Observed circumvention signals
    - Available capacity
    - Political and budgetary constraints
    
    Returns: effective audit rate (0-1)
    """
    # Base capacity
    capacity = (
        state_params.customs_capacity * 0.4 +
        state_params.governance_score * 0.3 +
        state_params.enforcement_budget * 0.3
    )
    
    # Response to observed circumvention (bounded by capacity)
    response_intensity = min(
        capacity * state_params.max_audit_rate * 2,
        observed_circumvention * state_params.political_will * 2
    )
    
    # Scenario boosts
    scenario_boost = (
        scenario.epa_tightening * 0.1 +
        scenario.digital_traceability * 0.15 +
        scenario.regional_harmonization * 0.05
    )
    
    audit_rate = np.clip(
        response_intensity + scenario_boost + rng.normal(0, 0.02),
        0.01, state_params.max_audit_rate
    )
    
    return audit_rate


def run_monte_carlo_simulation(
    firm_params: FirmParams,
    state_params: StateParams,
    scenario: ScenarioParams,
    n_simulations: int = 5000,
    n_periods: int = 5,
    seed: int = 42
) -> Dict:
    """
    Run Monte Carlo simulation of firm-state interaction over multiple periods.
    
    Generates distribution of:
    - Circumvention rates per period
    - Leakage values (% of trade volume)
    - State enforcement response
    - Adaptation dynamics
    
    Args:
        firm_params: Firm behavioral parameters
        state_params: State capacity parameters
        scenario: Scenario configuration
        n_simulations: Number of Monte Carlo iterations
        n_periods: Forward-looking periods (years)
        seed: Random seed
    
    Returns:
        Dict with simulation results and statistics
    """
    rng = np.random.RandomState(seed)
    
    # Storage arrays
    circumvention_rates = np.zeros((n_simulations, n_periods))
    leakage_rates = np.zeros((n_simulations, n_periods))
    audit_rates = np.zeros((n_simulations, n_periods))
    detected_rates = np.zeros((n_simulations, n_periods))
    
    for sim in range(n_simulations):
        sim_rng = np.random.RandomState(seed + sim)
        
        # Firm parameters with simulation-specific noise
        sim_firm = FirmParams(
            tariff_arbitrage_pp=firm_params.tariff_arbitrage_pp * sim_rng.lognormal(0, 0.1),
            rerouting_cost_pct=firm_params.rerouting_cost_pct * sim_rng.lognormal(0, 0.15),
            detection_probability=firm_params.detection_probability,
            penalty_multiplier=firm_params.penalty_multiplier * sim_rng.lognormal(0, 0.1),
            risk_aversion=firm_params.risk_aversion * sim_rng.lognormal(0, 0.1),
            compliance_baseline=np.clip(firm_params.compliance_baseline + sim_rng.normal(0, 0.05), 0.3, 0.9),
            adaptation_rate=firm_params.adaptation_rate,
        )
        
        prev_circumvention = 0.1  # Initial estimate
        
        for t in range(n_periods):
            # Firm decision
            circ_prob = compute_firm_circumvention_probability(
                sim_firm, state_params, scenario, sim_rng
            )
            
            # Adaptation: firms learn from previous period's enforcement
            if t > 0:
                enforcement_signal = audit_rates[sim, t-1]
                circ_prob *= (1 - sim_firm.adaptation_rate * enforcement_signal * 2)
                circ_prob += scenario.rerouting_pressure * sim_firm.adaptation_rate * (t / n_periods)
                circ_prob = np.clip(circ_prob, 0.01, 0.95)
            
            circumvention_rates[sim, t] = circ_prob
            
            # State response
            audit_rate = compute_state_audit_intensity(
                state_params, scenario, prev_circumvention, sim_rng
            )
            audit_rates[sim, t] = audit_rate
            
            # Detection
            detected = circ_prob * audit_rate * sim_rng.uniform(0.5, 1.5)
            detected_rates[sim, t] = np.clip(detected, 0, circ_prob)
            
            # Leakage = circumvention that escapes detection
            leakage = circ_prob - detected_rates[sim, t]
            leakage_rates[sim, t] = np.clip(leakage, 0, 1)
            
            prev_circumvention = circ_prob
    
    # Compute statistics
    results = {
        "scenario_name": scenario.name,
        "n_simulations": n_simulations,
        "n_periods": n_periods,
        # Per-period statistics
        "circumvention_mean": circumvention_rates.mean(axis=0).tolist(),
        "circumvention_p5": np.percentile(circumvention_rates, 5, axis=0).tolist(),
        "circumvention_p25": np.percentile(circumvention_rates, 25, axis=0).tolist(),
        "circumvention_p50": np.percentile(circumvention_rates, 50, axis=0).tolist(),
        "circumvention_p75": np.percentile(circumvention_rates, 75, axis=0).tolist(),
        "circumvention_p95": np.percentile(circumvention_rates, 95, axis=0).tolist(),
        "leakage_mean": leakage_rates.mean(axis=0).tolist(),
        "leakage_p5": np.percentile(leakage_rates, 5, axis=0).tolist(),
        "leakage_p25": np.percentile(leakage_rates, 25, axis=0).tolist(),
        "leakage_p50": np.percentile(leakage_rates, 50, axis=0).tolist(),
        "leakage_p75": np.percentile(leakage_rates, 75, axis=0).tolist(),
        "leakage_p95": np.percentile(leakage_rates, 95, axis=0).tolist(),
        "audit_mean": audit_rates.mean(axis=0).tolist(),
        "detected_mean": detected_rates.mean(axis=0).tolist(),
        # Overall summary
        "final_leakage_mean": float(leakage_rates[:, -1].mean()),
        "final_leakage_ci_90": (
            float(np.percentile(leakage_rates[:, -1], 5)),
            float(np.percentile(leakage_rates[:, -1], 95))
        ),
        "final_circumvention_mean": float(circumvention_rates[:, -1].mean()),
        # Raw arrays for downstream analysis
        "raw_circumvention": circumvention_rates,
        "raw_leakage": leakage_rates,
        "raw_audit": audit_rates,
    }
    
    return results


def run_multi_country_simulation(
    country_names: List[str],
    scenario: ScenarioParams,
    firm_params_override: Optional[Dict] = None,
    n_simulations: int = 3000,
    n_periods: int = 5,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Run Monte Carlo simulation for multiple countries simultaneously.
    
    Uses country-specific parameters from the config module.
    
    Returns:
        Dict mapping country name → simulation results
    """
    from config.countries import COUNTRIES
    from config.hs_codes import HS_RISK_CATEGORIES, COUNTRY_HS_HOTSPOTS
    
    all_results = {}
    
    for i, country_name in enumerate(country_names):
        if country_name not in COUNTRIES:
            continue
        
        cc = COUNTRIES[country_name]
        hotspots = COUNTRY_HS_HOTSPOTS.get(country_name, [])
        
        # Country-specific average tariff arbitrage
        avg_arbitrage = np.mean([
            HS_RISK_CATEGORIES[hs].avg_tariff_arbitrage_pp
            for hs in hotspots
            if hs in HS_RISK_CATEGORIES
        ]) if hotspots else 6.0
        
        # Country-specific firm params
        fp = FirmParams(
            tariff_arbitrage_pp=avg_arbitrage,
            rerouting_cost_pct=max(1.5, 5.0 - cc.port_volume_rank * 0.15),
            detection_probability=0.10 + cc.customs_capacity * 0.15,
            penalty_multiplier=2.0 + cc.governance_proxy * 2.0,
            risk_aversion=1.0 + cc.governance_proxy * 1.0,
            compliance_baseline=0.4 + cc.governance_proxy * 0.3,
            adaptation_rate=0.08 + cc.governance_proxy * 0.04,
        )
        
        # Apply overrides if provided
        if firm_params_override:
            for key, val in firm_params_override.items():
                if hasattr(fp, key):
                    setattr(fp, key, val)
        
        # Country-specific state params
        sp = StateParams(
            governance_score=cc.governance_proxy,
            customs_capacity=cc.customs_capacity,
            enforcement_budget=0.2 + cc.governance_proxy * 0.3,
            political_will=0.3 + cc.governance_proxy * 0.4,
            tech_adoption=cc.customs_capacity * 0.6,
            max_audit_rate=0.05 + cc.customs_capacity * 0.2,
        )
        
        results = run_monte_carlo_simulation(
            firm_params=fp,
            state_params=sp,
            scenario=scenario,
            n_simulations=n_simulations,
            n_periods=n_periods,
            seed=seed + i * 100,
        )
        
        results["country"] = country_name
        results["firm_params"] = fp
        results["state_params"] = sp
        all_results[country_name] = results
    
    return all_results


# ─── Predefined Scenarios ──────────────────────────────────────────────

PREDEFINED_SCENARIOS = {
    "Baseline": ScenarioParams(
        name="Baseline (Current Trajectory)",
        rerouting_pressure=0.0,
        afcfta_liberalization=0.3,
        epa_tightening=0.0,
        digital_traceability=0.0,
        regional_harmonization=0.0,
    ),
    "China Rerouting Shock": ScenarioParams(
        name="China Rerouting via West Africa",
        rerouting_pressure=0.6,
        afcfta_liberalization=0.3,
        epa_tightening=0.0,
        digital_traceability=0.0,
        regional_harmonization=0.0,
    ),
    "EU Enforcement Tightening": ScenarioParams(
        name="EU Tightens EPA RoO Enforcement",
        rerouting_pressure=0.0,
        afcfta_liberalization=0.3,
        epa_tightening=0.7,
        digital_traceability=0.2,
        regional_harmonization=0.1,
    ),
    "Digital Traceability Rollout": ScenarioParams(
        name="AfCFTA Digital Traceability Protocol",
        rerouting_pressure=0.0,
        afcfta_liberalization=0.5,
        epa_tightening=0.1,
        digital_traceability=0.8,
        regional_harmonization=0.4,
    ),
    "Full AfCFTA + Regional Harmonization": ScenarioParams(
        name="Full AfCFTA Implementation + Harmonized RoO",
        rerouting_pressure=0.1,
        afcfta_liberalization=0.8,
        epa_tightening=0.2,
        digital_traceability=0.5,
        regional_harmonization=0.8,
    ),
    "Worst Case: Multi-Shock": ScenarioParams(
        name="Worst Case: China Rerouting + Weak Enforcement",
        rerouting_pressure=0.8,
        afcfta_liberalization=0.6,
        epa_tightening=0.0,
        digital_traceability=0.0,
        regional_harmonization=0.0,
    ),
}
