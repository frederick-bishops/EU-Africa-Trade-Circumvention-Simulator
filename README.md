# African EPA-AfCFTA Overlap Integrity Platform

**Modular Multi-Country Rules-of-Origin Circumvention Simulator with Behavioral Forecasting**

A production-ready Streamlit application that models the overlap between EU Economic Partnership Agreements (EPAs) and the African Continental Free Trade Area (AfCFTA) as a dynamic strategic arbitrage game, identifying circumvention risks across 20 African countries.

## Features

### Core Capabilities
- **Multi-country anomaly detection**: Z-score export spike detection, production capacity mismatch analysis, origin shift detection, import-export ratio anomalies
- **Monte Carlo behavioral simulation**: Firm-level cost-driven rerouting decisions and state-level enforcement responses with game-theoretic interaction
- **6 predefined scenarios**: Baseline, China Rerouting Shock, EU Enforcement Tightening, Digital Traceability Rollout, Full AfCFTA + Regional Harmonization, Worst Case Multi-Shock
- **Custom scenario builder**: Interactive sliders for building custom what-if scenarios
- **Composite risk scoring**: Weighted combination of structural vulnerability (30%), anomaly signals (30%), MC leakage estimates (25%), and governance trajectory (15%)

### Dashboard Tabs
1. **Overview**: KPI cards, country risk ranking, risk distribution, heatmap
2. **Country Deep-Dive**: Per-country risk profile, trade flow trends, MC forecast fan charts, anomaly detail tables
3. **Comparative Analysis**: Regional radar charts, EPA group comparison, spillover corridor detection, HS vulnerability ranking
4. **Simulation Lab**: Multi-scenario comparison, time-series forecasts, custom scenario builder with real-time results
5. **Policy Menu**: Stakeholder-specific briefings (AfCFTA Secretariat, EU DG Trade), prioritized country recommendations, regional policy briefs
6. **Data Explorer**: Filterable data tables with CSV download, executive summary export

### Countries Covered (20)
- **West Africa**: Ghana, Côte d'Ivoire, Nigeria, Senegal, Togo
- **Central Africa**: Cameroon, Gabon
- **East Africa (EAC)**: Kenya, Tanzania, Uganda, Rwanda
- **ESA**: Mauritius, Madagascar, Zimbabwe, Seychelles, Comoros
- **SADC**: South Africa, Botswana, Mozambique, Namibia

### Monitored HS Categories (16)
Fish, vegetables/fruit, fats/oils, sugar, cocoa, tobacco, minerals/fuels, plastics, wood, cotton, apparel, precious metals, iron/steel, aluminium, machinery/electronics, vehicles

## Tech Stack
- **Frontend**: Streamlit
- **Data**: Pandas, NumPy
- **Analytics**: SciPy (statistical methods), scikit-learn
- **Visualization**: Plotly (interactive charts)
- **Simulation**: Custom Monte Carlo engine with behavioral game-theoretic framework

## Data Sources
- **Trade flows**: Synthetic data calibrated to UN Comtrade patterns (ready for live API integration)
- **Governance**: World Bank Worldwide Governance Indicators (WGI)
- **Tariff schedules**: EU Access2Markets EPA data
- **AfCFTA concessions**: AfCFTA e-Tariff Book
- **Enforcement calibration**: EU customs infringement statistics (EPPO/OLAF)

## Quick Start

### Local Development
```bash
# Clone the repository
git clone <repo-url>
cd epa-afcfta-platform

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Deploy to Streamlit Cloud
1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy

### Deploy to Hugging Face Spaces
1. Create a new Space (Streamlit SDK)
2. Upload all files
3. The app will auto-deploy

## Project Structure
```
epa-afcfta-platform/
├── app.py                      # Main Streamlit entry point
├── config/
│   ├── countries.py            # 20 country configurations
│   └── hs_codes.py             # 16 HS risk category definitions
├── data/
│   └── synthetic_data.py       # Calibrated synthetic data generator
├── engine/
│   ├── anomaly_detection.py    # 4 anomaly detection methods
│   ├── monte_carlo.py          # Behavioral MC simulation engine
│   ├── risk_scoring.py         # Composite risk scoring
│   └── behavioral_forecast.py  # Scenario forecasting
├── analysis/
│   ├── comparative.py          # Cross-country comparison
│   └── policy_menu.py          # Policy recommendation engine
├── ui/
│   └── components.py           # Reusable visualization components
├── utils/
│   └── export.py               # CSV/TXT export utilities
├── requirements.txt
└── README.md
```

## Methodology

### Anomaly Detection
- **Export spike detection**: Rolling Z-score with configurable threshold (default: 2.0 SD)
- **Capacity mismatch**: Export-to-production-capacity ratio analysis
- **Origin shift detection**: Pearson correlation between China import growth and EU export growth
- **Import-export ratio**: Transshipment indicator based on China import / EU export ratio

### Monte Carlo Simulation
- **Firm model**: Expected utility framework — firms circumvent when tariff saving minus rerouting cost exceeds risk-adjusted expected penalty
- **State model**: Audit intensity bounded by governance capacity, political will, and technology adoption
- **Interaction**: Multi-period adaptive dynamics where firms learn from enforcement and states respond to observed circumvention
- **Calibration**: Parameters anchored to public EPPO/OLAF enforcement statistics

### Risk Scoring
| Component | Weight | Source |
|-----------|--------|--------|
| Structural vulnerability | 30% | Country config (ports, governance, customs, EPA access) |
| Anomaly signals | 30% | Anomaly detection pipeline |
| MC leakage estimate | 25% | Monte Carlo simulation final-period mean |
| Governance trajectory | 15% | WGI time-series trend |

## Disclaimer
This analysis uses simulation-based modeling with synthetic data calibrated to public sources. Results should be interpreted as indicative risk assessments for capacity-building purposes, not precise predictions. Country comparisons are framed as neutral capacity diagnostics to support informed policy decisions.

## License
MIT
