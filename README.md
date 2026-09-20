# MatRisk AI: Atomistic-to-Financial Asset Resilience & Risk Engine

MatRisk AI is a unified physical-financial modeling framework that connects atomistic material properties to infrastructure project finance and actuarial catastrophe insurance pricing.

```
  [CIF Files] -> [Engine 1: Physics CGCNN] -> Physical Properties (K, G, K_Ic)
                                                    |
                                                    v
 [Load Cycles] -> [Engine 2: Stochastic SDE] -> Failure Trajectory P_f(t)
                                                    |
                                                    v
[CapEx / OpEx] -> [Engine 3: Project Finance] -> CFADS & DSCR Debt Sculpting
                                                    |
                                                    v
 [Tail Losses] -> [Engine 4: EVT & ESG Engine] -> Gross Premium & PARS Score
                                                    |
                                                    v
 [User Inputs] -> [Engine 5: MatRisk Lab UI] -> Interactive Stress Dashboard
```

---

## 5 Core Brain Engines

1. **Atomistic Physics & Dual-Attention CGCNN (`src/atomistic/`)**:
   - Converts raw Crystallographic Information Files (CIF) into crystal multigraphs.
   - Dual-Attention CGCNN predicting bulk modulus ($K$), shear modulus ($G$), and fracture toughness ($K_{Ic}$) subject to Born mechanical stability constraints.

2. **Microstructural Degradation & Stochastic SDE Engine (`src/degradation/`)**:
   - Time-dependent sub-critical crack propagation using Itô Stochastic Differential Equations (Paris' Law with Brownian noise).
   - High-performance vectorized Monte Carlo simulation (10,000 trajectories in $<2$s) producing time-dependent failure probabilities $P_f(t)$.

3. **Project Finance & Commodity Risk Engine (`src/financial/`)**:
   - Translates failure curves $P_f(t)$ into unplanned CapEx spikes and plant downtime.
   - Computes Cash Flow Available for Debt Service (CFADS) and sculpts principal repayment schedules to maintain $\text{DSCR} \ge 1.30\text{x}$.
   - Integrates live commodity futures curves via `yfinance`.

4. **Actuarial Insurance & ESG Resilience Engine (`src/insurance/`)**:
   - Catastrophe risk pricing using Extreme Value Theory (EVT) with Generalized Pareto Distribution (GPD).
   - Quantifies Expected Shortfall ($\text{ES}_{0.99}$) and calculates pure/gross actuarial insurance premiums.
   - Computes the Physical Asset Resilience Score ($\text{PARS}$) mapped to credit-style resilience ratings (AAA to D).

5. **MatRisk Lab Interactive Simulator (`src/lab/`)**:
   - Interactive Streamlit dashboard with real-time Plotly visualizations, shock generators, and sensitivity analysis.

---

## Repository Structure

```
matrisk_ai/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── raw_cifs/        # Raw CIF structure files & id_prop.csv
│   ├── processed/       # Pre-processed PyTorch Geometric graph objects (.pt)
│   └── finance/         # Financial schedules & commodity benchmark data
├── src/
│   ├── data/            # Materials Project fetcher & data utilities
│   ├── atomistic/       # CIF parser, DA-CGCNN, Born physics loss
│   ├── degradation/     # Paris' Law, vectorized Itô SDE solver
│   ├── financial/       # CFADS, DSCR sculpting, commodity desk
│   ├── insurance/       # EVT GPD pricing, PARS ESG resilience
│   ├── lab/             # Streamlit dashboard & shock scenarios
│   └── api/             # FastAPI microservice
├── tests/               # Unit tests & backtesting suites
└── docs/                # Architecture blueprint & documentation
```

---

## Quickstart

### 1. Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Phase 1: Ingest Benchmark Materials
```powershell
# Ingest Materials Project subset (or use offline calibrated benchmark mode)
python src/data/download_mp.py --offline --num-samples 500

# Parse CIFs into crystal graph tensors
python src/atomistic/cif_parser.py --cutoff 8.0
```

### 3. Run Unit Tests
```powershell
pytest tests/ -v
```

### 4. Launch MatRisk Lab
```powershell
streamlit run src/lab/app.py
```
