# Cross-Asset Energy Shock Transmission Engine

An institutional-grade quantitative macro research engine estimating the dynamic pass-through of pure crude oil supply shocks across liquid US financial assets. Built using Jordà (2005) Local Projections with Newey-West HAC covariance corrections and rendered in a Bloomberg Terminal interface.

---

## Econometric Framework

- **Shock Identification ($e_t$):** Daily high-frequency filter isolating cost-push supply jumps:
  - Brent daily return $> 1.5\sigma$ (dynamic 60-day rolling window)
  - S&P 500 Equal Weight daily return $\le 0.0\%$
  - CBOE Crude Oil Volatility Index ($\Delta\text{OVX}$) $> 0.0$
- **Exogenous Policy Controls ($D_t$):** Discrete dummy tracking official coordinated Strategic Petroleum Reserve (SPR) emergency announcements (Libya 2011, Post-COVID 2021, Ukraine 2022).
- **State-Dependent Local Projections:** Structural interaction with Federal Reserve policy stance ($I_t \in \{0, 1\}$), conditioning impulse responses to Restrictive/Hiking vs. Easing/Neutral regimes.
- **Robust Inference:** Equation-by-equation OLS with Newey-West HAC standard error correction using a dynamic lag bandwidth of $\text{maxlags} = h + 1$ to account for serial correlation induced by overlapping projection horizons.

---

## Asset Class Coverage

- **Front-End Inflation:** 5Y Inflation Breakeven (`T5YIE`)
- **Benchmark Sovereign Curve:** US Treasury 2-Year (`DGS2`) & 10-Year (`DGS10`) Yields
- **Real Rates:** 10-Year Real TIPS Yield (`DFII10`)
- **Broad Equities:** S&P 500 Equal Weight Index (`RSP`)
- **Currencies:** U.S. Dollar Index (`DXY`)

---

## Project Structure

```text
energy-shock-engine/
│
├── config.py                 # Core configurations, horizons, and API parameter definitions
├── data_loader.py            # Automated pipeline fetching data from FRED and Yahoo Finance
├── shock_identification.py   # High-frequency structural shock and Fed regime classification
├── local_projections.py      # Jordà (2005) Local Projections engine with Newey-West HAC
├── app.py                    # Interactive Streamlit dashboard styled as a Bloomberg Terminal
├── requirements.txt          # Python dependencies
├── .env.example              # Public environment template
└── .gitignore                # Rules excluding sensitive credentials (.env) and cache
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/energy-shock-engine.git](https://github.com/YOUR_USERNAME/energy-shock-engine.git)
cd energy-shock-engine
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure credentials
Create a `.env` file in the root directory and insert your free FRED API key:
```ini
FRED_API_KEY=your_actual_fred_api_key_here
```

### 4. Launch the terminal dashboard
```bash
streamlit run app.py
```
