# Portfolio Performance and Risk Analytics Dashboard

A Streamlit-based analytics dashboard for evaluating portfolio performance, risk metrics, and attribution relative to a benchmark (e.g., SPY).

---

## 1. Business Objective

The objective of this project is to develop an interactive portfolio analytics dashboard that:

- Evaluates portfolio performance relative to a benchmark
- Measures risk using industry-standard financial metrics
- Decomposes excess returns via attribution techniques
- Provides transparent and reproducible quantitative analysis

This project is designed for academic research and professional portfolio analysis.

---

## 2. Key Features

### Performance Metrics
- Cumulative Returns
- Annualized Return
- Sharpe Ratio
- Rolling Returns

### Risk Metrics
- Volatility (Annualized & Rolling)
- Value-at-Risk (VaR)
- Expected Shortfall (ES)
- Maximum Drawdown
- Tracking Error

### Attribution
- Brinson-Fachler Attribution (if applicable)
- Excess Return Decomposition

### Visualization
- Interactive Plotly charts
- Rolling risk analysis
- Performance comparison vs benchmark

---

## 3. Methodology

## Methodology

### Return Calculation

Daily returns are computed as:

$$
r_t = \frac{P_t}{P_{t-1}} - 1
$$

Annualization is performed using standard scaling factors.


### Volatility

Annualized volatility:

$$
\sigma_{annual} = \sigma_{daily} \times \sqrt{252}
$$


### Value-at-Risk (Historical)

$$
VaR_\alpha = \text{Quantile}(R, \alpha)
$$


### Expected Shortfall

$$
ES_\alpha = \mathbb{E}[R \mid R \le VaR_\alpha]
$$


### Sharpe Ratio

$$
Sharpe = \frac{R_p - R_f}{\sigma_p}
$$

---

## 4. Project Structure
```
portfolio-analytics-dashboard/
│
├── app.py
│
├── data/
│   ├── raw/                       # Original downloaded data (prices, factors, etc.)
│   ├── processed/                 # Cleaned and transformed datasets
│   ├── portfolio_weights/         # Generated portfolio allocation outputs
│   └── analytics/                 # Computed metrics and summary results
│
├── data_processing/
│   ├── ingestion.py               # Data loading (e.g., yfinance, CSV)
│   ├── preprocessing.py           # Cleaning, return calculation, resampling
│   ├── feature_engineering.py     # Rolling metrics, excess returns
│   └── portfolio_construction.py  # Weight calculation logic
│
├── src/
│   ├── performance.py             # Performance metrics
│   ├── risk.py                    # Risk metrics (VaR, ES, Volatility)
│   ├── attribution.py             # Return attribution analysis
│   └── visualization.py           # Plot functions for dashboard
│
├── requirements.txt
├── README.md
└── .gitignore
```
## 4. Data Flow Explanation

### Step 1 – Raw Data (`data/raw`)
Contains:
- Historical price data
- Benchmark data
- Factor datasets (if applicable)

These files are not modified directly.

---

### Step 2 – Data Processing (`data_processing/`)
Responsible for:

- Data cleaning
- Return computation
- Missing value handling
- Resampling (daily → monthly)
- Feature generation
- Portfolio weight construction

Outputs are saved into structured subfolders under `data/`.

---

### Step 3 – Processed Data (`data/processed`)
Contains:
- Cleaned price data
- Return series
- Rolling metrics

---

### Step 4 – Portfolio Weights (`data/portfolio_weights`)
Contains:
- Allocation weights
- Rebalancing results
- Strategy outputs

---

### Step 5 – Analytics Results (`data/analytics`)
Contains:
- Performance metrics
- Risk statistics
- Attribution tables
- Summary statistics

These outputs are consumed by `app.py`.

---

## 5. Methodology

### Return Calculation

$$
r_t = \frac{P_t}{P_{t-1}} - 1
$$

### Annualized Volatility

$$
\sigma_{annual} = \sigma_{daily} \times \sqrt{252}
$$

### Value-at-Risk (Historical)

$$
VaR_\alpha = \text{Quantile}(returns, \alpha)
$$

### Expected Shortfall

$$
ES_\alpha = \mathbb{E}[R \mid R \le VaR_\alpha]
$$

### Sharpe Ratio

$$
Sharpe = \frac{R_p - R_f}{\sigma_p}
$$

---


## 6. Future Improvements

- Regime-switching volatility models
- Machine learning-based volatility forecasting
- Factor model integration (e.g., Fama–French)
- Real-time API integration
- Portfolio optimization module (e.g., mean–variance, risk parity)

---

## 7. Limitations

- Historical simulation assumes a stable return distribution over time
- Transaction costs, slippage, and liquidity constraints are not modelled
- Risk-free rate is assumed constant (simplification)
- Limited macroeconomic or fundamental variables are included

---

## 8. Academic Context

This dashboard is part of a quantitative portfolio analytics project focusing on:

- Performance attribution
- Risk modelling
- Financial data visualization
- Applied quantitative finance
