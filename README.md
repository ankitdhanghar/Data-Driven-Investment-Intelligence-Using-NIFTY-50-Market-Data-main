# Data-Driven Investment Intelligence Using NIFTY-50 Market Data

An AI-powered investment intelligence platform that transforms historical NIFTY-50 market data into actionable insights, portfolio recommendations, and risk assessments.

## Features

| Module | Description | Type |
|--------|-------------|------|
| **Stock Predictor Engine** | XGBoost classifier for next-day direction prediction | Mandatory |
| **Portfolio Construction** | Mean-variance optimization for Conservative / Balanced / Aggressive profiles | Mandatory |
| **Risk Assessment** | Sharpe, Sortino, Max Drawdown, VaR, CVaR for stocks and portfolios | Mandatory |
| **Explainable AI** | SHAP-based feature importance + plain-English technical insights | Optional |
| **Anomaly Detection** | Z-score based detection of extreme returns, volume spikes, volatility regimes | Optional |
| **Forecasting** | Return forecasting, volatility estimation, price trend projection | Optional |
| **Streamlit Dashboard** | Interactive web app tying all modules together | Optional |

## Project Structure

```
├── app.py                  # Streamlit dashboard (deployment)
├── requirements.txt        # Python dependencies
├── data/
│   ├── raw/                # Original NIFTY-50 CSVs from Kaggle
│   └── processed/          # Cleaned / feature-engineered data
├── models/                 # Saved model files (.pkl)
├── notebooks/
│   ├── 1_eda.ipynb         # Exploratory Data Analysis
│   ├── 2_feature_engg.ipynb# Feature Engineering
│   ├── 3_prediction.ipynb  # Stock Prediction
│   ├── 4_portfolio.ipynb   # Portfolio Construction
│   ├── 5_risk_analysis.ipynb # Risk Assessment
│   └── 6_anomaly_detection.ipynb # Anomaly Detection
└── src/
    ├── __init__.py
    ├── preprocess.py       # Data loading, cleaning, and alignment
    ├── indicators.py       # Technical indicators (RSI, MACD, BB, etc.)
    └── explainability.py   # SHAP explanations + text insights
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Data-Driven-Investment-Intelligence-NIFTY50.git
cd Data-Driven-Investment-Intelligence-NIFTY50
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the NIFTY-50 dataset from [Kaggle](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data) and place all CSV files in `data/raw/`.

## Running the Application

### Streamlit Dashboard

```bash
streamlit run app.py
```

This launches the interactive dashboard with five modules:
- **Stock Explorer** — browse individual stocks with technical indicators and insights
- **Risk Assessment** — compare risk metrics across all 49 stocks
- **Portfolio Builder** — optimized portfolios for three investor profiles
- **Anomaly Detection** — flag unusual market events per stock
- **Forecasting** — linear trend projection for any stock

### Using Source Modules Directly

```python
from src.preprocess import load_all_stocks, load_stock, get_aligned_returns
from src.indicators import add_all_indicators
from src.explainability import generate_stock_insight, explain_with_shap

# Load data
stocks = load_all_stocks()
df = add_all_indicators(stocks["TCS"].copy())

# Technical insight for a stock
print(generate_stock_insight(df))

# Aligned returns for portfolio work
returns = get_aligned_returns()
```

For prediction, portfolio construction, risk assessment, and anomaly detection,
refer to the corresponding notebooks (3–6) which contain the full implementations.

## Reproducing Results

1. Complete the setup steps above.
2. Run the notebooks in order (`1_eda.ipynb` → `6_anomaly_detection.ipynb`).
3. Or use `src/` modules directly in Python scripts.
4. Launch the Streamlit dashboard to interact with all features.

## Methodology

- **Data Cleaning**: Handle duplicates, missing values (forward-fill ≤3 days), type coercion.
- **Feature Engineering**: 25+ technical indicators including RSI, MACD, Bollinger Bands, ATR, moving averages, lag features, and calendar features.
- **Prediction**: XGBoost classifier trained on pooled multi-stock data with time-aware train/test split.
- **Portfolio Optimization**: Mean-variance framework via `scipy.optimize.minimize` with profile-specific constraints.
- **Risk Assessment**: Standard financial risk metrics (Sharpe, Sortino, VaR, CVaR, Max Drawdown).
- **Anomaly Detection**: Statistical Z-score approach on returns and volume; volatility regime classification.
- **Explainability**: SHAP TreeExplainer for feature importance; rule-based plain-English technical summaries.

## Tech Stack

- Python 3.11+
- pandas, numpy, scipy
- scikit-learn, XGBoost, LightGBM
- SHAP
- Plotly, Matplotlib, Seaborn
- Streamlit

## Dataset

- [NIFTY-50 Stock Market Data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data) (Jan 2000 – Apr 2021)
- 49 stocks across Banking, IT, Energy, Consumer Goods, Pharma, Financial Services, Manufacturing
