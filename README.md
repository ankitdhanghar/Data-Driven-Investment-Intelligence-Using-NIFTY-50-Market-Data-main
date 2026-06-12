# Data-Driven Investment Intelligence Platform Using NIFTY-50 Market Data

## Overview

This project presents a comprehensive investment intelligence platform built using historical NIFTY-50 market data. The platform leverages machine learning, portfolio optimization, risk analytics, and anomaly detection techniques to assist investors in making data-driven financial decisions.

The system analyzes historical stock market data to generate actionable insights, evaluate investment risks, construct optimized portfolios, and identify unusual market events.

---

## Objectives

- Predict stock market movements using machine learning.
- Construct optimized portfolios for different investor profiles.
- Assess investment risk using financial metrics.
- Detect anomalous market behavior and extreme events.
- Provide an interactive dashboard for investment analysis.

---

## Dataset

The project utilizes historical NIFTY-50 market data spanning multiple sectors and companies.

### Dataset Characteristics

- Historical Data Period: 2000–2021
- Stocks Analyzed: 49
- Sectors Covered: 13
- Records: ~380,000
- Data Type: OHLCV (Open, High, Low, Close, Volume)

---

## Key Features

### Stock Prediction
Machine learning models are used to predict future market movements based on historical trends and technical indicators.

### Portfolio Optimization
Portfolio construction strategies are developed for:
- Conservative Investors
- Balanced Investors
- Aggressive Investors

### Risk Assessment
Investment risk is evaluated using:
- Sharpe Ratio
- Sortino Ratio
- Value-at-Risk (VaR)
- Maximum Drawdown

### Explainable AI
SHAP-based explainability is integrated to understand feature importance and improve model transparency.

### Anomaly Detection
The system identifies unusual market events, volatility spikes, and abnormal return patterns.

---

## Feature Engineering

Technical indicators generated include:

### Trend Indicators
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)

### Momentum Indicators
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)

### Volatility Indicators
- Bollinger Bands
- Average True Range (ATR)

Additional engineered features:
- Lagged Returns
- Volume Ratios
- Rolling Statistics
- Calendar Features

---

## Machine Learning Models

The following models were implemented:

- XGBoost
- LightGBM

### Performance

| Model | Accuracy |
|---------|---------|
| XGBoost | 51% |
| LightGBM | 51% |

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- SHAP
- Plotly
- Streamlit

---

## Project Structure

```text
├── data/
├── models/
├── notebooks/
├── src/
│   ├── preprocess.py
│   ├── indicators.py
│   └── explainability.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Dashboard Features

- Market Analysis
- Stock Prediction
- Portfolio Optimization
- Risk Assessment
- Explainable AI Insights
- Interactive Visualizations

---

## Key Findings

- Developed an end-to-end investment intelligence platform.
- Implemented machine learning-based stock prediction.
- Integrated explainable AI using SHAP.
- Designed optimized portfolios for multiple investor profiles.
- Detected major market anomalies, including the 2008 Financial Crisis and 2020 COVID-19 market crash.

---

## Future Enhancements

- Real-time market data integration
- Deep learning models (LSTM, Transformer)
- Financial news sentiment analysis
- Automated portfolio rebalancing
- Reinforcement learning-based investment strategies

---

## Author

Ankit Dhanghar
Anuj Mittal

