"""
app.py  —  Streamlit Dashboard for NIFTY-50 Investment Intelligence.

Self-contained dashboard — all logic is inlined so the src/ package
only needs preprocess.py, indicators.py, and explainability.py.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize

from src.preprocess import load_all_stocks, load_metadata, get_aligned_returns
from src.preprocess import RISK_FREE_RATE, TRADING_DAYS
from src.indicators import add_all_indicators
from src.explainability import generate_stock_insight

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(page_title="NIFTY-50 Investment Intelligence", layout="wide")
st.title("📈 NIFTY-50 Investment Intelligence Platform")


@st.cache_data
def load_data():
    stocks = load_all_stocks()
    meta = load_metadata()
    return stocks, meta


stocks, meta = load_data()
symbols = sorted(stocks.keys())

# ── Sidebar ────────────────────────────────────────────────────────
page = st.sidebar.selectbox("Module", [
    "Stock Explorer",
    "Risk Assessment",
    "Portfolio Builder",
    "Anomaly Detection",
    "Forecasting",
])


# ── Inline helpers (keep notebooks as the primary workspace) ──────

def _stock_risk_metrics(df):
    """Risk metrics for one stock."""
    returns = df["Close"].pct_change().dropna()
    ann_ret = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    down = returns[returns < 0]
    down_std = down.std() * np.sqrt(TRADING_DAYS) if len(down) > 0 else 1e-9
    sortino = (ann_ret - RISK_FREE_RATE) / down_std
    cum = (1 + returns).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    var95 = np.percentile(returns, 5)
    cvar95 = returns[returns <= var95].mean()
    return {
        "ann_return": ann_ret, "ann_vol": ann_vol,
        "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": max_dd, "VaR_95": var95, "CVaR_95": cvar95,
    }


def _portfolio_stats(w, mu, cov):
    ret = np.dot(w, mu) * TRADING_DAYS
    vol = np.sqrt(np.dot(w, np.dot(cov * TRADING_DAYS, w)))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0
    return ret, vol, sharpe


def _optimize(returns, profile):
    mu = returns.mean()
    cov = returns.cov()
    n = len(mu)
    w0 = np.ones(n) / n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bnd = [(0, 0.10 if profile == "conservative" else 1)] * n
    if profile == "aggressive":
        obj = lambda w: -np.dot(w, mu) * TRADING_DAYS
    elif profile == "conservative":
        obj = lambda w: np.dot(w, np.dot(cov * TRADING_DAYS, w))
    else:
        obj = lambda w: -(np.dot(w, mu) * TRADING_DAYS - RISK_FREE_RATE) / (
            np.sqrt(np.dot(w, np.dot(cov * TRADING_DAYS, w))) + 1e-9)
    res = minimize(obj, w0, method="SLSQP", bounds=bnd, constraints=cons)
    weights = pd.Series(res.x, index=returns.columns)
    ret, vol, sharpe = _portfolio_stats(res.x, mu, cov)
    return weights, ret, vol, sharpe


def _detect_anomalies(df):
    ret = df["Close"].pct_change().dropna()
    z = (ret - ret.mean()) / ret.std()
    mask = z.abs() > 3
    anom = df.loc[mask.index[mask]].copy()
    anom["z_score"] = z[mask]
    anom["type"] = "extreme_return"
    if "Volume" in df.columns:
        vol = df["Volume"]
        rm = vol.rolling(60).mean()
        rs = vol.rolling(60).std().replace(0, np.nan)
        vz = ((vol - rm) / rs).dropna()
        vm = vz.abs() > 3
        va = df.loc[vz.index[vm]].copy()
        va["z_score"] = vz[vm]
        va["type"] = "volume_spike"
        anom = pd.concat([anom[["Close", "z_score", "type"]],
                          va[["Close", "z_score", "type"]]])
    return anom


# ═══════════════════════════════════════════════════════════════════
# 1. STOCK EXPLORER
# ═══════════════════════════════════════════════════════════════════
if page == "Stock Explorer":
    sym = st.sidebar.selectbox("Stock", symbols)
    df = add_all_indicators(stocks[sym].copy())

    if sym in meta.index:
        info = meta.loc[sym]
        st.markdown(f"**{info.get('Company_Name', sym)}** — {info.get('Industry', 'N/A')}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], name="SMA 200", line=dict(dash="dot")))
    fig.update_layout(title=f"{sym} — Price History", xaxis_title="Date", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Technical Insight")
    st.code(generate_stock_insight(df))

    col1, col2, col3, col4 = st.columns(4)
    latest = df.iloc[-1]
    col1.metric("RSI", f"{latest.get('RSI', 0):.1f}")
    col2.metric("MACD Hist", f"{latest.get('MACD_Hist', 0):.2f}")
    col3.metric("BB Position", f"{latest.get('BB_Position', 0):.2f}")
    col4.metric("20d Volatility", f"{latest.get('Volatility_20', 0)*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════
# 2. RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════
elif page == "Risk Assessment":
    st.subheader("Risk Metrics — All Stocks")

    rows = []
    for sym, df in stocks.items():
        m = _stock_risk_metrics(df)
        m["symbol"] = sym
        rows.append(m)
    table = pd.DataFrame(rows).set_index("symbol").sort_values("sharpe", ascending=False)
    st.dataframe(table.style.format("{:.4f}"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 5 by Sharpe**")
        st.dataframe(table.head(5)[["sharpe", "ann_return", "max_drawdown"]])
    with col2:
        st.markdown("**Bottom 5 by Sharpe**")
        st.dataframe(table.tail(5)[["sharpe", "ann_return", "max_drawdown"]])


# ═══════════════════════════════════════════════════════════════════
# 3. PORTFOLIO BUILDER
# ═══════════════════════════════════════════════════════════════════
elif page == "Portfolio Builder":
    st.subheader("Portfolio Optimization")
    with st.spinner("Computing aligned returns..."):
        returns = get_aligned_returns()

    for profile in ["conservative", "balanced", "aggressive"]:
        weights, ret, vol, sharpe = _optimize(returns, profile)
        with st.expander(f"{profile.title()} Portfolio", expanded=(profile == "balanced")):
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected Return", f"{ret*100:.1f}%")
            c2.metric("Volatility", f"{vol*100:.1f}%")
            c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            top = weights[weights > 0.001].sort_values(ascending=False).head(10)
            fig = px.bar(x=top.index, y=top.values, labels={"x": "Stock", "y": "Weight"})
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# 4. ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════
elif page == "Anomaly Detection":
    sym = st.sidebar.selectbox("Stock", symbols)
    df = stocks[sym].copy()
    anomalies = _detect_anomalies(df)

    st.subheader(f"{sym} — Detected Anomalies")
    if len(anomalies) > 0:
        st.write(f"Found **{len(anomalies)}** anomalous events.")
        st.dataframe(anomalies.tail(20))
    else:
        st.info("No anomalies detected with default thresholds.")


# ═══════════════════════════════════════════════════════════════════
# 5. FORECASTING
# ═══════════════════════════════════════════════════════════════════
elif page == "Forecasting":
    sym = st.sidebar.selectbox("Stock", symbols)
    days = st.sidebar.slider("Forecast horizon (days)", 5, 60, 30)
    df = stocks[sym].copy()

    st.subheader(f"{sym} — Price Trend Forecast ({days} days)")

    # Simple linear trend on last 90 days
    from sklearn.linear_model import LinearRegression
    recent = df["Close"].iloc[-90:]
    x = np.arange(len(recent)).reshape(-1, 1)
    model = LinearRegression().fit(x, recent.values)
    future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)
    fx = np.arange(len(recent), len(recent) + len(future_dates)).reshape(-1, 1)
    proj = model.predict(fx)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-120:], y=df["Close"].iloc[-120:], name="Historical"))
    fig.add_trace(go.Scatter(x=future_dates, y=proj,
                             name="Forecast", line=dict(dash="dash", color="orange")))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⚠️ Linear trend projection for illustration — not financial advice.")
