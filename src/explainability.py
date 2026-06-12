"""
explainability.py  —  Explain model predictions using SHAP + generate text insights.

Functions
---------
get_feature_importance(model, feature_names) -> DataFrame of feature importances
explain_with_shap(model, X_test)            -> SHAP values + plots
generate_stock_insight(df)                   -> plain-English technical summary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# shap is imported lazily inside explain_with_shap() so that importing this
# module (e.g. from app.py) never crashes when shap is not yet installed.


def get_feature_importance(model, feature_names):
    """
    Get feature importance from a trained tree model (XGBoost / LightGBM).
    Returns a sorted DataFrame.
    """
    importance = model.feature_importances_
    result = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return result


def explain_with_shap(model, X_data, max_display=15):
    """
    Compute SHAP values and create summary + bar plots.
    Pass your trained model and the test features.
    Returns the SHAP values array.

    shap is imported lazily here so app.py stays alive even if shap
    is not installed in the current environment.
    """
    import shap  # noqa: PLC0415  (lazy import intentional)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)

    # Summary plot (dot plot showing impact of each feature)
    plt.figure()
    shap.summary_plot(shap_values, X_data, max_display=max_display, show=False)
    plt.title("SHAP Feature Impact")
    plt.tight_layout()
    plt.show()

    # Bar plot (average absolute SHAP value per feature)
    plt.figure()
    shap.summary_plot(shap_values, X_data, plot_type="bar", max_display=max_display, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()

    return shap_values


def generate_stock_insight(df):
    """
    Generate a plain-English summary of a stock's current technical state.
    Pass a DataFrame that already has indicators added (from add_all_indicators).
    Looks at the last row.
    """
    latest = df.iloc[-1]
    lines = []

    # RSI
    rsi = latest.get("RSI", None)
    if rsi is not None and not np.isnan(rsi):
        if rsi > 70:
            lines.append(f"RSI = {rsi:.1f} → OVERBOUGHT (may be overvalued)")
        elif rsi < 30:
            lines.append(f"RSI = {rsi:.1f} → OVERSOLD (may be undervalued)")
        else:
            lines.append(f"RSI = {rsi:.1f} → Neutral")

    # MACD
    macd_hist = latest.get("MACD_Hist", None)
    if macd_hist is not None and not np.isnan(macd_hist):
        if macd_hist > 0:
            lines.append(f"MACD Histogram = {macd_hist:.2f} → Bullish momentum")
        else:
            lines.append(f"MACD Histogram = {macd_hist:.2f} → Bearish momentum")

    # Price vs SMAs
    for ma in ["SMA_20", "SMA_50", "SMA_200"]:
        key = f"Price_vs_{ma}"
        pct = latest.get(key, None)
        if pct is not None and not np.isnan(pct):
            word = "ABOVE" if pct > 0 else "BELOW"
            lines.append(f"Price is {abs(pct):.1f}% {word} {ma}")

    # Bollinger
    bb = latest.get("BB_Position", None)
    if bb is not None and not np.isnan(bb):
        if bb > 0.9:
            lines.append(f"Bollinger Position = {bb:.2f} → Near upper band (overextended)")
        elif bb < 0.1:
            lines.append(f"Bollinger Position = {bb:.2f} → Near lower band (oversold)")
        else:
            lines.append(f"Bollinger Position = {bb:.2f} → Within normal range")

    # Volatility
    vol = latest.get("Volatility_20", None)
    if vol is not None and not np.isnan(vol):
        lines.append(f"20-day Volatility = {vol*100:.1f}% annualized")

    return "\n".join(lines)
