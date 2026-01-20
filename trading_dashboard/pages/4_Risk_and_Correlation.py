"""
Risk and Correlation page - Risk metrics, VaR, CVaR, and correlation analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasource import DemoDataSource, DEMO_ASSETS
from src.portfolio import Portfolio, portfolio_correlation_matrix
from src.analytics import (
    calculate_returns, historical_var, historical_cvar, calculate_var_cvar,
    rolling_volatility, annualized_volatility, correlation_matrix, covariance_matrix,
    stress_test
)
from src.utils import format_percent

st.set_page_config(page_title="Risk & Correlation", page_icon="📉", layout="wide")


@st.cache_data(ttl=300)
def get_multi_asset_data(tickers: list, start: str, end: str):
    """Get data for multiple assets."""
    source = DemoDataSource()
    data = {}
    for ticker in tickers:
        try:
            df = source.get_price_history(ticker, start, end)
            if not df.empty:
                data[ticker] = df
        except Exception:
            pass
    return data


def main():
    st.title("📉 Risk & Correlation Analysis")
    st.markdown("Analyze risk metrics, Value at Risk, and correlations between assets.")

    # Initialize session state
    if "start_date" not in st.session_state:
        st.session_state.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.now().strftime("%Y-%m-%d")
    if "portfolio" not in st.session_state:
        portfolio = Portfolio(name="My Portfolio", initial_value=100000.0)
        portfolio.add_position("DEMO_TECH", 0.25)
        portfolio.add_position("DEMO_BANK", 0.20)
        portfolio.add_position("DEMO_PHARMA", 0.20)
        portfolio.add_position("DEMO_OIL", 0.15)
        portfolio.add_position("DEMO_SPY", 0.20)
        st.session_state.portfolio = portfolio

    source = DemoDataSource()

    # Sidebar - Asset Selection
    with st.sidebar:
        st.subheader("Asset Selection")

        available_tickers = source.get_available_tickers()

        # Use portfolio tickers by default
        default_tickers = list(st.session_state.portfolio.tickers)

        selected_tickers = st.multiselect(
            "Select Assets for Analysis",
            available_tickers,
            default=default_tickers,
        )

        st.markdown("---")
        st.subheader("Date Range")

        start_date = st.date_input(
            "Start Date",
            value=datetime.strptime(st.session_state.start_date, "%Y-%m-%d"),
        )
        end_date = st.date_input(
            "End Date",
            value=datetime.strptime(st.session_state.end_date, "%Y-%m-%d"),
        )

        st.session_state.start_date = start_date.strftime("%Y-%m-%d")
        st.session_state.end_date = end_date.strftime("%Y-%m-%d")

        st.markdown("---")
        st.subheader("Risk Parameters")

        confidence_95 = st.checkbox("Show 95% Confidence", value=True)
        confidence_99 = st.checkbox("Show 99% Confidence", value=True)

    if not selected_tickers:
        st.info("Please select at least one asset from the sidebar.")
        return

    # Get data for selected assets
    asset_data = get_multi_asset_data(
        selected_tickers,
        st.session_state.start_date,
        st.session_state.end_date,
    )

    if not asset_data:
        st.error("Could not load data for selected assets.")
        return

    # Calculate returns for all assets
    returns_dict = {}
    for ticker, df in asset_data.items():
        returns_dict[ticker] = calculate_returns(df["Close"]).dropna()

    # VaR and CVaR Summary
    st.subheader("Value at Risk (VaR) & Conditional VaR (CVaR)")

    var_data = []
    for ticker, returns in returns_dict.items():
        var_cvar = calculate_var_cvar(returns)
        var_data.append({
            "Ticker": ticker,
            "VaR 95%": var_cvar["var_95"],
            "VaR 99%": var_cvar["var_99"],
            "CVaR 95%": var_cvar["cvar_95"],
            "CVaR 99%": var_cvar["cvar_99"],
            "Ann. Volatility": annualized_volatility(returns),
        })

    var_df = pd.DataFrame(var_data)

    # Display as metrics
    cols = st.columns(min(len(selected_tickers), 4))
    for i, ticker in enumerate(selected_tickers[:4]):
        with cols[i]:
            if ticker in returns_dict:
                var_cvar = calculate_var_cvar(returns_dict[ticker])
                st.markdown(f"**{ticker}**")
                if confidence_95:
                    st.metric("VaR (95%)", format_percent(var_cvar["var_95"]))
                    st.metric("CVaR (95%)", format_percent(var_cvar["cvar_95"]))
                if confidence_99:
                    st.metric("VaR (99%)", format_percent(var_cvar["var_99"]))

    # VaR comparison chart
    st.markdown("---")
    st.subheader("VaR Comparison")

    fig = go.Figure()
    if confidence_95:
        fig.add_trace(go.Bar(
            name="VaR 95%",
            x=var_df["Ticker"],
            y=var_df["VaR 95%"] * 100,
            marker_color="#FFA726",
        ))
    if confidence_99:
        fig.add_trace(go.Bar(
            name="VaR 99%",
            x=var_df["Ticker"],
            y=var_df["VaR 99%"] * 100,
            marker_color="#EF5350",
        ))

    fig.update_layout(
        title="Value at Risk by Asset (Daily)",
        xaxis_title="Asset",
        yaxis_title="VaR (%)",
        barmode="group",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling Volatility
    st.markdown("---")
    st.subheader("Rolling Volatility")

    col1, col2 = st.columns([3, 1])

    with col2:
        vol_window_20 = st.checkbox("20-day Window", value=True)
        vol_window_60 = st.checkbox("60-day Window", value=True)

    with col1:
        fig = go.Figure()

        for ticker, returns in returns_dict.items():
            if vol_window_20:
                vol_20 = rolling_volatility(returns, 20, annualize=True)
                fig.add_trace(go.Scatter(
                    x=vol_20.index,
                    y=vol_20 * 100,
                    name=f"{ticker} (20d)",
                    mode="lines",
                ))

            if vol_window_60:
                vol_60 = rolling_volatility(returns, 60, annualize=True)
                fig.add_trace(go.Scatter(
                    x=vol_60.index,
                    y=vol_60 * 100,
                    name=f"{ticker} (60d)",
                    mode="lines",
                    line=dict(dash="dash"),
                ))

        fig.update_layout(
            title="Rolling Annualized Volatility",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Matrix
    st.markdown("---")
    st.subheader("Correlation Matrix")

    corr_matrix = correlation_matrix(returns_dict)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False,
    ))

    fig.update_layout(
        title="Asset Correlation Matrix",
        height=500,
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling Correlation (for pairs)
    if len(selected_tickers) >= 2:
        st.markdown("---")
        st.subheader("Rolling Correlation Analysis")

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            asset1 = st.selectbox("First Asset", selected_tickers, key="corr_asset1")

        with col2:
            remaining = [t for t in selected_tickers if t != asset1]
            asset2 = st.selectbox("Second Asset", remaining, key="corr_asset2")

        with col3:
            corr_window = st.slider("Rolling Window (days)", 20, 120, 60)

        if asset1 in returns_dict and asset2 in returns_dict:
            combined = pd.concat([returns_dict[asset1], returns_dict[asset2]], axis=1).dropna()
            combined.columns = [asset1, asset2]

            rolling_corr = combined[asset1].rolling(window=corr_window).corr(combined[asset2])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr.values,
                mode="lines",
                name=f"Rolling Correlation ({corr_window}d)",
                line=dict(color="#2196F3", width=2),
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.5)
            fig.add_hline(y=-1, line_dash="dot", line_color="red", opacity=0.5)

            fig.update_layout(
                title=f"Rolling Correlation: {asset1} vs {asset2}",
                xaxis_title="Date",
                yaxis_title="Correlation",
                yaxis=dict(range=[-1.1, 1.1]),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Stress Testing
    st.markdown("---")
    st.subheader("Stress Test Scenarios")

    st.markdown(
        "These scenarios show hypothetical portfolio impacts based on historical market events. "
        "Actual results may differ significantly."
    )

    # Calculate weighted portfolio returns for stress test
    if len(selected_tickers) > 0:
        # Simple equal-weight for demonstration
        portfolio_returns = pd.DataFrame()
        for ticker in selected_tickers:
            if ticker in returns_dict:
                portfolio_returns[ticker] = returns_dict[ticker]

        if not portfolio_returns.empty:
            equal_weight_returns = portfolio_returns.mean(axis=1)
            current_vol = annualized_volatility(equal_weight_returns)

            scenarios = stress_test(equal_weight_returns)

            stress_data = []
            for scenario, impact in scenarios.items():
                loss_pct = (1 - impact) * 100
                stress_data.append({
                    "Scenario": scenario,
                    "Impact": f"-{loss_pct:.1f}%",
                    "Value After (from $100K)": f"${impact * 100000:,.0f}",
                })

            stress_df = pd.DataFrame(stress_data)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.dataframe(stress_df, use_container_width=True, hide_index=True)

            with col2:
                fig = go.Figure(data=[
                    go.Bar(
                        x=[s["Scenario"] for s in stress_data],
                        y=[(1 - scenarios[s["Scenario"]]) * 100 for s in stress_data],
                        marker_color="#EF5350",
                    )
                ])
                fig.update_layout(
                    title="Stress Test Scenarios",
                    xaxis_title="Scenario",
                    yaxis_title="Loss (%)",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

    # Return Distribution
    st.markdown("---")
    st.subheader("Return Distribution Analysis")

    if len(selected_tickers) > 0:
        # Pick first asset for detailed distribution
        selected_for_dist = st.selectbox(
            "Select Asset for Distribution Analysis",
            selected_tickers,
            key="dist_asset",
        )

        if selected_for_dist in returns_dict:
            returns = returns_dict[selected_for_dist].dropna()

            col1, col2 = st.columns(2)

            with col1:
                # Histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=returns * 100,
                    nbinsx=50,
                    name="Returns",
                    marker_color="#2196F3",
                    opacity=0.7,
                ))

                # Add VaR lines
                var_95 = historical_var(returns, 0.95)
                var_99 = historical_var(returns, 0.99)

                fig.add_vline(x=-var_95 * 100, line_dash="dash", line_color="orange",
                              annotation_text=f"VaR 95%: {-var_95*100:.1f}%")
                fig.add_vline(x=-var_99 * 100, line_dash="dash", line_color="red",
                              annotation_text=f"VaR 99%: {-var_99*100:.1f}%")

                fig.update_layout(
                    title=f"Return Distribution: {selected_for_dist}",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Distribution statistics
                st.markdown("**Distribution Statistics**")
                st.write(f"Mean: {returns.mean()*100:.3f}%")
                st.write(f"Std Dev: {returns.std()*100:.3f}%")
                st.write(f"Skewness: {returns.skew():.3f}")
                st.write(f"Kurtosis: {returns.kurtosis():.3f}")
                st.write(f"Min: {returns.min()*100:.2f}%")
                st.write(f"Max: {returns.max()*100:.2f}%")
                st.write(f"5th Percentile: {np.percentile(returns, 5)*100:.2f}%")
                st.write(f"95th Percentile: {np.percentile(returns, 95)*100:.2f}%")

    # Disclaimer
    st.markdown("---")
    st.caption(
        "**Disclaimer:** Risk metrics are based on historical data and may not predict future results. "
        "VaR and CVaR are statistical measures and do not account for all possible risks. "
        "This analysis is for educational purposes only."
    )


if __name__ == "__main__":
    main()
