"""
Portfolio Builder page - Create and analyze custom portfolios.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasource import DemoDataSource, DEMO_ASSETS
from src.portfolio import (
    Portfolio, load_portfolio_data, calculate_portfolio_equity_curve,
    calculate_portfolio_metrics, get_allocation_by_category,
    portfolio_risk_metrics, optimize_weights_minvar
)
from src.analytics import calculate_returns, drawdown
from src.analytics.performance import performance_attribution
from src.utils import format_percent, format_currency

st.set_page_config(page_title="Portfolio Builder", page_icon="💼", layout="wide")


def init_portfolio():
    """Initialize default portfolio in session state."""
    if "portfolio" not in st.session_state:
        portfolio = Portfolio(name="My Portfolio", initial_value=100000.0)
        portfolio.add_position("DEMO_TECH", 0.25)
        portfolio.add_position("DEMO_BANK", 0.20)
        portfolio.add_position("DEMO_PHARMA", 0.20)
        portfolio.add_position("DEMO_OIL", 0.15)
        portfolio.add_position("DEMO_SPY", 0.20)
        st.session_state.portfolio = portfolio


def main():
    st.title("💼 Portfolio Builder")
    st.markdown("Create and analyze custom portfolios with detailed performance metrics.")

    init_portfolio()

    if "start_date" not in st.session_state:
        st.session_state.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.now().strftime("%Y-%m-%d")

    source = DemoDataSource()
    portfolio = st.session_state.portfolio

    # Sidebar - Portfolio Management
    with st.sidebar:
        st.subheader("Portfolio Management")

        # Portfolio name
        new_name = st.text_input("Portfolio Name", value=portfolio.name)
        if new_name != portfolio.name:
            portfolio.name = new_name

        # Initial value
        new_value = st.number_input(
            "Initial Value ($)",
            min_value=1000.0,
            max_value=10000000.0,
            value=portfolio.initial_value,
            step=1000.0,
        )
        portfolio.initial_value = new_value

        st.markdown("---")
        st.subheader("Add Position")

        available_tickers = source.get_available_tickers()
        # Filter out already added tickers
        available_to_add = [t for t in available_tickers if t not in portfolio.tickers]

        if available_to_add:
            add_ticker = st.selectbox("Select Ticker", available_to_add)
            add_weight = st.slider("Weight (%)", 1, 100, 10) / 100

            if st.button("Add Position", use_container_width=True):
                portfolio.add_position(add_ticker, add_weight)
                st.rerun()
        else:
            st.info("All available tickers are in the portfolio.")

        st.markdown("---")
        st.subheader("Quick Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Equal Weight", use_container_width=True):
                portfolio.set_equal_weights()
                st.rerun()

        with col2:
            if st.button("Normalize", use_container_width=True):
                portfolio.normalize_weights()
                st.rerun()

        if st.button("Optimize (Min Vol)", use_container_width=True):
            # Load data and optimize
            portfolio_data = {}
            for ticker in portfolio.tickers:
                try:
                    df = source.get_price_history(
                        ticker,
                        st.session_state.start_date,
                        st.session_state.end_date
                    )
                    if not df.empty:
                        portfolio_data[ticker] = df
                except Exception:
                    pass

            if portfolio_data:
                new_weights = optimize_weights_minvar(portfolio, portfolio_data)
                for ticker, weight in new_weights.items():
                    portfolio.positions[ticker].weight = weight
                st.rerun()

    # Main content - Current Holdings
    st.subheader("Current Holdings")

    if not portfolio.positions:
        st.info("No positions in portfolio. Add some positions from the sidebar.")
        return

    # Holdings table with remove buttons
    holdings_data = []
    for ticker, pos in portfolio.positions.items():
        metadata = source.get_metadata(ticker)
        holdings_data.append({
            "Ticker": ticker,
            "Name": metadata.name,
            "Sector": metadata.sector,
            "Weight": pos.weight,
        })

    col1, col2 = st.columns([3, 1])

    with col1:
        holdings_df = pd.DataFrame(holdings_data)
        holdings_df["Weight %"] = holdings_df["Weight"].apply(lambda x: f"{x*100:.1f}%")

        # Use data editor for weights
        edited_df = st.data_editor(
            holdings_df[["Ticker", "Name", "Sector", "Weight"]],
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
                "Name": st.column_config.TextColumn("Name", disabled=True),
                "Sector": st.column_config.TextColumn("Sector", disabled=True),
                "Weight": st.column_config.NumberColumn(
                    "Weight",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f",
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

        # Update weights from editor
        for i, row in edited_df.iterrows():
            ticker = row["Ticker"]
            new_weight = row["Weight"]
            if ticker in portfolio.positions:
                portfolio.positions[ticker].weight = new_weight

    with col2:
        st.markdown("**Remove Positions**")
        for ticker in list(portfolio.positions.keys()):
            if st.button(f"Remove {ticker}", key=f"remove_{ticker}", use_container_width=True):
                portfolio.remove_position(ticker)
                st.rerun()

    # Weight summary
    total_weight = portfolio.total_weight
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"Total weight: {total_weight*100:.1f}% (should be 100%)")
    else:
        st.success(f"Total weight: {total_weight*100:.1f}%")

    st.markdown("---")

    # Load portfolio data for analysis
    portfolio_data = {}
    for ticker in portfolio.tickers:
        try:
            df = source.get_price_history(
                ticker,
                st.session_state.start_date,
                st.session_state.end_date
            )
            if not df.empty:
                portfolio_data[ticker] = df
                # Update metadata
                if portfolio.positions[ticker].metadata is None:
                    portfolio.positions[ticker].metadata = source.get_metadata(ticker)
        except Exception:
            pass

    if not portfolio_data:
        st.error("Could not load data for portfolio positions.")
        return

    # Portfolio Performance
    st.subheader("Portfolio Performance")

    equity_curve = calculate_portfolio_equity_curve(portfolio, portfolio_data)
    metrics = calculate_portfolio_metrics(portfolio, portfolio_data)
    risk_metrics = portfolio_risk_metrics(portfolio, portfolio_data)

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        final_value = equity_curve.iloc[-1] if not equity_curve.empty else portfolio.initial_value
        st.metric("Portfolio Value", format_currency(final_value))

    with col2:
        st.metric("Total Return", format_percent(metrics.total_return))

    with col3:
        st.metric("Ann. Return", format_percent(metrics.annualized_return))

    with col4:
        st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")

    with col5:
        st.metric("Max Drawdown", format_percent(metrics.max_drawdown))

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Equity curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#2196F3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33, 150, 243, 0.1)",
        ))
        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Allocation pie chart
        weights = portfolio.get_weights()
        fig = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            hole=0.4,
            textinfo="label+percent",
        )])
        fig.update_layout(
            title="Portfolio Allocation",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Sector/Country breakdown
    st.markdown("---")
    st.subheader("Allocation Breakdown")

    col1, col2, col3 = st.columns(3)

    with col1:
        sector_alloc = get_allocation_by_category(portfolio, "sector")
        fig = px.treemap(
            names=list(sector_alloc.keys()),
            parents=[""] * len(sector_alloc),
            values=list(sector_alloc.values()),
            title="By Sector",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        country_alloc = get_allocation_by_category(portfolio, "country")
        fig = px.treemap(
            names=list(country_alloc.keys()),
            parents=[""] * len(country_alloc),
            values=list(country_alloc.values()),
            title="By Country",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        asset_class_alloc = get_allocation_by_category(portfolio, "asset_class")
        fig = px.treemap(
            names=list(asset_class_alloc.keys()),
            parents=[""] * len(asset_class_alloc),
            values=list(asset_class_alloc.values()),
            title="By Asset Class",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Performance Attribution
    st.markdown("---")
    st.subheader("Performance Attribution")

    asset_returns = {}
    for ticker, df in portfolio_data.items():
        asset_returns[ticker] = calculate_returns(df["Close"])

    attribution_df = performance_attribution(
        portfolio.get_weights(),
        asset_returns,
        st.session_state.start_date,
        st.session_state.end_date,
    )

    if not attribution_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Attribution table
            display_df = attribution_df.copy()
            display_df["Weight"] = display_df["Weight"].map("{:.1%}".format)
            display_df["Asset Return"] = display_df["Asset Return"].map("{:.2%}".format)
            display_df["Contribution"] = display_df["Contribution"].map("{:.2%}".format)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        with col2:
            # Contribution bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=attribution_df["Ticker"],
                    y=attribution_df["Contribution"] * 100,
                    marker_color=[
                        "#26a69a" if c >= 0 else "#ef5350"
                        for c in attribution_df["Contribution"]
                    ],
                )
            ])
            fig.update_layout(
                title="Return Contribution by Asset (%)",
                xaxis_title="Ticker",
                yaxis_title="Contribution (%)",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Risk Metrics
    st.markdown("---")
    st.subheader("Risk Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("VaR (95%)", format_percent(risk_metrics.get("var_95", 0)))

    with col2:
        st.metric("VaR (99%)", format_percent(risk_metrics.get("var_99", 0)))

    with col3:
        st.metric("CVaR (95%)", format_percent(risk_metrics.get("cvar_95", 0)))

    with col4:
        st.metric("Ann. Volatility", format_percent(risk_metrics.get("annualized_vol", 0)))

    # Drawdown chart
    if not equity_curve.empty:
        dd = drawdown(equity_curve)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd * 100,
            fill="tozeroy",
            fillcolor="rgba(239, 83, 80, 0.3)",
            line=dict(color="#ef5350", width=1),
            name="Drawdown",
        ))
        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Disclaimer
    st.markdown("---")
    st.caption(
        "**Disclaimer:** This portfolio analysis is for educational purposes only. "
        "It does not constitute financial or investment advice. Past performance does not guarantee future results."
    )


if __name__ == "__main__":
    main()
