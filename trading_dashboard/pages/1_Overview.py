"""
Overview page - Dashboard summary with key metrics and market overview.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasource import DemoDataSource, DEMO_ASSETS
from src.analytics import rsi, calculate_returns, annualized_volatility
from src.portfolio import (
    load_portfolio_data, calculate_portfolio_equity_curve,
    calculate_portfolio_metrics, portfolio_risk_metrics
)
from src.utils import format_percent, format_currency, format_change

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")


@st.cache_data(ttl=300)
def get_market_data(tickers: list, start: str, end: str, source_name: str):
    """Cached function to get market data for multiple tickers."""
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


def create_mini_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """Create a small sparkline-style chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        line=dict(width=2, color="#1f77b4"),
        fill="tozeroy",
        fillcolor="rgba(31, 119, 180, 0.1)",
    ))
    fig.update_layout(
        height=120,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=title, font=dict(size=12)),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def main():
    st.title("📊 Market Overview")
    st.markdown("Dashboard summary with key metrics and market overview.")

    # Initialize session state if needed
    if "portfolio" not in st.session_state:
        from src.portfolio import Portfolio
        portfolio = Portfolio(name="My Portfolio", initial_value=100000.0)
        portfolio.add_position("DEMO_TECH", 0.25)
        portfolio.add_position("DEMO_BANK", 0.20)
        portfolio.add_position("DEMO_PHARMA", 0.20)
        portfolio.add_position("DEMO_OIL", 0.15)
        portfolio.add_position("DEMO_SPY", 0.20)
        st.session_state.portfolio = portfolio

    if "start_date" not in st.session_state:
        st.session_state.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.now().strftime("%Y-%m-%d")

    # Get data
    source = DemoDataSource()
    portfolio = st.session_state.portfolio
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date

    # Market Overview Tickers
    market_tickers = ["DEMO_SPY", "DEMO_QQQ", "DEMO_TECH", "DEMO_BANK", "DEMO_OIL", "DEMO_BTC"]

    # Fetch market data
    market_data = get_market_data(market_tickers, start_date, end_date, "demo")

    # Market Summary Cards
    st.subheader("Market Summary")
    cols = st.columns(len(market_tickers))

    for i, ticker in enumerate(market_tickers):
        with cols[i]:
            if ticker in market_data:
                df = market_data[ticker]
                if len(df) > 1:
                    latest = df["Close"].iloc[-1]
                    prev = df["Close"].iloc[-2]
                    change = (latest / prev - 1)

                    metadata = source.get_metadata(ticker)
                    st.metric(
                        metadata.name[:15] + "..." if len(metadata.name) > 15 else metadata.name,
                        f"${latest:.2f}",
                        f"{change*100:+.2f}%"
                    )
            else:
                st.metric(ticker, "N/A", "")

    st.markdown("---")

    # Portfolio Summary
    st.subheader("Portfolio Summary")

    # Load portfolio data
    portfolio_data = {}
    for ticker in portfolio.tickers:
        try:
            df = source.get_price_history(ticker, start_date, end_date)
            if not df.empty:
                portfolio_data[ticker] = df
        except Exception:
            pass

    if portfolio_data:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Portfolio equity curve
            equity_curve = calculate_portfolio_equity_curve(portfolio, portfolio_data)

            if not equity_curve.empty:
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
                    height=350,
                    margin=dict(l=50, r=20, t=50, b=50),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Portfolio metrics
            metrics = calculate_portfolio_metrics(portfolio, portfolio_data)

            st.metric("Portfolio Value", format_currency(equity_curve.iloc[-1] if not equity_curve.empty else 100000))
            st.metric("Total Return", format_percent(metrics.total_return))
            st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
            st.metric("Max Drawdown", format_percent(metrics.max_drawdown))

        # Portfolio Allocation
        st.subheader("Portfolio Allocation")
        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            weights = portfolio.get_weights()
            fig = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=0.4,
                textinfo="label+percent",
                textposition="outside",
            )])
            fig.update_layout(
                title="Allocation by Asset",
                height=350,
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Allocation table
            allocation_data = []
            for ticker, weight in weights.items():
                metadata = source.get_metadata(ticker)
                if ticker in portfolio_data:
                    df = portfolio_data[ticker]
                    current_price = df["Close"].iloc[-1]
                    returns = calculate_returns(df["Close"])
                    period_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1)
                else:
                    current_price = 0
                    period_return = 0

                allocation_data.append({
                    "Ticker": ticker,
                    "Name": metadata.name[:20],
                    "Weight": f"{weight*100:.1f}%",
                    "Price": f"${current_price:.2f}",
                    "Return": format_percent(period_return),
                })

            st.dataframe(
                pd.DataFrame(allocation_data),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    # Market Movers
    st.subheader("Top Movers (Demo Data)")

    # Calculate daily changes for all demo assets
    movers_data = []
    all_tickers = list(DEMO_ASSETS.keys())

    for ticker in all_tickers:
        try:
            df = source.get_price_history(ticker, start_date, end_date)
            if len(df) > 1:
                latest = df["Close"].iloc[-1]
                prev = df["Close"].iloc[-2]
                change = (latest / prev - 1)
                metadata = source.get_metadata(ticker)
                movers_data.append({
                    "ticker": ticker,
                    "name": metadata.name,
                    "price": latest,
                    "change": change,
                })
        except Exception:
            pass

    if movers_data:
        movers_df = pd.DataFrame(movers_data)
        movers_df = movers_df.sort_values("change", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top Gainers**")
            gainers = movers_df.head(5)
            for _, row in gainers.iterrows():
                st.markdown(f"🟢 **{row['ticker']}** - ${row['price']:.2f} ({row['change']*100:+.2f}%)")

        with col2:
            st.markdown("**Top Losers**")
            losers = movers_df.tail(5).iloc[::-1]
            for _, row in losers.iterrows():
                st.markdown(f"🔴 **{row['ticker']}** - ${row['price']:.2f} ({row['change']*100:+.2f}%)")

    # Mini charts for key assets
    st.markdown("---")
    st.subheader("Key Asset Charts")

    chart_tickers = ["DEMO_SPY", "DEMO_QQQ", "DEMO_BTC", "DEMO_TECH"]
    cols = st.columns(len(chart_tickers))

    for i, ticker in enumerate(chart_tickers):
        with cols[i]:
            if ticker in market_data:
                df = market_data[ticker]
                metadata = source.get_metadata(ticker)
                fig = create_mini_chart(df, metadata.name)
                st.plotly_chart(fig, use_container_width=True)

    # Disclaimer
    st.markdown("---")
    st.caption(
        "**Disclaimer:** This dashboard is for educational and analytical purposes only. "
        "It does not constitute financial advice. All data shown is demo/simulated data."
    )


if __name__ == "__main__":
    main()
