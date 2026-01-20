"""
Asset Explorer page - Detailed analysis of individual assets with charts and indicators.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasource import DemoDataSource, DEMO_ASSETS
from src.analytics import (
    sma, ema, rsi, macd, bollinger_bands, atr,
    calculate_returns, annualized_volatility, historical_var, historical_cvar
)
from src.analytics.performance import calculate_all_metrics, drawdown
from src.utils import format_percent, format_currency

st.set_page_config(page_title="Asset Explorer", page_icon="🔍", layout="wide")


@st.cache_data(ttl=300)
def get_asset_data(ticker: str, start: str, end: str, source_name: str = "demo"):
    """Cached function to get asset data."""
    source = DemoDataSource()
    return source.get_price_history(ticker, start, end)


def create_candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    show_sma: bool = True,
    show_ema: bool = False,
    show_bb: bool = False,
    sma_period: int = 20,
    ema_period: int = 12,
) -> go.Figure:
    """Create a candlestick chart with optional indicators."""

    # Create subplots: candlestick, volume, RSI, MACD
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(f"{ticker} Price", "Volume", "RSI (14)", "MACD"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1
    )

    # SMA
    if show_sma:
        sma_values = sma(df["Close"], sma_period)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sma_values,
                name=f"SMA({sma_period})",
                line=dict(color="orange", width=1.5),
            ),
            row=1, col=1
        )

    # EMA
    if show_ema:
        ema_values = ema(df["Close"], ema_period)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ema_values,
                name=f"EMA({ema_period})",
                line=dict(color="purple", width=1.5),
            ),
            row=1, col=1
        )

    # Bollinger Bands
    if show_bb:
        bb_mid, bb_upper, bb_lower = bollinger_bands(df["Close"])
        fig.add_trace(
            go.Scatter(
                x=df.index, y=bb_upper, name="BB Upper",
                line=dict(color="gray", width=1, dash="dash"),
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=bb_lower, name="BB Lower",
                line=dict(color="gray", width=1, dash="dash"),
                fill="tonexty", fillcolor="rgba(128,128,128,0.1)",
            ),
            row=1, col=1
        )

    # Volume bars
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.7,
        ),
        row=2, col=1
    )

    # RSI
    rsi_values = rsi(df["Close"], 14)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=rsi_values,
            name="RSI",
            line=dict(color="#2196F3", width=1.5),
        ),
        row=3, col=1
    )
    # RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # MACD
    macd_line, signal_line, histogram = macd(df["Close"])
    fig.add_trace(
        go.Scatter(
            x=df.index, y=macd_line, name="MACD",
            line=dict(color="#2196F3", width=1.5),
        ),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=signal_line, name="Signal",
            line=dict(color="#FF9800", width=1.5),
        ),
        row=4, col=1
    )
    hist_colors = ["#26a69a" if h >= 0 else "#ef5350" for h in histogram]
    fig.add_trace(
        go.Bar(
            x=df.index, y=histogram, name="Histogram",
            marker_color=hist_colors, opacity=0.5,
        ),
        row=4, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    return fig


def main():
    st.title("🔍 Asset Explorer")
    st.markdown("Detailed analysis of individual assets with charts and technical indicators.")

    # Initialize session state
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = "DEMO_TECH"
    if "start_date" not in st.session_state:
        st.session_state.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.now().strftime("%Y-%m-%d")

    source = DemoDataSource()

    # Sidebar controls
    with st.sidebar:
        st.subheader("Asset Selection")

        # Ticker selection
        available_tickers = source.get_available_tickers()
        selected_ticker = st.selectbox(
            "Select Ticker",
            available_tickers,
            index=available_tickers.index(st.session_state.selected_ticker)
            if st.session_state.selected_ticker in available_tickers else 0,
        )
        st.session_state.selected_ticker = selected_ticker

        st.markdown("---")
        st.subheader("Indicators")

        show_sma = st.checkbox("Show SMA", value=True)
        sma_period = st.slider("SMA Period", 5, 200, 20) if show_sma else 20

        show_ema = st.checkbox("Show EMA", value=False)
        ema_period = st.slider("EMA Period", 5, 200, 12) if show_ema else 12

        show_bb = st.checkbox("Show Bollinger Bands", value=False)

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

    # Get asset data
    df = get_asset_data(
        selected_ticker,
        st.session_state.start_date,
        st.session_state.end_date,
    )

    if df.empty:
        st.error("No data available for the selected ticker and date range.")
        return

    # Asset metadata
    metadata = source.get_metadata(selected_ticker)

    # Header with asset info
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Asset", metadata.name)

    with col2:
        st.metric("Sector", metadata.sector)

    with col3:
        st.metric("Country", metadata.country)

    with col4:
        st.metric("Asset Class", metadata.asset_class)

    # Price metrics
    st.markdown("---")
    st.subheader("Price Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    latest_price = df["Close"].iloc[-1]
    prev_price = df["Close"].iloc[-2] if len(df) > 1 else latest_price
    daily_change = (latest_price / prev_price - 1)
    period_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1)

    with col1:
        st.metric("Current Price", f"${latest_price:.2f}", f"{daily_change*100:+.2f}%")

    with col2:
        st.metric("Period High", f"${df['High'].max():.2f}")

    with col3:
        st.metric("Period Low", f"${df['Low'].min():.2f}")

    with col4:
        st.metric("Period Return", format_percent(period_return))

    with col5:
        avg_volume = df["Volume"].mean()
        st.metric("Avg Volume", f"{avg_volume/1e6:.2f}M")

    # Main chart
    st.markdown("---")
    st.subheader("Price Chart with Indicators")

    fig = create_candlestick_chart(
        df, selected_ticker,
        show_sma=show_sma,
        show_ema=show_ema,
        show_bb=show_bb,
        sma_period=sma_period,
        ema_period=ema_period,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Performance metrics
    st.markdown("---")
    st.subheader("Performance Metrics")

    metrics = calculate_all_metrics(df["Close"])
    returns = calculate_returns(df["Close"]).dropna()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Return Metrics**")
        st.write(f"Total Return: {format_percent(metrics.total_return)}")
        st.write(f"Annualized Return: {format_percent(metrics.annualized_return)}")
        st.write(f"Best Day: {format_percent(metrics.best_day)}")
        st.write(f"Worst Day: {format_percent(metrics.worst_day)}")

    with col2:
        st.markdown("**Risk Metrics**")
        st.write(f"Volatility (Ann.): {format_percent(metrics.annualized_volatility)}")
        st.write(f"Max Drawdown: {format_percent(metrics.max_drawdown)}")
        var_95 = historical_var(returns, 0.95)
        st.write(f"VaR (95%): {format_percent(var_95)}")

    with col3:
        st.markdown("**Risk-Adjusted**")
        st.write(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        st.write(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
        st.write(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")

    with col4:
        st.markdown("**Distribution**")
        st.write(f"Win Rate: {format_percent(metrics.win_rate)}")
        st.write(f"Positive Days: {metrics.num_positive_days}")
        st.write(f"Negative Days: {metrics.num_negative_days}")
        st.write(f"Skewness: {metrics.skewness:.2f}")

    # Drawdown chart
    st.markdown("---")
    st.subheader("Drawdown Analysis")

    dd = drawdown(df["Close"])
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index,
        y=dd * 100,
        fill="tozeroy",
        fillcolor="rgba(239, 83, 80, 0.3)",
        line=dict(color="#ef5350", width=1),
        name="Drawdown %",
    ))
    fig_dd.update_layout(
        title="Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    fig_dd.add_hline(y=metrics.max_drawdown * 100, line_dash="dash", line_color="red",
                     annotation_text=f"Max DD: {metrics.max_drawdown*100:.1f}%")
    st.plotly_chart(fig_dd, use_container_width=True)

    # Historical data table
    st.markdown("---")
    st.subheader("Recent Price Data")

    display_df = df.tail(20).copy()
    display_df["Daily Return"] = calculate_returns(df["Close"]).tail(20)
    display_df["RSI"] = rsi(df["Close"], 14).tail(20)

    # Format for display
    display_df = display_df.reset_index()
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    display_df["Open"] = display_df["Open"].map("${:.2f}".format)
    display_df["High"] = display_df["High"].map("${:.2f}".format)
    display_df["Low"] = display_df["Low"].map("${:.2f}".format)
    display_df["Close"] = display_df["Close"].map("${:.2f}".format)
    display_df["Volume"] = display_df["Volume"].map("{:,.0f}".format)
    display_df["Daily Return"] = display_df["Daily Return"].map("{:.2%}".format)
    display_df["RSI"] = display_df["RSI"].map("{:.1f}".format)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Disclaimer
    st.markdown("---")
    st.caption(
        "**Disclaimer:** This analysis is for educational purposes only. "
        "Past performance does not guarantee future results."
    )


if __name__ == "__main__":
    main()
