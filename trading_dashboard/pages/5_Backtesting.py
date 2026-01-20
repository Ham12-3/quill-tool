"""
Backtesting page - Test trading strategies on historical data.
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

from src.datasource import DemoDataSource
from src.backtest import (
    BuyAndHoldStrategy, MovingAverageCrossoverStrategy,
    RSIMeanReversionStrategy, MACDStrategy,
    run_backtest, compare_strategies, get_trade_statistics
)
from src.analytics.performance import drawdown
from src.utils import format_percent, format_currency

st.set_page_config(page_title="Backtesting", page_icon="📈", layout="wide")


@st.cache_data(ttl=300)
def get_asset_data(ticker: str, start: str, end: str):
    """Cached function to get asset data."""
    source = DemoDataSource()
    return source.get_price_history(ticker, start, end)


def create_backtest_chart(df: pd.DataFrame, result, strategy_name: str) -> go.Figure:
    """Create backtest visualization chart."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"Price with Signals ({strategy_name})", "Equity Curve", "Drawdown"),
    )

    # Price chart with buy/sell markers
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Price",
            line=dict(color="#2196F3", width=1.5),
        ),
        row=1, col=1
    )

    # Buy signals
    if not result.buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=result.buy_signals["Date"],
                y=result.buy_signals["Price"],
                mode="markers",
                name="Buy",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="#26a69a",
                ),
            ),
            row=1, col=1
        )

    # Sell signals
    if not result.sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=result.sell_signals["Date"],
                y=result.sell_signals["Price"],
                mode="markers",
                name="Sell",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="#ef5350",
                ),
            ),
            row=1, col=1
        )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#4CAF50", width=2),
            fill="tozeroy",
            fillcolor="rgba(76, 175, 80, 0.1)",
        ),
        row=2, col=1
    )

    # Drawdown
    dd = drawdown(result.equity_curve)
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd * 100,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            fillcolor="rgba(239, 83, 80, 0.3)",
            line=dict(color="#ef5350", width=1),
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)

    return fig


def main():
    st.title("📈 Strategy Backtesting")
    st.markdown("Test trading strategies on historical data with detailed performance analysis.")

    # Initialize session state
    if "start_date" not in st.session_state:
        st.session_state.start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.now().strftime("%Y-%m-%d")

    source = DemoDataSource()

    # Sidebar - Backtest Settings
    with st.sidebar:
        st.subheader("Asset Selection")

        available_tickers = source.get_available_tickers()
        selected_ticker = st.selectbox("Select Ticker", available_tickers)

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
        st.subheader("Capital & Costs")

        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=1000,
        )

        transaction_cost = st.slider(
            "Transaction Cost (%)",
            0.0, 1.0, 0.1, 0.01,
        ) / 100

        slippage = st.slider(
            "Slippage (%)",
            0.0, 1.0, 0.05, 0.01,
        ) / 100

    # Main content - Strategy Selection
    st.subheader("Select Strategy")

    strategy_type = st.selectbox(
        "Strategy Type",
        ["Buy and Hold", "Moving Average Crossover", "RSI Mean Reversion", "MACD Crossover"],
    )

    # Strategy parameters
    col1, col2 = st.columns(2)

    strategy = None

    if strategy_type == "Buy and Hold":
        strategy = BuyAndHoldStrategy()
        st.info("Buy and Hold strategy: Buy on the first day and hold throughout the period.")

    elif strategy_type == "Moving Average Crossover":
        with col1:
            fast_period = st.slider("Fast MA Period", 5, 50, 10)
            use_ema = st.checkbox("Use EMA instead of SMA", value=False)
        with col2:
            slow_period = st.slider("Slow MA Period", 20, 200, 30)

        strategy = MovingAverageCrossoverStrategy(fast_period, slow_period, use_ema)
        st.info(
            f"{'EMA' if use_ema else 'SMA'} Crossover: Buy when fast ({fast_period}) crosses above slow ({slow_period}), "
            f"sell when fast crosses below slow."
        )

    elif strategy_type == "RSI Mean Reversion":
        with col1:
            rsi_period = st.slider("RSI Period", 7, 28, 14)
            oversold = st.slider("Oversold Threshold", 10, 40, 30)
        with col2:
            overbought = st.slider("Overbought Threshold", 60, 90, 70)

        strategy = RSIMeanReversionStrategy(rsi_period, oversold, overbought)
        st.info(
            f"RSI Mean Reversion: Buy when RSI({rsi_period}) crosses below {oversold}, "
            f"sell when RSI crosses above {overbought}."
        )

    elif strategy_type == "MACD Crossover":
        with col1:
            macd_fast = st.slider("MACD Fast Period", 8, 20, 12)
            macd_slow = st.slider("MACD Slow Period", 20, 40, 26)
        with col2:
            macd_signal = st.slider("MACD Signal Period", 5, 15, 9)

        strategy = MACDStrategy(macd_fast, macd_slow, macd_signal)
        st.info(
            f"MACD Crossover: Buy when MACD({macd_fast},{macd_slow}) crosses above signal({macd_signal}), "
            f"sell when MACD crosses below signal."
        )

    # Run backtest button
    st.markdown("---")

    if st.button("Run Backtest", use_container_width=True, type="primary"):
        with st.spinner("Running backtest..."):
            # Get data
            df = get_asset_data(
                selected_ticker,
                st.session_state.start_date,
                st.session_state.end_date,
            )

            if df.empty:
                st.error("No data available for the selected ticker and date range.")
                return

            # Run backtest
            result = run_backtest(
                df,
                strategy,
                initial_cash=float(initial_capital),
                transaction_cost=transaction_cost,
                slippage=slippage,
            )

            # Store result in session state
            st.session_state.backtest_result = result
            st.session_state.backtest_df = df

    # Display results if available
    if "backtest_result" in st.session_state:
        result = st.session_state.backtest_result
        df = st.session_state.backtest_df

        st.markdown("---")
        st.subheader("Backtest Results")

        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Final Value", format_currency(result.final_value))

        with col2:
            st.metric("Total Return", format_percent(result.metrics.total_return))

        with col3:
            st.metric("Sharpe Ratio", f"{result.metrics.sharpe_ratio:.2f}")

        with col4:
            st.metric("Max Drawdown", format_percent(result.metrics.max_drawdown))

        with col5:
            trade_stats = get_trade_statistics(result.trades)
            st.metric("Win Rate", format_percent(trade_stats.get("win_rate", 0)))

        # Charts
        fig = create_backtest_chart(df, result, result.strategy_name)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed metrics
        st.markdown("---")
        st.subheader("Performance Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Return Metrics**")
            st.write(f"Initial Capital: {format_currency(initial_capital)}")
            st.write(f"Final Value: {format_currency(result.final_value)}")
            st.write(f"Total Return: {format_percent(result.metrics.total_return)}")
            st.write(f"Annualized Return: {format_percent(result.metrics.annualized_return)}")
            st.write(f"Best Day: {format_percent(result.metrics.best_day)}")
            st.write(f"Worst Day: {format_percent(result.metrics.worst_day)}")

        with col2:
            st.markdown("**Risk Metrics**")
            st.write(f"Annualized Volatility: {format_percent(result.metrics.annualized_volatility)}")
            st.write(f"Max Drawdown: {format_percent(result.metrics.max_drawdown)}")
            st.write(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
            st.write(f"Sortino Ratio: {result.metrics.sortino_ratio:.2f}")
            st.write(f"Calmar Ratio: {result.metrics.calmar_ratio:.2f}")

        with col3:
            st.markdown("**Trade Statistics**")
            trade_stats = get_trade_statistics(result.trades)
            st.write(f"Total Trades: {trade_stats.get('total_trades', 0)}")
            st.write(f"Winning Trades: {trade_stats.get('winning_trades', 0)}")
            st.write(f"Losing Trades: {trade_stats.get('losing_trades', 0)}")
            st.write(f"Win Rate: {format_percent(trade_stats.get('win_rate', 0))}")
            st.write(f"Profit Factor: {trade_stats.get('profit_factor', 0):.2f}")
            st.write(f"Avg Trade Duration: {trade_stats.get('avg_trade_duration', 0):.1f} days")

        # Trade log
        st.markdown("---")
        st.subheader("Trade Log")

        if result.trades:
            trade_data = []
            for trade in result.trades:
                trade_data.append({
                    "Entry Date": trade.entry_date.strftime("%Y-%m-%d") if trade.entry_date else "",
                    "Exit Date": trade.exit_date.strftime("%Y-%m-%d") if trade.exit_date else "Open",
                    "Direction": trade.direction.upper(),
                    "Entry Price": f"${trade.entry_price:.2f}",
                    "Exit Price": f"${trade.exit_price:.2f}" if trade.exit_price else "-",
                    "Shares": f"{trade.position_size:.0f}",
                    "P&L": f"${trade.pnl:.2f}" if trade.pnl else "-",
                    "P&L %": format_percent(trade.pnl_pct) if trade.pnl_pct else "-",
                })

            trade_df = pd.DataFrame(trade_data)
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades executed during this backtest.")

    # Strategy comparison
    st.markdown("---")
    st.subheader("Strategy Comparison")

    if st.button("Compare All Strategies", use_container_width=True):
        with st.spinner("Running strategy comparison..."):
            df = get_asset_data(
                selected_ticker,
                st.session_state.start_date,
                st.session_state.end_date,
            )

            if df.empty:
                st.error("No data available.")
            else:
                strategies = [
                    BuyAndHoldStrategy(),
                    MovingAverageCrossoverStrategy(10, 30),
                    RSIMeanReversionStrategy(14, 30, 70),
                    MACDStrategy(12, 26, 9),
                ]

                results = compare_strategies(
                    df, strategies, initial_capital, transaction_cost, slippage
                )

                # Comparison chart
                fig = go.Figure()
                for name, result in results.items():
                    fig.add_trace(go.Scatter(
                        x=result.equity_curve.index,
                        y=result.equity_curve.values,
                        mode="lines",
                        name=name,
                    ))

                fig.update_layout(
                    title="Strategy Equity Curves Comparison",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Comparison table
                comparison_data = []
                for name, result in results.items():
                    trade_stats = get_trade_statistics(result.trades)
                    comparison_data.append({
                        "Strategy": name,
                        "Final Value": format_currency(result.final_value),
                        "Total Return": format_percent(result.metrics.total_return),
                        "Ann. Return": format_percent(result.metrics.annualized_return),
                        "Volatility": format_percent(result.metrics.annualized_volatility),
                        "Sharpe": f"{result.metrics.sharpe_ratio:.2f}",
                        "Max DD": format_percent(result.metrics.max_drawdown),
                        "Trades": trade_stats.get("total_trades", 0),
                        "Win Rate": format_percent(trade_stats.get("win_rate", 0)),
                    })

                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Disclaimer
    st.markdown("---")
    st.caption(
        "**Disclaimer:** Backtesting results are hypothetical and do not guarantee future performance. "
        "Past results do not predict future returns. Transaction costs and slippage may vary in real trading. "
        "This analysis is for educational purposes only and does not constitute trading advice."
    )


if __name__ == "__main__":
    main()
