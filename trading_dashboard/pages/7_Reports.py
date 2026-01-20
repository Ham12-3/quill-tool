"""
Reports page - Generate PDF reports for portfolios and backtests.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasource import DemoDataSource
from src.portfolio import (
    Portfolio, load_portfolio_data, calculate_portfolio_equity_curve,
    calculate_portfolio_metrics, portfolio_risk_metrics
)
from src.reports import PortfolioReportGenerator, chart_to_image, format_percent, format_currency
from src.analytics import calculate_returns

st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")


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
    st.title("📄 Reports")
    st.markdown("Generate PDF reports for your portfolio and backtest analysis.")

    init_portfolio()

    if "start_date" not in st.session_state:
        st.session_state.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.now().strftime("%Y-%m-%d")

    source = DemoDataSource()
    portfolio = st.session_state.portfolio

    # Tabs for different report types
    tab1, tab2 = st.tabs(["📊 Portfolio Report", "📈 Backtest Report"])

    # ==================== PORTFOLIO REPORT ====================
    with tab1:
        st.subheader("Generate Portfolio Report")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Report Preview")

            # Load portfolio data
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
                        if portfolio.positions[ticker].metadata is None:
                            portfolio.positions[ticker].metadata = source.get_metadata(ticker)
                except Exception:
                    pass

            if not portfolio_data:
                st.error("Could not load portfolio data.")
            else:
                # Calculate metrics
                equity_curve = calculate_portfolio_equity_curve(portfolio, portfolio_data)
                metrics = calculate_portfolio_metrics(portfolio, portfolio_data)
                risk_metrics = portfolio_risk_metrics(portfolio, portfolio_data)

                # Preview metrics
                st.markdown("**Performance Summary**")
                preview_data = {
                    "Metric": [
                        "Total Return",
                        "Annualized Return",
                        "Annualized Volatility",
                        "Sharpe Ratio",
                        "Maximum Drawdown",
                        "Calmar Ratio",
                    ],
                    "Value": [
                        format_percent(metrics.total_return),
                        format_percent(metrics.annualized_return),
                        format_percent(metrics.annualized_volatility),
                        f"{metrics.sharpe_ratio:.2f}",
                        format_percent(metrics.max_drawdown),
                        f"{metrics.calmar_ratio:.2f}",
                    ]
                }
                st.dataframe(pd.DataFrame(preview_data), hide_index=True, use_container_width=True)

                st.markdown("**Portfolio Allocation**")
                weights = portfolio.get_weights()
                alloc_data = {
                    "Ticker": list(weights.keys()),
                    "Weight": [format_percent(w) for w in weights.values()],
                }
                st.dataframe(pd.DataFrame(alloc_data), hide_index=True, use_container_width=True)

                # Equity curve preview
                st.markdown("**Equity Curve Preview**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve.values,
                    mode="lines",
                    line=dict(color="#2196F3", width=2),
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(l=50, r=20, t=20, b=50),
                    xaxis_title="Date",
                    yaxis_title="Value ($)",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Report Settings")

            report_title = st.text_input(
                "Report Title",
                value="Portfolio Analytics Report",
            )

            include_charts = st.checkbox("Include Charts", value=True)

            st.markdown("---")
            st.markdown("### Date Range")
            st.write(f"Start: {st.session_state.start_date}")
            st.write(f"End: {st.session_state.end_date}")

            st.markdown("---")

            # Generate PDF button
            if st.button("📥 Generate PDF Report", use_container_width=True, type="primary"):
                with st.spinner("Generating PDF report..."):
                    try:
                        # Create chart images
                        chart_images = {}

                        if include_charts:
                            # Equity curve chart
                            eq_fig = go.Figure()
                            eq_fig.add_trace(go.Scatter(
                                x=equity_curve.index,
                                y=equity_curve.values,
                                mode="lines",
                                fill="tozeroy",
                                line=dict(color="#2196F3", width=2),
                            ))
                            eq_fig.update_layout(
                                title="Portfolio Equity Curve",
                                xaxis_title="Date",
                                yaxis_title="Value ($)",
                                height=400,
                                width=700,
                            )

                            try:
                                eq_img = chart_to_image(eq_fig)
                                if eq_img:
                                    chart_images["Equity Curve"] = eq_img
                            except Exception as e:
                                st.warning(f"Could not export equity chart: {e}")

                            # Allocation pie chart
                            pie_fig = go.Figure(data=[go.Pie(
                                labels=list(weights.keys()),
                                values=list(weights.values()),
                                hole=0.3,
                            )])
                            pie_fig.update_layout(
                                title="Portfolio Allocation",
                                height=400,
                                width=700,
                            )

                            try:
                                pie_img = chart_to_image(pie_fig)
                                if pie_img:
                                    chart_images["Allocation"] = pie_img
                            except Exception as e:
                                st.warning(f"Could not export allocation chart: {e}")

                        # Generate PDF
                        generator = PortfolioReportGenerator(title=report_title)
                        pdf_bytes = generator.generate_report(
                            portfolio_name=portfolio.name,
                            positions=weights,
                            metrics=metrics,
                            risk_metrics=risk_metrics,
                            start_date=st.session_state.start_date,
                            end_date=st.session_state.end_date,
                            chart_images=chart_images if include_charts else None,
                        )

                        # Offer download
                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_bytes,
                            file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )

                        st.success("PDF report generated successfully!")

                    except Exception as e:
                        st.error(f"Error generating PDF: {e}")

    # ==================== BACKTEST REPORT ====================
    with tab2:
        st.subheader("Generate Backtest Report")

        if "backtest_result" not in st.session_state:
            st.info(
                "No backtest results available. Please run a backtest first on the Backtesting page."
            )

            # Link to backtesting page
            st.markdown("[Go to Backtesting Page →](5_Backtesting)")

        else:
            result = st.session_state.backtest_result

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### Backtest Preview")

                st.write(f"**Strategy:** {result.strategy_name}")
                st.write(f"**Ticker:** {result.ticker}")
                st.write(f"**Period:** {result.start_date} to {result.end_date}")

                # Preview metrics
                preview_data = {
                    "Metric": [
                        "Initial Capital",
                        "Final Value",
                        "Total Return",
                        "Sharpe Ratio",
                        "Max Drawdown",
                    ],
                    "Value": [
                        format_currency(result.initial_cash),
                        format_currency(result.final_value),
                        format_percent(result.metrics.total_return),
                        f"{result.metrics.sharpe_ratio:.2f}",
                        format_percent(result.metrics.max_drawdown),
                    ]
                }
                st.dataframe(pd.DataFrame(preview_data), hide_index=True, use_container_width=True)

                # Equity curve preview
                st.markdown("**Equity Curve Preview**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve.values,
                    mode="lines",
                    line=dict(color="#4CAF50", width=2),
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(l=50, r=20, t=20, b=50),
                    xaxis_title="Date",
                    yaxis_title="Value ($)",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Report Settings")

                bt_report_title = st.text_input(
                    "Report Title",
                    value=f"Backtest Report: {result.strategy_name}",
                    key="bt_title",
                )

                include_bt_charts = st.checkbox("Include Charts", value=True, key="bt_charts")

                st.markdown("---")

                if st.button("📥 Generate Backtest PDF", use_container_width=True, type="primary"):
                    with st.spinner("Generating backtest report..."):
                        try:
                            from src.reports import generate_backtest_report
                            from src.backtest import get_trade_statistics

                            trade_stats = get_trade_statistics(result.trades)

                            # Create chart images
                            chart_images = {}

                            if include_bt_charts:
                                # Equity curve
                                eq_fig = go.Figure()
                                eq_fig.add_trace(go.Scatter(
                                    x=result.equity_curve.index,
                                    y=result.equity_curve.values,
                                    mode="lines",
                                    fill="tozeroy",
                                    line=dict(color="#4CAF50", width=2),
                                ))
                                eq_fig.update_layout(
                                    title="Equity Curve",
                                    xaxis_title="Date",
                                    yaxis_title="Value ($)",
                                    height=400,
                                    width=700,
                                )

                                try:
                                    eq_img = chart_to_image(eq_fig)
                                    if eq_img:
                                        chart_images["Equity Curve"] = eq_img
                                except Exception:
                                    pass

                            # Generate PDF
                            pdf_bytes = generate_backtest_report(
                                strategy_name=result.strategy_name,
                                ticker=result.ticker,
                                metrics=result.metrics,
                                trade_stats=trade_stats,
                                start_date=result.start_date,
                                end_date=result.end_date,
                                initial_capital=result.initial_cash,
                                final_value=result.final_value,
                                chart_images=chart_images if include_bt_charts else None,
                            )

                            # Offer download
                            st.download_button(
                                label="📥 Download Backtest PDF",
                                data=pdf_bytes,
                                file_name=f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )

                            st.success("Backtest PDF generated successfully!")

                        except Exception as e:
                            st.error(f"Error generating PDF: {e}")

    # Quick export options
    st.markdown("---")
    st.subheader("Quick Data Export")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Portfolio Weights (CSV)**")
        if portfolio.positions:
            weights = portfolio.get_weights()
            weights_df = pd.DataFrame({
                "Ticker": list(weights.keys()),
                "Weight": list(weights.values()),
            })
            csv = weights_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="portfolio_weights.csv",
                mime="text/csv",
            )

    with col2:
        st.markdown("**Performance Metrics (CSV)**")
        if portfolio_data:
            metrics_dict = {
                "Metric": [
                    "Total Return",
                    "Annualized Return",
                    "Volatility",
                    "Sharpe Ratio",
                    "Max Drawdown",
                    "Calmar Ratio",
                    "VaR 95%",
                    "CVaR 95%",
                ],
                "Value": [
                    metrics.total_return,
                    metrics.annualized_return,
                    metrics.annualized_volatility,
                    metrics.sharpe_ratio,
                    metrics.max_drawdown,
                    metrics.calmar_ratio,
                    risk_metrics.get("var_95", 0),
                    risk_metrics.get("cvar_95", 0),
                ]
            }
            metrics_df = pd.DataFrame(metrics_dict)
            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="performance_metrics.csv",
                mime="text/csv",
            )

    with col3:
        st.markdown("**Equity Curve (CSV)**")
        if portfolio_data and not equity_curve.empty:
            eq_df = equity_curve.reset_index()
            eq_df.columns = ["Date", "Value"]
            csv = eq_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="equity_curve.csv",
                mime="text/csv",
            )

    # Disclaimer
    st.markdown("---")
    st.caption(
        "**Disclaimer:** Reports generated are for educational and analytical purposes only. "
        "They do not constitute financial advice. Past performance does not guarantee future results."
    )


if __name__ == "__main__":
    main()
