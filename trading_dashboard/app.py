"""
Trading and Portfolio Analytics Dashboard
Main entry point for the Streamlit application.
"""
import streamlit as st
from datetime import datetime, timedelta
import os
import sys
import importlib.util

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasource import DemoDataSource
from src.alerts import AlertStore
from src.portfolio import Portfolio


def check_package_installed(package_name: str) -> bool:
    """Check if a package is installed and importable."""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return False
    try:
        importlib.import_module(package_name)
        return True
    except Exception:
        return False


# Check optional dependencies
YFINANCE_AVAILABLE = check_package_installed("yfinance")
CCXT_AVAILABLE = check_package_installed("ccxt")


# Page configuration
st.set_page_config(
    page_title="Trading & Portfolio Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize session state variables."""
    # Data source settings
    if "data_source" not in st.session_state:
        st.session_state.data_source = "demo"

    if "data_source_instance" not in st.session_state:
        st.session_state.data_source_instance = DemoDataSource()

    # Date range defaults
    if "start_date" not in st.session_state:
        st.session_state.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.now().strftime("%Y-%m-%d")

    # Selected ticker
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = "DEMO_TECH"

    # Portfolio
    if "portfolio" not in st.session_state:
        portfolio = Portfolio(name="My Portfolio", initial_value=100000.0)
        # Add default demo positions
        portfolio.add_position("DEMO_TECH", 0.25)
        portfolio.add_position("DEMO_BANK", 0.20)
        portfolio.add_position("DEMO_PHARMA", 0.20)
        portfolio.add_position("DEMO_OIL", 0.15)
        portfolio.add_position("DEMO_SPY", 0.20)
        st.session_state.portfolio = portfolio

    # Alerts store
    if "alert_store" not in st.session_state:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        st.session_state.alert_store = AlertStore(data_dir)

    # Auto-refresh settings
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False

    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 60

    # Risk-free rate for calculations
    if "risk_free_rate" not in st.session_state:
        st.session_state.risk_free_rate = 0.0

    # Last refresh time
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()


def get_data_source():
    """Get the current data source instance with fallback to demo."""
    source_name = st.session_state.get("data_source", "demo")

    if source_name == "yfinance" and YFINANCE_AVAILABLE:
        try:
            from src.datasource import YFinanceDataSource
            return YFinanceDataSource()
        except Exception as e:
            st.warning(f"Failed to load Yahoo Finance: {e}. Using demo data.")
            st.session_state.data_source = "demo"
            return DemoDataSource()
    elif source_name == "ccxt" and CCXT_AVAILABLE:
        try:
            from src.datasource import CCXTDataSource
            exchange = st.session_state.get("ccxt_exchange", "binance")
            return CCXTDataSource(exchange_id=exchange)
        except Exception as e:
            st.warning(f"Failed to load CCXT: {e}. Using demo data.")
            st.session_state.data_source = "demo"
            return DemoDataSource()
    else:
        # Fallback to demo if selected source is unavailable
        if source_name != "demo":
            st.session_state.data_source = "demo"
        return DemoDataSource()


def main():
    """Main application entry point."""
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("📈 Trading Dashboard")
        st.markdown("---")

        # Data source info
        source = get_data_source()
        st.caption(f"Data Source: **{source.name}**")

        # Quick ticker selection
        st.subheader("Quick Select")

        available_tickers = []
        if isinstance(source, DemoDataSource):
            available_tickers = source.get_available_tickers()
        else:
            available_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ"]

        selected = st.selectbox(
            "Select Ticker",
            available_tickers,
            index=available_tickers.index(st.session_state.selected_ticker)
            if st.session_state.selected_ticker in available_tickers else 0,
            key="ticker_select"
        )

        if selected != st.session_state.selected_ticker:
            st.session_state.selected_ticker = selected

        # Date range
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input(
                "Start",
                value=datetime.strptime(st.session_state.start_date, "%Y-%m-%d"),
                key="start_input"
            )
            st.session_state.start_date = start.strftime("%Y-%m-%d")

        with col2:
            end = st.date_input(
                "End",
                value=datetime.strptime(st.session_state.end_date, "%Y-%m-%d"),
                key="end_input"
            )
            st.session_state.end_date = end.strftime("%Y-%m-%d")

        # Refresh button
        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state.last_refresh = datetime.now()
            st.cache_data.clear()
            st.rerun()

        # Last refresh time
        st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

        # Footer
        st.markdown("---")
        st.caption("📊 Analytics & Education Tool")
        st.caption("Not financial advice")

    # Main content area - landing page
    st.title("Welcome to the Trading & Portfolio Analytics Dashboard")

    st.markdown("""
    This dashboard provides comprehensive tools for analyzing financial markets and portfolios.
    **This is an educational and analytics tool only - it does not provide investment advice.**

    ### Getting Started

    Use the **sidebar** to navigate between pages:

    - **📊 Overview** - Dashboard summary with key metrics and market overview
    - **🔍 Asset Explorer** - Detailed analysis of individual assets with charts and indicators
    - **💼 Portfolio Builder** - Create and analyze custom portfolios
    - **📉 Risk & Correlation** - Risk metrics, VaR, CVaR, and correlation analysis
    - **📈 Backtesting** - Test trading strategies on historical data
    - **🔔 Alerts & Watchlist** - Set price alerts and manage watchlists
    - **📄 Reports** - Generate PDF reports for portfolios and backtests
    - **⚙️ Settings** - Configure data sources and preferences

    ### Current Configuration
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Data Source", source.name)

    with col2:
        st.metric("Selected Ticker", st.session_state.selected_ticker)

    with col3:
        portfolio = st.session_state.portfolio
        st.metric("Portfolio Assets", len(portfolio.positions))

    # Quick stats from demo data
    st.markdown("### Quick Market Overview")

    try:
        # Get data for selected ticker
        df = source.get_price_history(
            st.session_state.selected_ticker,
            st.session_state.start_date,
            st.session_state.end_date
        )

        if not df.empty:
            col1, col2, col3, col4 = st.columns(4)

            latest_price = df["Close"].iloc[-1]
            prev_price = df["Close"].iloc[-2] if len(df) > 1 else latest_price
            daily_change = (latest_price / prev_price - 1) * 100

            with col1:
                st.metric(
                    "Current Price",
                    f"${latest_price:.2f}",
                    f"{daily_change:+.2f}%"
                )

            with col2:
                period_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
                st.metric("Period Return", f"{period_return:+.2f}%")

            with col3:
                high = df["High"].max()
                st.metric("Period High", f"${high:.2f}")

            with col4:
                low = df["Low"].min()
                st.metric("Period Low", f"${low:.2f}")

            # Simple price chart
            st.markdown("### Price Chart")
            st.line_chart(df["Close"])

    except Exception as e:
        st.error(f"Error loading data: {e}")

    # Data source availability
    st.markdown("### Data Source Availability")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("✅ Demo Data - Always Available")

    with col2:
        if YFINANCE_AVAILABLE:
            st.success("✅ Yahoo Finance - Installed")
        else:
            st.warning("⚠️ Yahoo Finance - Not Installed")
            st.code("pip install yfinance", language="bash")

    with col3:
        if CCXT_AVAILABLE:
            st.success("✅ CCXT (Crypto) - Installed")
        else:
            st.warning("⚠️ CCXT (Crypto) - Not Installed")
            st.code("pip install ccxt", language="bash")

    # Show install all command if any packages are missing
    if not YFINANCE_AVAILABLE or not CCXT_AVAILABLE:
        st.markdown("---")
        st.info("💡 **Tip:** Install all dependencies with:")
        st.code("pip install -r requirements.txt", language="bash")


if __name__ == "__main__":
    main()
