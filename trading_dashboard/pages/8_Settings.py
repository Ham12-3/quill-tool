"""
Settings page - Configure data sources and preferences.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_package_installed(package_name: str) -> tuple[bool, str | None]:
    """
    Check if a package is installed and importable.

    Returns:
        Tuple of (is_installed, error_message)
    """
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return False, f"Package '{package_name}' is not installed"

    # Try to actually import it to catch any import errors
    try:
        importlib.import_module(package_name)
        return True, None
    except ImportError as e:
        return False, f"Package '{package_name}' found but failed to import: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error importing '{package_name}': {str(e)}"


# Check package availability with detailed error info
YFINANCE_INSTALLED, YFINANCE_ERROR = check_package_installed("yfinance")
CCXT_INSTALLED, CCXT_ERROR = check_package_installed("ccxt")

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")


def main():
    st.title("⚙️ Settings")
    st.markdown("Configure data sources, preferences, and application settings.")

    # Data Source Settings
    st.subheader("Data Source Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Primary Data Source")

        # Show available data sources
        data_sources = ["Demo Data (Always Available)"]

        if YFINANCE_INSTALLED:
            data_sources.append("Yahoo Finance (yfinance)")
        if CCXT_INSTALLED:
            data_sources.append("CCXT (Cryptocurrency)")

        # Get current selection
        current_source = st.session_state.get("data_source", "demo")

        # Validate current source is still available
        if current_source == "yfinance" and not YFINANCE_INSTALLED:
            current_source = "demo"
            st.session_state.data_source = "demo"
        if current_source == "ccxt" and not CCXT_INSTALLED:
            current_source = "demo"
            st.session_state.data_source = "demo"

        current_index = 0
        if current_source == "yfinance" and YFINANCE_INSTALLED:
            current_index = 1
        elif current_source == "ccxt" and CCXT_INSTALLED:
            current_index = 2 if YFINANCE_INSTALLED else 1

        selected_source = st.selectbox(
            "Select Data Source",
            data_sources,
            index=min(current_index, len(data_sources) - 1),
        )

        # Map selection to source name
        if "Demo" in selected_source:
            st.session_state.data_source = "demo"
        elif "Yahoo" in selected_source:
            st.session_state.data_source = "yfinance"
        elif "CCXT" in selected_source:
            st.session_state.data_source = "ccxt"

        # Show status
        if st.session_state.data_source == "demo":
            st.success("✅ Demo data source active - no API keys required")
        elif st.session_state.data_source == "yfinance":
            st.success("✅ Yahoo Finance data source active")
        elif st.session_state.data_source == "ccxt":
            st.success("✅ CCXT cryptocurrency data source active")

    with col2:
        st.markdown("### Data Source Status")

        # Demo - always available
        st.markdown("**Demo Data**")
        st.success("✅ Available - Synthetic data, always works")

        # yfinance
        st.markdown("**Yahoo Finance**")
        if YFINANCE_INSTALLED:
            st.success("✅ Installed - Real stock data")
        else:
            st.warning("⚠️ Not Installed")
            st.code("pip install yfinance", language="bash")
            if YFINANCE_ERROR:
                with st.expander("Error details"):
                    st.error(YFINANCE_ERROR)

        # CCXT
        st.markdown("**CCXT (Crypto)**")
        if CCXT_INSTALLED:
            st.success("✅ Installed - Cryptocurrency data")
        else:
            st.warning("⚠️ Not Installed")
            st.code("pip install ccxt", language="bash")
            if CCXT_ERROR:
                with st.expander("Error details"):
                    st.error(CCXT_ERROR)

    # Show install all command if packages are missing
    if not YFINANCE_INSTALLED or not CCXT_INSTALLED:
        st.markdown("---")
        st.markdown("### Install Missing Packages")
        st.markdown("Run this command to install all dependencies:")
        st.code("pip install -r requirements.txt", language="bash")

    # CCXT Exchange Settings (if available)
    if CCXT_INSTALLED and st.session_state.data_source == "ccxt":
        st.markdown("---")
        st.subheader("CCXT Exchange Settings")

        exchanges = ["binance", "coinbase", "kraken", "bitfinex", "huobi"]
        current_exchange = st.session_state.get("ccxt_exchange", "binance")

        selected_exchange = st.selectbox(
            "Select Exchange",
            exchanges,
            index=exchanges.index(current_exchange) if current_exchange in exchanges else 0,
        )
        st.session_state.ccxt_exchange = selected_exchange

        st.info(
            "Note: For public market data, no API keys are required. "
            "API keys would only be needed for trading (not supported in this app)."
        )

    # Analytics Settings
    st.markdown("---")
    st.subheader("Analytics Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Risk Calculations")

        risk_free_rate = st.number_input(
            "Risk-Free Rate (% annualized)",
            min_value=0.0,
            max_value=20.0,
            value=st.session_state.get("risk_free_rate", 0.0) * 100,
            step=0.1,
            help="Used for Sharpe ratio and other risk-adjusted calculations"
        )
        st.session_state.risk_free_rate = risk_free_rate / 100

    with col2:
        st.markdown("### Display Settings")

        show_disclaimer = st.checkbox(
            "Show Disclaimer on All Pages",
            value=st.session_state.get("show_disclaimer", True),
        )
        st.session_state.show_disclaimer = show_disclaimer

    # Cache Settings
    st.markdown("---")
    st.subheader("Cache Management")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Clear Cache")
        st.markdown(
            "Clearing the cache will force fresh data to be loaded on the next request. "
            "This is useful if you're seeing stale data."
        )

        if st.button("🗑️ Clear All Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared successfully!")
            st.rerun()

    with col2:
        st.markdown("### Cache Info")
        st.info(
            "Data is cached for 5 minutes (300 seconds) by default to improve performance "
            "and reduce API calls. Cache is automatically invalidated when the date range changes."
        )

    # Data Export Settings
    st.markdown("---")
    st.subheader("Data Management")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Export All Settings")

        if st.button("📥 Export Settings to JSON", use_container_width=True):
            import json

            settings = {
                "data_source": st.session_state.get("data_source", "demo"),
                "ccxt_exchange": st.session_state.get("ccxt_exchange", "binance"),
                "risk_free_rate": st.session_state.get("risk_free_rate", 0.0),
                "start_date": st.session_state.get("start_date", ""),
                "end_date": st.session_state.get("end_date", ""),
                "show_disclaimer": st.session_state.get("show_disclaimer", True),
            }

            json_str = json.dumps(settings, indent=2)
            st.download_button(
                label="📥 Download Settings JSON",
                data=json_str,
                file_name="dashboard_settings.json",
                mime="application/json",
            )

    with col2:
        st.markdown("### Reset to Defaults")

        if st.button("🔄 Reset All Settings", use_container_width=True):
            # Reset to defaults
            st.session_state.data_source = "demo"
            st.session_state.ccxt_exchange = "binance"
            st.session_state.risk_free_rate = 0.0
            st.session_state.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            st.session_state.end_date = datetime.now().strftime("%Y-%m-%d")
            st.session_state.show_disclaimer = True

            st.success("Settings reset to defaults!")
            st.rerun()

    # About Section
    st.markdown("---")
    st.subheader("About")

    st.markdown("""
    ### Trading & Portfolio Analytics Dashboard

    **Version:** 1.0.0

    **Features:**
    - Market overview with key metrics
    - Detailed asset analysis with technical indicators
    - Portfolio construction and optimization
    - Risk analysis including VaR, CVaR, and correlations
    - Strategy backtesting with multiple strategies
    - Price alerts and watchlist management
    - PDF report generation

    **Data Sources:**
    - Demo Data: Synthetic data for testing (always available)
    - Yahoo Finance: Real stock market data (requires yfinance)
    - CCXT: Cryptocurrency data (requires ccxt)

    **Libraries Used:**
    - Streamlit for the web interface
    - Pandas & NumPy for data processing
    - Plotly for interactive charts
    - SciPy for statistical calculations
    - ReportLab for PDF generation

    ---

    **Disclaimer:**
    This application is for educational and analytical purposes only.
    It does not constitute financial, investment, or trading advice.
    Past performance does not guarantee future results.
    Always consult with a qualified financial advisor before making investment decisions.
    """)

    # Installation Instructions
    st.markdown("---")
    st.subheader("Installation Instructions")

    with st.expander("How to install dependencies"):
        st.markdown("""
        **Install all dependencies at once:**
        ```bash
        pip install -r requirements.txt
        ```

        **Or install individually:**

        Yahoo Finance (yfinance):
        ```bash
        pip install yfinance
        ```

        CCXT (Cryptocurrency):
        ```bash
        pip install ccxt
        ```

        Chart Image Export (for PDF reports):
        ```bash
        pip install kaleido
        ```
        """)

    with st.expander("How to run the application"):
        st.markdown("""
        **Start the dashboard:**
        ```bash
        streamlit run app.py
        ```

        **Or with custom port:**
        ```bash
        streamlit run app.py --server.port 8080
        ```

        **Environment variables (optional):**
        - `STREAMLIT_SERVER_PORT`: Custom port
        - `STREAMLIT_SERVER_HEADLESS`: Run in headless mode
        """)


if __name__ == "__main__":
    main()
