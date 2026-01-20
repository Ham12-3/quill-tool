"""
Alerts and Watchlist page - Set price alerts and manage watchlists.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasource import DemoDataSource
from src.alerts import (
    AlertStore, AlertType, AlertStatus, Alert,
    evaluate_alerts, format_alert_message
)
from src.analytics import rsi, calculate_returns
from src.utils import format_percent, format_currency

st.set_page_config(page_title="Alerts & Watchlist", page_icon="🔔", layout="wide")


def get_alert_store():
    """Get or create alert store."""
    if "alert_store" not in st.session_state:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        st.session_state.alert_store = AlertStore(data_dir)
    return st.session_state.alert_store


def main():
    st.title("🔔 Alerts & Watchlist")
    st.markdown("Set price alerts and manage your watchlists.")

    source = DemoDataSource()
    alert_store = get_alert_store()

    if "start_date" not in st.session_state:
        st.session_state.start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.now().strftime("%Y-%m-%d")

    # Tabs for Alerts and Watchlist
    tab1, tab2 = st.tabs(["📊 Alerts", "👁️ Watchlist"])

    # ==================== ALERTS TAB ====================
    with tab1:
        st.subheader("Price & Indicator Alerts")

        col1, col2 = st.columns([2, 1])

        with col2:
            st.markdown("### Create New Alert")

            available_tickers = source.get_available_tickers()
            alert_ticker = st.selectbox("Ticker", available_tickers, key="alert_ticker")

            alert_type = st.selectbox(
                "Alert Type",
                [
                    ("Price Above", AlertType.PRICE_ABOVE.value),
                    ("Price Below", AlertType.PRICE_BELOW.value),
                    ("Daily Change Above %", AlertType.CHANGE_ABOVE.value),
                    ("Daily Change Below %", AlertType.CHANGE_BELOW.value),
                    ("RSI Above", AlertType.RSI_ABOVE.value),
                    ("RSI Below", AlertType.RSI_BELOW.value),
                ],
                format_func=lambda x: x[0],
                key="alert_type",
            )

            # Get current price for reference
            try:
                df = source.get_price_history(
                    alert_ticker,
                    st.session_state.start_date,
                    st.session_state.end_date,
                )
                if not df.empty:
                    current_price = df["Close"].iloc[-1]
                    st.caption(f"Current price: ${current_price:.2f}")

                    if "RSI" in alert_type[0]:
                        current_rsi = rsi(df["Close"], 14).iloc[-1]
                        st.caption(f"Current RSI: {current_rsi:.1f}")
                else:
                    current_price = 100
            except Exception:
                current_price = 100

            # Threshold input based on type
            if "Price" in alert_type[0]:
                threshold = st.number_input(
                    "Price Threshold ($)",
                    min_value=0.01,
                    value=float(current_price),
                    step=1.0,
                    key="alert_threshold",
                )
            elif "Change" in alert_type[0]:
                threshold = st.number_input(
                    "Change Threshold (%)",
                    min_value=-50.0,
                    max_value=50.0,
                    value=5.0,
                    step=0.5,
                    key="alert_threshold_pct",
                )
            else:  # RSI
                threshold = st.number_input(
                    "RSI Threshold",
                    min_value=0.0,
                    max_value=100.0,
                    value=70.0 if "Above" in alert_type[0] else 30.0,
                    step=1.0,
                    key="alert_threshold_rsi",
                )

            alert_message = st.text_input("Custom Message (optional)", key="alert_message")

            if st.button("Create Alert", use_container_width=True, type="primary"):
                alert = alert_store.create_alert(
                    ticker=alert_ticker,
                    alert_type=alert_type[1],
                    threshold=threshold,
                    message=alert_message if alert_message else None,
                )
                st.success(f"Alert created for {alert_ticker}!")
                st.rerun()

        with col1:
            # Active Alerts
            st.markdown("### Active Alerts")

            active_alerts = alert_store.get_active_alerts()

            if active_alerts:
                alert_data = []
                for alert in active_alerts:
                    alert_data.append({
                        "ID": alert.id,
                        "Ticker": alert.ticker,
                        "Type": alert.alert_type.replace("_", " ").title(),
                        "Threshold": alert.threshold,
                        "Created": alert.created_at[:10],
                    })

                alert_df = pd.DataFrame(alert_data)
                st.dataframe(alert_df, use_container_width=True, hide_index=True)

                # Delete alert
                col_del1, col_del2 = st.columns([3, 1])
                with col_del1:
                    alert_to_delete = st.selectbox(
                        "Select alert to delete",
                        [a.id for a in active_alerts],
                        format_func=lambda x: f"{x} - {[a for a in active_alerts if a.id == x][0].ticker}",
                        key="delete_alert_select",
                    )
                with col_del2:
                    if st.button("Delete Alert", use_container_width=True):
                        alert_store.delete_alert(alert_to_delete)
                        st.rerun()
            else:
                st.info("No active alerts. Create one from the sidebar.")

            # Check Alerts
            st.markdown("---")
            st.markdown("### Check Alerts Now")

            if st.button("Evaluate Alerts", use_container_width=True):
                with st.spinner("Checking alerts..."):
                    # Get current prices and indicators
                    current_prices = {}
                    daily_changes = {}
                    rsi_values = {}

                    for alert in active_alerts:
                        ticker = alert.ticker
                        if ticker not in current_prices:
                            try:
                                df = source.get_price_history(
                                    ticker,
                                    st.session_state.start_date,
                                    st.session_state.end_date,
                                )
                                if not df.empty and len(df) > 1:
                                    current_prices[ticker] = df["Close"].iloc[-1]
                                    daily_changes[ticker] = (
                                        (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
                                    )
                                    rsi_values[ticker] = rsi(df["Close"], 14).iloc[-1]
                            except Exception:
                                pass

                    triggered = evaluate_alerts(
                        active_alerts, current_prices, daily_changes, rsi_values
                    )

                    if triggered:
                        st.warning(f"🔔 {len(triggered)} alert(s) triggered!")
                        for alert in triggered:
                            alert_store.trigger_alert(alert.id, alert.triggered_value)
                            st.error(format_alert_message(alert))
                        st.rerun()
                    else:
                        st.success("No alerts triggered.")

            # Triggered Alerts History
            st.markdown("---")
            st.markdown("### Triggered Alerts History")

            triggered_alerts = alert_store.get_triggered_alerts()

            if triggered_alerts:
                triggered_data = []
                for alert in triggered_alerts:
                    triggered_data.append({
                        "Ticker": alert.ticker,
                        "Type": alert.alert_type.replace("_", " ").title(),
                        "Threshold": alert.threshold,
                        "Triggered Value": alert.triggered_value,
                        "Triggered At": alert.triggered_at[:19] if alert.triggered_at else "",
                    })

                triggered_df = pd.DataFrame(triggered_data)
                st.dataframe(triggered_df, use_container_width=True, hide_index=True)

                if st.button("Clear Triggered Alerts"):
                    cleared = alert_store.clear_triggered_alerts()
                    st.success(f"Cleared {cleared} triggered alerts.")
                    st.rerun()
            else:
                st.info("No triggered alerts.")

    # ==================== WATCHLIST TAB ====================
    with tab2:
        st.subheader("Watchlists")

        col1, col2 = st.columns([2, 1])

        with col2:
            st.markdown("### Manage Watchlists")

            # Create new watchlist
            new_watchlist_name = st.text_input("New Watchlist Name", key="new_watchlist")
            if st.button("Create Watchlist", use_container_width=True):
                if new_watchlist_name:
                    alert_store.create_watchlist(new_watchlist_name)
                    st.success(f"Watchlist '{new_watchlist_name}' created!")
                    st.rerun()

            st.markdown("---")

            # Add to watchlist
            st.markdown("### Add to Watchlist")
            watchlist_names = alert_store.get_watchlist_names()
            selected_watchlist = st.selectbox("Select Watchlist", watchlist_names, key="add_watchlist")

            available_tickers = source.get_available_tickers()
            ticker_to_add = st.selectbox("Ticker to Add", available_tickers, key="watchlist_ticker")
            notes = st.text_input("Notes (optional)", key="watchlist_notes")

            if st.button("Add to Watchlist", use_container_width=True, type="primary"):
                if alert_store.add_to_watchlist(selected_watchlist, ticker_to_add, notes):
                    st.success(f"Added {ticker_to_add} to {selected_watchlist}!")
                    st.rerun()
                else:
                    st.warning(f"{ticker_to_add} is already in {selected_watchlist}.")

        with col1:
            # Display watchlists
            st.markdown("### Your Watchlists")

            watchlist_names = alert_store.get_watchlist_names()

            for watchlist_name in watchlist_names:
                with st.expander(f"📋 {watchlist_name}", expanded=(watchlist_name == "Default")):
                    items = alert_store.get_watchlist(watchlist_name)

                    if items:
                        watchlist_data = []
                        for item in items:
                            # Get current price
                            try:
                                df = source.get_price_history(
                                    item.ticker,
                                    st.session_state.start_date,
                                    st.session_state.end_date,
                                )
                                if not df.empty and len(df) > 1:
                                    current_price = df["Close"].iloc[-1]
                                    prev_price = df["Close"].iloc[-2]
                                    daily_change = (current_price / prev_price - 1)
                                    metadata = source.get_metadata(item.ticker)
                                else:
                                    current_price = 0
                                    daily_change = 0
                                    metadata = source.get_metadata(item.ticker)
                            except Exception:
                                current_price = 0
                                daily_change = 0
                                metadata = source.get_metadata(item.ticker)

                            watchlist_data.append({
                                "Ticker": item.ticker,
                                "Name": metadata.name[:20],
                                "Price": f"${current_price:.2f}" if current_price else "N/A",
                                "Change": format_percent(daily_change) if daily_change else "N/A",
                                "Notes": item.notes or "",
                                "Added": item.added_at[:10],
                            })

                        watchlist_df = pd.DataFrame(watchlist_data)
                        st.dataframe(watchlist_df, use_container_width=True, hide_index=True)

                        # Remove from watchlist
                        col_r1, col_r2 = st.columns([3, 1])
                        with col_r1:
                            ticker_to_remove = st.selectbox(
                                "Remove ticker",
                                [item.ticker for item in items],
                                key=f"remove_{watchlist_name}",
                            )
                        with col_r2:
                            if st.button("Remove", key=f"remove_btn_{watchlist_name}"):
                                alert_store.remove_from_watchlist(watchlist_name, ticker_to_remove)
                                st.rerun()
                    else:
                        st.info("No items in this watchlist.")

                    # Delete watchlist (except Default)
                    if watchlist_name != "Default":
                        if st.button(f"Delete Watchlist", key=f"delete_wl_{watchlist_name}"):
                            alert_store.delete_watchlist(watchlist_name)
                            st.rerun()

            # Watchlist performance overview
            st.markdown("---")
            st.markdown("### Watchlist Performance Overview")

            selected_for_overview = st.selectbox(
                "Select Watchlist for Overview",
                watchlist_names,
                key="overview_watchlist",
            )

            items = alert_store.get_watchlist(selected_for_overview)
            if items:
                performance_data = []
                for item in items:
                    try:
                        df = source.get_price_history(
                            item.ticker,
                            st.session_state.start_date,
                            st.session_state.end_date,
                        )
                        if not df.empty and len(df) > 1:
                            current = df["Close"].iloc[-1]
                            prev = df["Close"].iloc[-2]
                            period_start = df["Close"].iloc[0]
                            daily_change = (current / prev - 1)
                            period_return = (current / period_start - 1)
                            high = df["High"].max()
                            low = df["Low"].min()

                            performance_data.append({
                                "Ticker": item.ticker,
                                "Price": current,
                                "Daily": daily_change,
                                "Period Return": period_return,
                                "High": high,
                                "Low": low,
                            })
                    except Exception:
                        pass

                if performance_data:
                    perf_df = pd.DataFrame(performance_data)

                    # Format for display
                    display_df = perf_df.copy()
                    display_df["Price"] = display_df["Price"].map("${:.2f}".format)
                    display_df["Daily"] = display_df["Daily"].map("{:.2%}".format)
                    display_df["Period Return"] = display_df["Period Return"].map("{:.2%}".format)
                    display_df["High"] = display_df["High"].map("${:.2f}".format)
                    display_df["Low"] = display_df["Low"].map("${:.2f}".format)

                    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Disclaimer
    st.markdown("---")
    st.caption(
        "**Disclaimer:** Alerts are evaluated manually when you click 'Evaluate Alerts'. "
        "For real-time alerts, consider integrating with a dedicated alerting service. "
        "This tool is for educational purposes only."
    )


if __name__ == "__main__":
    main()
