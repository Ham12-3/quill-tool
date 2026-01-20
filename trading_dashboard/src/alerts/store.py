"""
Alerts management and storage.
Supports price alerts, percent change alerts, and RSI alerts.
"""
import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd


class AlertType(Enum):
    """Types of alerts."""
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    CHANGE_ABOVE = "change_above"
    CHANGE_BELOW = "change_below"
    RSI_ABOVE = "rsi_above"
    RSI_BELOW = "rsi_below"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    DISABLED = "disabled"


@dataclass
class Alert:
    """Represents a price/indicator alert."""
    id: str
    ticker: str
    alert_type: str
    threshold: float
    created_at: str
    status: str = "active"
    triggered_at: Optional[str] = None
    triggered_value: Optional[float] = None
    message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Alert":
        return cls(**data)


@dataclass
class WatchlistItem:
    """Represents an item in a watchlist."""
    ticker: str
    added_at: str
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WatchlistItem":
        return cls(**data)


class AlertStore:
    """
    Persistent storage for alerts and watchlists.
    Uses JSON file for simplicity, can be extended to SQLite.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.alerts_file = os.path.join(data_dir, "alerts.json")
        self.watchlist_file = os.path.join(data_dir, "watchlists.json")

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Load existing data
        self.alerts: Dict[str, Alert] = self._load_alerts()
        self.watchlists: Dict[str, List[WatchlistItem]] = self._load_watchlists()

    def _load_alerts(self) -> Dict[str, Alert]:
        """Load alerts from file."""
        if os.path.exists(self.alerts_file):
            try:
                with open(self.alerts_file, "r") as f:
                    data = json.load(f)
                    return {k: Alert.from_dict(v) for k, v in data.items()}
            except Exception:
                pass
        return {}

    def _save_alerts(self) -> None:
        """Save alerts to file."""
        data = {k: v.to_dict() for k, v in self.alerts.items()}
        with open(self.alerts_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_watchlists(self) -> Dict[str, List[WatchlistItem]]:
        """Load watchlists from file."""
        if os.path.exists(self.watchlist_file):
            try:
                with open(self.watchlist_file, "r") as f:
                    data = json.load(f)
                    result = {}
                    for name, items in data.items():
                        result[name] = [WatchlistItem.from_dict(item) for item in items]
                    return result
            except Exception:
                pass
        return {"Default": []}

    def _save_watchlists(self) -> None:
        """Save watchlists to file."""
        data = {}
        for name, items in self.watchlists.items():
            data[name] = [item.to_dict() for item in items]
        with open(self.watchlist_file, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    # Alert methods
    def create_alert(
        self,
        ticker: str,
        alert_type: str,
        threshold: float,
        message: Optional[str] = None
    ) -> Alert:
        """Create a new alert."""
        alert_id = self._generate_id()
        alert = Alert(
            id=alert_id,
            ticker=ticker.upper(),
            alert_type=alert_type,
            threshold=threshold,
            created_at=datetime.now().isoformat(),
            status=AlertStatus.ACTIVE.value,
            message=message,
        )
        self.alerts[alert_id] = alert
        self._save_alerts()
        return alert

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID."""
        return self.alerts.get(alert_id)

    def get_alerts_for_ticker(self, ticker: str) -> List[Alert]:
        """Get all alerts for a ticker."""
        ticker = ticker.upper()
        return [a for a in self.alerts.values() if a.ticker == ticker]

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE.value]

    def get_triggered_alerts(self) -> List[Alert]:
        """Get all triggered alerts."""
        return [a for a in self.alerts.values() if a.status == AlertStatus.TRIGGERED.value]

    def trigger_alert(self, alert_id: str, value: float) -> Optional[Alert]:
        """Mark an alert as triggered."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.TRIGGERED.value
            alert.triggered_at = datetime.now().isoformat()
            alert.triggered_value = value
            self._save_alerts()
            return alert
        return None

    def disable_alert(self, alert_id: str) -> bool:
        """Disable an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].status = AlertStatus.DISABLED.value
            self._save_alerts()
            return True
        return False

    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert."""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            self._save_alerts()
            return True
        return False

    def clear_triggered_alerts(self) -> int:
        """Clear all triggered alerts. Returns count of cleared alerts."""
        to_delete = [a.id for a in self.alerts.values() if a.status == AlertStatus.TRIGGERED.value]
        for alert_id in to_delete:
            del self.alerts[alert_id]
        self._save_alerts()
        return len(to_delete)

    # Watchlist methods
    def create_watchlist(self, name: str) -> None:
        """Create a new watchlist."""
        if name not in self.watchlists:
            self.watchlists[name] = []
            self._save_watchlists()

    def delete_watchlist(self, name: str) -> bool:
        """Delete a watchlist."""
        if name in self.watchlists and name != "Default":
            del self.watchlists[name]
            self._save_watchlists()
            return True
        return False

    def get_watchlist_names(self) -> List[str]:
        """Get all watchlist names."""
        return list(self.watchlists.keys())

    def get_watchlist(self, name: str) -> List[WatchlistItem]:
        """Get items in a watchlist."""
        return self.watchlists.get(name, [])

    def add_to_watchlist(
        self,
        watchlist_name: str,
        ticker: str,
        notes: Optional[str] = None
    ) -> bool:
        """Add a ticker to a watchlist."""
        if watchlist_name not in self.watchlists:
            self.watchlists[watchlist_name] = []

        ticker = ticker.upper()

        # Check if already exists
        for item in self.watchlists[watchlist_name]:
            if item.ticker == ticker:
                return False

        item = WatchlistItem(
            ticker=ticker,
            added_at=datetime.now().isoformat(),
            notes=notes,
        )
        self.watchlists[watchlist_name].append(item)
        self._save_watchlists()
        return True

    def remove_from_watchlist(self, watchlist_name: str, ticker: str) -> bool:
        """Remove a ticker from a watchlist."""
        if watchlist_name not in self.watchlists:
            return False

        ticker = ticker.upper()
        original_len = len(self.watchlists[watchlist_name])
        self.watchlists[watchlist_name] = [
            item for item in self.watchlists[watchlist_name]
            if item.ticker != ticker
        ]

        if len(self.watchlists[watchlist_name]) < original_len:
            self._save_watchlists()
            return True
        return False

    def get_watchlist_tickers(self, watchlist_name: str) -> List[str]:
        """Get list of tickers in a watchlist."""
        items = self.watchlists.get(watchlist_name, [])
        return [item.ticker for item in items]


def evaluate_alerts(
    alerts: List[Alert],
    current_prices: Dict[str, float],
    daily_changes: Dict[str, float] = None,
    rsi_values: Dict[str, float] = None
) -> List[Alert]:
    """
    Evaluate alerts against current market data.

    Args:
        alerts: List of active alerts
        current_prices: Dictionary of ticker to current price
        daily_changes: Dictionary of ticker to daily percent change
        rsi_values: Dictionary of ticker to RSI value

    Returns:
        List of alerts that should be triggered
    """
    triggered = []

    for alert in alerts:
        if alert.status != AlertStatus.ACTIVE.value:
            continue

        ticker = alert.ticker
        should_trigger = False
        value = None

        if alert.alert_type == AlertType.PRICE_ABOVE.value:
            if ticker in current_prices:
                value = current_prices[ticker]
                should_trigger = value > alert.threshold

        elif alert.alert_type == AlertType.PRICE_BELOW.value:
            if ticker in current_prices:
                value = current_prices[ticker]
                should_trigger = value < alert.threshold

        elif alert.alert_type == AlertType.CHANGE_ABOVE.value:
            if daily_changes and ticker in daily_changes:
                value = daily_changes[ticker]
                should_trigger = value > alert.threshold

        elif alert.alert_type == AlertType.CHANGE_BELOW.value:
            if daily_changes and ticker in daily_changes:
                value = daily_changes[ticker]
                should_trigger = value < alert.threshold

        elif alert.alert_type == AlertType.RSI_ABOVE.value:
            if rsi_values and ticker in rsi_values:
                value = rsi_values[ticker]
                should_trigger = value > alert.threshold

        elif alert.alert_type == AlertType.RSI_BELOW.value:
            if rsi_values and ticker in rsi_values:
                value = rsi_values[ticker]
                should_trigger = value < alert.threshold

        if should_trigger:
            alert.triggered_value = value
            triggered.append(alert)

    return triggered


def format_alert_message(alert: Alert) -> str:
    """Format an alert for display."""
    type_messages = {
        AlertType.PRICE_ABOVE.value: f"Price crossed above ${alert.threshold:.2f}",
        AlertType.PRICE_BELOW.value: f"Price crossed below ${alert.threshold:.2f}",
        AlertType.CHANGE_ABOVE.value: f"Daily change exceeded {alert.threshold:.1f}%",
        AlertType.CHANGE_BELOW.value: f"Daily change fell below {alert.threshold:.1f}%",
        AlertType.RSI_ABOVE.value: f"RSI crossed above {alert.threshold:.0f}",
        AlertType.RSI_BELOW.value: f"RSI crossed below {alert.threshold:.0f}",
    }

    base_message = type_messages.get(alert.alert_type, "Alert triggered")

    if alert.triggered_value is not None:
        if "Price" in base_message:
            base_message += f" (current: ${alert.triggered_value:.2f})"
        elif "change" in base_message:
            base_message += f" (current: {alert.triggered_value:.2f}%)"
        elif "RSI" in base_message:
            base_message += f" (current: {alert.triggered_value:.1f})"

    return f"{alert.ticker}: {base_message}"
