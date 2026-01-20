"""
Base data source interface for market data.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class AssetMetadata:
    """Metadata about an asset."""
    ticker: str
    name: str
    sector: str = "Unknown"
    country: str = "Unknown"
    currency: str = "USD"
    asset_class: str = "Equity"
    exchange: str = "Unknown"


class DataSourceBase(ABC):
    """Abstract base class for data sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this data source."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this data source is available (dependencies installed, API keys set, etc.)."""
        pass

    @abstractmethod
    def get_price_history(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get OHLCV price history for a ticker.

        Args:
            ticker: The ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
            Date should be the index.
        """
        pass

    @abstractmethod
    def get_metadata(self, ticker: str) -> AssetMetadata:
        """
        Get metadata for a ticker.

        Args:
            ticker: The ticker symbol

        Returns:
            AssetMetadata object with ticker info
        """
        pass

    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get the current/latest price for a ticker.

        Args:
            ticker: The ticker symbol

        Returns:
            Current price or None if unavailable
        """
        try:
            from datetime import datetime, timedelta
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            df = self.get_price_history(ticker, start, end, "1d")
            if not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception:
            pass
        return None

    def search_tickers(self, query: str) -> list[str]:
        """
        Search for tickers matching a query.

        Args:
            query: Search string

        Returns:
            List of matching ticker symbols
        """
        return []
