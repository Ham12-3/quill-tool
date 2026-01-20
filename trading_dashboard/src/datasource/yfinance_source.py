"""
Yahoo Finance data source using yfinance library.
Optional dependency - falls back to demo data if not installed.
"""
import pandas as pd
from typing import Optional
import logging

from .base import DataSourceBase, AssetMetadata

logger = logging.getLogger(__name__)

# Check if yfinance is available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None


class YFinanceDataSource(DataSourceBase):
    """
    Data source using Yahoo Finance via yfinance library.
    Provides real market data for stocks, ETFs, and indices.
    """

    def __init__(self, rate_limit_seconds: float = 1.0):
        """
        Initialize the Yahoo Finance data source.

        Args:
            rate_limit_seconds: Minimum time between API calls
        """
        self._rate_limit = rate_limit_seconds
        self._last_call = 0
        self._ticker_cache: dict = {}

    @property
    def name(self) -> str:
        return "Yahoo Finance"

    @property
    def is_available(self) -> bool:
        return YFINANCE_AVAILABLE

    def _get_ticker_obj(self, ticker: str):
        """Get or create a yfinance Ticker object."""
        if not YFINANCE_AVAILABLE:
            return None

        if ticker not in self._ticker_cache:
            self._ticker_cache[ticker] = yf.Ticker(ticker)
        return self._ticker_cache[ticker]

    def _apply_rate_limit(self):
        """Apply rate limiting between API calls."""
        import time
        current = time.time()
        elapsed = current - self._last_call
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_call = time.time()

    def get_price_history(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get OHLCV price history from Yahoo Finance.

        Args:
            ticker: The ticker symbol (e.g., "AAPL", "MSFT")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, 1wk, 1mo)

        Returns:
            DataFrame with OHLCV data
        """
        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance is not installed")

        self._apply_rate_limit()

        # Map interval names
        interval_map = {
            "1d": "1d",
            "1h": "1h",
            "4h": "1h",  # yfinance doesn't have 4h, use 1h
            "1w": "1wk",
            "1m": "1mo",
        }
        yf_interval = interval_map.get(interval, "1d")

        try:
            ticker_obj = self._get_ticker_obj(ticker)
            df = ticker_obj.history(start=start, end=end, interval=yf_interval)

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

            # Standardize column names
            df = df.rename(columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            })

            # Select only OHLCV columns
            columns = ["Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in columns if c in df.columns]]

            # Ensure index name
            df.index.name = "Date"

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            raise

    def get_metadata(self, ticker: str) -> AssetMetadata:
        """Get metadata for a ticker from Yahoo Finance."""
        if not YFINANCE_AVAILABLE:
            return AssetMetadata(
                ticker=ticker,
                name=ticker,
                sector="Unknown",
                country="Unknown",
            )

        self._apply_rate_limit()

        try:
            ticker_obj = self._get_ticker_obj(ticker)
            info = ticker_obj.info

            return AssetMetadata(
                ticker=ticker,
                name=info.get("shortName", info.get("longName", ticker)),
                sector=info.get("sector", "Unknown"),
                country=info.get("country", "Unknown"),
                currency=info.get("currency", "USD"),
                asset_class=self._determine_asset_class(info),
                exchange=info.get("exchange", "Unknown"),
            )
        except Exception as e:
            logger.error(f"Error fetching metadata for {ticker}: {e}")
            return AssetMetadata(
                ticker=ticker,
                name=ticker,
                sector="Unknown",
                country="Unknown",
            )

    def _determine_asset_class(self, info: dict) -> str:
        """Determine asset class from yfinance info."""
        quote_type = info.get("quoteType", "").upper()
        if quote_type == "ETF":
            return "ETF"
        elif quote_type == "CRYPTOCURRENCY":
            return "Crypto"
        elif quote_type == "FUTURE":
            return "Futures"
        elif quote_type == "INDEX":
            return "Index"
        elif quote_type == "MUTUALFUND":
            return "Mutual Fund"
        else:
            return "Equity"

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for a ticker."""
        if not YFINANCE_AVAILABLE:
            return None

        self._apply_rate_limit()

        try:
            ticker_obj = self._get_ticker_obj(ticker)
            info = ticker_obj.info
            return info.get("regularMarketPrice") or info.get("previousClose")
        except Exception as e:
            logger.error(f"Error fetching current price for {ticker}: {e}")
            return None

    def search_tickers(self, query: str) -> list[str]:
        """
        Search for tickers matching a query.
        Note: yfinance doesn't have a built-in search, so this is limited.
        """
        # Common tickers to search through
        common_tickers = [
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
            "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "BAC",
            "NFLX", "INTC", "CSCO", "VZ", "ADBE", "CRM", "ABT", "CMCSA",
            "PFE", "KO", "PEP", "MRK", "T", "NKE", "WMT", "XOM", "CVX",
            "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "BND", "GLD", "SLV",
            "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
        ]

        query_upper = query.upper()
        matches = [t for t in common_tickers if query_upper in t]
        return matches[:20]  # Limit results
