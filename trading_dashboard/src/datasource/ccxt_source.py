"""
CCXT data source for cryptocurrency data.
Optional dependency - falls back to demo data if not installed.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import logging

from .base import DataSourceBase, AssetMetadata

logger = logging.getLogger(__name__)

# Check if ccxt is available
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None


# Common crypto assets metadata
CRYPTO_METADATA = {
    "BTC/USDT": {"name": "Bitcoin", "sector": "Cryptocurrency"},
    "ETH/USDT": {"name": "Ethereum", "sector": "Cryptocurrency"},
    "BNB/USDT": {"name": "Binance Coin", "sector": "Cryptocurrency"},
    "XRP/USDT": {"name": "Ripple", "sector": "Cryptocurrency"},
    "ADA/USDT": {"name": "Cardano", "sector": "Cryptocurrency"},
    "SOL/USDT": {"name": "Solana", "sector": "Cryptocurrency"},
    "DOGE/USDT": {"name": "Dogecoin", "sector": "Cryptocurrency"},
    "DOT/USDT": {"name": "Polkadot", "sector": "Cryptocurrency"},
    "MATIC/USDT": {"name": "Polygon", "sector": "Cryptocurrency"},
    "LINK/USDT": {"name": "Chainlink", "sector": "Cryptocurrency"},
    "AVAX/USDT": {"name": "Avalanche", "sector": "Cryptocurrency"},
    "UNI/USDT": {"name": "Uniswap", "sector": "DeFi"},
    "AAVE/USDT": {"name": "Aave", "sector": "DeFi"},
    "ATOM/USDT": {"name": "Cosmos", "sector": "Cryptocurrency"},
    "LTC/USDT": {"name": "Litecoin", "sector": "Cryptocurrency"},
}


class CCXTDataSource(DataSourceBase):
    """
    Data source for cryptocurrency data using CCXT library.
    Supports multiple exchanges, defaults to Binance.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = None,
        api_secret: str = None,
        rate_limit_seconds: float = 1.0
    ):
        """
        Initialize the CCXT data source.

        Args:
            exchange_id: CCXT exchange identifier (e.g., "binance", "coinbase")
            api_key: Optional API key for authenticated requests
            api_secret: Optional API secret
            rate_limit_seconds: Minimum time between API calls
        """
        self._exchange_id = exchange_id
        self._api_key = api_key
        self._api_secret = api_secret
        self._rate_limit = rate_limit_seconds
        self._last_call = 0
        self._exchange = None

    @property
    def name(self) -> str:
        return f"CCXT ({self._exchange_id.title()})"

    @property
    def is_available(self) -> bool:
        return CCXT_AVAILABLE

    def _get_exchange(self):
        """Get or create the exchange instance."""
        if not CCXT_AVAILABLE:
            return None

        if self._exchange is None:
            exchange_class = getattr(ccxt, self._exchange_id)
            config = {
                "enableRateLimit": True,
                "rateLimit": int(self._rate_limit * 1000),
            }
            if self._api_key:
                config["apiKey"] = self._api_key
            if self._api_secret:
                config["secret"] = self._api_secret

            self._exchange = exchange_class(config)

        return self._exchange

    def _apply_rate_limit(self):
        """Apply rate limiting between API calls."""
        import time
        current = time.time()
        elapsed = current - self._last_call
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_call = time.time()

    def _normalize_symbol(self, ticker: str) -> str:
        """Normalize ticker to CCXT symbol format."""
        ticker = ticker.upper()

        # If already in CCXT format (e.g., "BTC/USDT")
        if "/" in ticker:
            return ticker

        # Common conversions
        if ticker.endswith("USDT"):
            base = ticker[:-4]
            return f"{base}/USDT"
        elif ticker.endswith("USD"):
            base = ticker[:-3]
            return f"{base}/USDT"
        elif ticker.endswith("BTC"):
            base = ticker[:-3]
            return f"{base}/BTC"
        else:
            # Default to USDT pair
            return f"{ticker}/USDT"

    def get_price_history(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get OHLCV price history from the exchange.

        Args:
            ticker: The ticker symbol (e.g., "BTC/USDT", "BTCUSDT", "BTC")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, 4h, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        if not CCXT_AVAILABLE:
            raise RuntimeError("ccxt is not installed")

        exchange = self._get_exchange()
        if not exchange:
            raise RuntimeError("Failed to initialize exchange")

        symbol = self._normalize_symbol(ticker)

        # Map interval to CCXT timeframe
        timeframe_map = {
            "1d": "1d",
            "1h": "1h",
            "4h": "4h",
            "1w": "1w",
            "1m": "1M",
            "15m": "15m",
            "5m": "5m",
        }
        timeframe = timeframe_map.get(interval, "1d")

        # Convert dates to timestamps
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        since = int(start_dt.timestamp() * 1000)

        self._apply_rate_limit()

        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=1000
            )

            if not ohlcv:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
            )

            # Convert timestamp to datetime
            df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
            df = df.set_index("Date")
            df = df.drop(columns=["Timestamp"])

            # Filter by end date
            df = df[df.index <= end_dt]

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise

    def get_metadata(self, ticker: str) -> AssetMetadata:
        """Get metadata for a cryptocurrency."""
        symbol = self._normalize_symbol(ticker)

        if symbol in CRYPTO_METADATA:
            info = CRYPTO_METADATA[symbol]
            return AssetMetadata(
                ticker=symbol,
                name=info["name"],
                sector=info.get("sector", "Cryptocurrency"),
                country="Global",
                asset_class="Crypto",
                currency="USDT",
                exchange=self._exchange_id.title(),
            )
        else:
            # Extract base currency from symbol
            base = symbol.split("/")[0] if "/" in symbol else symbol
            return AssetMetadata(
                ticker=symbol,
                name=f"{base} Token",
                sector="Cryptocurrency",
                country="Global",
                asset_class="Crypto",
                currency="USDT",
                exchange=self._exchange_id.title(),
            )

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for a cryptocurrency."""
        if not CCXT_AVAILABLE:
            return None

        exchange = self._get_exchange()
        if not exchange:
            return None

        symbol = self._normalize_symbol(ticker)
        self._apply_rate_limit()

        try:
            ticker_data = exchange.fetch_ticker(symbol)
            return ticker_data.get("last")
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    def search_tickers(self, query: str) -> List[str]:
        """Search for cryptocurrency pairs matching a query."""
        query = query.upper()
        matches = []

        # Search in known metadata
        for symbol in CRYPTO_METADATA.keys():
            if query in symbol:
                matches.append(symbol)

        # Add common variations
        common_pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT"]
        for pair in common_pairs:
            if query in pair and pair not in matches:
                matches.append(pair)

        return matches[:20]

    def get_available_pairs(self) -> List[str]:
        """Get list of available trading pairs from the exchange."""
        if not CCXT_AVAILABLE:
            return list(CRYPTO_METADATA.keys())

        exchange = self._get_exchange()
        if not exchange:
            return list(CRYPTO_METADATA.keys())

        try:
            self._apply_rate_limit()
            exchange.load_markets()
            return list(exchange.symbols)[:100]  # Limit to first 100
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            return list(CRYPTO_METADATA.keys())
