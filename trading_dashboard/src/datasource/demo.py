"""
Demo data source that generates realistic synthetic OHLCV data.
Always available, no external dependencies required.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import hashlib

from .base import DataSourceBase, AssetMetadata


# Demo asset definitions with realistic characteristics
DEMO_ASSETS: Dict[str, Dict] = {
    # Tech stocks
    "DEMO_TECH": {
        "name": "Demo Tech Corp",
        "sector": "Technology",
        "country": "USA",
        "asset_class": "Equity",
        "base_price": 150.0,
        "volatility": 0.02,
        "drift": 0.0003,
    },
    "DEMO_SOFT": {
        "name": "Demo Software Inc",
        "sector": "Technology",
        "country": "USA",
        "asset_class": "Equity",
        "base_price": 350.0,
        "volatility": 0.018,
        "drift": 0.0002,
    },
    "DEMO_CHIP": {
        "name": "Demo Semiconductor",
        "sector": "Technology",
        "country": "USA",
        "asset_class": "Equity",
        "base_price": 85.0,
        "volatility": 0.025,
        "drift": 0.0004,
    },
    # Finance
    "DEMO_BANK": {
        "name": "Demo Bank Holdings",
        "sector": "Financial Services",
        "country": "USA",
        "asset_class": "Equity",
        "base_price": 45.0,
        "volatility": 0.015,
        "drift": 0.0001,
    },
    "DEMO_INS": {
        "name": "Demo Insurance Group",
        "sector": "Financial Services",
        "country": "UK",
        "asset_class": "Equity",
        "base_price": 120.0,
        "volatility": 0.012,
        "drift": 0.00015,
    },
    # Healthcare
    "DEMO_PHARMA": {
        "name": "Demo Pharmaceuticals",
        "sector": "Healthcare",
        "country": "USA",
        "asset_class": "Equity",
        "base_price": 180.0,
        "volatility": 0.016,
        "drift": 0.00025,
    },
    "DEMO_BIO": {
        "name": "Demo Biotech",
        "sector": "Healthcare",
        "country": "Germany",
        "asset_class": "Equity",
        "base_price": 65.0,
        "volatility": 0.03,
        "drift": 0.0001,
    },
    # Energy
    "DEMO_OIL": {
        "name": "Demo Energy Corp",
        "sector": "Energy",
        "country": "USA",
        "asset_class": "Equity",
        "base_price": 75.0,
        "volatility": 0.022,
        "drift": 0.0001,
    },
    "DEMO_RENEW": {
        "name": "Demo Renewables",
        "sector": "Energy",
        "country": "Spain",
        "asset_class": "Equity",
        "base_price": 28.0,
        "volatility": 0.028,
        "drift": 0.0003,
    },
    # Consumer
    "DEMO_RETAIL": {
        "name": "Demo Retail Inc",
        "sector": "Consumer Discretionary",
        "country": "USA",
        "asset_class": "Equity",
        "base_price": 95.0,
        "volatility": 0.017,
        "drift": 0.0002,
    },
    "DEMO_FOOD": {
        "name": "Demo Foods Ltd",
        "sector": "Consumer Staples",
        "country": "UK",
        "asset_class": "Equity",
        "base_price": 55.0,
        "volatility": 0.01,
        "drift": 0.00012,
    },
    # Crypto (higher volatility)
    "DEMO_BTC": {
        "name": "Demo Bitcoin",
        "sector": "Cryptocurrency",
        "country": "Global",
        "asset_class": "Crypto",
        "base_price": 45000.0,
        "volatility": 0.04,
        "drift": 0.0002,
    },
    "DEMO_ETH": {
        "name": "Demo Ethereum",
        "sector": "Cryptocurrency",
        "country": "Global",
        "asset_class": "Crypto",
        "base_price": 2500.0,
        "volatility": 0.045,
        "drift": 0.00025,
    },
    # Index ETFs
    "DEMO_SPY": {
        "name": "Demo S&P 500 ETF",
        "sector": "Index",
        "country": "USA",
        "asset_class": "ETF",
        "base_price": 450.0,
        "volatility": 0.012,
        "drift": 0.0002,
    },
    "DEMO_QQQ": {
        "name": "Demo Nasdaq ETF",
        "sector": "Index",
        "country": "USA",
        "asset_class": "ETF",
        "base_price": 380.0,
        "volatility": 0.015,
        "drift": 0.00025,
    },
}


class DemoDataSource(DataSourceBase):
    """
    Demo data source that generates realistic synthetic OHLCV data.
    Uses geometric Brownian motion with mean reversion for price simulation.
    """

    def __init__(self, seed: int = None):
        """
        Initialize the demo data source.

        Args:
            seed: Random seed for reproducibility (optional)
        """
        self._seed = seed

    @property
    def name(self) -> str:
        return "Demo Data"

    @property
    def is_available(self) -> bool:
        return True

    def _get_seed_for_ticker(self, ticker: str) -> int:
        """Generate a consistent seed for a ticker."""
        if self._seed is not None:
            return self._seed
        # Use hash to get consistent results for same ticker
        return int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)

    def _generate_ohlcv(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data using geometric Brownian motion.
        """
        # Get asset characteristics or use defaults
        if ticker in DEMO_ASSETS:
            asset = DEMO_ASSETS[ticker]
            base_price = asset["base_price"]
            volatility = asset["volatility"]
            drift = asset["drift"]
        else:
            # Generate characteristics based on ticker hash
            seed = self._get_seed_for_ticker(ticker)
            rng = np.random.default_rng(seed)
            base_price = rng.uniform(20, 500)
            volatility = rng.uniform(0.01, 0.03)
            drift = rng.uniform(-0.0001, 0.0003)

        # Determine time delta based on interval
        interval_map = {
            "1d": timedelta(days=1),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1w": timedelta(weeks=1),
            "1m": timedelta(days=30),
        }
        delta = interval_map.get(interval, timedelta(days=1))

        # Generate date range
        dates = []
        current = start_date
        while current <= end_date:
            # Skip weekends for daily data (equities)
            if interval == "1d" and current.weekday() < 5:
                dates.append(current)
            elif interval != "1d":
                dates.append(current)
            current += delta

        if not dates:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        n_periods = len(dates)

        # Set seed for reproducibility
        seed = self._get_seed_for_ticker(ticker + start_date.strftime("%Y%m"))
        rng = np.random.default_rng(seed)

        # Generate returns using GBM with mean reversion
        returns = rng.normal(drift, volatility, n_periods)

        # Add some autocorrelation and mean reversion
        for i in range(1, n_periods):
            # Mean reversion component
            if i > 20:
                avg_ret = np.mean(returns[i-20:i])
                returns[i] -= 0.1 * avg_ret

        # Generate close prices
        close_prices = np.zeros(n_periods)
        close_prices[0] = base_price
        for i in range(1, n_periods):
            close_prices[i] = close_prices[i-1] * (1 + returns[i])

        # Generate OHLC from close
        open_prices = np.zeros(n_periods)
        high_prices = np.zeros(n_periods)
        low_prices = np.zeros(n_periods)

        for i in range(n_periods):
            # Intraday range
            intraday_vol = volatility * rng.uniform(0.5, 1.5)

            if i == 0:
                open_prices[i] = close_prices[i] * (1 + rng.normal(0, intraday_vol * 0.5))
            else:
                # Open near previous close with some gap
                gap = rng.normal(0, volatility * 0.3)
                open_prices[i] = close_prices[i-1] * (1 + gap)

            # High and low
            high_factor = abs(rng.normal(0, intraday_vol)) + 0.001
            low_factor = abs(rng.normal(0, intraday_vol)) + 0.001

            high_prices[i] = max(open_prices[i], close_prices[i]) * (1 + high_factor)
            low_prices[i] = min(open_prices[i], close_prices[i]) * (1 - low_factor)

        # Generate volume with some patterns
        base_volume = rng.uniform(1e6, 1e8)
        volume = np.zeros(n_periods)
        for i in range(n_periods):
            # Volume tends to be higher on bigger price moves
            price_change = abs(returns[i]) if i > 0 else 0
            vol_factor = 1 + price_change * 10
            volume[i] = base_volume * vol_factor * rng.uniform(0.5, 1.5)

        # Create DataFrame
        df = pd.DataFrame({
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volume.astype(int),
        }, index=pd.DatetimeIndex(dates, name="Date"))

        return df

    def get_price_history(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get synthetic OHLCV data for a ticker."""
        ticker = ticker.upper()
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")

        return self._generate_ohlcv(ticker, start_date, end_date, interval)

    def get_metadata(self, ticker: str) -> AssetMetadata:
        """Get metadata for a ticker."""
        ticker = ticker.upper()

        if ticker in DEMO_ASSETS:
            asset = DEMO_ASSETS[ticker]
            return AssetMetadata(
                ticker=ticker,
                name=asset["name"],
                sector=asset["sector"],
                country=asset["country"],
                asset_class=asset["asset_class"],
                currency="USD",
                exchange="Demo Exchange",
            )
        else:
            # Generate metadata for unknown tickers
            return AssetMetadata(
                ticker=ticker,
                name=f"{ticker} Inc.",
                sector="Unknown",
                country="Unknown",
                asset_class="Equity",
                currency="USD",
                exchange="Demo Exchange",
            )

    def search_tickers(self, query: str) -> List[str]:
        """Search for tickers matching a query."""
        query = query.upper()
        matches = []
        for ticker, info in DEMO_ASSETS.items():
            if query in ticker or query.lower() in info["name"].lower():
                matches.append(ticker)
        return matches

    def get_available_tickers(self) -> List[str]:
        """Get list of all available demo tickers."""
        return list(DEMO_ASSETS.keys())
