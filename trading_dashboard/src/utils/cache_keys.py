"""
Cache key generation utilities for Streamlit caching.
"""
import hashlib
from typing import Any, List, Optional


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a consistent cache key from arguments.

    Args:
        *args: Positional arguments to include in key
        **kwargs: Keyword arguments to include in key

    Returns:
        Cache key string
    """
    parts = [str(arg) for arg in args]
    parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_str = "|".join(parts)
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def price_data_key(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    source: str
) -> str:
    """
    Generate cache key for price data.

    Args:
        ticker: Ticker symbol
        start: Start date
        end: End date
        interval: Data interval
        source: Data source name

    Returns:
        Cache key
    """
    return generate_cache_key(
        "price",
        ticker=ticker.upper(),
        start=start,
        end=end,
        interval=interval,
        source=source
    )


def metadata_key(ticker: str, source: str) -> str:
    """
    Generate cache key for asset metadata.

    Args:
        ticker: Ticker symbol
        source: Data source name

    Returns:
        Cache key
    """
    return generate_cache_key("metadata", ticker=ticker.upper(), source=source)


def indicator_key(
    ticker: str,
    indicator: str,
    period: int,
    start: str = None,
    end: str = None
) -> str:
    """
    Generate cache key for indicator data.

    Args:
        ticker: Ticker symbol
        indicator: Indicator name (e.g., "RSI", "SMA")
        period: Indicator period
        start: Optional start date
        end: Optional end date

    Returns:
        Cache key
    """
    return generate_cache_key(
        "indicator",
        ticker=ticker.upper(),
        indicator=indicator,
        period=period,
        start=start,
        end=end
    )


def portfolio_key(
    tickers: List[str],
    weights: List[float],
    start: str,
    end: str
) -> str:
    """
    Generate cache key for portfolio calculations.

    Args:
        tickers: List of ticker symbols
        weights: List of weights
        start: Start date
        end: End date

    Returns:
        Cache key
    """
    tickers_str = ",".join(sorted(tickers))
    weights_str = ",".join(f"{w:.4f}" for w in weights)
    return generate_cache_key(
        "portfolio",
        tickers=tickers_str,
        weights=weights_str,
        start=start,
        end=end
    )


def backtest_key(
    ticker: str,
    strategy: str,
    start: str,
    end: str,
    params: dict
) -> str:
    """
    Generate cache key for backtest results.

    Args:
        ticker: Ticker symbol
        strategy: Strategy name
        start: Start date
        end: End date
        params: Strategy parameters

    Returns:
        Cache key
    """
    params_str = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return generate_cache_key(
        "backtest",
        ticker=ticker.upper(),
        strategy=strategy,
        start=start,
        end=end,
        params=params_str
    )
