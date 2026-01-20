"""
Technical indicators for price analysis.
Includes SMA, EMA, RSI, MACD, and other common indicators.
"""
import numpy as np
import pandas as pd
from typing import Tuple


def sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        prices: Series of prices (typically Close)
        period: Lookback period

    Returns:
        Series with SMA values
    """
    return prices.rolling(window=period, min_periods=1).mean()


def ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        prices: Series of prices (typically Close)
        period: Lookback period

    Returns:
        Series with EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Args:
        prices: Series of prices (typically Close)
        period: Lookback period (default 14)

    Returns:
        Series with RSI values (0-100)
    """
    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    # Calculate average gains and losses using EMA
    avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi_values = 100 - (100 / (1 + rs))

    # Handle edge cases
    rsi_values = rsi_values.fillna(50)  # Neutral when no data
    rsi_values = rsi_values.replace([np.inf, -np.inf], 50)

    return rsi_values


def macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Series of prices (typically Close)
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    # Calculate EMAs
    fast_ema = ema(prices, fast_period)
    slow_ema = ema(prices, slow_period)

    # MACD line
    macd_line = fast_ema - slow_ema

    # Signal line
    signal_line = ema(macd_line, signal_period)

    # Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Series of prices (typically Close)
        period: SMA period (default 20)
        std_dev: Number of standard deviations (default 2.0)

    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle = sma(prices, period)
    rolling_std = prices.rolling(window=period, min_periods=1).std()

    upper = middle + (rolling_std * std_dev)
    lower = middle - (rolling_std * std_dev)

    return middle, upper, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range.

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        period: Lookback period (default 14)

    Returns:
        Series with ATR values
    """
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR as EMA of True Range
    return true_range.ewm(span=period, adjust=False).mean()


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        k_period: %K lookback period (default 14)
        d_period: %D smoothing period (default 3)

    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()

    # %K
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = k.fillna(50)

    # %D (SMA of %K)
    d = sma(k, d_period)

    return k, d


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume.

    Args:
        close: Series of close prices
        volume: Series of volume

    Returns:
        Series with OBV values
    """
    # Determine direction
    direction = np.sign(close.diff())
    direction.iloc[0] = 0

    # Calculate OBV
    obv_values = (direction * volume).cumsum()

    return obv_values


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (intraday only).

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        volume: Series of volume

    Returns:
        Series with VWAP values
    """
    typical_price = (high + low + close) / 3
    cumulative_tp_vol = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum()

    return cumulative_tp_vol / cumulative_vol


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all common indicators to a DataFrame.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with indicator columns added
    """
    result = df.copy()

    # Moving averages
    result["SMA_20"] = sma(df["Close"], 20)
    result["SMA_50"] = sma(df["Close"], 50)
    result["SMA_200"] = sma(df["Close"], 200)
    result["EMA_12"] = ema(df["Close"], 12)
    result["EMA_26"] = ema(df["Close"], 26)

    # RSI
    result["RSI"] = rsi(df["Close"], 14)

    # MACD
    macd_line, signal_line, histogram = macd(df["Close"])
    result["MACD"] = macd_line
    result["MACD_Signal"] = signal_line
    result["MACD_Hist"] = histogram

    # Bollinger Bands
    bb_mid, bb_upper, bb_lower = bollinger_bands(df["Close"])
    result["BB_Middle"] = bb_mid
    result["BB_Upper"] = bb_upper
    result["BB_Lower"] = bb_lower

    # ATR
    result["ATR"] = atr(df["High"], df["Low"], df["Close"])

    return result
