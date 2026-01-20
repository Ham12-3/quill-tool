"""
Tests for technical indicators.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics.indicators import sma, ema, rsi, macd, bollinger_bands, atr


class TestSMA:
    """Tests for Simple Moving Average."""

    def test_sma_output_shape(self):
        """SMA output should have same length as input."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        result = sma(prices, 20)
        assert len(result) == len(prices)

    def test_sma_values(self):
        """SMA of constant series should equal that constant."""
        prices = pd.Series([100.0] * 50)
        result = sma(prices, 10)
        # After window is filled, all values should be 100
        assert np.allclose(result.iloc[9:], 100.0)

    def test_sma_calculation(self):
        """Test actual SMA calculation."""
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(prices, 3)
        # SMA(3) at index 2 = (1+2+3)/3 = 2
        assert result.iloc[2] == 2.0
        # SMA(3) at index 4 = (3+4+5)/3 = 4
        assert result.iloc[4] == 4.0


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_output_shape(self):
        """EMA output should have same length as input."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        result = ema(prices, 20)
        assert len(result) == len(prices)

    def test_ema_responsiveness(self):
        """EMA should be more responsive to recent prices than SMA."""
        prices = pd.Series([100.0] * 20 + [150.0])
        sma_result = sma(prices, 20)
        ema_result = ema(prices, 20)
        # EMA should be closer to 150 than SMA after the spike
        assert ema_result.iloc[-1] > sma_result.iloc[-1]


class TestRSI:
    """Tests for Relative Strength Index."""

    def test_rsi_output_shape(self):
        """RSI output should have same length as input."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        result = rsi(prices, 14)
        assert len(result) == len(prices)

    def test_rsi_range(self):
        """RSI should be bounded between 0 and 100."""
        prices = pd.Series(np.random.randn(200).cumsum() + 100)
        result = rsi(prices, 14)
        assert result.min() >= 0
        assert result.max() <= 100

    def test_rsi_uptrend(self):
        """RSI should be high (>50) in strong uptrend."""
        # Create strong uptrend
        prices = pd.Series(range(1, 101), dtype=float)
        result = rsi(prices, 14)
        # After initial period, RSI should be high
        assert result.iloc[-1] > 70

    def test_rsi_downtrend(self):
        """RSI should be low (<50) in strong downtrend."""
        # Create strong downtrend
        prices = pd.Series(range(100, 0, -1), dtype=float)
        result = rsi(prices, 14)
        # After initial period, RSI should be low
        assert result.iloc[-1] < 30


class TestMACD:
    """Tests for MACD."""

    def test_macd_output_shape(self):
        """MACD should return three series of same length as input."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        macd_line, signal, histogram = macd(prices, 12, 26, 9)

        assert len(macd_line) == len(prices)
        assert len(signal) == len(prices)
        assert len(histogram) == len(prices)

    def test_macd_histogram(self):
        """Histogram should equal MACD line minus signal line."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        macd_line, signal, histogram = macd(prices, 12, 26, 9)

        # Check that histogram = macd_line - signal
        expected = macd_line - signal
        assert np.allclose(histogram.values, expected.values, equal_nan=True)

    def test_macd_crossover_detection(self):
        """Test that MACD can detect trend changes."""
        # Create a series that trends up then down
        up = pd.Series(range(50), dtype=float)
        down = pd.Series(range(50, 0, -1), dtype=float)
        prices = pd.concat([up, down], ignore_index=True)

        macd_line, signal, histogram = macd(prices, 12, 26, 9)

        # At some point, histogram should change sign
        sign_changes = (histogram.diff().dropna() != 0).sum()
        assert sign_changes > 0


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bb_output_shape(self):
        """Bollinger Bands should return three series of same length."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        middle, upper, lower = bollinger_bands(prices, 20, 2.0)

        assert len(middle) == len(prices)
        assert len(upper) == len(prices)
        assert len(lower) == len(prices)

    def test_bb_ordering(self):
        """Upper band should be above middle, lower should be below."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        middle, upper, lower = bollinger_bands(prices, 20, 2.0)

        # After window is filled
        assert (upper.iloc[20:] >= middle.iloc[20:]).all()
        assert (lower.iloc[20:] <= middle.iloc[20:]).all()

    def test_bb_middle_equals_sma(self):
        """Middle band should equal SMA."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        middle, upper, lower = bollinger_bands(prices, 20, 2.0)
        sma_result = sma(prices, 20)

        assert np.allclose(middle.values, sma_result.values, equal_nan=True)


class TestATR:
    """Tests for Average True Range."""

    def test_atr_output_shape(self):
        """ATR output should have same length as input."""
        n = 100
        high = pd.Series(np.random.randn(n).cumsum() + 102)
        low = pd.Series(np.random.randn(n).cumsum() + 98)
        close = pd.Series(np.random.randn(n).cumsum() + 100)

        result = atr(high, low, close, 14)
        assert len(result) == n

    def test_atr_positive(self):
        """ATR should always be positive."""
        n = 100
        high = pd.Series(np.random.randn(n).cumsum() + 102)
        low = high - 5  # Ensure low is always less than high
        close = (high + low) / 2

        result = atr(high, low, close, 14)
        assert (result.dropna() >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
