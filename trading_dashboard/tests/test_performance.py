"""
Tests for performance analytics.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics.performance import (
    total_return, annualized_return, sharpe_ratio,
    drawdown, max_drawdown, calmar_ratio, calculate_all_metrics
)


class TestTotalReturn:
    """Tests for total return calculation."""

    def test_positive_return(self):
        """Test positive total return."""
        prices = pd.Series([100.0, 110.0, 120.0, 130.0])
        ret = total_return(prices)

        assert abs(ret - 0.30) < 1e-10  # 30% return

    def test_negative_return(self):
        """Test negative total return."""
        prices = pd.Series([100.0, 90.0, 80.0, 70.0])
        ret = total_return(prices)

        assert abs(ret - (-0.30)) < 1e-10  # -30% return

    def test_zero_return(self):
        """Test zero return."""
        prices = pd.Series([100.0, 110.0, 90.0, 100.0])
        ret = total_return(prices)

        assert abs(ret) < 1e-10  # 0% return


class TestAnnualizedReturn:
    """Tests for annualized return (CAGR)."""

    def test_one_year_return(self):
        """Test CAGR for exactly one year of data (252 trading days)."""
        # If we have 252 days and 26% return, CAGR should be ~26%
        prices = pd.Series([100.0] + [None] * 250 + [126.0])
        prices = prices.interpolate()

        cagr = annualized_return(prices, periods_per_year=252)

        # Should be close to 26% (one year)
        assert 0.24 < cagr < 0.28

    def test_multi_year_return(self):
        """Test CAGR for multiple years."""
        # 2 years, doubling (100% total return) = ~41.4% CAGR
        prices = pd.Series([100.0] + [None] * 503 + [200.0])
        prices = prices.interpolate()

        cagr = annualized_return(prices, periods_per_year=252)

        # sqrt(2) - 1 ≈ 0.414
        assert 0.38 < cagr < 0.44


class TestSharpeRatio:
    """Tests for Sharpe ratio."""

    def test_sharpe_positive(self):
        """Test Sharpe for positive returns with risk."""
        np.random.seed(42)
        # Returns averaging 0.05% daily with 1% daily vol
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252))

        sharpe = sharpe_ratio(returns, risk_free_rate=0.0)

        # Should be around sqrt(252) * 0.0005 / 0.01 ≈ 0.79
        assert sharpe > 0

    def test_sharpe_with_risk_free(self):
        """Sharpe with risk-free rate should be lower."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))

        sharpe_0 = sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_5 = sharpe_ratio(returns, risk_free_rate=0.05)

        assert sharpe_5 < sharpe_0


class TestDrawdown:
    """Tests for drawdown calculations."""

    def test_drawdown_shape(self):
        """Drawdown series should have same length as prices."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        dd = drawdown(prices)

        assert len(dd) == len(prices)

    def test_drawdown_always_negative_or_zero(self):
        """Drawdown should always be <= 0."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        dd = drawdown(prices)

        assert (dd <= 0).all()

    def test_drawdown_at_peak(self):
        """Drawdown should be 0 at all-time high."""
        prices = pd.Series([100, 110, 120, 130, 140])  # Monotonically increasing
        dd = drawdown(prices)

        # All points are peaks, so drawdown should be 0 everywhere
        assert (dd == 0).all()

    def test_drawdown_calculation(self):
        """Test specific drawdown calculation."""
        prices = pd.Series([100, 110, 100, 90, 95])
        dd = drawdown(prices)

        # At index 3 (price=90), peak was 110
        # Drawdown = (90 - 110) / 110 = -0.1818...
        assert abs(dd.iloc[3] - (-0.1818)) < 0.01


class TestMaxDrawdown:
    """Tests for maximum drawdown."""

    def test_max_drawdown_value(self):
        """Test maximum drawdown calculation."""
        prices = pd.Series([100, 110, 100, 90, 95])
        mdd = max_drawdown(prices)

        # Max DD at price=90, peak=110
        # MDD = (90 - 110) / 110 = -0.1818...
        assert abs(mdd - (-0.1818)) < 0.01

    def test_max_drawdown_no_decline(self):
        """Max drawdown should be 0 for monotonically increasing prices."""
        prices = pd.Series([100, 110, 120, 130, 140])
        mdd = max_drawdown(prices)

        assert mdd == 0

    def test_max_drawdown_total_loss(self):
        """Test max drawdown approaching -100%."""
        prices = pd.Series([100, 50, 25, 10, 5])
        mdd = max_drawdown(prices)

        # MDD = (5 - 100) / 100 = -0.95
        assert mdd == -0.95


class TestCalmarRatio:
    """Tests for Calmar ratio."""

    def test_calmar_positive(self):
        """Test Calmar for positive return with drawdown."""
        # Create prices with 20% return and 10% max drawdown
        prices = pd.Series([100, 110, 99, 108, 120])
        cal = calmar_ratio(prices)

        # CAGR / |MDD|
        assert cal > 0

    def test_calmar_no_drawdown(self):
        """Calmar should be infinite with no drawdown."""
        prices = pd.Series([100, 110, 120, 130, 140])
        cal = calmar_ratio(prices)

        assert cal == np.inf


class TestCalculateAllMetrics:
    """Tests for combined metrics calculation."""

    def test_metrics_object(self):
        """Test that all metrics are calculated."""
        prices = pd.Series(np.random.randn(252).cumsum() + 100)
        metrics = calculate_all_metrics(prices)

        # Check all attributes exist
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'annualized_return')
        assert hasattr(metrics, 'annualized_volatility')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'calmar_ratio')
        assert hasattr(metrics, 'win_rate')
        assert hasattr(metrics, 'num_positive_days')
        assert hasattr(metrics, 'num_negative_days')

    def test_win_rate_range(self):
        """Win rate should be between 0 and 1."""
        prices = pd.Series(np.random.randn(252).cumsum() + 100)
        metrics = calculate_all_metrics(prices)

        assert 0 <= metrics.win_rate <= 1

    def test_day_counts(self):
        """Positive + negative days should equal total trading days minus 1."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        metrics = calculate_all_metrics(prices)

        # Total days = 99 (100 prices = 99 returns, some might be 0)
        assert metrics.num_positive_days + metrics.num_negative_days <= 99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
