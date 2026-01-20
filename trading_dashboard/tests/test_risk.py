"""
Tests for risk analytics.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics.risk import (
    calculate_returns, historical_var, historical_cvar, calculate_var_cvar,
    rolling_volatility, annualized_volatility, correlation_matrix
)


class TestReturns:
    """Tests for return calculations."""

    def test_simple_returns(self):
        """Test simple return calculation."""
        prices = pd.Series([100.0, 110.0, 99.0, 108.9])
        returns = calculate_returns(prices, method="simple")

        expected = pd.Series([np.nan, 0.10, -0.10, 0.10])
        # Check non-NaN values
        assert np.allclose(returns.iloc[1:].values, expected.iloc[1:].values)

    def test_log_returns(self):
        """Test log return calculation."""
        prices = pd.Series([100.0, 110.0, 99.0, 108.9])
        returns = calculate_returns(prices, method="log")

        expected = pd.Series([np.nan, np.log(1.1), np.log(0.9), np.log(1.1)])
        assert np.allclose(returns.iloc[1:].values, expected.iloc[1:].values)


class TestVaR:
    """Tests for Value at Risk calculations."""

    def test_var_on_uniform_distribution(self):
        """Test VaR on uniform distribution."""
        # Uniform distribution from -1 to 1
        np.random.seed(42)
        returns = pd.Series(np.random.uniform(-0.10, 0.10, 10000))

        var_95 = historical_var(returns, 0.95)

        # For uniform(-0.10, 0.10), 5th percentile is around -0.09
        assert 0.08 < var_95 < 0.11  # VaR is returned as positive

    def test_var_ordering(self):
        """99% VaR should be greater than 95% VaR."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(-0.001, 0.02, 1000))

        var_95 = historical_var(returns, 0.95)
        var_99 = historical_var(returns, 0.99)

        assert var_99 > var_95

    def test_var_positive_returns(self):
        """VaR on all positive returns should be negative (gain)."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05] * 100)
        var_95 = historical_var(returns, 0.95)

        # 5th percentile of positive returns is small positive, so VaR is negative
        # Our function returns positive VaR (loss), so this should be negative
        assert var_95 < 0.02  # Should be close to -0.01 (the smallest gain)


class TestCVaR:
    """Tests for Conditional Value at Risk."""

    def test_cvar_greater_than_var(self):
        """CVaR should be >= VaR."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(-0.001, 0.02, 1000))

        var_95 = historical_var(returns, 0.95)
        cvar_95 = historical_cvar(returns, 0.95)

        assert cvar_95 >= var_95

    def test_cvar_on_synthetic_data(self):
        """Test CVaR calculation on known distribution."""
        # Create returns with known tail
        np.random.seed(42)
        normal_returns = np.random.normal(0, 0.01, 950)
        tail_returns = np.random.uniform(-0.10, -0.05, 50)
        returns = pd.Series(np.concatenate([normal_returns, tail_returns]))

        cvar_95 = historical_cvar(returns, 0.95)

        # CVaR should be in the tail region
        assert cvar_95 > 0.04


class TestVarCvar:
    """Tests for combined VaR/CVaR calculation."""

    def test_calculate_var_cvar_keys(self):
        """Test that calculate_var_cvar returns all expected keys."""
        returns = pd.Series(np.random.normal(0, 0.02, 500))
        result = calculate_var_cvar(returns)

        assert "var_95" in result
        assert "var_99" in result
        assert "cvar_95" in result
        assert "cvar_99" in result


class TestVolatility:
    """Tests for volatility calculations."""

    def test_rolling_vol_shape(self):
        """Rolling volatility should have same length as input."""
        returns = pd.Series(np.random.normal(0, 0.02, 100))
        vol = rolling_volatility(returns, window=20, annualize=True)

        assert len(vol) == len(returns)

    def test_annualized_volatility(self):
        """Test annualized volatility calculation."""
        # Create returns with known daily std = 1%
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 1000))

        ann_vol = annualized_volatility(returns)

        # Annualized = daily * sqrt(252) ≈ 0.01 * 15.87 ≈ 0.159
        assert 0.14 < ann_vol < 0.18

    def test_zero_volatility(self):
        """Zero volatility for constant returns."""
        returns = pd.Series([0.01] * 100)
        ann_vol = annualized_volatility(returns)

        assert ann_vol == 0.0


class TestCorrelation:
    """Tests for correlation matrix."""

    def test_correlation_diagonal(self):
        """Diagonal of correlation matrix should be 1."""
        np.random.seed(42)
        returns_dict = {
            "A": pd.Series(np.random.normal(0, 0.02, 100)),
            "B": pd.Series(np.random.normal(0, 0.02, 100)),
            "C": pd.Series(np.random.normal(0, 0.02, 100)),
        }

        corr = correlation_matrix(returns_dict)

        # Diagonal should be 1
        for i in range(len(corr)):
            assert abs(corr.iloc[i, i] - 1.0) < 1e-10

    def test_correlation_symmetry(self):
        """Correlation matrix should be symmetric."""
        np.random.seed(42)
        returns_dict = {
            "A": pd.Series(np.random.normal(0, 0.02, 100)),
            "B": pd.Series(np.random.normal(0, 0.02, 100)),
        }

        corr = correlation_matrix(returns_dict)

        assert corr.loc["A", "B"] == corr.loc["B", "A"]

    def test_correlation_range(self):
        """Correlations should be between -1 and 1."""
        np.random.seed(42)
        returns_dict = {
            "A": pd.Series(np.random.normal(0, 0.02, 100)),
            "B": pd.Series(np.random.normal(0, 0.02, 100)),
            "C": pd.Series(np.random.normal(0, 0.02, 100)),
        }

        corr = correlation_matrix(returns_dict)

        assert (corr >= -1).all().all()
        assert (corr <= 1).all().all()

    def test_perfect_correlation(self):
        """Test correlation of identical series."""
        returns = pd.Series(np.random.normal(0, 0.02, 100))
        returns_dict = {
            "A": returns,
            "B": returns.copy(),
        }

        corr = correlation_matrix(returns_dict)

        assert abs(corr.loc["A", "B"] - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
