"""
Tests for formatting utilities.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.formatting import (
    format_percent, format_currency, format_number, format_change,
    format_ratio, format_date, format_volume, color_metric,
    style_dataframe_returns, truncate_string, format_time_period
)


class TestFormatPercent:
    """Tests for percent formatting."""

    def test_positive_percent(self):
        """Test positive percentage."""
        result = format_percent(0.15)
        assert result == "15.00%"

    def test_negative_percent(self):
        """Test negative percentage."""
        result = format_percent(-0.05)
        assert result == "-5.00%"

    def test_with_sign(self):
        """Test with sign inclusion."""
        result = format_percent(0.10, include_sign=True)
        assert result == "+10.00%"

    def test_nan_handling(self):
        """Test NaN handling."""
        result = format_percent(float('nan'))
        assert result == "N/A"


class TestFormatCurrency:
    """Tests for currency formatting."""

    def test_basic_currency(self):
        """Test basic currency formatting."""
        result = format_currency(1234.56)
        assert result == "$1,234.56"

    def test_custom_symbol(self):
        """Test custom currency symbol."""
        result = format_currency(100.00, symbol="€")
        assert result == "€100.00"


class TestFormatNumber:
    """Tests for number formatting."""

    def test_basic_number(self):
        """Test basic number formatting."""
        result = format_number(1234.567, decimals=2)
        assert result == "1,234.57"

    def test_abbreviate_millions(self):
        """Test abbreviation for millions."""
        result = format_number(1500000, abbreviate=True)
        assert result == "1.50M"

    def test_abbreviate_billions(self):
        """Test abbreviation for billions."""
        result = format_number(2500000000, abbreviate=True)
        assert result == "2.50B"


class TestStyleDataframeReturns:
    """Tests for DataFrame styling."""

    def test_import_does_not_crash(self):
        """Test that importing the function does not crash."""
        # This test verifies the fix for pd.io.formats.style issue
        from src.utils.formatting import style_dataframe_returns
        assert callable(style_dataframe_returns)

    def test_basic_styling(self):
        """Test basic DataFrame styling."""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT'],
            'return': [0.05, -0.03, 0.00]
        })

        result = style_dataframe_returns(df, ['return'])

        # Result should be either a Styler or DataFrame (fallback)
        assert result is not None
        # Should have same data
        if hasattr(result, 'data'):
            # It's a Styler
            assert len(result.data) == 3
        else:
            # It's a plain DataFrame (fallback)
            assert len(result) == 3

    def test_nonexistent_columns(self):
        """Test with nonexistent columns."""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL'],
            'price': [150.0, 140.0]
        })

        # Should not crash with nonexistent column
        result = style_dataframe_returns(df, ['nonexistent'])
        assert result is not None

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = style_dataframe_returns(df, ['return'])
        assert result is not None


class TestTruncateString:
    """Tests for string truncation."""

    def test_no_truncation_needed(self):
        """Test when string is short enough."""
        result = truncate_string("hello", max_length=10)
        assert result == "hello"

    def test_truncation_with_ellipsis(self):
        """Test truncation adds ellipsis."""
        result = truncate_string("hello world this is long", max_length=10)
        assert len(result) == 10
        assert result.endswith("...")


class TestFormatTimePeriod:
    """Tests for time period formatting."""

    def test_days(self):
        """Test days formatting."""
        result = format_time_period(5)
        assert result == "5 days"

    def test_weeks(self):
        """Test weeks formatting."""
        result = format_time_period(14)
        assert "weeks" in result

    def test_months(self):
        """Test months formatting."""
        result = format_time_period(60)
        assert "months" in result

    def test_years(self):
        """Test years formatting."""
        result = format_time_period(400)
        assert "years" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
