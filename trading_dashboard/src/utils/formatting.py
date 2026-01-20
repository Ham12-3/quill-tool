"""
Formatting utilities for display.
"""
from typing import Optional, Any, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler


def format_percent(value: float, decimals: int = 2, include_sign: bool = False) -> str:
    """
    Format a decimal as percentage string.

    Args:
        value: Value as decimal (e.g., 0.15 for 15%)
        decimals: Number of decimal places
        include_sign: Whether to include + sign for positive values

    Returns:
        Formatted percentage string
    """
    if pd.isna(value):
        return "N/A"

    pct = value * 100
    if include_sign and pct > 0:
        return f"+{pct:.{decimals}f}%"
    return f"{pct:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2, symbol: str = "$") -> str:
    """
    Format a number as currency.

    Args:
        value: Numeric value
        decimals: Number of decimal places
        symbol: Currency symbol

    Returns:
        Formatted currency string
    """
    if pd.isna(value):
        return "N/A"
    return f"{symbol}{value:,.{decimals}f}"


def format_number(value: float, decimals: int = 2, abbreviate: bool = False) -> str:
    """
    Format a number with specified decimals.

    Args:
        value: Numeric value
        decimals: Number of decimal places
        abbreviate: Whether to abbreviate large numbers (K, M, B)

    Returns:
        Formatted number string
    """
    if pd.isna(value):
        return "N/A"

    if abbreviate:
        if abs(value) >= 1e9:
            return f"{value/1e9:.{decimals}f}B"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.{decimals}f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.{decimals}f}K"

    return f"{value:,.{decimals}f}"


def format_change(value: float, decimals: int = 2) -> str:
    """
    Format a change value with color indicator.

    Args:
        value: Change value as decimal
        decimals: Number of decimal places

    Returns:
        Formatted change string with emoji indicator
    """
    if pd.isna(value):
        return "N/A"

    pct = value * 100
    if pct > 0:
        return f"🟢 +{pct:.{decimals}f}%"
    elif pct < 0:
        return f"🔴 {pct:.{decimals}f}%"
    else:
        return f"⚪ {pct:.{decimals}f}%"


def format_ratio(value: float, decimals: int = 2) -> str:
    """
    Format a ratio value.

    Args:
        value: Ratio value
        decimals: Number of decimal places

    Returns:
        Formatted ratio string
    """
    if pd.isna(value):
        return "N/A"
    if value == float('inf'):
        return "∞"
    if value == float('-inf'):
        return "-∞"
    return f"{value:.{decimals}f}"


def format_date(date, include_time: bool = False) -> str:
    """
    Format a date/datetime for display.

    Args:
        date: Date or datetime object
        include_time: Whether to include time

    Returns:
        Formatted date string
    """
    if pd.isna(date):
        return "N/A"

    try:
        if isinstance(date, str):
            date = pd.to_datetime(date)

        if include_time:
            return date.strftime("%Y-%m-%d %H:%M")
        return date.strftime("%Y-%m-%d")
    except Exception:
        return str(date)


def format_volume(value: float) -> str:
    """
    Format trading volume with abbreviation.

    Args:
        value: Volume value

    Returns:
        Formatted volume string
    """
    if pd.isna(value):
        return "N/A"

    if value >= 1e9:
        return f"{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{value/1e3:.1f}K"
    return f"{value:.0f}"


def color_metric(value: float, positive_is_good: bool = True) -> str:
    """
    Return CSS color based on value sign.

    Args:
        value: Numeric value
        positive_is_good: Whether positive values should be green

    Returns:
        CSS color string
    """
    if pd.isna(value):
        return "gray"

    is_positive = value > 0
    if positive_is_good:
        return "green" if is_positive else "red"
    else:
        return "red" if is_positive else "green"


def style_dataframe_returns(df: pd.DataFrame, return_columns: list) -> Any:
    """
    Style a DataFrame with colored return columns.

    Args:
        df: DataFrame to style
        return_columns: List of column names containing returns

    Returns:
        Styled DataFrame if styling available, otherwise plain DataFrame
    """
    def color_returns(val):
        if pd.isna(val):
            return ""
        color = "green" if val > 0 else "red" if val < 0 else "black"
        return f"color: {color}"

    # Only apply to columns that exist
    existing_columns = [c for c in return_columns if c in df.columns]

    try:
        # Use applymap for older pandas, map for newer versions
        styler = df.style
        if hasattr(styler, 'map'):
            # pandas >= 2.1.0
            return styler.map(color_returns, subset=existing_columns)
        else:
            # pandas < 2.1.0
            return styler.applymap(color_returns, subset=existing_columns)
    except Exception:
        # Fallback to plain DataFrame if styling unavailable
        return df


def truncate_string(s: str, max_length: int = 50) -> str:
    """
    Truncate a string with ellipsis.

    Args:
        s: String to truncate
        max_length: Maximum length

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."


def format_time_period(days: int) -> str:
    """
    Format a number of days as a readable time period.

    Args:
        days: Number of days

    Returns:
        Formatted time period string
    """
    if days < 7:
        return f"{days} days"
    elif days < 30:
        weeks = days / 7
        return f"{weeks:.1f} weeks"
    elif days < 365:
        months = days / 30
        return f"{months:.1f} months"
    else:
        years = days / 365
        return f"{years:.1f} years"
