"""
Performance analytics calculations.
Includes returns, Sharpe ratio, drawdown, and attribution analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .risk import TRADING_DAYS_PER_YEAR, calculate_returns


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_return: float
    annualized_return: float  # CAGR
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    best_day: float
    worst_day: float
    avg_daily_return: float
    skewness: float
    kurtosis: float
    num_positive_days: int
    num_negative_days: int


def total_return(prices: pd.Series) -> float:
    """
    Calculate total return from price series.

    Args:
        prices: Series of prices

    Returns:
        Total return as decimal (e.g., 0.25 for 25%)
    """
    prices = prices.dropna()
    if len(prices) < 2:
        return 0.0

    return (prices.iloc[-1] / prices.iloc[0]) - 1


def annualized_return(
    prices: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate annualized return (CAGR).

    Args:
        prices: Series of prices
        periods_per_year: Number of periods per year

    Returns:
        Annualized return as decimal
    """
    prices = prices.dropna()
    if len(prices) < 2:
        return 0.0

    total_ret = total_return(prices)
    num_periods = len(prices) - 1

    if num_periods == 0:
        return 0.0

    years = num_periods / periods_per_year

    if years <= 0:
        return 0.0

    # CAGR formula
    if total_ret <= -1:
        return -1.0  # Total loss

    return (1 + total_ret) ** (1 / years) - 1


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    # Convert annual risk-free to period rate
    period_rf = risk_free_rate / periods_per_year

    # Excess returns
    excess_returns = returns - period_rf

    # Standard deviation
    std = excess_returns.std()
    if std == 0:
        return 0.0

    # Annualized Sharpe
    return (excess_returns.mean() / std) * np.sqrt(periods_per_year)


def drawdown(prices: pd.Series) -> pd.Series:
    """
    Calculate drawdown series.

    Args:
        prices: Series of prices (or equity curve)

    Returns:
        Series of drawdown values (negative numbers)
    """
    # Running maximum
    running_max = prices.expanding().max()

    # Drawdown
    dd = (prices - running_max) / running_max

    return dd


def max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        prices: Series of prices (or equity curve)

    Returns:
        Maximum drawdown as negative decimal (e.g., -0.25 for -25%)
    """
    dd = drawdown(prices)
    return dd.min()


def drawdown_duration(prices: pd.Series) -> Tuple[int, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown duration.

    Args:
        prices: Series of prices

    Returns:
        Tuple of (duration_in_days, peak_date, recovery_date or None)
    """
    dd = drawdown(prices)

    # Find periods where not at peak
    in_drawdown = dd < 0

    if not in_drawdown.any():
        return 0, None, None

    # Find max drawdown point
    max_dd_idx = dd.idxmin()

    # Find the peak before max drawdown
    peak_date = prices[:max_dd_idx].idxmax()

    # Find recovery (if any)
    peak_value = prices[peak_date]
    after_trough = prices[max_dd_idx:]
    recovered = after_trough[after_trough >= peak_value]

    if len(recovered) > 0:
        recovery_date = recovered.index[0]
        duration = (recovery_date - peak_date).days
    else:
        recovery_date = None
        duration = (prices.index[-1] - peak_date).days

    return duration, peak_date, recovery_date


def calmar_ratio(
    prices: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate Calmar ratio (CAGR / |Max Drawdown|).

    Args:
        prices: Series of prices
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    ann_ret = annualized_return(prices, periods_per_year)
    max_dd = max_drawdown(prices)

    if max_dd == 0:
        return np.inf if ann_ret > 0 else 0.0

    return ann_ret / abs(max_dd)


def calculate_all_metrics(
    prices: pd.Series,
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """
    Calculate all performance metrics.

    Args:
        prices: Series of prices (or equity curve)
        risk_free_rate: Annualized risk-free rate

    Returns:
        PerformanceMetrics object
    """
    prices = prices.dropna()
    if len(prices) < 2:
        return PerformanceMetrics(
            total_return=0, annualized_return=0, annualized_volatility=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0, calmar_ratio=0,
            win_rate=0, profit_factor=0, best_day=0, worst_day=0,
            avg_daily_return=0, skewness=0, kurtosis=0,
            num_positive_days=0, num_negative_days=0
        )

    returns = calculate_returns(prices)
    returns = returns.dropna()

    # Basic returns
    tot_ret = total_return(prices)
    ann_ret = annualized_return(prices)

    # Volatility
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Risk-adjusted
    sharp = sharpe_ratio(returns, risk_free_rate)

    # Sortino
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    downside = returns[returns < daily_rf]
    if len(downside) > 0:
        downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sortino = (ann_ret - risk_free_rate) / downside_std if downside_std > 0 else 0
    else:
        sortino = np.inf if ann_ret > risk_free_rate else 0

    # Drawdown
    max_dd = max_drawdown(prices)
    calmar = calmar_ratio(prices)

    # Win rate
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    num_positive = len(positive_returns)
    num_negative = len(negative_returns)
    win_rate = num_positive / len(returns) if len(returns) > 0 else 0

    # Profit factor
    total_gains = positive_returns.sum() if len(positive_returns) > 0 else 0
    total_losses = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0
    profit_factor = total_gains / total_losses if total_losses > 0 else np.inf

    # Distribution stats
    best = returns.max()
    worst = returns.min()
    avg = returns.mean()
    skew = returns.skew()
    kurt = returns.kurtosis()

    return PerformanceMetrics(
        total_return=tot_ret,
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharp,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        best_day=best,
        worst_day=worst,
        avg_daily_return=avg,
        skewness=skew,
        kurtosis=kurt,
        num_positive_days=num_positive,
        num_negative_days=num_negative,
    )


def performance_attribution(
    portfolio_weights: Dict[str, float],
    asset_returns: Dict[str, pd.Series],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate performance attribution by asset.

    Args:
        portfolio_weights: Dictionary mapping ticker to weight
        asset_returns: Dictionary mapping ticker to returns series
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        DataFrame with attribution by asset
    """
    attribution_data = []

    for ticker, weight in portfolio_weights.items():
        if ticker not in asset_returns:
            continue

        returns = asset_returns[ticker]

        # Filter by date if specified
        if start_date:
            returns = returns[returns.index >= start_date]
        if end_date:
            returns = returns[returns.index <= end_date]

        if len(returns) == 0:
            continue

        # Asset contribution
        asset_return = (1 + returns).prod() - 1
        contribution = weight * asset_return
        weighted_vol = weight * returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        attribution_data.append({
            "Ticker": ticker,
            "Weight": weight,
            "Asset Return": asset_return,
            "Contribution": contribution,
            "Weighted Volatility": weighted_vol,
        })

    df = pd.DataFrame(attribution_data)
    if len(df) > 0:
        df["% of Total Return"] = df["Contribution"] / df["Contribution"].sum() * 100

    return df


def time_period_attribution(
    portfolio_returns: pd.Series,
    frequency: str = "M"
) -> pd.DataFrame:
    """
    Calculate performance attribution by time period.

    Args:
        portfolio_returns: Series of portfolio returns
        frequency: Resampling frequency ('D', 'W', 'M', 'Q', 'Y')

    Returns:
        DataFrame with period returns
    """
    # Resample returns
    period_returns = portfolio_returns.resample(frequency).apply(
        lambda x: (1 + x).prod() - 1
    )

    df = period_returns.to_frame("Return")
    df["Cumulative"] = (1 + df["Return"]).cumprod() - 1

    return df


def rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    risk_free_rate: float = 0.0
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.

    Args:
        returns: Series of returns
        window: Rolling window in periods
        risk_free_rate: Annualized risk-free rate

    Returns:
        Series of rolling Sharpe ratios
    """
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf

    rolling_mean = excess.rolling(window=window).mean()
    rolling_std = excess.rolling(window=window).std()

    # Annualized
    return (rolling_mean / rolling_std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def underwater_chart(prices: pd.Series) -> pd.Series:
    """
    Create underwater equity curve (drawdown over time).

    Args:
        prices: Series of prices (equity curve)

    Returns:
        Series of drawdowns for underwater chart
    """
    return drawdown(prices)
