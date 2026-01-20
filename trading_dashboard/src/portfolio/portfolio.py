"""
Portfolio management and analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..analytics.risk import (
    calculate_returns, annualized_volatility, correlation_matrix,
    historical_var, historical_cvar, covariance_matrix
)
from ..analytics.performance import (
    total_return, annualized_return, sharpe_ratio, max_drawdown,
    calmar_ratio, calculate_all_metrics, PerformanceMetrics
)
from ..datasource.base import DataSourceBase, AssetMetadata


@dataclass
class PortfolioPosition:
    """Represents a position in a portfolio."""
    ticker: str
    weight: float
    metadata: Optional[AssetMetadata] = None
    current_price: Optional[float] = None
    returns: Optional[pd.Series] = None


@dataclass
class Portfolio:
    """
    Portfolio container with positions and analytics.
    """
    name: str = "My Portfolio"
    positions: Dict[str, PortfolioPosition] = field(default_factory=dict)
    initial_value: float = 100000.0
    benchmark_ticker: Optional[str] = None

    def add_position(
        self,
        ticker: str,
        weight: float,
        metadata: Optional[AssetMetadata] = None
    ) -> None:
        """Add or update a position."""
        self.positions[ticker] = PortfolioPosition(
            ticker=ticker,
            weight=weight,
            metadata=metadata,
        )

    def remove_position(self, ticker: str) -> None:
        """Remove a position."""
        if ticker in self.positions:
            del self.positions[ticker]

    def get_weights(self) -> Dict[str, float]:
        """Get dictionary of ticker to weight."""
        return {t: p.weight for t, p in self.positions.items()}

    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(p.weight for p in self.positions.values())
        if total > 0:
            for pos in self.positions.values():
                pos.weight /= total

    def set_equal_weights(self) -> None:
        """Set all positions to equal weight."""
        n = len(self.positions)
        if n > 0:
            weight = 1.0 / n
            for pos in self.positions.values():
                pos.weight = weight

    @property
    def tickers(self) -> List[str]:
        """Get list of tickers in portfolio."""
        return list(self.positions.keys())

    @property
    def total_weight(self) -> float:
        """Get sum of all weights."""
        return sum(p.weight for p in self.positions.values())


def load_portfolio_data(
    portfolio: Portfolio,
    data_source: DataSourceBase,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    Load price data for all positions in a portfolio.

    Args:
        portfolio: Portfolio object
        data_source: Data source to use
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval

    Returns:
        Dictionary mapping ticker to OHLCV DataFrame
    """
    data = {}
    for ticker in portfolio.tickers:
        try:
            df = data_source.get_price_history(ticker, start_date, end_date, interval)
            if not df.empty:
                data[ticker] = df

                # Update position with metadata
                if portfolio.positions[ticker].metadata is None:
                    portfolio.positions[ticker].metadata = data_source.get_metadata(ticker)

                # Calculate returns
                returns = calculate_returns(df["Close"])
                portfolio.positions[ticker].returns = returns

                # Get current price
                portfolio.positions[ticker].current_price = float(df["Close"].iloc[-1])

        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")

    return data


def calculate_portfolio_returns(
    portfolio: Portfolio,
    price_data: Dict[str, pd.DataFrame]
) -> pd.Series:
    """
    Calculate weighted portfolio returns.

    Args:
        portfolio: Portfolio object
        price_data: Dictionary of price DataFrames

    Returns:
        Series of portfolio returns
    """
    weights = portfolio.get_weights()
    returns_dict = {}

    for ticker, df in price_data.items():
        if ticker in weights:
            returns_dict[ticker] = calculate_returns(df["Close"])

    if not returns_dict:
        return pd.Series()

    # Align all returns to common index
    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna(how="all")

    # Calculate weighted returns
    portfolio_returns = pd.Series(0.0, index=returns_df.index)
    for ticker, weight in weights.items():
        if ticker in returns_df.columns:
            portfolio_returns += weight * returns_df[ticker].fillna(0)

    return portfolio_returns


def calculate_portfolio_equity_curve(
    portfolio: Portfolio,
    price_data: Dict[str, pd.DataFrame],
    initial_value: Optional[float] = None
) -> pd.Series:
    """
    Calculate portfolio equity curve from returns.

    Args:
        portfolio: Portfolio object
        price_data: Dictionary of price DataFrames
        initial_value: Starting portfolio value

    Returns:
        Series of portfolio values over time
    """
    if initial_value is None:
        initial_value = portfolio.initial_value

    returns = calculate_portfolio_returns(portfolio, price_data)
    if returns.empty:
        return pd.Series()

    # Convert returns to equity curve
    equity = (1 + returns).cumprod() * initial_value

    # Insert initial value at the beginning
    equity = pd.concat([
        pd.Series([initial_value], index=[returns.index[0] - pd.Timedelta(days=1)]),
        equity
    ])

    return equity


def calculate_portfolio_metrics(
    portfolio: Portfolio,
    price_data: Dict[str, pd.DataFrame],
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """
    Calculate all performance metrics for a portfolio.

    Args:
        portfolio: Portfolio object
        price_data: Dictionary of price DataFrames
        risk_free_rate: Annualized risk-free rate

    Returns:
        PerformanceMetrics object
    """
    equity = calculate_portfolio_equity_curve(portfolio, price_data)
    if equity.empty:
        return PerformanceMetrics(
            total_return=0, annualized_return=0, annualized_volatility=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0, calmar_ratio=0,
            win_rate=0, profit_factor=0, best_day=0, worst_day=0,
            avg_daily_return=0, skewness=0, kurtosis=0,
            num_positive_days=0, num_negative_days=0
        )

    return calculate_all_metrics(equity, risk_free_rate)


def get_allocation_by_category(
    portfolio: Portfolio,
    category: str = "sector"
) -> Dict[str, float]:
    """
    Get portfolio allocation by category.

    Args:
        portfolio: Portfolio object
        category: Category to group by ("sector", "country", "asset_class")

    Returns:
        Dictionary mapping category value to weight
    """
    allocation = {}

    for pos in portfolio.positions.values():
        if pos.metadata:
            if category == "sector":
                key = pos.metadata.sector
            elif category == "country":
                key = pos.metadata.country
            elif category == "asset_class":
                key = pos.metadata.asset_class
            else:
                key = "Unknown"
        else:
            key = "Unknown"

        allocation[key] = allocation.get(key, 0) + pos.weight

    return allocation


def portfolio_correlation_matrix(
    portfolio: Portfolio,
    price_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Calculate correlation matrix for portfolio assets.

    Args:
        portfolio: Portfolio object
        price_data: Dictionary of price DataFrames

    Returns:
        Correlation matrix DataFrame
    """
    returns_dict = {}
    for ticker, df in price_data.items():
        if ticker in portfolio.tickers:
            returns_dict[ticker] = calculate_returns(df["Close"])

    return correlation_matrix(returns_dict)


def portfolio_risk_metrics(
    portfolio: Portfolio,
    price_data: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    """
    Calculate risk metrics for a portfolio.

    Args:
        portfolio: Portfolio object
        price_data: Dictionary of price DataFrames

    Returns:
        Dictionary with risk metrics
    """
    returns = calculate_portfolio_returns(portfolio, price_data)
    returns = returns.dropna()

    if len(returns) == 0:
        return {
            "var_95": 0.0, "var_99": 0.0,
            "cvar_95": 0.0, "cvar_99": 0.0,
            "annualized_vol": 0.0,
        }

    return {
        "var_95": historical_var(returns, 0.95),
        "var_99": historical_var(returns, 0.99),
        "cvar_95": historical_cvar(returns, 0.95),
        "cvar_99": historical_cvar(returns, 0.99),
        "annualized_vol": annualized_volatility(returns),
    }


def compare_to_benchmark(
    portfolio: Portfolio,
    price_data: Dict[str, pd.DataFrame],
    benchmark_data: pd.DataFrame
) -> Dict[str, any]:
    """
    Compare portfolio performance to a benchmark.

    Args:
        portfolio: Portfolio object
        price_data: Dictionary of price DataFrames
        benchmark_data: OHLCV DataFrame for benchmark

    Returns:
        Dictionary with comparison metrics
    """
    portfolio_equity = calculate_portfolio_equity_curve(portfolio, price_data)
    benchmark_returns = calculate_returns(benchmark_data["Close"])

    if portfolio_equity.empty or benchmark_returns.empty:
        return {}

    portfolio_returns = calculate_returns(portfolio_equity)

    # Calculate tracking error
    combined = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(combined) < 2:
        return {}

    combined.columns = ["portfolio", "benchmark"]
    tracking_diff = combined["portfolio"] - combined["benchmark"]
    tracking_error = tracking_diff.std() * np.sqrt(252)

    # Information ratio
    active_return = (combined["portfolio"].mean() - combined["benchmark"].mean()) * 252
    info_ratio = active_return / tracking_error if tracking_error > 0 else 0

    # Beta
    cov = combined["portfolio"].cov(combined["benchmark"])
    var = combined["benchmark"].var()
    beta = cov / var if var > 0 else 1.0

    # Alpha
    alpha = active_return - beta * (combined["benchmark"].mean() * 252)

    return {
        "tracking_error": tracking_error,
        "information_ratio": info_ratio,
        "beta": beta,
        "alpha": alpha,
        "correlation": combined["portfolio"].corr(combined["benchmark"]),
    }


def optimize_weights_minvar(
    portfolio: Portfolio,
    price_data: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    """
    Simple minimum variance optimization (equal risk contribution approximation).

    Args:
        portfolio: Portfolio object
        price_data: Dictionary of price DataFrames

    Returns:
        Dictionary of optimized weights
    """
    returns_dict = {}
    for ticker, df in price_data.items():
        if ticker in portfolio.tickers:
            returns_dict[ticker] = calculate_returns(df["Close"])

    if not returns_dict:
        return portfolio.get_weights()

    # Calculate volatilities
    vols = {}
    for ticker, returns in returns_dict.items():
        vol = returns.std() * np.sqrt(252)
        vols[ticker] = vol if vol > 0 else 0.0001

    # Inverse volatility weighting
    total_inv_vol = sum(1 / v for v in vols.values())

    weights = {}
    for ticker, vol in vols.items():
        weights[ticker] = (1 / vol) / total_inv_vol

    return weights
