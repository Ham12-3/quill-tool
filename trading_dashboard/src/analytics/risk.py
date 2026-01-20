"""
Risk analytics calculations.
Includes VaR, CVaR, volatility, correlation, and drawdown calculations.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats


# Trading days per year for annualization
TRADING_DAYS_PER_YEAR = 252


def calculate_returns(
    prices: pd.Series,
    method: str = "simple"
) -> pd.Series:
    """
    Calculate returns from price series.

    Args:
        prices: Series of prices
        method: "simple" for arithmetic returns, "log" for logarithmic returns

    Returns:
        Series of returns
    """
    if method == "log":
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()


def historical_var(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Historical Value at Risk.

    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        VaR as a positive number (loss threshold)
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    # VaR is the (1-confidence) percentile of returns
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return -var  # Return as positive number


def historical_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Historical Conditional Value at Risk (Expected Shortfall).

    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        CVaR as a positive number (expected loss beyond VaR)
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    # Get the VaR threshold
    var_threshold = np.percentile(returns, (1 - confidence_level) * 100)

    # CVaR is the mean of returns below VaR threshold
    tail_returns = returns[returns <= var_threshold]
    if len(tail_returns) == 0:
        return -var_threshold

    cvar = tail_returns.mean()
    return -cvar  # Return as positive number


def calculate_var_cvar(
    returns: pd.Series
) -> Dict[str, float]:
    """
    Calculate VaR and CVaR at both 95% and 99% confidence levels.

    Args:
        returns: Series of returns

    Returns:
        Dictionary with VaR and CVaR values
    """
    return {
        "var_95": historical_var(returns, 0.95),
        "var_99": historical_var(returns, 0.99),
        "cvar_95": historical_cvar(returns, 0.95),
        "cvar_99": historical_cvar(returns, 0.99),
    }


def rolling_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling volatility.

    Args:
        returns: Series of returns
        window: Rolling window size (in trading days)
        annualize: Whether to annualize the volatility

    Returns:
        Series of rolling volatility values
    """
    vol = returns.rolling(window=window, min_periods=window // 2).std()

    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)

    return vol


def annualized_volatility(returns: pd.Series) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Series of returns

    Returns:
        Annualized volatility as a decimal
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def correlation_matrix(
    returns_dict: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple assets.

    Args:
        returns_dict: Dictionary mapping ticker to returns series

    Returns:
        DataFrame with correlation matrix
    """
    # Combine all returns into a DataFrame
    df = pd.DataFrame(returns_dict)

    # Calculate correlation
    return df.corr()


def covariance_matrix(
    returns_dict: Dict[str, pd.Series],
    annualize: bool = True
) -> pd.DataFrame:
    """
    Calculate covariance matrix for multiple assets.

    Args:
        returns_dict: Dictionary mapping ticker to returns series
        annualize: Whether to annualize the covariances

    Returns:
        DataFrame with covariance matrix
    """
    df = pd.DataFrame(returns_dict)
    cov = df.cov()

    if annualize:
        cov = cov * TRADING_DAYS_PER_YEAR

    return cov


def calculate_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series
) -> float:
    """
    Calculate beta of an asset relative to the market.

    Args:
        asset_returns: Series of asset returns
        market_returns: Series of market returns

    Returns:
        Beta coefficient
    """
    # Align the series
    combined = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(combined) < 2:
        return 1.0

    asset_ret = combined.iloc[:, 0]
    market_ret = combined.iloc[:, 1]

    covariance = asset_ret.cov(market_ret)
    market_variance = market_ret.var()

    if market_variance == 0:
        return 1.0

    return covariance / market_variance


def calculate_alpha(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Jensen's alpha.

    Args:
        asset_returns: Series of asset returns
        market_returns: Series of market returns
        risk_free_rate: Annualized risk-free rate

    Returns:
        Annualized alpha
    """
    # Convert annual risk-free to daily
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    beta = calculate_beta(asset_returns, market_returns)

    asset_mean = asset_returns.mean()
    market_mean = market_returns.mean()

    # Daily alpha
    daily_alpha = (asset_mean - daily_rf) - beta * (market_mean - daily_rf)

    # Annualize
    return daily_alpha * TRADING_DAYS_PER_YEAR


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino ratio (using downside deviation).

    Args:
        returns: Series of returns
        risk_free_rate: Annualized risk-free rate
        target_return: Target return for downside calculation

    Returns:
        Sortino ratio
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    # Calculate excess returns
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = returns - daily_rf

    # Calculate downside deviation
    downside_returns = excess_returns[excess_returns < target_return]
    if len(downside_returns) == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0

    downside_std = np.sqrt(np.mean(downside_returns ** 2))

    if downside_std == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0

    # Annualize
    annualized_excess = excess_returns.mean() * TRADING_DAYS_PER_YEAR
    annualized_downside = downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)

    return annualized_excess / annualized_downside


def risk_parity_weights(
    returns_dict: Dict[str, pd.Series]
) -> Dict[str, float]:
    """
    Calculate risk parity weights (equal risk contribution).

    Args:
        returns_dict: Dictionary mapping ticker to returns series

    Returns:
        Dictionary mapping ticker to weight
    """
    tickers = list(returns_dict.keys())
    if not tickers:
        return {}

    # Calculate volatilities
    volatilities = {}
    for ticker, returns in returns_dict.items():
        vol = annualized_volatility(returns)
        volatilities[ticker] = vol if vol > 0 else 0.0001  # Avoid division by zero

    # Inverse volatility weighting
    total_inv_vol = sum(1 / v for v in volatilities.values())

    weights = {}
    for ticker, vol in volatilities.items():
        weights[ticker] = (1 / vol) / total_inv_vol

    return weights


def portfolio_var(
    weights: Dict[str, float],
    returns_dict: Dict[str, pd.Series],
    confidence_level: float = 0.95
) -> float:
    """
    Calculate portfolio VaR.

    Args:
        weights: Dictionary mapping ticker to weight
        returns_dict: Dictionary mapping ticker to returns series
        confidence_level: Confidence level

    Returns:
        Portfolio VaR
    """
    # Calculate portfolio returns
    portfolio_returns = None
    for ticker, weight in weights.items():
        if ticker in returns_dict:
            if portfolio_returns is None:
                portfolio_returns = weight * returns_dict[ticker]
            else:
                portfolio_returns = portfolio_returns + weight * returns_dict[ticker]

    if portfolio_returns is None:
        return 0.0

    return historical_var(portfolio_returns.dropna(), confidence_level)


def stress_test(
    returns: pd.Series,
    scenarios: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Perform simple stress testing.

    Args:
        returns: Series of returns
        scenarios: Dictionary of scenario names to shock percentages

    Returns:
        Dictionary of scenario names to stressed values
    """
    if scenarios is None:
        scenarios = {
            "2008 Financial Crisis": -0.50,
            "COVID Crash (Mar 2020)": -0.34,
            "10% Market Decline": -0.10,
            "20% Market Decline": -0.20,
            "Black Monday (1987)": -0.22,
        }

    current_value = 1.0  # Normalized
    results = {}

    for name, shock in scenarios.items():
        results[name] = current_value * (1 + shock)

    return results
