"""
Analytics module for technical indicators, risk metrics, and performance calculations.
"""
from .indicators import (
    sma, ema, rsi, macd, bollinger_bands, atr, stochastic, obv, vwap,
    add_all_indicators
)
from .risk import (
    calculate_returns, historical_var, historical_cvar, calculate_var_cvar,
    rolling_volatility, annualized_volatility, correlation_matrix, covariance_matrix,
    calculate_beta, calculate_alpha, calculate_sortino_ratio, risk_parity_weights,
    portfolio_var, stress_test, TRADING_DAYS_PER_YEAR
)
from .performance import (
    PerformanceMetrics, total_return, annualized_return, sharpe_ratio,
    drawdown, max_drawdown, drawdown_duration, calmar_ratio,
    calculate_all_metrics, performance_attribution, time_period_attribution,
    rolling_sharpe, underwater_chart
)

__all__ = [
    # Indicators
    "sma", "ema", "rsi", "macd", "bollinger_bands", "atr", "stochastic",
    "obv", "vwap", "add_all_indicators",
    # Risk
    "calculate_returns", "historical_var", "historical_cvar", "calculate_var_cvar",
    "rolling_volatility", "annualized_volatility", "correlation_matrix", "covariance_matrix",
    "calculate_beta", "calculate_alpha", "calculate_sortino_ratio", "risk_parity_weights",
    "portfolio_var", "stress_test", "TRADING_DAYS_PER_YEAR",
    # Performance
    "PerformanceMetrics", "total_return", "annualized_return", "sharpe_ratio",
    "drawdown", "max_drawdown", "drawdown_duration", "calmar_ratio",
    "calculate_all_metrics", "performance_attribution", "time_period_attribution",
    "rolling_sharpe", "underwater_chart",
]
