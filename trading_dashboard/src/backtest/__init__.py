"""
Backtesting module for trading strategies.
"""
from .engine import (
    SignalType, Trade, BacktestResult, Strategy,
    BuyAndHoldStrategy, MovingAverageCrossoverStrategy,
    RSIMeanReversionStrategy, MACDStrategy,
    run_backtest, compare_strategies, get_trade_statistics
)

__all__ = [
    "SignalType", "Trade", "BacktestResult", "Strategy",
    "BuyAndHoldStrategy", "MovingAverageCrossoverStrategy",
    "RSIMeanReversionStrategy", "MACDStrategy",
    "run_backtest", "compare_strategies", "get_trade_statistics"
]
