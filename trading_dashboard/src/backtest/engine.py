"""
Backtesting engine for trading strategies.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from ..analytics.indicators import sma, ema, rsi, macd
from ..analytics.performance import (
    calculate_all_metrics, PerformanceMetrics,
    drawdown, max_drawdown
)
from ..analytics.risk import calculate_returns


class SignalType(Enum):
    """Trading signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    direction: str  # "long" or "short"
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

    def close(self, exit_date: pd.Timestamp, exit_price: float) -> None:
        """Close the trade."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        if self.direction == "long":
            self.pnl = (exit_price - self.entry_price) * self.position_size
            self.pnl_pct = (exit_price / self.entry_price) - 1
        else:
            self.pnl = (self.entry_price - exit_price) * self.position_size
            self.pnl_pct = (self.entry_price / exit_price) - 1


@dataclass
class BacktestResult:
    """Results from a backtest."""
    strategy_name: str
    ticker: str
    start_date: str
    end_date: str
    initial_cash: float
    final_value: float
    equity_curve: pd.Series
    trades: List[Trade]
    signals: pd.Series
    metrics: PerformanceMetrics
    buy_signals: pd.DataFrame
    sell_signals: pd.DataFrame
    transaction_cost: float
    slippage: float


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from OHLCV data.

        Args:
            df: OHLCV DataFrame

        Returns:
            Series with signals (1 for buy, -1 for sell, 0 for hold)
        """
        pass


class BuyAndHoldStrategy(Strategy):
    """Simple buy and hold strategy."""

    @property
    def name(self) -> str:
        return "Buy and Hold"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals.iloc[0] = 1  # Buy on first day
        return signals


class MovingAverageCrossoverStrategy(Strategy):
    """Moving average crossover strategy."""

    def __init__(self, fast_period: int = 10, slow_period: int = 30, use_ema: bool = False):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_ema = use_ema

    @property
    def name(self) -> str:
        ma_type = "EMA" if self.use_ema else "SMA"
        return f"{ma_type} Crossover ({self.fast_period}/{self.slow_period})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        prices = df["Close"]

        if self.use_ema:
            fast_ma = ema(prices, self.fast_period)
            slow_ma = ema(prices, self.slow_period)
        else:
            fast_ma = sma(prices, self.fast_period)
            slow_ma = sma(prices, self.slow_period)

        # Generate signals on crossover
        signals = pd.Series(0, index=df.index)

        # Fast crosses above slow = buy
        # Fast crosses below slow = sell
        for i in range(1, len(df)):
            if fast_ma.iloc[i] > slow_ma.iloc[i] and fast_ma.iloc[i-1] <= slow_ma.iloc[i-1]:
                signals.iloc[i] = 1  # Buy signal
            elif fast_ma.iloc[i] < slow_ma.iloc[i] and fast_ma.iloc[i-1] >= slow_ma.iloc[i-1]:
                signals.iloc[i] = -1  # Sell signal

        return signals


class RSIMeanReversionStrategy(Strategy):
    """RSI mean reversion strategy."""

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70
    ):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return f"RSI Mean Reversion ({self.oversold}/{self.overbought})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        prices = df["Close"]
        rsi_values = rsi(prices, self.period)

        signals = pd.Series(0, index=df.index)

        for i in range(1, len(df)):
            if rsi_values.iloc[i] < self.oversold and rsi_values.iloc[i-1] >= self.oversold:
                signals.iloc[i] = 1  # Buy when RSI crosses below oversold
            elif rsi_values.iloc[i] > self.overbought and rsi_values.iloc[i-1] <= self.overbought:
                signals.iloc[i] = -1  # Sell when RSI crosses above overbought

        return signals


class MACDStrategy(Strategy):
    """MACD crossover strategy."""

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    @property
    def name(self) -> str:
        return f"MACD ({self.fast_period}/{self.slow_period}/{self.signal_period})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        prices = df["Close"]
        macd_line, signal_line, histogram = macd(
            prices, self.fast_period, self.slow_period, self.signal_period
        )

        signals = pd.Series(0, index=df.index)

        for i in range(1, len(df)):
            if macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i-1] <= signal_line.iloc[i-1]:
                signals.iloc[i] = 1  # Buy on MACD crossover
            elif macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i-1] >= signal_line.iloc[i-1]:
                signals.iloc[i] = -1  # Sell on MACD crossunder

        return signals


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_cash: float = 100000.0,
    transaction_cost: float = 0.001,  # 0.1% per trade
    slippage: float = 0.0005,  # 0.05% slippage
    position_size: float = 1.0,  # Fraction of equity per trade
    allow_short: bool = False
) -> BacktestResult:
    """
    Run a backtest on OHLCV data.

    Args:
        df: OHLCV DataFrame
        strategy: Strategy instance
        initial_cash: Starting capital
        transaction_cost: Transaction cost as decimal (0.001 = 0.1%)
        slippage: Slippage as decimal
        position_size: Fraction of equity to use per trade
        allow_short: Whether to allow short positions

    Returns:
        BacktestResult object
    """
    df = df.copy()
    signals = strategy.generate_signals(df)

    # Initialize tracking variables
    cash = initial_cash
    shares = 0
    equity_values = []
    trades: List[Trade] = []
    current_trade: Optional[Trade] = None

    buy_signals_data = []
    sell_signals_data = []

    for i, (date, row) in enumerate(df.iterrows()):
        price = row["Close"]
        signal = signals.iloc[i]

        # Apply slippage
        buy_price = price * (1 + slippage)
        sell_price = price * (1 - slippage)

        # Process signals
        if signal == 1 and shares == 0:  # Buy signal and no position
            # Calculate shares to buy
            available = cash * position_size
            cost_per_share = buy_price * (1 + transaction_cost)
            shares_to_buy = available // cost_per_share

            if shares_to_buy > 0:
                total_cost = shares_to_buy * buy_price * (1 + transaction_cost)
                cash -= total_cost
                shares = shares_to_buy

                current_trade = Trade(
                    entry_date=date,
                    exit_date=None,
                    entry_price=buy_price,
                    exit_price=None,
                    position_size=shares_to_buy,
                    direction="long"
                )

                buy_signals_data.append({
                    "Date": date,
                    "Price": buy_price,
                    "Shares": shares_to_buy,
                })

        elif signal == -1 and shares > 0:  # Sell signal and have position
            # Sell all shares
            proceeds = shares * sell_price * (1 - transaction_cost)
            cash += proceeds

            if current_trade:
                current_trade.close(date, sell_price)
                trades.append(current_trade)
                current_trade = None

            sell_signals_data.append({
                "Date": date,
                "Price": sell_price,
                "Shares": shares,
            })

            shares = 0

        # Calculate daily equity
        equity = cash + shares * price
        equity_values.append({"Date": date, "Equity": equity})

    # Close any open position at the end
    if shares > 0 and current_trade:
        final_price = df["Close"].iloc[-1] * (1 - slippage)
        proceeds = shares * final_price * (1 - transaction_cost)
        cash += proceeds
        current_trade.close(df.index[-1], final_price)
        trades.append(current_trade)

    # Create equity curve
    equity_df = pd.DataFrame(equity_values)
    equity_df = equity_df.set_index("Date")
    equity_curve = equity_df["Equity"]

    # Calculate metrics
    metrics = calculate_all_metrics(equity_curve)

    # Create signal DataFrames
    buy_signals_df = pd.DataFrame(buy_signals_data) if buy_signals_data else pd.DataFrame()
    sell_signals_df = pd.DataFrame(sell_signals_data) if sell_signals_data else pd.DataFrame()

    return BacktestResult(
        strategy_name=strategy.name,
        ticker=df.attrs.get("ticker", "Unknown"),
        start_date=str(df.index[0].date()) if len(df) > 0 else "",
        end_date=str(df.index[-1].date()) if len(df) > 0 else "",
        initial_cash=initial_cash,
        final_value=equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_cash,
        equity_curve=equity_curve,
        trades=trades,
        signals=signals,
        metrics=metrics,
        buy_signals=buy_signals_df,
        sell_signals=sell_signals_df,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )


def compare_strategies(
    df: pd.DataFrame,
    strategies: List[Strategy],
    initial_cash: float = 100000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005
) -> Dict[str, BacktestResult]:
    """
    Run multiple strategies and compare results.

    Args:
        df: OHLCV DataFrame
        strategies: List of Strategy instances
        initial_cash: Starting capital
        transaction_cost: Transaction cost as decimal
        slippage: Slippage as decimal

    Returns:
        Dictionary mapping strategy name to BacktestResult
    """
    results = {}

    for strategy in strategies:
        result = run_backtest(
            df, strategy, initial_cash, transaction_cost, slippage
        )
        results[strategy.name] = result

    return results


def get_trade_statistics(trades: List[Trade]) -> Dict[str, float]:
    """
    Calculate statistics from a list of trades.

    Args:
        trades: List of Trade objects

    Returns:
        Dictionary with trade statistics
    """
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "avg_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "avg_trade_duration": 0,
        }

    closed_trades = [t for t in trades if t.exit_date is not None]
    if not closed_trades:
        return {"total_trades": len(trades), "closed_trades": 0}

    winning = [t for t in closed_trades if t.pnl and t.pnl > 0]
    losing = [t for t in closed_trades if t.pnl and t.pnl < 0]

    total_wins = sum(t.pnl for t in winning) if winning else 0
    total_losses = abs(sum(t.pnl for t in losing)) if losing else 0

    # Calculate average trade duration
    durations = []
    for t in closed_trades:
        if t.entry_date and t.exit_date:
            duration = (t.exit_date - t.entry_date).days
            durations.append(duration)

    return {
        "total_trades": len(trades),
        "closed_trades": len(closed_trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": len(winning) / len(closed_trades) if closed_trades else 0,
        "avg_pnl": sum(t.pnl for t in closed_trades if t.pnl) / len(closed_trades),
        "avg_win": total_wins / len(winning) if winning else 0,
        "avg_loss": total_losses / len(losing) if losing else 0,
        "profit_factor": total_wins / total_losses if total_losses > 0 else float('inf'),
        "avg_trade_duration": np.mean(durations) if durations else 0,
        "max_consecutive_wins": _max_consecutive(closed_trades, positive=True),
        "max_consecutive_losses": _max_consecutive(closed_trades, positive=False),
    }


def _max_consecutive(trades: List[Trade], positive: bool) -> int:
    """Calculate maximum consecutive wins or losses."""
    max_count = 0
    current_count = 0

    for trade in trades:
        if trade.pnl is None:
            continue

        is_win = trade.pnl > 0
        if is_win == positive:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0

    return max_count
