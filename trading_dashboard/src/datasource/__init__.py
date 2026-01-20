"""
Data source module for market data.
"""
from .base import DataSourceBase, AssetMetadata
from .demo import DemoDataSource, DEMO_ASSETS
from .yfinance_source import YFinanceDataSource, YFINANCE_AVAILABLE
from .ccxt_source import CCXTDataSource, CCXT_AVAILABLE

__all__ = [
    "DataSourceBase",
    "AssetMetadata",
    "DemoDataSource",
    "DEMO_ASSETS",
    "YFinanceDataSource",
    "YFINANCE_AVAILABLE",
    "CCXTDataSource",
    "CCXT_AVAILABLE",
]
