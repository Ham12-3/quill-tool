"""
Report generation module.
"""
from .pdf import (
    PortfolioReportGenerator, generate_backtest_report, chart_to_image,
    format_percent, format_currency, format_number
)

__all__ = [
    "PortfolioReportGenerator", "generate_backtest_report", "chart_to_image",
    "format_percent", "format_currency", "format_number"
]
