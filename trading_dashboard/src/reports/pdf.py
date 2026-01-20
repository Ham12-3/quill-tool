"""
PDF report generation for portfolio analytics.
"""
import io
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from ..analytics.performance import PerformanceMetrics


def format_percent(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a number as currency."""
    return f"${value:,.{decimals}f}"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with specified decimals."""
    return f"{value:,.{decimals}f}"


class PortfolioReportGenerator:
    """Generate PDF reports for portfolio analysis."""

    def __init__(
        self,
        title: str = "Portfolio Analytics Report",
        pagesize=letter
    ):
        self.title = title
        self.pagesize = pagesize
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        """Setup custom styles."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
        ))

        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#1f77b4'),
        ))

        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.gray,
            alignment=TA_CENTER,
            spaceAfter=20,
        ))

        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            spaceBefore=20,
        ))

    def generate_report(
        self,
        portfolio_name: str,
        positions: Dict[str, float],
        metrics: PerformanceMetrics,
        risk_metrics: Dict[str, float],
        start_date: str,
        end_date: str,
        chart_images: Dict[str, bytes] = None,
        additional_info: Dict[str, Any] = None
    ) -> bytes:
        """
        Generate a PDF report.

        Args:
            portfolio_name: Name of the portfolio
            positions: Dictionary of ticker to weight
            metrics: PerformanceMetrics object
            risk_metrics: Dictionary with VaR, CVaR, etc.
            start_date: Report start date
            end_date: Report end date
            chart_images: Dictionary of chart name to PNG bytes
            additional_info: Any additional information to include

        Returns:
            PDF as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self.pagesize,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        elements = []

        # Title
        elements.append(Paragraph(self.title, self.styles['ReportTitle']))
        elements.append(Paragraph(
            f"Portfolio: {portfolio_name}<br/>Period: {start_date} to {end_date}<br/>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            self.styles['Subtitle']
        ))

        # Summary Section
        elements.append(Paragraph("Executive Summary", self.styles['SectionTitle']))
        summary_data = [
            ["Metric", "Value"],
            ["Total Return", format_percent(metrics.total_return)],
            ["Annualized Return (CAGR)", format_percent(metrics.annualized_return)],
            ["Annualized Volatility", format_percent(metrics.annualized_volatility)],
            ["Sharpe Ratio", format_number(metrics.sharpe_ratio)],
            ["Maximum Drawdown", format_percent(metrics.max_drawdown)],
            ["Calmar Ratio", format_number(metrics.calmar_ratio)],
        ]
        elements.append(self._create_table(summary_data))
        elements.append(Spacer(1, 20))

        # Portfolio Allocation
        elements.append(Paragraph("Portfolio Allocation", self.styles['SectionTitle']))
        allocation_data = [["Ticker", "Weight"]]
        for ticker, weight in sorted(positions.items(), key=lambda x: -x[1]):
            allocation_data.append([ticker, format_percent(weight)])
        elements.append(self._create_table(allocation_data))
        elements.append(Spacer(1, 20))

        # Risk Metrics
        elements.append(Paragraph("Risk Metrics", self.styles['SectionTitle']))
        risk_data = [
            ["Metric", "Value"],
            ["Value at Risk (95%)", format_percent(risk_metrics.get('var_95', 0))],
            ["Value at Risk (99%)", format_percent(risk_metrics.get('var_99', 0))],
            ["Conditional VaR (95%)", format_percent(risk_metrics.get('cvar_95', 0))],
            ["Conditional VaR (99%)", format_percent(risk_metrics.get('cvar_99', 0))],
        ]
        elements.append(self._create_table(risk_data))
        elements.append(Spacer(1, 20))

        # Performance Statistics
        elements.append(Paragraph("Performance Statistics", self.styles['SectionTitle']))
        perf_data = [
            ["Metric", "Value"],
            ["Best Day", format_percent(metrics.best_day)],
            ["Worst Day", format_percent(metrics.worst_day)],
            ["Average Daily Return", format_percent(metrics.avg_daily_return)],
            ["Win Rate", format_percent(metrics.win_rate)],
            ["Positive Days", str(metrics.num_positive_days)],
            ["Negative Days", str(metrics.num_negative_days)],
            ["Skewness", format_number(metrics.skewness)],
            ["Kurtosis", format_number(metrics.kurtosis)],
        ]
        elements.append(self._create_table(perf_data))

        # Charts (if provided)
        if chart_images:
            elements.append(PageBreak())
            elements.append(Paragraph("Charts", self.styles['SectionTitle']))

            for name, img_bytes in chart_images.items():
                try:
                    img_buffer = io.BytesIO(img_bytes)
                    img = Image(img_buffer, width=6.5 * inch, height=4 * inch)
                    elements.append(Paragraph(name, self.styles['Heading3']))
                    elements.append(img)
                    elements.append(Spacer(1, 20))
                except Exception as e:
                    elements.append(Paragraph(
                        f"Could not load chart: {name}",
                        self.styles['Normal']
                    ))

        # Disclaimer
        elements.append(Spacer(1, 30))
        elements.append(Paragraph(
            "DISCLAIMER: This report is for educational and informational purposes only. "
            "It does not constitute financial, investment, or trading advice. "
            "Past performance does not guarantee future results. "
            "Always consult with a qualified financial advisor before making investment decisions.",
            self.styles['Disclaimer']
        ))

        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    def _create_table(self, data: List[List[str]], col_widths: List[float] = None) -> Table:
        """Create a styled table."""
        if col_widths is None:
            col_widths = [2.5 * inch, 2 * inch]

        table = Table(data, colWidths=col_widths)

        style = TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f7f7')]),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ])

        table.setStyle(style)
        return table


def generate_backtest_report(
    strategy_name: str,
    ticker: str,
    metrics: PerformanceMetrics,
    trade_stats: Dict[str, float],
    start_date: str,
    end_date: str,
    initial_capital: float,
    final_value: float,
    chart_images: Dict[str, bytes] = None
) -> bytes:
    """
    Generate a PDF report for a backtest.

    Args:
        strategy_name: Name of the strategy
        ticker: Ticker symbol
        metrics: PerformanceMetrics from backtest
        trade_stats: Trade statistics dictionary
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        final_value: Final portfolio value
        chart_images: Dictionary of chart images

    Returns:
        PDF as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    # Title
    elements.append(Paragraph(
        f"Backtest Report: {strategy_name}",
        styles['Heading1']
    ))
    elements.append(Paragraph(
        f"Ticker: {ticker} | Period: {start_date} to {end_date}",
        styles['Normal']
    ))
    elements.append(Spacer(1, 20))

    # Summary
    elements.append(Paragraph("Performance Summary", styles['Heading2']))

    summary_data = [
        ["Metric", "Value"],
        ["Initial Capital", format_currency(initial_capital)],
        ["Final Value", format_currency(final_value)],
        ["Total Return", format_percent(metrics.total_return)],
        ["Annualized Return", format_percent(metrics.annualized_return)],
        ["Annualized Volatility", format_percent(metrics.annualized_volatility)],
        ["Sharpe Ratio", format_number(metrics.sharpe_ratio)],
        ["Maximum Drawdown", format_percent(metrics.max_drawdown)],
        ["Calmar Ratio", format_number(metrics.calmar_ratio)],
    ]

    table = Table(summary_data, colWidths=[2.5 * inch, 2 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2196F3')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # Trade Statistics
    elements.append(Paragraph("Trade Statistics", styles['Heading2']))

    trade_data = [
        ["Metric", "Value"],
        ["Total Trades", str(int(trade_stats.get('total_trades', 0)))],
        ["Winning Trades", str(int(trade_stats.get('winning_trades', 0)))],
        ["Losing Trades", str(int(trade_stats.get('losing_trades', 0)))],
        ["Win Rate", format_percent(trade_stats.get('win_rate', 0))],
        ["Average P&L", format_currency(trade_stats.get('avg_pnl', 0))],
        ["Average Win", format_currency(trade_stats.get('avg_win', 0))],
        ["Average Loss", format_currency(trade_stats.get('avg_loss', 0))],
        ["Profit Factor", format_number(trade_stats.get('profit_factor', 0))],
        ["Avg Trade Duration (days)", format_number(trade_stats.get('avg_trade_duration', 0), 1)],
    ]

    trade_table = Table(trade_data, colWidths=[2.5 * inch, 2 * inch])
    trade_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    elements.append(trade_table)

    # Charts
    if chart_images:
        elements.append(PageBreak())
        elements.append(Paragraph("Charts", styles['Heading2']))

        for name, img_bytes in chart_images.items():
            try:
                img_buffer = io.BytesIO(img_bytes)
                img = Image(img_buffer, width=6 * inch, height=3.5 * inch)
                elements.append(Paragraph(name, styles['Heading3']))
                elements.append(img)
                elements.append(Spacer(1, 15))
            except Exception:
                pass

    # Disclaimer
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(
        "DISCLAIMER: This backtest is for educational purposes only. "
        "Past performance does not guarantee future results. "
        "Backtests may not account for all real-world trading costs and market conditions.",
        ParagraphStyle('Disclaimer', fontSize=8, textColor=colors.gray)
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def chart_to_image(fig, width: int = 800, height: int = 500) -> bytes:
    """
    Convert a Plotly figure to PNG bytes.

    Args:
        fig: Plotly figure object
        width: Image width
        height: Image height

    Returns:
        PNG image as bytes
    """
    try:
        return fig.to_image(format="png", width=width, height=height)
    except Exception as e:
        # If kaleido is not available, try alternative
        print(f"Warning: Could not export chart to image: {e}")
        return None
