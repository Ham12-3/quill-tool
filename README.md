# Trading & Portfolio Analytics Dashboard

A Streamlit-based dashboard for analysing financial markets and portfolios. Built for educational and analytical purposes.

## Features

- **Market Overview** - Key metrics and market summary
- **Asset Explorer** - Detailed analysis with charts and technical indicators
- **Portfolio Builder** - Create and analyse custom portfolios
- **Risk & Correlation** - VaR, CVaR, and correlation analysis
- **Backtesting** - Test trading strategies on historical data
- **Alerts & Watchlist** - Set price alerts and manage watchlists
- **PDF Reports** - Generate reports for portfolios and backtests

## Data Sources

- **Demo Data** - Built-in simulated data (always available)
- **Yahoo Finance** - Real market data via yfinance
- **CCXT** - Cryptocurrency data from multiple exchanges

## Installation

```bash
cd trading_dashboard
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Project Structure

```
trading_dashboard/
├── app.py                 # Main entry point
├── pages/                 # Streamlit pages
├── src/
│   ├── analytics/         # Technical indicators and metrics
│   ├── backtest/          # Backtesting engine
│   ├── datasource/        # Data source implementations
│   ├── portfolio/         # Portfolio management
│   ├── alerts/            # Alert system
│   └── reports/           # PDF report generation
└── tests/                 # Unit tests
```

## Running Tests

```bash
pytest
```

## Disclaimer

This dashboard is for educational and analytical purposes only. It does not constitute financial advice.
