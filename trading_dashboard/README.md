# Trading and Portfolio Analytics Dashboard

A comprehensive Streamlit-based dashboard for financial market analysis, portfolio management, and strategy backtesting. This tool is designed for **educational and analytical purposes only** - it does not provide investment or trading advice.

## Features

### Market Analysis
- **Candlestick Charts** with volume bars
- **Technical Indicators**: SMA, EMA, RSI (14), MACD (12, 26, 9), Bollinger Bands, ATR
- Interactive charts with Plotly

### Portfolio Management
- Create custom portfolios with multiple assets
- Portfolio equity curve visualization
- Allocation breakdown by sector, country, and asset class
- Performance attribution analysis
- Equal-weight and minimum variance optimization

### Risk Analytics
- **Value at Risk (VaR)** - Historical simulation at 95% and 99% confidence
- **Conditional VaR (CVaR)** - Expected shortfall
- Rolling volatility (20-day, 60-day windows)
- Correlation matrix heatmaps
- Stress testing scenarios

### Backtesting
- **Buy and Hold** baseline strategy
- **Moving Average Crossover** strategy (configurable fast/slow periods)
- **RSI Mean Reversion** strategy (configurable oversold/overbought levels)
- **MACD Crossover** strategy
- Transaction costs and slippage modeling
- Comprehensive trade statistics

### Alerts & Watchlists
- Price alerts (above/below threshold)
- Daily change alerts (percentage)
- RSI alerts (overbought/oversold)
- Multiple watchlists with notes
- Alert history tracking

### Reports
- PDF report generation for portfolios
- Backtest report generation
- CSV data export
- Chart image export

## Quick Start

```bash
# 1. Navigate to the project directory
cd trading_dashboard

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On macOS/Linux:
source .venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Run the application
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## Installation Details

### Prerequisites
- Python 3.10 or higher

### What Gets Installed

The `requirements.txt` includes all dependencies:
- **Core**: streamlit, pandas, numpy, plotly, scipy, reportlab
- **Data Sources**: yfinance (stocks), ccxt (crypto)
- **PDF Export**: kaleido (chart images)
- **Testing**: pytest, pytest-cov

### Troubleshooting

**PowerShell execution policy error on Windows:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**If `python` command not found, try:**
```bash
py -m venv .venv
```

**The app works without yfinance/ccxt** - it will use demo data automatically.

## Project Structure

```
trading_dashboard/
├── app.py                      # Main entry point
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── pages/                      # Streamlit pages
│   ├── 1_Overview.py          # Market overview
│   ├── 2_Asset_Explorer.py    # Individual asset analysis
│   ├── 3_Portfolio_Builder.py # Portfolio construction
│   ├── 4_Risk_and_Correlation.py # Risk analytics
│   ├── 5_Backtesting.py       # Strategy backtesting
│   ├── 6_Alerts_and_Watchlist.py # Alerts management
│   ├── 7_Reports.py           # PDF report generation
│   └── 8_Settings.py          # Configuration
├── src/                        # Source modules
│   ├── datasource/            # Data source implementations
│   │   ├── base.py            # Abstract base class
│   │   ├── demo.py            # Demo data generator
│   │   ├── yfinance_source.py # Yahoo Finance
│   │   └── ccxt_source.py     # Cryptocurrency
│   ├── analytics/             # Analytics calculations
│   │   ├── indicators.py      # Technical indicators
│   │   ├── risk.py            # Risk metrics
│   │   └── performance.py     # Performance metrics
│   ├── portfolio/             # Portfolio management
│   │   └── portfolio.py
│   ├── backtest/              # Backtesting engine
│   │   └── engine.py
│   ├── alerts/                # Alert system
│   │   └── store.py
│   ├── reports/               # PDF generation
│   │   └── pdf.py
│   └── utils/                 # Utilities
│       ├── cache_keys.py
│       └── formatting.py
├── data/                       # Local data storage
│   ├── alerts.json            # Saved alerts
│   └── watchlists.json        # Saved watchlists
└── tests/                      # Unit tests
    ├── test_indicators.py
    ├── test_risk.py
    └── test_performance.py
```

## Data Sources

### Demo Data (Default)
- Always available, no API keys required
- Generates realistic synthetic OHLCV data
- Includes 15 demo assets across different sectors
- Perfect for testing and learning

### Yahoo Finance (Optional)
- Real stock market data
- Requires `yfinance` package
- No API key needed for basic usage
- Supports stocks, ETFs, indices, and crypto

### CCXT (Optional)
- Cryptocurrency data from major exchanges
- Requires `ccxt` package
- Supports Binance, Coinbase, Kraken, etc.
- Public data only (no trading)

## Configuration

Settings can be configured via the Settings page:
- Data source selection
- Risk-free rate for Sharpe ratio calculations
- Cache management
- Date range defaults

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_indicators.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Technical Details

### Performance Metrics
- **Total Return**: (End Value / Start Value) - 1
- **Annualized Return (CAGR)**: Compound Annual Growth Rate
- **Sharpe Ratio**: (Return - Risk-Free Rate) / Volatility (annualized)
- **Sortino Ratio**: Uses downside deviation instead of total volatility
- **Max Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: CAGR / |Max Drawdown|

### Risk Metrics
- **VaR (Historical)**: Percentile of returns distribution
- **CVaR (Expected Shortfall)**: Average of returns below VaR threshold
- **Volatility**: Annualized standard deviation (×√252)
- **Correlation**: Pearson correlation of daily returns

### Technical Indicators
- **SMA**: Simple Moving Average
- **EMA**: Exponential Moving Average
- **RSI**: Relative Strength Index (Wilder smoothing)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: SMA ± 2×standard deviation

## Disclaimer

**This application is for educational and analytical purposes only.**

- It does not constitute financial, investment, or trading advice
- Past performance does not guarantee future results
- Demo data is synthetic and does not represent real market conditions
- Always consult with a qualified financial advisor before making investment decisions

## License

This project is provided for educational purposes. Use at your own risk.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- All tests pass
- New features include appropriate tests
- Documentation is updated

## Support

For issues and feature requests, please open an issue on the project repository.
