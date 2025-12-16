# 📈 Trend Terminal

A professional-grade stock scanning, signal generation, and backtesting application with an intuitive Streamlit GUI.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

### 📊 Real-Time Market Dashboard
- Market overview with major indices
- Live scanning across S&P 500 and NASDAQ 100
- Top movers and signal highlights
- AI-powered market summaries

### 🔍 Advanced Stock Screener
- Filter by price, volume, market cap
- Signal direction filtering (bullish/bearish)
- News sentiment integration
- Sortable multi-column results

### 📈 Detailed Stock Analysis
- Interactive candlestick charts with volume
- RSI and technical indicator overlays
- Signal breakdown with confidence scoring
- News feed with sentiment classification
- AI-generated analysis explanations

### 🔬 Backtesting Engine
- No look-ahead bias enforcement
- Multiple exit strategies (fixed days, signal flip)
- Transaction costs and slippage modeling
- Performance metrics (Sharpe, win rate, max drawdown)
- Equity curves and monthly return heatmaps

### 📱 Telegram Notifications
- Real-time signal alerts
- Price and volume threshold alerts
- Customizable notification preferences
- Rate limiting and deduplication

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone or download the repository:**
```bash
cd "Trend Terminal"
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your API keys
```

5. **Run the application:**
```bash
streamlit run app/main.py
```

6. **Open in browser:**
Navigate to `http://localhost:8501`

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI API (for AI summaries)
OPENAI_API_KEY=sk-your-api-key-here

# Telegram Bot (for notifications)
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

### Application Settings

Edit `config/config.yaml` to customize:

```yaml
# Indicator settings
indicators:
  ema_short: 9
  ema_medium: 21
  ema_long: 50
  rsi_period: 14
  volume_ma_period: 20
  
# Signal weights (must sum to 100)
signal_weights:
  technical: 60
  volume: 20
  news: 20
  
# Backtest defaults
backtest:
  initial_capital: 10000
  transaction_cost: 0.001  # 0.1%
  slippage: 0.0005         # 0.05%
```

### Watchlist

Edit `config/watchlist.txt` to add your personal stocks:

```
AAPL
GOOGL
MSFT
NVDA
TSLA
```

---

## 📱 Setting Up Telegram Notifications

### Step 1: Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Start a chat and send `/newbot`
3. Follow the prompts:
   - Choose a name for your bot (e.g., "My Stock Alerts")
   - Choose a username (must end in `bot`, e.g., `mystockalerts_bot`)
4. Copy the **API token** provided (looks like `123456789:ABCdefGHI...`)

### Step 2: Get Your Chat ID

**Option A: Using @userinfobot**
1. Search for **@userinfobot** on Telegram
2. Start the bot
3. It will display your **Chat ID**

**Option B: Using @RawDataBot**
1. Search for **@RawDataBot** on Telegram
2. Forward any message to it
3. Look for the `chat.id` field in the response

### Step 3: Start Your Bot

1. Find your new bot by its username
2. Press **Start** to activate it
3. The bot can now send you messages!

### Step 4: Configure in Trend Terminal

**Option A: Via Settings Page**
1. Navigate to Settings in the app
2. Enter your Bot Token and Chat ID
3. Click "Test Connection"
4. Save settings

**Option B: Via Environment Variables**
```env
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### For Group Notifications

1. Add your bot to the group
2. Send a message in the group
3. Use `@RawDataBot` to find the group's chat ID (will be a negative number)
4. Use that ID as your `TELEGRAM_CHAT_ID`

---

## 📊 Signal Logic

### Score Calculation (0-100)

The signal score combines three components:

| Component | Weight | Factors |
|-----------|--------|---------|
| **Technical** | 60% | EMA alignment, RSI levels, price momentum |
| **Volume** | 20% | Volume vs average, trend confirmation |
| **News** | 20% | Sentiment, freshness, relevance |

### Technical Scoring (5 factors × 12 points each)

1. **Trend Alignment** - Price vs EMAs
2. **EMA Stack** - Short > Medium > Long EMAs
3. **RSI Zone** - Oversold/overbought conditions
4. **Momentum** - Rate of change
5. **Volatility** - ATR-based assessment

### Direction Classification

| Score | Direction | Interpretation |
|-------|-----------|----------------|
| ≥ 65 | Bullish | Strong upward momentum |
| ≤ 35 | Bearish | Strong downward momentum |
| 36-64 | Neutral | Mixed or sideways |

### News Sentiment

- **Positive**: Earnings beats, upgrades, new products
- **Negative**: Misses, downgrades, investigations
- **Neutral**: Routine filings, management changes

---

## 🔬 Backtesting Methodology

### No Look-Ahead Bias

The backtest engine enforces strict temporal ordering:

1. **Signals generated**: Using only data available at signal time
2. **Entry on next day**: After signal is generated
3. **Exit conditions**: Based on rules, not future knowledge

### Exit Strategies

| Strategy | Description |
|----------|-------------|
| `fixed_days` | Hold for N days after entry |
| `signal_flip` | Exit when signal direction changes |
| `trailing_stop` | Exit on X% decline from peak |

### Metrics Calculated

- **Total Return**: Overall portfolio growth
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns
- **Avg Win/Loss**: Average gain on wins vs losses

---

## 🗂️ Project Structure

```
Trend Terminal/
├── app/
│   ├── main.py              # Streamlit entry point
│   ├── components/
│   │   └── dashboard.py     # Dashboard rendering
│   └── pages/
│       ├── 2_screener.py    # Stock screener
│       ├── 3_stock_detail.py # Individual stock view
│       ├── 4_backtest.py    # Backtesting interface
│       └── 5_settings.py    # Configuration
├── core/
│   ├── data_fetcher.py      # Yahoo Finance wrapper
│   ├── indicators.py        # Technical indicators
│   ├── signals.py           # Signal generation
│   ├── news_classifier.py   # News sentiment
│   ├── universes.py         # Stock universes
│   └── llm_analyzer.py      # AI analysis
├── backtest/
│   ├── engine.py            # Backtest execution
│   └── metrics.py           # Performance metrics
├── storage/
│   ├── database.py          # SQLite operations
│   ├── cache.py             # Caching layer
│   └── models.py            # Data models
├── notifications/
│   └── telegram.py          # Telegram integration
├── tests/
│   ├── test_indicators.py
│   ├── test_signals.py
│   ├── test_news_classifier.py
│   └── test_backtest.py
├── config/
│   ├── config.yaml          # Configuration
│   └── watchlist.txt        # User watchlist
├── requirements.txt         # Dependencies
└── .env.example             # Environment template
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_indicators.py -v

# Run with coverage
pytest tests/ --cov=core --cov=backtest --cov-report=html
```

---

## 🔧 Development

### Adding New Indicators

1. Add calculation to `core/indicators.py`
2. Update `core/signals.py` to use the indicator
3. Add tests in `tests/test_indicators.py`

### Adding New Signal Components

1. Create scoring logic in `core/signals.py`
2. Update weights in `config/config.yaml`
3. Add to reasons generation

### Adding New Pages

1. Create `app/pages/N_pagename.py`
2. Add navigation in `app/main.py` sidebar
3. Follow existing page patterns

---

## ⚠️ Disclaimer

**This software is for educational and informational purposes only.**

- Not financial advice
- Past performance does not guarantee future results
- Always do your own research
- Consult a financial advisor before investing
- The authors are not responsible for any financial losses

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for market data
- [Streamlit](https://streamlit.io/) for the web framework
- [Plotly](https://plotly.com/) for interactive charts
- [OpenAI](https://openai.com/) for AI capabilities

---

## 📞 Support

For issues and feature requests, please open a GitHub issue.

Happy Trading! 📈
