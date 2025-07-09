# ETH Polymarket Trading Bot

An automated trading bot that uses machine learning to predict ETH candle direction and places limit orders on Polymarket prediction markets.

## 🚀 Features

- **ML-Powered Predictions**: Uses trained models (Random Forest, Gradient Boosting, Logistic Regression) to predict ETH candle direction
- **45-Minute Analysis**: Makes predictions based on the first 45 minutes of hourly candle data using official Binance ETH/USDT data
- **Official ETHUSDT Markets**: Specifically targets official Polymarket ETHUSDT hourly prediction markets
- **Risk Management**: Built-in position sizing, exposure limits, and order timeouts
- **Multiple Data Sources**: Fallback data sources for robust operation
- **Real-time Monitoring**: Continuous monitoring with detailed logging

## 📁 Project Structure

```
poly-reward/
├── eth_polymarket_trading_bot.py    # Main trading bot
├── trading_bot_manager.py           # Management utilities
├── config/
│   └── trading_config.yaml         # Configuration file
├── src/
│   ├── eth_candle_predictor.py     # ML prediction model
│   ├── polymarket_client.py        # Polymarket API client
│   ├── market_data_fetcher.py      # Alternative data sources
│   └── ...
└── eth_candle_analysis.ipynb       # Analysis notebook
```

## 🛠 Installation

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Set up environment variables:**
   ```bash
   # Create .env file
   PK=your_polymarket_private_key
   BROWSER_ADDRESS=your_wallet_address
   ```

3. **Configure the bot:**
   Edit `config/trading_config.yaml` to adjust settings.

## 🎯 How It Works

### 1. Data Collection
- Fetches real-time ETH/USDT 1-minute candle data from Binance (official source for Polymarket)
- Uses CCXT library for reliable exchange data access
- Falls back to Coinbase and CoinGecko if Binance is unavailable
- Maintains rolling window of recent data for predictions

### 2. Model Training
- Trains ML models using historical ETH data
- Extracts 39+ features from first 45 minutes of each hour:
  - Price action (OHLC, momentum, position in range)
  - Technical indicators (RSI, MACD, moving averages)
  - Volume patterns and trends
  - Volatility measures
  - Time-based features
  - Market microstructure signals

### 3. Prediction Logic
- At 43-47 minutes of each hour, analyzes current data
- Makes prediction for candle direction (green/red)
- Calculates confidence score based on model probabilities
- Only trades when confidence exceeds threshold (default 75%)

### 4. Market Analysis
- Uses existing Polymarket client to fetch official ETHUSDT hourly markets
- Targets markets from the "ethusdt" series (defined in constants)
- Filters for markets closing in 30-90 minutes with sufficient liquidity
- Analyzes order books to determine optimal entry prices

### 5. Order Placement
- Calculates position size using Kelly criterion (conservative)
- Places limit orders at favorable prices
- Monitors and cancels expired orders (default 30 minutes)

## 🚀 Usage

### Quick Start
```bash
# Start the trading bot
uv run python trading_bot_manager.py start

# Or with custom config
uv run python trading_bot_manager.py --config my_config.yaml start
```

### Testing and Validation
```bash
# Test prediction model
uv run python trading_bot_manager.py test-prediction

# Check available markets
uv run python trading_bot_manager.py check-markets

# Check current positions
uv run python trading_bot_manager.py check-positions

# Validate configuration
uv run python trading_bot_manager.py validate-config
```

### Data Source Testing
```bash
# Test all data sources
uv run python src/market_data_fetcher.py
```

## ⚙️ Configuration

The bot is configured via `config/trading_config.yaml`:

### Model Settings
```yaml
model:
  type: "logistic"              # Model type: logistic, random_forest, gradient_boost
  confidence_threshold: 0.75    # Minimum confidence to trade (75%)
  retrain_interval_hours: 24    # Retrain model every 24 hours
```

### Trading Settings
```yaml
trading:
  max_position_size: 50.0       # Max USD per trade
  max_daily_trades: 5           # Max trades per day
  order_timeout_minutes: 30     # Cancel orders after 30 minutes
```

### Risk Management
```yaml
risk:
  max_total_exposure: 200.0     # Max total exposure across positions
  kelly_fraction: 0.25          # Conservative Kelly sizing
  max_position_percentage: 0.1  # Max 10% of bankroll per trade
```

## 📊 Model Performance

Based on historical analysis:
- **Accuracy**: 84.8% (logistic regression)
- **AUC Score**: 0.927
- **Flip Rate**: 14.7% (candles changing direction after 45 minutes)
- **Best Features**: Price change %, momentum, technical indicators

### Key Insights
- Higher confidence predictions tend to be more accurate
- Model performs better during certain hours (market structure dependent)
- 45-minute mark provides good signal for final candle direction

## 🛡️ Risk Management

### Position Sizing
- Uses conservative Kelly criterion for position sizing
- Maximum 10% of bankroll per position
- Daily trade limits to prevent overtrading

### Order Management
- Orders automatically canceled after timeout period
- Maximum exposure limits across all positions
- Stop-loss through position sizing rather than explicit stops

### Market Filtering
- Only trades markets with sufficient liquidity
- Avoids markets close to expiration
- Filters for reasonable bid-ask spreads

## 📈 Monitoring and Logging

### Logging Levels
- **DEBUG**: Detailed execution information
- **INFO**: General operation status (default)
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors requiring attention

### Log Files
- Main log: `eth_trading_bot.log`
- Automatic rotation when file size exceeds limits
- Structured format for easy parsing

### Key Metrics Tracked
- Prediction accuracy over time
- Trade win/loss ratio
- Total exposure and P&L
- Order fill rates and slippage
- Market data source reliability

## 🔧 Troubleshooting

### Common Issues

1. **"No markets found for prediction"**
   - Check market keywords in config
   - Verify Polymarket has active ETH markets
   - Adjust minimum liquidity requirements

2. **"Confidence below threshold"**
   - Lower confidence threshold in config
   - Check model performance on recent data
   - Ensure sufficient historical data for training

3. **"Error placing order"**
   - Verify Polymarket API credentials
   - Check wallet balance and allowances
   - Ensure market is still active

4. **"Failed to get ETH price"**
   - Check internet connection
   - Verify API rate limits not exceeded
   - Test alternative data sources

### Data Source Fallbacks
If primary data sources fail:
1. **Binance** (primary) → **Coinbase** → **CoinGecko**
2. Automatic switching with health checks
3. Graceful degradation of functionality

## 📋 Development

### Adding New Features
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Update configuration schema if needed
5. Submit pull request

### Testing
```bash
# Run unit tests
uv run python -m pytest tests/

# Test specific components
uv run python -m pytest tests/test_predictor.py

# Integration tests
uv run python trading_bot_manager.py test-prediction
```

## ⚠️ Disclaimers

- **Not Financial Advice**: This bot is for educational purposes only
- **Risk Warning**: Cryptocurrency trading involves substantial risk
- **No Guarantees**: Past performance does not guarantee future results
- **Regulatory Compliance**: Ensure compliance with local regulations
- **Test First**: Always test with small amounts before full deployment

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines and submit pull requests for any improvements.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
1. Check the troubleshooting section
2. Review logs for error messages
3. Test individual components
4. Open an issue with detailed information

---

**Remember**: Start with small position sizes and thoroughly test the bot before increasing exposure. Monitor performance regularly and adjust configuration as needed.