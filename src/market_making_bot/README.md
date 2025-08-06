# Market Making Bot for Polymarket

A sophisticated market making bot designed to profit from spreads in hourly Ethereum price prediction markets on Polymarket.

## Overview

This bot automatically:
- ✅ Finds active hourly ETH prediction markets
- ✅ Analyzes spreads and market conditions  
- ✅ Places competitive bid/ask orders to profit from spreads
- ✅ Manages risk across multiple positions
- ✅ Handles order lifecycle and refreshing
- ✅ Monitors volatility and market health

## Architecture

The bot consists of several key components:

### Core Components

1. **MarketMakingBot** (`bot.py`) - Main orchestrator
2. **MarketMakingStrategy** (`strategy.py`) - Market discovery and opportunity evaluation  
3. **OrderManager** (`order_manager.py`) - Order placement and lifecycle management
4. **RiskManager** (`risk_manager.py`) - Risk monitoring and position sizing
5. **MarketMakingConfig** (`config.py`) - Configuration parameters

### Key Features

- **Multi-Market Support**: Automatically finds and trades multiple hourly ETH markets
- **Risk Management**: Position limits, stop losses, volatility monitoring
- **Smart Order Placement**: Competitive pricing with profit margins
- **Emergency Stops**: Automatic shutdown on consecutive losses or high volatility
- **Dry Run Mode**: Test strategies without real money
- **Comprehensive Logging**: Detailed performance and health metrics

## Quick Start

### Basic Usage

```python
from src.market_making_bot import MarketMakingBot, MarketMakingConfig

# Create conservative configuration
config = MarketMakingConfig(
    crypto="ethereum",
    dry_run=True,  # Safe mode - no real orders
    base_position_size=10.0,
    min_spread_threshold=0.02,  # Only trade >2% spreads
    target_profit_margin=0.008  # Target 0.8% profit
)

# Start bot
bot = MarketMakingBot(config)
await bot.start()  # Or use bot.run_for_duration(hours)
```

### Example Scripts

Run the example script for guided setup:

```bash
cd /Users/luca/dev/poly-reward
python -m src.market_making_bot.example
```

Options:
1. **Conservative Bot** - Safe settings for beginners
2. **Aggressive Bot** - Higher risk/reward settings  
3. **Market Discovery Test** - Test market finding functionality
4. **Market Analysis** - Analyze opportunities without trading

## Configuration Parameters

### Trading Parameters
- `base_position_size`: Base position size in USD (default: 10.0)
- `max_position_size`: Maximum position per market (default: 100.0)
- `min_spread_threshold`: Minimum spread to trade (default: 0.02 = 2%)
- `target_profit_margin`: Target profit per trade (default: 0.005 = 0.5%)

### Risk Management
- `max_exposure_per_market`: Max exposure per market (default: 50.0)
- `max_total_exposure`: Max total exposure (default: 200.0)
- `stop_loss_percentage`: Stop loss threshold (default: 0.05 = 5%)
- `volatility_threshold`: Volatility pause threshold (default: 0.03 = 3%)

### Order Management
- `order_refresh_interval`: Order refresh frequency in seconds (default: 30)
- `tick_buffer_size`: Distance from best bid/ask in ticks (default: 2)
- `order_ttl_seconds`: Order time-to-live (default: 300)

### Safety Features
- `dry_run`: Enable dry run mode (default: True)
- `enable_emergency_stop`: Auto-stop on losses (default: True)
- `max_consecutive_losses`: Emergency stop trigger (default: 5)

## Market Selection

The bot targets hourly Ethereum price prediction markets with patterns like:
- `ethereum-up-or-down-january-15-3pm-et`
- `eth-price-february-22-9am-et`

It automatically discovers:
- Current hour market (if still active)
- Next hour market (if available)
- Other active hourly ETH markets

## Strategy Logic

### Market Making Approach

1. **Market Discovery**: Find active hourly ETH markets
2. **Opportunity Evaluation**: 
   - Check spread > minimum threshold
   - Verify market is accepting orders
   - Ensure sufficient time to expiry
   - Assess volatility conditions
3. **Order Placement**:
   - Calculate competitive bid/ask prices
   - Ensure target profit margin
   - Place both buy and sell orders
4. **Order Management**:
   - Refresh orders based on market conditions
   - Cancel expired orders
   - Update positions on fills

### Risk Management

- **Position Limits**: Per-market and total exposure caps
- **Stop Losses**: Automatic position closure on losses
- **Volatility Monitoring**: Pause trading during high volatility
- **Time Management**: Stop trading near market expiry
- **Emergency Stops**: Automatic shutdown triggers

## Performance Monitoring

The bot provides comprehensive metrics:

### Health Checks (every 5 minutes)
- Uptime and cycle completion
- Total exposure and PnL
- Order fill rates and activity
- Market opportunities found

### Real-time Monitoring
- Active positions across markets
- Order book analysis
- Volatility measurements
- Risk limit utilization

## Safety Features

### Dry Run Mode
- **Always start in dry run mode** when testing
- No real orders placed, but full strategy simulation
- Perfect for testing configurations and market discovery

### Emergency Stops
- Automatic shutdown on consecutive losses
- Volatility-based trading pauses
- Manual emergency stop capability
- Graceful cleanup of all active orders

### Risk Limits
- Multiple layers of position size limits
- Real-time exposure monitoring
- Automatic risk assessment per market

## Prerequisites

### Environment Setup

1. **Polymarket Account**: Set up account and get API credentials
2. **Environment Variables**:
   ```bash
   export PK="your_private_key"
   export BROWSER_ADDRESS="your_wallet_address"
   ```

3. **Dependencies**: All required packages are in `pyproject.toml`

### Market Access
- The bot requires access to Polymarket's CLOB API
- Ensure your account has trading permissions
- Start with small position sizes and dry run mode

## Monitoring and Logs

### Log Files
- Automatic log file creation with timestamps
- Configurable log levels (INFO, DEBUG, etc.)
- Separate logs for different components

### Key Metrics to Monitor
- **Fill Rate**: Percentage of orders that get filled
- **Opportunity Rate**: Percentage of markets with profitable spreads
- **PnL Tracking**: Realized and unrealized profit/loss
- **Risk Utilization**: How close to risk limits

### Health Check Outputs
```
=== HEALTH CHECK SUMMARY ===
Uptime: 2.3 hours
Cycles completed: 276
Risk - Total exposure: $150.25
Risk - Realized PnL: $12.45
Orders - Fill rate: 67.8%
Strategy - Opportunity rate: 23.4%
=== END HEALTH CHECK ===
```

## Advanced Usage

### Custom Strategies
Extend the `MarketMakingStrategy` class to implement custom logic:

```python
class CustomStrategy(MarketMakingStrategy):
    async def evaluate_market_opportunity(self, market):
        # Custom market evaluation logic
        opportunity = await super().evaluate_market_opportunity(market)
        
        # Add custom filters
        if opportunity and self.custom_filter(market):
            return opportunity
        return None
```

### Multiple Crypto Support
While currently focused on Ethereum, the bot can be extended for other cryptocurrencies by modifying the `MARKETS` constant and slug patterns.

## Troubleshooting

### Common Issues

1. **No Markets Found**
   - Check if hourly ETH markets are currently active
   - Verify API credentials and connection
   - Run market discovery test

2. **No Orders Placed**
   - Check if spreads meet minimum threshold
   - Verify risk limits aren't exceeded
   - Ensure dry_run=False for live trading

3. **High Error Rates**
   - Check network connectivity
   - Verify API rate limits
   - Review log files for specific errors

### Debug Mode
Set `log_level="DEBUG"` in configuration for detailed troubleshooting information.

## Performance Expectations

### Typical Results (varies by market conditions)
- **Fill Rate**: 40-80% depending on market activity
- **Opportunity Rate**: 15-40% of evaluated markets
- **Profit per Trade**: 0.3-1.5% when successful
- **Daily Trades**: 10-50 depending on market availability

### Market Conditions Impact
- **High Volatility**: Fewer opportunities, more risk
- **Low Volume**: Better spreads, fewer fills
- **Market Hours**: More activity during US trading hours

## Disclaimer

⚠️ **Important Risk Warnings**:

- This is experimental software for educational purposes
- Market making involves significant risk of loss
- Always start with dry run mode and small position sizes  
- Cryptocurrency prediction markets are highly volatile
- Past performance does not guarantee future results
- Use at your own risk and never invest more than you can afford to lose

The authors are not responsible for any trading losses incurred using this software.

## License

This software is provided as-is for educational and research purposes. Use responsibly and in accordance with applicable laws and regulations.