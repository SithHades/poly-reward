#!/usr/bin/env python3
"""
Example script demonstrating how to use the Market Making Bot
for hourly Ethereum price prediction markets on Polymarket.
"""

import asyncio
import logging
import sys
from datetime import datetime

from src.polymarket_client import PolymarketClient
from .bot import MarketMakingBot
from .config import MarketMakingConfig


async def run_conservative_bot():
    """Run bot with conservative settings"""
    
    config = MarketMakingConfig(
        # Market settings
        crypto="ethereum",
        
        # Conservative trading parameters
        base_position_size=10.0,          # Small position sizes
        max_position_size=25.0,           # Low max per market
        min_spread_threshold=0.025,       # Only trade markets with >2.5% spread
        target_profit_margin=0.010,       # Target 1% profit per trade
        
        # Conservative risk management
        max_exposure_per_market=30.0,     # Low exposure per market
        max_total_exposure=100.0,         # Low total exposure
        stop_loss_percentage=0.03,        # 3% stop loss
        volatility_threshold=0.02,        # Low volatility threshold
        
        # Order management
        order_refresh_interval=45,        # Refresh every 45 seconds
        tick_buffer_size=3,               # Stay further from best prices
        order_ttl_seconds=180,            # Shorter TTL
        
        # Safety settings
        dry_run=True,                     # Start in dry run mode
        market_close_buffer_minutes=10,   # Stop trading 10 min before close
        min_time_to_expiry_hours=1.0,     # Only trade markets >1hr from expiry
        
        # Logging
        enable_logging=True,
        log_level="INFO"
    )
    
    # Create bot
    bot = MarketMakingBot(config)
    
    print("Starting Conservative Market Making Bot")
    print(f"Mode: {'DRY RUN' if config.dry_run else 'LIVE TRADING'}")
    print(f"Crypto: {config.crypto}")
    print(f"Min spread: {config.min_spread_threshold:.2%}")
    print(f"Target margin: {config.target_profit_margin:.2%}")
    print()
    
    # Run for 2 hours
    await bot.run_for_duration(2.0)


async def run_aggressive_bot():
    """Run bot with more aggressive settings (for experienced users)"""
    
    config = MarketMakingConfig(
        # Market settings
        crypto="ethereum",
        
        # Aggressive trading parameters  
        base_position_size=50.0,          # Larger positions
        max_position_size=100.0,          # Higher max per market
        min_spread_threshold=0.015,       # Trade smaller spreads
        target_profit_margin=0.006,       # Lower profit target for more fills
        
        # Aggressive risk management
        max_exposure_per_market=75.0,     # Higher exposure per market
        max_total_exposure=300.0,         # Higher total exposure
        stop_loss_percentage=0.05,        # 5% stop loss
        volatility_threshold=0.035,       # Higher volatility tolerance
        
        # Order management
        order_refresh_interval=20,        # More frequent refreshes
        tick_buffer_size=1,               # Closer to best prices
        order_ttl_seconds=300,            # Longer TTL
        
        # Safety settings
        dry_run=True,                     # Still start in dry run
        market_close_buffer_minutes=5,    # Trade closer to close
        min_time_to_expiry_hours=0.5,     # Trade shorter-term markets
        
        # Logging
        enable_logging=True,
        log_level="DEBUG"
    )
    
    bot = MarketMakingBot(config)
    
    print("Starting Aggressive Market Making Bot")
    print("⚠️  WARNING: This uses aggressive settings!")
    print(f"Mode: {'DRY RUN' if config.dry_run else 'LIVE TRADING'}")
    print(f"Crypto: {config.crypto}")
    print(f"Min spread: {config.min_spread_threshold:.2%}")
    print(f"Target margin: {config.target_profit_margin:.2%}")
    print()
    
    # Run for 1 hour
    await bot.run_for_duration(1.0)


async def test_market_finding():
    """Test the market discovery functionality"""
    
    print("Testing market discovery...")
    
    client = PolymarketClient()
    
    try:
        # Test getting markets by slug
        from src.parsing_utils import get_current_market_slug, get_next_market_slug
        
        current_slug = get_current_market_slug("ethereum")
        next_slug = get_next_market_slug("ethereum")
        
        print(f"Current ETH market slug: {current_slug}")
        print(f"Next ETH market slug: {next_slug}")
        
        # Try to fetch these markets
        try:
            current_market = client.get_market_by_slug(current_slug)
            if current_market:
                print(f"✅ Found current market: {current_market.market_slug}")
                print(f"   Active: {current_market.active}")
                print(f"   Accepting orders: {current_market.accepting_orders}")
                print(f"   Tokens: {len(current_market.tokens)}")
                if current_market.end_date_iso:
                    time_left = current_market.end_date_iso - datetime.now(current_market.end_date_iso.tzinfo)
                    print(f"   Time to expiry: {time_left}")
            else:
                print(f"❌ Current market not found: {current_slug}")
        except Exception as e:
            print(f"❌ Error fetching current market: {e}")
            
        try:
            next_market = client.get_market_by_slug(next_slug)
            if next_market:
                print(f"✅ Found next market: {next_market.market_slug}")
                print(f"   Active: {next_market.active}")
                print(f"   Accepting orders: {next_market.accepting_orders}")
                print(f"   Tokens: {len(next_market.tokens)}")
                if next_market.end_date_iso:
                    time_left = next_market.end_date_iso - datetime.now(next_market.end_date_iso.tzinfo)
                    print(f"   Time to expiry: {time_left}")
            else:
                print(f"❌ Next market not found: {next_slug}")
        except Exception as e:
            print(f"❌ Error fetching next market: {e}")
            
    except Exception as e:
        print(f"❌ Error in market discovery test: {e}")


async def run_market_analyzer():
    """Run market analysis without trading"""
    
    config = MarketMakingConfig(
        crypto="ethereum",
        dry_run=True,  # Analysis only
        enable_logging=True,
        log_level="INFO"
    )
    
    bot = MarketMakingBot(config)
    
    print("Running Market Analysis (no trading)")
    print("This will find and analyze ETH hourly markets for opportunities...")
    print()
    
    # Just run a few strategy cycles to analyze markets
    for i in range(5):
        print(f"Analysis cycle {i+1}/5")
        await bot.strategy.run_strategy_cycle()
        
        # Print metrics
        metrics = bot.strategy.get_strategy_metrics()
        print(f"  Markets found: {metrics['active_markets']}")
        print(f"  Markets evaluated: {metrics['markets_evaluated']}")
        print(f"  Opportunities found: {metrics['profitable_spreads_found']}")
        if metrics['markets_evaluated'] > 0:
            print(f"  Opportunity rate: {metrics['opportunity_rate']:.1%}")
        print()
        
        await asyncio.sleep(30)  # Wait 30 seconds between cycles
        
    print("Market analysis complete!")


def setup_logging():
    """Setup logging configuration"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'market_making_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main function with menu selection"""
    
    setup_logging()
    
    print("🤖 Polymarket Market Making Bot Examples")
    print("=" * 50)
    print("1. Conservative Bot (2 hours)")
    print("2. Aggressive Bot (1 hour)") 
    print("3. Test Market Discovery")
    print("4. Market Analysis Only")
    print("5. Exit")
    print()
    
    try:
        choice = input("Select option (1-5): ").strip()
        
        if choice == "1":
            asyncio.run(run_conservative_bot())
        elif choice == "2":
            print("⚠️  Are you sure you want to run aggressive settings? (y/N): ")
            confirm = input().strip().lower()
            if confirm == 'y':
                asyncio.run(run_aggressive_bot())
            else:
                print("Cancelled.")
        elif choice == "3":
            asyncio.run(test_market_finding())
        elif choice == "4":
            asyncio.run(run_market_analyzer())
        elif choice == "5":
            print("Goodbye!")
        else:
            print("Invalid choice.")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()