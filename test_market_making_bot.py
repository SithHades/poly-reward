#!/usr/bin/env python3
"""
Simple test script for the Market Making Bot
"""

import asyncio
import logging
import sys
from src.market_making_bot import MarketMakingBot, MarketMakingConfig


async def test_bot_initialization():
    """Test that the bot initializes correctly"""
    print("Testing bot initialization...")
    
    config = MarketMakingConfig(
        crypto="ethereum",
        dry_run=True,
        base_position_size=5.0,
    )
    
    try:
        bot = MarketMakingBot(config)
        print("✅ Bot initialized successfully")
        
        # Test getting status
        status = bot.get_status()
        print(f"✅ Bot status retrieved: {status['is_running']}")
        
        return True
    except Exception as e:
        print(f"❌ Bot initialization failed: {e}")
        return False


async def test_market_discovery():
    """Test market discovery functionality"""
    print("\nTesting market discovery...")
    
    config = MarketMakingConfig(crypto="ethereum", dry_run=True)
    bot = MarketMakingBot(config)
    
    try:
        # Test finding markets
        markets = await bot.strategy.find_active_markets()
        print(f"✅ Market discovery completed: found {len(markets)} markets")
        
        for market in markets[:3]:  # Show first 3 markets
            print(f"   - {market.market_slug}")
            print(f"     Active: {market.active}, Tokens: {len(market.tokens)}")
            
        return True
    except Exception as e:
        print(f"❌ Market discovery failed: {e}")
        return False


async def test_strategy_cycle():
    """Test running one strategy cycle"""
    print("\nTesting strategy cycle...")
    
    config = MarketMakingConfig(
        crypto="ethereum", 
        dry_run=True,
        min_spread_threshold=0.01  # Lower threshold for testing
    )
    bot = MarketMakingBot(config)
    
    try:
        # Run one cycle
        await bot.strategy.run_strategy_cycle()
        
        # Get metrics
        metrics = bot.strategy.get_strategy_metrics()
        print("✅ Strategy cycle completed")
        print(f"   Markets evaluated: {metrics['markets_evaluated']}")
        print(f"   Opportunities found: {metrics['profitable_spreads_found']}")
        
        return True
    except Exception as e:
        print(f"❌ Strategy cycle failed: {e}")
        return False


async def run_all_tests():
    """Run all tests"""
    print("🤖 Market Making Bot Test Suite")
    print("=" * 40)
    
    tests = [
        test_bot_initialization,
        test_market_discovery, 
        test_strategy_cycle
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test error: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check configuration and API access")
    
    return passed == total


def main():
    """Main test runner"""
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest suite error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()