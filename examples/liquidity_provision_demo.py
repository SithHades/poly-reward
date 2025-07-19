#!/usr/bin/env python3
"""
Polymarket Liquidity Provision Demo

This script demonstrates how to use the complete liquidity provision system:
1. Screen markets for high-reward, low-competition opportunities
2. Analyze specific markets for LP viability
3. Calculate optimal order placement strategies
4. Monitor and manage risk

This is a demonstration script - run with paper trading enabled for safety.
"""

import sys
import os
import logging
import json
from typing import List
from datetime import datetime, timezone

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.polymarket_client import PolymarketClient
from src.market_screener import MarketScreener, ScreeningCriteria, MarketOpportunity
from src.strategy import PolymarketLiquidityStrategy, LiquidityProvisionConfig
from src.core.models import Position


def setup_logging():
    """Configure logging for the demo"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def demo_market_screening():
    """Demonstrate the market screening functionality"""
    print("\n" + "=" * 60)
    print("DEMO: Market Screening for LP Opportunities")
    print("=" * 60)

    # Initialize client in paper trading mode
    client = PolymarketClient()

    # Set up screening criteria
    criteria = ScreeningCriteria(
        min_daily_rewards=30,  # Minimum $30/day rewards
        max_min_order_size=200,  # Max $200 minimum order
        max_competition_density=0.7,  # Max 70% competition
        min_spread_budget=0.02,  # Min 2c spread
        max_risk_level="medium",  # Medium risk or lower
    )

    # Create screener
    screener = MarketScreener(client, criteria)

    print("Screening criteria:")
    print(f"  - Min daily rewards: ${criteria.min_daily_rewards}")
    print(f"  - Max competition: {criteria.max_competition_density * 100}%")
    print(f"  - Min spread: {criteria.min_spread_budget * 100}c")
    print(f"  - Max risk level: {criteria.max_risk_level}")

    # Find opportunities
    print("\nSearching for opportunities...")
    opportunities = screener.find_opportunities(max_markets=50)

    print(f"\nFound {len(opportunities)} qualifying opportunities:")
    print("-" * 80)

    for i, opp in enumerate(opportunities[:5], 1):  # Show top 5
        print(f"{i}. {opp.market.question[:60]}...")
        print(f"   Reward Score: {opp.reward_score:.1f}")
        print(f"   Daily Rewards: ${opp.estimated_daily_rewards}")
        print(f"   Competition: {opp.competition_density * 100:.1f}%")
        print(f"   Min Capital: ${opp.min_capital_required}")
        print(f"   Max Spread: {opp.max_spread_allowed * 100:.1f}c")
        print(f"   Risk Level: {opp.risk_level}")
        print(
            f"   YES Token: {opp.yes_token['token_id'] if isinstance(opp.yes_token, dict) else opp.yes_token.token_id}"
        )
        print(
            f"   NO Token: {opp.no_token['token_id'] if isinstance(opp.no_token, dict) else opp.no_token.token_id}"
        )
        print()

    return opportunities


def demo_strategy_analysis(opportunities: List[MarketOpportunity]):
    """Demonstrate strategy analysis for selected opportunities"""
    print("\n" + "=" * 60)
    print("DEMO: Liquidity Provision Strategy Analysis")
    print("=" * 60)

    if not opportunities:
        print("No opportunities available for strategy analysis")
        return

    # Configure strategy
    config = LiquidityProvisionConfig(
        max_spread_from_midpoint=0.03,  # 3c max from midpoint
        optimal_distance_from_midpoint=0.015,  # 1.5c optimal distance
        min_order_size=50,  # $50 minimum orders
        max_position_size=500,  # $500 max per market
        volatility_exit_threshold=0.02,  # 2% volatility exit
        enable_yes_no_hedging=True,  # Enable hedging
        order_refresh_interval_minutes=30,  # Refresh every 30 min
    )

    strategy = PolymarketLiquidityStrategy(config)

    print("Strategy configuration:")
    print(f"  - Max spread from midpoint: {config.max_spread_from_midpoint * 100}c")
    print(f"  - Optimal distance: {config.optimal_distance_from_midpoint * 100}c")
    print(f"  - Min order size: ${config.min_order_size}")
    print(f"  - Max position size: ${config.max_position_size}")
    print(f"  - Hedging enabled: {config.enable_yes_no_hedging}")

    # Analyze top opportunity
    top_opportunity = opportunities[0]
    print("\nAnalyzing top opportunity:")
    print(f"Market: {top_opportunity.market.question[:80]}...")

    # Initialize client for orderbook data
    client = PolymarketClient()
    screener = MarketScreener(client)

    # Get orderbooks
    yes_orderbook, no_orderbook = screener.get_market_orderbooks(top_opportunity.market)

    if not yes_orderbook or not no_orderbook:
        print("Could not retrieve orderbook data")
        return

    print("\nOrderbook Analysis:")
    print("YES Market:")
    print(f"  - Midpoint: {yes_orderbook.midpoint}")
    print(f"  - Spread: {yes_orderbook.spread}")
    print(f"  - Best Bid: {yes_orderbook.best_bid()}")
    print(f"  - Best Ask: {yes_orderbook.best_ask()}")

    print("NO Market:")
    print(f"  - Midpoint: {no_orderbook.midpoint}")
    print(f"  - Spread: {no_orderbook.spread}")
    print(f"  - Best Bid: {no_orderbook.best_bid()}")
    print(f"  - Best Ask: {no_orderbook.best_ask()}")

    # Analyze market condition
    condition = strategy.analyze_market_condition(yes_orderbook, no_orderbook)
    print(f"\nMarket Condition: {condition.value.upper()}")

    if condition.name == "ATTRACTIVE":
        # Calculate optimal orders
        current_positions = {}  # No existing positions
        available_capital = 2000  # $2000 available

        orders = strategy.calculate_optimal_orders(
            yes_orderbook, no_orderbook, current_positions, available_capital
        )

        print("\nOptimal Order Recommendations:")
        print(f"Available Capital: ${available_capital}")
        print(f"Recommended Orders: {len(orders)}")

        for i, order in enumerate(orders, 1):
            print(f"  {i}. {order['side'].value.upper()} {order['size']} shares")
            print(f"     Price: {order['price']}")
            print(f"     Market: {order['market_type']}")
            print(f"     Token: {order['asset_id']}")
            print()
    else:
        print(f"Market not suitable for LP: {condition.value}")


def demo_risk_management():
    """Demonstrate risk management features"""
    print("\n" + "=" * 60)
    print("DEMO: Risk Management & Order Lifecycle")
    print("=" * 60)

    config = LiquidityProvisionConfig()
    strategy = PolymarketLiquidityStrategy(config)

    # Simulate some orders for risk management demo
    from core.models import Order, OrderSide, OrderStatus
    from datetime import datetime, timezone, timedelta

    current_time = datetime.now(timezone.utc)

    mock_orders = [
        Order(
            id="order_1",
            market_id="test_market",
            side=OrderSide.BUY,
            price=0.52,
            size=100,
            status=OrderStatus.OPEN,
            timestamp=current_time - timedelta(minutes=35),  # Old order
        ),
        Order(
            id="order_2",
            market_id="test_market",
            side=OrderSide.SELL,
            price=0.54,
            size=100,
            status=OrderStatus.OPEN,
            timestamp=current_time - timedelta(minutes=10),  # Recent order
        ),
        Order(
            id="order_3",
            market_id="test_market",
            side=OrderSide.BUY,
            price=0.55,
            size=100,
            status=OrderStatus.FILLED,
            timestamp=current_time - timedelta(minutes=5),
            filled_size=100,
            metadata={"market_type": "YES"},
        ),
    ]

    print("Current Orders:")
    for order in mock_orders:
        age_minutes = (current_time - order.timestamp).total_seconds() / 60
        print(f"  {order.id}: {order.side.value} {order.size} @ {order.price}")
        print(f"    Status: {order.status.value}, Age: {age_minutes:.1f} minutes")

    # Check which orders should be cancelled
    orders_to_cancel = strategy.should_cancel_orders("test_market", mock_orders)

    print("\nRisk Management Analysis:")
    print(f"Orders to cancel: {orders_to_cancel}")

    # Demonstrate hedge calculation for filled order
    filled_order = mock_orders[2]  # The filled order

    print("\nHedge Order Calculation:")
    print(
        f"Filled Order: {filled_order.side.value} {filled_order.filled_size} @ {filled_order.price}"
    )

    # Mock orderbooks for hedge calculation
    from src.strategy import OrderbookSnapshot, OrderbookLevel

    yes_orderbook = OrderbookSnapshot(
        asset_id="yes_token",
        bids=[OrderbookLevel(0.52, 100)],
        asks=[OrderbookLevel(0.54, 100)],
        midpoint=0.53,
        spread=0.02,
    )

    no_orderbook = OrderbookSnapshot(
        asset_id="no_token",
        bids=[OrderbookLevel(0.46, 100)],
        asks=[OrderbookLevel(0.48, 100)],
        midpoint=0.47,
        spread=0.02,
    )

    hedge_orders = strategy.calculate_hedge_orders(
        filled_order, yes_orderbook, no_orderbook
    )

    for hedge in hedge_orders:
        print(f"Hedge Order: {hedge['side'].value} {hedge['size']} @ {hedge['price']}")
        print(f"  Market: {hedge['market_type']}")
        print(f"  Reason: {hedge['reason']}")


def demo_portfolio_analysis():
    """Demonstrate portfolio analysis capabilities"""
    print("\n" + "=" * 60)
    print("DEMO: Portfolio Analysis")
    print("=" * 60)

    # Mock portfolio positions
    positions = {
        "market_1": Position(market_id="market_1", size=150, entry_price=0.55),
        "market_2": Position(
            market_id="market_2",
            size=-200,  # Short position
            entry_price=0.45,
        ),
        "market_3": Position(market_id="market_3", size=100, entry_price=0.60),
    }

    # Mock current prices
    current_prices = {
        "market_1": 0.58,  # +5.5% gain
        "market_2": 0.42,  # +6.7% gain (short position)
        "market_3": 0.57,  # -5% loss
    }

    print("Portfolio Positions:")
    total_pnl = 0
    total_value = 0

    for market_id, position in positions.items():
        current_price = current_prices[market_id]
        pnl = position.pnl(current_price)
        pnl_pct = position.pnl_percentage(current_price)
        current_val = position.current_value(current_price)

        position_type = (
            "LONG" if position.is_long() else "SHORT" if position.is_short() else "FLAT"
        )

        print(f"  {market_id}:")
        print(
            f"    Position: {position_type} {abs(position.size)} @ {position.entry_price}"
        )
        print(f"    Current Price: {current_price}")
        print(f"    Current Value: ${current_val}")
        print(f"    P&L: ${pnl} ({pnl_pct:+.1f}%)")
        print()

        total_pnl += pnl
        total_value += current_val

    print("Portfolio Summary:")
    print(f"  Total Value: ${total_value}")
    print(f"  Total P&L: ${total_pnl}")
    print(
        f"  Overall Return: {(total_pnl / total_value * 100) if total_value > 0 else 0:+.2f}%"
    )


def save_demo_results(opportunities: List[MarketOpportunity]):
    """Save demo results to JSON file"""
    print("\n" + "=" * 60)
    print("DEMO: Saving Results")
    print("=" * 60)

    if not opportunities:
        print("No opportunities to save")
        return

    # Convert opportunities to serializable format
    results = {
        "timestamp": str(datetime.now(timezone.utc)),
        "total_opportunities": len(opportunities),
        "top_opportunities": [opp.to_dict() for opp in opportunities[:10]],
    }

    filename = "demo_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {filename}")
    print(f"Saved {len(results['top_opportunities'])} top opportunities")


def main():
    """Run the complete liquidity provision demo"""
    print("Polymarket Liquidity Provision System Demo")
    print("=" * 80)
    print("This demo showcases the complete LP system functionality.")
    print("Running in PAPER TRADING mode - no real orders will be placed.")
    print("=" * 80)

    # Setup
    setup_logging()

    try:
        # 1. Market Screening
        opportunities = demo_market_screening()

        # 2. Strategy Analysis
        demo_strategy_analysis(opportunities)

        # 3. Risk Management
        demo_risk_management()

        # 4. Portfolio Analysis
        demo_portfolio_analysis()

        # 5. Save Results
        save_demo_results(opportunities)

        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print("The demo has showcased all major components:")
        print("✓ Market screening and opportunity identification")
        print("✓ Strategy analysis and order optimization")
        print("✓ Risk management and order lifecycle")
        print("✓ Portfolio tracking and P&L analysis")
        print("✓ Data persistence and reporting")
        print("\nNext steps:")
        print("- Review demo_results.json for detailed opportunity data")
        print("- Customize ScreeningCriteria and LiquidityProvisionConfig")
        print("- Implement websocket integration for real-time updates")
        print("- Add OrderManager for complete order lifecycle management")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
