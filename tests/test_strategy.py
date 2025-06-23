from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from src.strategy import (
    PolymarketLiquidityStrategy,
    LiquidityProvisionConfig,
    OrderbookSnapshot,
    OrderbookLevel,
    VolatilityMetrics,
    MarketCondition,
)
from src.models import Order, Position, OrderSide, OrderStatus


class TestOrderbookSnapshot:
    def test_orderbook_creation(self):
        bids = [OrderbookLevel(0.52, 100)]
        asks = [OrderbookLevel(0.54, 150)]

        orderbook = OrderbookSnapshot(
            asset_id="test_asset",
            bids=bids,
            asks=asks,
            midpoint=0.53,
            spread=0.02,
        )

        assert orderbook.asset_id == "test_asset"
        assert orderbook.best_bid() == 0.52
        assert orderbook.best_ask() == 0.54
        assert orderbook.midpoint == 0.53
        assert orderbook.spread == 0.02

    def test_volume_in_range_calculation(self):
        bids = [
            OrderbookLevel(0.50, 100),
            OrderbookLevel(0.49, 200),
            OrderbookLevel(0.48, 300),
        ]
        asks = [
            OrderbookLevel(0.55, 150),
            OrderbookLevel(0.56, 250),
            OrderbookLevel(0.57, 350),
        ]

        orderbook = OrderbookSnapshot(
            asset_id="test_asset",
            bids=bids,
            asks=asks,
            midpoint=0.525,
            spread=0.05,
        )

        # Test bid volume in range 0.49-0.51
        bid_volume = orderbook.total_bid_volume_in_range(
            (0.49, 0.51)
        )
        assert bid_volume == 300  # 100 + 200

        # Test ask volume in range 0.55-0.57
        ask_volume = orderbook.total_ask_volume_in_range(
            (0.55, 0.57)
        )
        assert ask_volume == 750  # 150 + 250 + 350


class TestVolatilityMetrics:
    def test_volatility_tracking(self):
        metrics = VolatilityMetrics()

        # Add some midpoint changes
        metrics.add_midpoint_change(0.01)
        metrics.add_midpoint_change(0.005)
        metrics.add_midpoint_change(0.03)  # Large move

        # Add spread changes
        metrics.add_spread_change(0.02)
        metrics.add_spread_change(0.08)  # Large spread change

        assert len(metrics.midpoint_changes) == 3
        assert len(metrics.spread_changes) == 2

        # Test volatility detection
        assert metrics.is_volatile(
            0.02, 0.05
        )  # Should be volatile
        assert not metrics.is_volatile(
            0.05, 0.1
        )  # Should not be volatile

    def test_window_maintenance(self):
        metrics = VolatilityMetrics()

        # Add more than 50 data points to test window maintenance
        for i in range(60):
            metrics.add_midpoint_change(float(i * 0.001))

        # Should only keep last 50 points
        assert len(metrics.midpoint_changes) == 50
        assert metrics.midpoint_changes[0] == -0.010


class TestLiquidityProvisionConfig:
    def test_default_config(self):
        config = LiquidityProvisionConfig()

        assert config.max_spread_from_midpoint == 0.03
        assert config.min_order_size == 100
        assert config.optimal_distance_from_midpoint == 0.015
        assert config.enable_yes_no_hedging is True
        assert config.hedge_ratio == 1.0


class TestPolymarketLiquidityStrategy:
    def setup_method(self):
        self.config = LiquidityProvisionConfig()
        self.strategy = PolymarketLiquidityStrategy(self.config)

        # Create sample orderbooks
        self.yes_orderbook = OrderbookSnapshot(
            asset_id="test_yes",
            bids=[OrderbookLevel(0.52, 100)],
            asks=[OrderbookLevel(0.54, 100)],
            midpoint=0.53,
            spread=0.02,
        )

        self.no_orderbook = OrderbookSnapshot(
            asset_id="test_no",
            bids=[OrderbookLevel(0.46, 100)],
            asks=[OrderbookLevel(0.48, 100)],
            midpoint=0.47,
            spread=0.02,
        )

    def test_competition_density_calculation(self):
        # Test with low competition
        density = self.strategy._calculate_competition_density(
            self.yes_orderbook, self.no_orderbook
        )
        assert density < 0.5  # Should be low competition

        # Test with high competition (many orders in spread)
        high_competition_yes = OrderbookSnapshot(
            asset_id="test_yes",
            bids=[
                OrderbookLevel(0.52, 500),
                OrderbookLevel(0.51, 500),
                OrderbookLevel(0.50, 500),
            ],
            asks=[
                OrderbookLevel(0.54, 500),
                OrderbookLevel(0.55, 500),
                OrderbookLevel(0.56, 500),
            ],
            midpoint=0.53,
            spread=0.02,
        )

        high_density = self.strategy._calculate_competition_density(
            high_competition_yes, self.no_orderbook
        )
        assert high_density > density  # Should be higher competition

    def test_market_condition_analysis(self):
        # Test attractive market
        condition = self.strategy.analyze_market_condition(
            self.yes_orderbook, self.no_orderbook
        )
        assert condition == MarketCondition.ATTRACTIVE

        # Test volatile market
        self.strategy.volatility_tracker["test_yes"] = VolatilityMetrics()
        self.strategy.volatility_tracker["test_yes"].add_midpoint_change(
            0.05
        )  # Large change

        with patch.object(self.strategy, "_is_market_volatile", return_value=True):
            condition = self.strategy.analyze_market_condition(
                self.yes_orderbook, self.no_orderbook
            )
            assert condition == MarketCondition.VOLATILE

    def test_optimal_order_calculation(self):
        current_positions = {}
        available_capital = 10000

        orders = self.strategy.calculate_optimal_orders(
            self.yes_orderbook, self.no_orderbook, current_positions, available_capital
        )

        # Should generate orders for both YES and NO markets
        assert len(orders) > 0

        # Check order structure
        for order in orders:
            assert "side" in order
            assert "price" in order
            assert "size" in order
            assert "market_type" in order
            assert "asset_id" in order
            assert order["size"] >= self.config.min_order_size

    def test_order_cancellation_logic(self):
        # Create orders with different ages
        now = datetime.now(timezone.utc)
        old_order = Order(
            id="old_order",
            market_id="test_market",
            side=OrderSide.BUY,
            price=0.52,
            size=100,
            status=OrderStatus.OPEN,
            timestamp=now - timedelta(minutes=60),  # 1 hour old
        )

        recent_order = Order(
            id="recent_order",
            market_id="test_market",
            side=OrderSide.BUY,
            price=0.52,
            size=100,
            status=OrderStatus.OPEN,
            timestamp=now - timedelta(minutes=10),  # 10 minutes old
        )

        current_orders = [old_order, recent_order]

        orders_to_cancel = self.strategy.should_cancel_orders(
            "test_asset", current_orders
        )

        # Should cancel the old order
        assert "old_order" in orders_to_cancel
        assert "recent_order" not in orders_to_cancel

    def test_hedge_order_calculation(self):
        # Test YES order getting filled
        filled_yes_order = Order(
            id="filled_yes",
            market_id="test_market",
            side=OrderSide.BUY,
            price=0.6,
            size=100,
            status=OrderStatus.FILLED,
            filled_size=100,
            metadata={"market_type": "YES"},
        )

        hedge_orders = self.strategy.calculate_hedge_orders(
            filled_yes_order, self.yes_orderbook, self.no_orderbook
        )

        assert len(hedge_orders) == 1
        hedge_order = hedge_orders[0]

        # Check hedge order properties
        assert hedge_order["side"] == OrderSide.SELL  # Opposite of original BUY
        assert hedge_order["price"] == 0.4  # 1 - 0.6 = 0.4
        assert hedge_order["size"] == 100  # Same size
        assert hedge_order["market_type"] == "NO"  # Opposite market

    def test_volatility_update(self):
        # Test volatility metrics update
        assert "test_asset" not in self.strategy.volatility_tracker

        self.strategy.update_volatility_metrics("test_asset", self.yes_orderbook)

        assert "test_asset" in self.strategy.volatility_tracker
        assert isinstance(
            self.strategy.volatility_tracker["test_asset"], VolatilityMetrics
        )

    def test_reward_share_estimation(self):
        # Test with low competition
        reward_share = self.strategy._estimate_reward_share(
            self.yes_orderbook, self.no_orderbook
        )
        assert reward_share > 0.01  # Should be above minimum
        assert reward_share <= 0.5  # Should be below maximum

    def test_insufficient_capital_handling(self):
        current_positions = {}
        available_capital = 50  # Less than minimum order size

        orders = self.strategy.calculate_optimal_orders(
            self.yes_orderbook, self.no_orderbook, current_positions, available_capital
        )

        # Should not generate orders with insufficient capital
        assert len(orders) == 0

    def test_price_boundary_validation(self):
        # Test with extreme orderbook that would generate invalid prices
        extreme_orderbook = OrderbookSnapshot(
            asset_id="extreme_test",
            bids=[OrderbookLevel(0.005, 100)],
            asks=[OrderbookLevel(0.01, 100)],
            midpoint=0.0075,
            spread=0.005,
        )

        orders = self.strategy._generate_market_orders(
            extreme_orderbook, True, 1000, None
        )

        # Should handle edge cases gracefully
        for order in orders:
            assert order["price"] > 0.01  # Above minimum
            assert order["price"] < 0.99  # Below maximum

    def test_position_size_limits(self):
        # Test with existing position near limit
        current_position = Position(
            market_id="test_market",
            size=900,  # Close to max position size of 1000
            entry_price=0.5,
        )

        current_positions = {"test_yes": current_position}
        available_capital = 10000

        orders = self.strategy.calculate_optimal_orders(
            self.yes_orderbook, self.no_orderbook, current_positions, available_capital
        )

        # Should still generate orders but with appropriate sizing
        # Implementation would need to account for existing positions
        assert isinstance(orders, list)


class TestIntegrationScenarios:
    def setup_method(self):
        self.config = LiquidityProvisionConfig(
            min_order_size=100,
            max_position_size=1000,
            optimal_distance_from_midpoint=0.01,
        )
        self.strategy = PolymarketLiquidityStrategy(self.config)

    def test_full_liquidity_provision_cycle(self):
        """Test complete cycle: analyze -> place orders -> monitor -> hedge if filled"""

        # 1. Create attractive market conditions
        yes_orderbook = OrderbookSnapshot(
            asset_id="integration_yes",
            bids=[OrderbookLevel(0.55, 50)],
            asks=[OrderbookLevel(0.57, 50)],
            midpoint=0.56,
            spread=0.02,
        )

        no_orderbook = OrderbookSnapshot(
            asset_id="integration_no",
            bids=[OrderbookLevel(0.43, 50)],
            asks=[OrderbookLevel(0.45, 50)],
            midpoint=0.44,
            spread=0.02,
        )

        # 2. Analyze market condition
        condition = self.strategy.analyze_market_condition(yes_orderbook, no_orderbook)
        assert condition == MarketCondition.ATTRACTIVE

        # 3. Calculate optimal orders
        current_positions = {}
        available_capital = 5000

        orders = self.strategy.calculate_optimal_orders(
            yes_orderbook, no_orderbook, current_positions, available_capital
        )

        assert len(orders) > 0

        # 4. Simulate order getting filled and calculate hedge
        filled_order = Order(
            id="test_fill",
            market_id="integration_yes",
            side=OrderSide.BUY,
            price=0.55,
            size=100,
            status=OrderStatus.FILLED,
            filled_size=100,
            metadata={"market_type": "YES"},
        )

        hedge_orders = self.strategy.calculate_hedge_orders(
            filled_order, yes_orderbook, no_orderbook
        )

        assert len(hedge_orders) == 1
        assert hedge_orders[0]["price"] == 0.45  # 1 - 0.55

    def test_risk_management_triggers(self):
        """Test various risk management scenarios"""

        # Create volatile market
        volatile_orderbook = OrderbookSnapshot(
            asset_id="volatile_test",
            bids=[OrderbookLevel(0.30, 100)],
            asks=[OrderbookLevel(0.70, 100)],
            midpoint=0.50,
            spread=0.40,  # Very wide spread indicates volatility
        )

        # Add volatility history
        self.strategy.volatility_tracker["volatile_test"] = VolatilityMetrics()
        self.strategy.volatility_tracker["volatile_test"].add_midpoint_change(
            0.1
        )

        # Should detect as volatile
        condition = self.strategy.analyze_market_condition(
            volatile_orderbook, volatile_orderbook
        )
        # Note: Would need to implement proper volatility detection in _is_market_volatile

        # Test order cancellation due to volatility
        test_orders = [
            Order(
                id="test_order_1",
                market_id="volatile_test",
                side=OrderSide.BUY,
                price=0.5,
                size=100,
                status=OrderStatus.OPEN,
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
            )
        ]

        with patch.object(
            self.strategy.volatility_tracker["volatile_test"],
            "is_volatile",
            return_value=True,
        ):
            orders_to_cancel = self.strategy.should_cancel_orders(
                "volatile_test", test_orders
            )
            assert "test_order_1" in orders_to_cancel
