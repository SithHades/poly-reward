import pytest
from unittest.mock import Mock
from datetime import datetime, timezone, timedelta

from src.order_manager import (
    OrderManager,
    ManagedOrder,
    OrderMetrics,
    OrderLifecycleEvent,
)
from src.models import Order, Position, OrderSide, OrderStatus
from src.client import Client
from src.strategy import (
    PolymarketLiquidityStrategy,
    OrderbookSnapshot,
    OrderbookLevel,
)
from py_clob_client.clob_types import OrderType


class TestOrderManager:
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_client = Mock(spec=Client)
        self.mock_strategy = Mock(spec=PolymarketLiquidityStrategy)

        # Setup mock client responses
        self.mock_client.paper_trading = False
        self.mock_client.place_order.return_value = {
            "id": "test_order_123",
            "status": "placed",
        }
        self.mock_client.cancel_order.return_value = {"status": "cancelled"}
        self.mock_client.get_order_status.return_value = {
            "status": "open",
            "size_matched": "0",
        }

        # Setup mock strategy responses
        self.mock_strategy.should_cancel_orders.return_value = []
        self.mock_strategy.calculate_optimal_orders.return_value = []

        self.order_manager = OrderManager(
            client=self.mock_client,
            strategy=self.mock_strategy,
            max_active_orders=10,
            status_check_interval=5,
        )

    def test_initialization(self):
        """Test OrderManager initialization"""
        assert self.order_manager.client == self.mock_client
        assert self.order_manager.strategy == self.mock_strategy
        assert self.order_manager.max_active_orders == 10
        assert self.order_manager.status_check_interval == 5
        assert len(self.order_manager.active_orders) == 0
        assert len(self.order_manager.completed_orders) == 0
        assert not self.order_manager._running

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping order manager"""
        # Test start
        await self.order_manager.start()
        assert self.order_manager._running
        assert self.order_manager._status_check_task is not None

        # Test stop
        await self.order_manager.stop()
        assert not self.order_manager._running

    @pytest.mark.asyncio
    async def test_place_order_success(self):
        """Test successful order placement"""
        order_id = await self.order_manager.place_order(
            price=0.55,
            size=100,
            side=OrderSide.BUY,
            token_id="token_123",
        )

        assert order_id == "test_order_123"
        assert len(self.order_manager.active_orders) == 1
        assert self.order_manager.order_metrics.total_orders_placed == 1

        # Verify client was called correctly
        self.mock_client.place_order.assert_called_once_with(
            price=0.55,
            size=100.0,
            side="buy",
            token_id="token_123",
            order_type=OrderType.GTC,
        )

    @pytest.mark.asyncio
    async def test_place_order_validation_failures(self):
        """Test order placement validation"""
        # Test negative price
        order_id = await self.order_manager.place_order(
            price=-0.10,
            size=100,
            side=OrderSide.BUY,
            token_id="token_123",
        )
        assert order_id is None

        # Test zero size
        order_id = await self.order_manager.place_order(
            price=0.55,
            size=0,
            side=OrderSide.BUY,
            token_id="token_123",
        )
        assert order_id is None

        # Test price too high
        order_id = await self.order_manager.place_order(
            price=1.50,
            size=100,
            side=OrderSide.BUY,
            token_id="token_123",
        )
        assert order_id is None

        # Test empty token_id
        order_id = await self.order_manager.place_order(
            price=0.55, size=100, side=OrderSide.BUY, token_id=""
        )
        assert order_id is None

    @pytest.mark.asyncio
    async def test_place_order_max_orders_limit(self):
        """Test maximum active orders limit"""
        # Fill up to max orders
        for i in range(10):
            self.mock_client.place_order.return_value = {
                "id": f"order_{i}",
                "status": "placed",
            }
            order_id = await self.order_manager.place_order(
                price=0.55,
                size=100,
                side=OrderSide.BUY,
                token_id="token_123",
            )
            assert order_id == f"order_{i}"

        # Try to place one more - should fail
        self.mock_client.place_order.return_value = {
            "id": "order_overflow",
            "status": "placed",
        }
        order_id = await self.order_manager.place_order(
            price=0.55,
            size=100,
            side=OrderSide.BUY,
            token_id="token_123",
        )
        assert order_id is None
        assert len(self.order_manager.active_orders) == 10

    @pytest.mark.asyncio
    async def test_place_order_client_failure(self):
        """Test order placement when client fails"""
        self.mock_client.place_order.return_value = {"error": "API Error"}

        order_id = await self.order_manager.place_order(
            price=0.55,
            size=100,
            side=OrderSide.BUY,
            token_id="token_123",
        )

        assert order_id is None
        assert len(self.order_manager.active_orders) == 0
        assert self.order_manager.order_metrics.total_orders_placed == 0

    @pytest.mark.asyncio
    async def test_place_order_paper_trading(self):
        """Test order placement in paper trading mode"""
        self.mock_client.paper_trading = True
        self.mock_client.place_order.return_value = {
            "status": "paper",
            "price": 0.55,
            "size": 100,
            "side": "buy",
            "token_id": "token_123",
        }

        order_id = await self.order_manager.place_order(
            price=0.55,
            size=100,
            side=OrderSide.BUY,
            token_id="token_123",
        )

        assert order_id is not None
        assert order_id.startswith("paper_")
        assert len(self.order_manager.active_orders) == 1

    @pytest.mark.asyncio
    async def test_cancel_order_success(self):
        """Test successful order cancellation"""
        # First place an order
        order_id = await self.order_manager.place_order(
            price=0.55,
            size=100,
            side=OrderSide.BUY,
            token_id="token_123",
        )
        assert len(self.order_manager.active_orders) == 1

        # Cancel the order
        result = await self.order_manager.cancel_order(order_id, "test_cancellation")

        assert result is True
        assert len(self.order_manager.active_orders) == 0
        assert len(self.order_manager.completed_orders) == 1
        assert self.order_manager.order_metrics.total_orders_cancelled == 1

        # Verify client was called
        self.mock_client.cancel_order.assert_called_once_with(order_id)

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self):
        """Test cancelling non-existent order"""
        result = await self.order_manager.cancel_order("nonexistent_order", "test")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_order_client_failure(self):
        """Test order cancellation when client fails"""
        # Place an order first
        order_id = await self.order_manager.place_order(
            price=0.55,
            size=100,
            side=OrderSide.BUY,
            token_id="token_123",
        )

        # Mock client failure
        self.mock_client.cancel_order.return_value = {"error": "Cancellation failed"}

        result = await self.order_manager.cancel_order(order_id, "test")
        assert result is False
        assert len(self.order_manager.active_orders) == 1  # Order still active

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self):
        """Test cancelling all active orders"""
        # Place multiple orders
        order_ids = []
        for i in range(3):
            self.mock_client.place_order.return_value = {
                "id": f"order_{i}",
                "status": "placed",
            }
            order_id = await self.order_manager.place_order(
                price=0.55,
                size=100,
                side=OrderSide.BUY,
                token_id="token_123",
            )
            order_ids.append(order_id)

        assert len(self.order_manager.active_orders) == 3

        # Cancel all orders
        cancelled_count = await self.order_manager.cancel_all_orders(
            "mass_cancellation"
        )

        assert cancelled_count == 3
        assert len(self.order_manager.active_orders) == 0
        assert len(self.order_manager.completed_orders) == 3

    @pytest.mark.asyncio
    async def test_update_order_status_to_filled(self):
        """Test updating order status to filled"""
        # Place an order
        order_id = await self.order_manager.place_order(
            price=0.55,
            size=100,
            side=OrderSide.BUY,
            token_id="token_123",
        )

        # Mock status response as filled
        self.mock_client.get_order_status.return_value = {
            "status": "matched",
            "size_matched": "100",
        }

        result = await self.order_manager.update_order_status(order_id)

        assert result is True
        assert len(self.order_manager.active_orders) == 0
        assert len(self.order_manager.completed_orders) == 1
        assert self.order_manager.order_metrics.total_orders_filled == 1
        assert self.order_manager.order_metrics.total_volume_filled == 100

    @pytest.mark.asyncio
    async def test_update_order_status_partially_filled(self):
        """Test updating order status to partially filled"""
        # Place an order
        order_id = await self.order_manager.place_order(
            price=0.55,
            size=100,
            side=OrderSide.BUY,
            token_id="token_123",
        )

        # Mock status response as partially filled
        self.mock_client.get_order_status.return_value = {
            "status": "open",
            "size_matched": "50",
        }

        result = await self.order_manager.update_order_status(order_id)

        assert result is True
        assert len(self.order_manager.active_orders) == 1  # Still active

        managed_order = list(self.order_manager.active_orders.values())[0]
        assert managed_order.order.status == OrderStatus.PARTIALLY_FILLED
        assert managed_order.order.filled_size == 50

    @pytest.mark.asyncio
    async def test_update_order_status_client_error(self):
        """Test updating order status when client returns error"""
        # Place an order
        order_id = await self.order_manager.place_order(
            price=0.55,
            size=100,
            side=OrderSide.BUY,
            token_id="token_123",
        )

        # Mock client error
        self.mock_client.get_order_status.return_value = {"error": "Order not found"}

        result = await self.order_manager.update_order_status(order_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_manage_order_lifecycle(self):
        """Test complete order lifecycle management"""
        # Setup mock strategy to return new orders
        mock_orderbook = OrderbookSnapshot(
            asset_id="token_123",
            bids=[OrderbookLevel(0.52, 100)],
            asks=[OrderbookLevel(0.54, 100)],
            midpoint=0.53,
            spread=0.02,
        )

        self.mock_strategy.calculate_optimal_orders.return_value = [
            {
                "price": 0.52,
                "size": 100,
                "side": OrderSide.BUY,
                "asset_id": "token_123",
                "market_type": "YES",
            }
        ]

        market_data = {
            "asset_id": "token_123",
            "yes_orderbook": mock_orderbook,
            "no_orderbook": mock_orderbook,
            "available_capital": 1000,
        }

        summary = await self.order_manager.manage_order_lifecycle(market_data)

        assert summary["orders_placed"] == 1
        assert len(summary["errors"]) == 0
        assert len(self.order_manager.active_orders) == 1

    @pytest.mark.asyncio
    async def test_manage_order_lifecycle_with_cancellations(self):
        """Test lifecycle management with risk-based cancellations"""
        # Place an order first
        order_id = await self.order_manager.place_order(
            price=0.55,
            size=100,
            side=OrderSide.BUY,
            token_id="token_123",
        )

        # Setup strategy to recommend cancellation
        self.mock_strategy.should_cancel_orders.return_value = [order_id]

        market_data = {
            "asset_id": "token_123",
            "yes_orderbook": None,
            "no_orderbook": None,
        }

        summary = await self.order_manager.manage_order_lifecycle(market_data)

        assert summary["orders_cancelled"] == 1
        assert len(self.order_manager.active_orders) == 0

    def test_get_order_status_summary(self):
        """Test getting order status summary"""
        # Create some test data
        self.order_manager.order_metrics.total_orders_placed = 5
        self.order_manager.order_metrics.total_orders_filled = 3
        self.order_manager.order_metrics.total_orders_cancelled = 1
        self.order_manager.order_metrics.total_volume_filled = 300
        self.order_manager.order_metrics.update_fill_rate()

        # Add a position
        self.order_manager.positions["token_123"] = Position(
            market_id="token_123", size=100, entry_price=0.55
        )

        summary = self.order_manager.get_order_status_summary()

        assert summary["active_orders"] == 0
        assert summary["completed_orders"] == 0
        assert summary["metrics"]["total_placed"] == 5
        assert summary["metrics"]["total_filled"] == 3
        assert summary["metrics"]["fill_rate"] == 0.6
        assert "token_123" in summary["positions"]

    def test_get_active_orders(self):
        """Test getting active orders list"""
        # Mock an active order
        mock_order = Order(
            id="test_order",
            market_id="token_123",
            side=OrderSide.BUY,
            price=0.55,
            size=100,
            status=OrderStatus.OPEN,
            filled_size=50,
        )

        managed_order = ManagedOrder(
            order=mock_order,
            placement_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            last_status_check=datetime.now(timezone.utc),
        )

        self.order_manager.active_orders["test_order"] = managed_order

        active_orders = self.order_manager.get_active_orders()

        assert len(active_orders) == 1
        order_data = active_orders[0]
        assert order_data["id"] == "test_order"
        assert order_data["side"] == "buy"
        assert order_data["price"] == 0.55
        assert order_data["size"] == 100.0
        assert order_data["filled_size"] == 50.0
        assert order_data["age_minutes"] == pytest.approx(5, abs=1)

    def test_validate_order(self):
        """Test order validation"""
        # Valid order
        assert (
            self.order_manager._validate_order(
                0.55, 100, OrderSide.BUY, "token_123"
            )
            is True
        )

        # Invalid orders
        assert (
            self.order_manager._validate_order(
                0, 100, OrderSide.BUY, "token_123"
            )
            is False
        )

        assert (
            self.order_manager._validate_order(
                0.55, 0, OrderSide.BUY, "token_123"
            )
            is False
        )

        assert (
            self.order_manager._validate_order(
                1.50, 100, OrderSide.BUY, "token_123"
            )
            is False
        )

        assert (
            self.order_manager._validate_order(
                0.55, 100, OrderSide.BUY, ""
            )
            is False
        )

    def test_update_position_buy_order(self):
        """Test position update for buy order"""
        order = Order(
            id="test_order",
            market_id="token_123",
            side=OrderSide.BUY,
            price=0.55,
            size=100,
            status=OrderStatus.FILLED,
            filled_size=100,
        )

        self.order_manager._update_position(order)

        assert "token_123" in self.order_manager.positions
        position = self.order_manager.positions["token_123"]
        assert position.size == 100
        assert position.entry_price == 0.55

    def test_update_position_sell_order(self):
        """Test position update for sell order"""
        # First create a long position
        buy_order = Order(
            id="buy_order",
            market_id="token_123",
            side=OrderSide.BUY,
            price=0.55,
            size=150,
            status=OrderStatus.FILLED,
            filled_size=150,
        )
        self.order_manager._update_position(buy_order)

        # Then sell some
        sell_order = Order(
            id="sell_order",
            market_id="token_123",
            side=OrderSide.SELL,
            price=0.60,
            size=50,
            status=OrderStatus.FILLED,
            filled_size=50,
        )
        self.order_manager._update_position(sell_order)

        position = self.order_manager.positions["token_123"]
        assert position.size == 100  # 150 - 50


class TestManagedOrder:
    def test_managed_order_creation(self):
        """Test ManagedOrder creation and lifecycle events"""
        order = Order(
            id="test_order",
            market_id="token_123",
            side=OrderSide.BUY,
            price=0.55,
            size=100,
            status=OrderStatus.PENDING,
        )

        managed_order = ManagedOrder(
            order=order,
            placement_time=datetime.now(timezone.utc),
            last_status_check=datetime.now(timezone.utc),
        )

        assert managed_order.order == order
        assert managed_order.retry_count == 0
        assert not managed_order.hedge_pending
        assert len(managed_order.lifecycle_events) == 0

        # Add lifecycle event
        managed_order.add_lifecycle_event(
            OrderLifecycleEvent.PLACED, "Order placed successfully"
        )

        assert len(managed_order.lifecycle_events) == 1
        event_time, event_type, details = managed_order.lifecycle_events[0]
        assert event_type == OrderLifecycleEvent.PLACED
        assert details == "Order placed successfully"


class TestOrderMetrics:
    def test_order_metrics_initialization(self):
        """Test OrderMetrics initialization"""
        metrics = OrderMetrics()

        assert metrics.total_orders_placed == 0
        assert metrics.total_orders_filled == 0
        assert metrics.total_orders_cancelled == 0
        assert metrics.total_volume_filled == 0
        assert metrics.fill_rate == 0

    def test_fill_rate_calculation(self):
        """Test fill rate calculation"""
        metrics = OrderMetrics()
        metrics.total_orders_placed = 10
        metrics.total_orders_filled = 7

        metrics.update_fill_rate()

        assert metrics.fill_rate == 0.7

    def test_fill_rate_zero_orders(self):
        """Test fill rate when no orders placed"""
        metrics = OrderMetrics()
        metrics.update_fill_rate()

        assert metrics.fill_rate == 0
