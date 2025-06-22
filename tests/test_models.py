import pytest
from decimal import Decimal
from datetime import datetime, timezone
from src.models import Market, Order, Position, OrderSide, OrderStatus


class TestMarket:
    def test_market_creation(self):
        market = Market(
            id="market_1",
            name="Test Market",
            current_price=Decimal("100.50"),
            volume=Decimal("1000"),
            bid=Decimal("100.00"),
            ask=Decimal("101.00"),
        )
        assert market.id == "market_1"
        assert market.name == "Test Market"
        assert market.current_price == Decimal("100.50")
        assert market.volume == Decimal("1000")
        assert market.bid == Decimal("100.00")
        assert market.ask == Decimal("101.00")

    def test_market_validation_negative_price(self):
        with pytest.raises(ValueError, match="Current price cannot be negative"):
            Market(
                id="market_1",
                name="Test Market",
                current_price=Decimal("-100.50"),
                volume=Decimal("1000"),
                bid=Decimal("100.00"),
                ask=Decimal("101.00"),
            )

    def test_market_validation_negative_volume(self):
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            Market(
                id="market_1",
                name="Test Market",
                current_price=Decimal("100.50"),
                volume=Decimal("-1000"),
                bid=Decimal("100.00"),
                ask=Decimal("101.00"),
            )

    def test_market_validation_negative_bid(self):
        with pytest.raises(ValueError, match="Bid cannot be negative"):
            Market(
                id="market_1",
                name="Test Market",
                current_price=Decimal("100.50"),
                volume=Decimal("1000"),
                bid=Decimal("-100.00"),
                ask=Decimal("101.00"),
            )

    def test_market_validation_negative_ask(self):
        with pytest.raises(ValueError, match="Ask cannot be negative"):
            Market(
                id="market_1",
                name="Test Market",
                current_price=Decimal("100.50"),
                volume=Decimal("1000"),
                bid=Decimal("100.00"),
                ask=Decimal("-101.00"),
            )

    def test_market_validation_bid_greater_than_ask(self):
        with pytest.raises(ValueError, match="Bid cannot be greater than ask"):
            Market(
                id="market_1",
                name="Test Market",
                current_price=Decimal("100.50"),
                volume=Decimal("1000"),
                bid=Decimal("102.00"),
                ask=Decimal("101.00"),
            )

    def test_market_spread(self):
        market = Market(
            id="market_1",
            name="Test Market",
            current_price=Decimal("100.50"),
            volume=Decimal("1000"),
            bid=Decimal("100.00"),
            ask=Decimal("101.00"),
        )
        assert market.spread() == Decimal("1.00")

    def test_market_serialization(self):
        market = Market(
            id="market_1",
            name="Test Market",
            current_price=Decimal("100.50"),
            volume=Decimal("1000"),
            bid=Decimal("100.00"),
            ask=Decimal("101.00"),
            metadata={"exchange": "polymarket"},
        )

        # Test to_dict
        market_dict = market.to_dict()
        expected_dict = {
            "id": "market_1",
            "name": "Test Market",
            "current_price": "100.50",
            "volume": "1000",
            "bid": "100.00",
            "ask": "101.00",
            "metadata": {"exchange": "polymarket"},
        }
        assert market_dict == expected_dict

        # Test from_dict
        reconstructed_market = Market.from_dict(market_dict)
        assert reconstructed_market.id == market.id
        assert reconstructed_market.name == market.name
        assert reconstructed_market.current_price == market.current_price
        assert reconstructed_market.volume == market.volume
        assert reconstructed_market.bid == market.bid
        assert reconstructed_market.ask == market.ask
        assert reconstructed_market.metadata == market.metadata

    def test_market_json_serialization(self):
        market = Market(
            id="market_1",
            name="Test Market",
            current_price=Decimal("100.50"),
            volume=Decimal("1000"),
            bid=Decimal("100.00"),
            ask=Decimal("101.00"),
        )

        # Test JSON round-trip
        json_str = market.to_json()
        reconstructed_market = Market.from_json(json_str)

        assert reconstructed_market.id == market.id
        assert reconstructed_market.name == market.name
        assert reconstructed_market.current_price == market.current_price
        assert reconstructed_market.volume == market.volume
        assert reconstructed_market.bid == market.bid
        assert reconstructed_market.ask == market.ask


class TestOrder:
    def test_order_creation(self):
        timestamp = datetime.now(timezone.utc)
        order = Order(
            id="order_1",
            market_id="market_1",
            side=OrderSide.BUY,
            price=Decimal("100.50"),
            size=Decimal("10"),
            status=OrderStatus.OPEN,
            timestamp=timestamp,
        )
        assert order.id == "order_1"
        assert order.market_id == "market_1"
        assert order.side == OrderSide.BUY
        assert order.price == Decimal("100.50")
        assert order.size == Decimal("10")
        assert order.status == OrderStatus.OPEN
        assert order.timestamp == timestamp
        assert order.filled_size == Decimal("0")

    def test_order_validation_zero_price(self):
        with pytest.raises(ValueError, match="Price must be positive"):
            Order(
                id="order_1",
                market_id="market_1",
                side=OrderSide.BUY,
                price=Decimal("0"),
                size=Decimal("10"),
                status=OrderStatus.OPEN,
            )

    def test_order_validation_negative_price(self):
        with pytest.raises(ValueError, match="Price must be positive"):
            Order(
                id="order_1",
                market_id="market_1",
                side=OrderSide.BUY,
                price=Decimal("-100.50"),
                size=Decimal("10"),
                status=OrderStatus.OPEN,
            )

    def test_order_validation_zero_size(self):
        with pytest.raises(ValueError, match="Size must be positive"):
            Order(
                id="order_1",
                market_id="market_1",
                side=OrderSide.BUY,
                price=Decimal("100.50"),
                size=Decimal("0"),
                status=OrderStatus.OPEN,
            )

    def test_order_validation_negative_filled_size(self):
        with pytest.raises(ValueError, match="Filled size cannot be negative"):
            Order(
                id="order_1",
                market_id="market_1",
                side=OrderSide.BUY,
                price=Decimal("100.50"),
                size=Decimal("10"),
                status=OrderStatus.OPEN,
                filled_size=Decimal("-5"),
            )

    def test_order_validation_filled_size_exceeds_size(self):
        with pytest.raises(ValueError, match="Filled size cannot exceed order size"):
            Order(
                id="order_1",
                market_id="market_1",
                side=OrderSide.BUY,
                price=Decimal("100.50"),
                size=Decimal("10"),
                status=OrderStatus.OPEN,
                filled_size=Decimal("15"),
            )

    def test_order_remaining_size(self):
        order = Order(
            id="order_1",
            market_id="market_1",
            side=OrderSide.BUY,
            price=Decimal("100.50"),
            size=Decimal("10"),
            status=OrderStatus.PARTIALLY_FILLED,
            filled_size=Decimal("3"),
        )
        assert order.remaining_size() == Decimal("7")

    def test_order_is_filled(self):
        # Not filled
        order = Order(
            id="order_1",
            market_id="market_1",
            side=OrderSide.BUY,
            price=Decimal("100.50"),
            size=Decimal("10"),
            status=OrderStatus.OPEN,
            filled_size=Decimal("0"),
        )
        assert not order.is_filled()

        # Fully filled
        order.filled_size = Decimal("10")
        assert order.is_filled()

    def test_order_fill_percentage(self):
        order = Order(
            id="order_1",
            market_id="market_1",
            side=OrderSide.BUY,
            price=Decimal("100.50"),
            size=Decimal("10"),
            status=OrderStatus.PARTIALLY_FILLED,
            filled_size=Decimal("3"),
        )
        assert order.fill_percentage() == Decimal("30")

        # Edge case: zero size
        order.size = Decimal("0")
        order.filled_size = Decimal("0")
        assert order.fill_percentage() == Decimal("0")

    def test_order_serialization(self):
        timestamp = datetime.now(timezone.utc)
        order = Order(
            id="order_1",
            market_id="market_1",
            side=OrderSide.SELL,
            price=Decimal("100.50"),
            size=Decimal("10"),
            status=OrderStatus.FILLED,
            timestamp=timestamp,
            filled_size=Decimal("10"),
            metadata={"broker": "test"},
        )

        # Test to_dict
        order_dict = order.to_dict()
        expected_dict = {
            "id": "order_1",
            "market_id": "market_1",
            "side": "sell",
            "price": "100.50",
            "size": "10",
            "status": "filled",
            "timestamp": timestamp.isoformat(),
            "filled_size": "10",
            "metadata": {"broker": "test"},
        }
        assert order_dict == expected_dict

        # Test from_dict
        reconstructed_order = Order.from_dict(order_dict)
        assert reconstructed_order.id == order.id
        assert reconstructed_order.market_id == order.market_id
        assert reconstructed_order.side == order.side
        assert reconstructed_order.price == order.price
        assert reconstructed_order.size == order.size
        assert reconstructed_order.status == order.status
        assert reconstructed_order.timestamp == order.timestamp
        assert reconstructed_order.filled_size == order.filled_size
        assert reconstructed_order.metadata == order.metadata

    def test_order_json_serialization(self):
        order = Order(
            id="order_1",
            market_id="market_1",
            side=OrderSide.BUY,
            price=Decimal("100.50"),
            size=Decimal("10"),
            status=OrderStatus.OPEN,
        )

        # Test JSON round-trip
        json_str = order.to_json()
        reconstructed_order = Order.from_json(json_str)

        assert reconstructed_order.id == order.id
        assert reconstructed_order.market_id == order.market_id
        assert reconstructed_order.side == order.side
        assert reconstructed_order.price == order.price
        assert reconstructed_order.size == order.size
        assert reconstructed_order.status == order.status
        assert reconstructed_order.filled_size == order.filled_size


class TestPosition:
    def test_position_creation(self):
        position = Position(
            market_id="market_1", size=Decimal("10"), entry_price=Decimal("100.50")
        )
        assert position.market_id == "market_1"
        assert position.size == Decimal("10")
        assert position.entry_price == Decimal("100.50")

    def test_position_validation_zero_entry_price(self):
        with pytest.raises(ValueError, match="Entry price must be positive"):
            Position(market_id="market_1", size=Decimal("10"), entry_price=Decimal("0"))

    def test_position_validation_negative_entry_price(self):
        with pytest.raises(ValueError, match="Entry price must be positive"):
            Position(
                market_id="market_1", size=Decimal("10"), entry_price=Decimal("-100.50")
            )

    def test_position_current_value(self):
        position = Position(
            market_id="market_1", size=Decimal("10"), entry_price=Decimal("100.50")
        )
        current_price = Decimal("105.00")
        assert position.current_value(current_price) == Decimal("1050.00")

        # Test with negative position (short)
        position.size = Decimal("-10")
        assert position.current_value(current_price) == Decimal("1050.00")

    def test_position_pnl_long(self):
        # Long position
        position = Position(
            market_id="market_1", size=Decimal("10"), entry_price=Decimal("100.00")
        )

        # Price goes up - profit
        current_price = Decimal("105.00")
        assert position.pnl(current_price) == Decimal("50.00")

        # Price goes down - loss
        current_price = Decimal("95.00")
        assert position.pnl(current_price) == Decimal("-50.00")

    def test_position_pnl_short(self):
        # Short position
        position = Position(
            market_id="market_1", size=Decimal("-10"), entry_price=Decimal("100.00")
        )

        # Price goes down - profit
        current_price = Decimal("95.00")
        assert position.pnl(current_price) == Decimal("50.00")

        # Price goes up - loss
        current_price = Decimal("105.00")
        assert position.pnl(current_price) == Decimal("-50.00")

    def test_position_pnl_zero_size(self):
        position = Position(
            market_id="market_1", size=Decimal("0"), entry_price=Decimal("100.00")
        )
        current_price = Decimal("105.00")
        assert position.pnl(current_price) == Decimal("0")

    def test_position_pnl_percentage(self):
        position = Position(
            market_id="market_1", size=Decimal("10"), entry_price=Decimal("100.00")
        )

        # 5% gain
        current_price = Decimal("105.00")
        assert position.pnl_percentage(current_price) == Decimal("5.00")

        # 10% loss
        current_price = Decimal("90.00")
        assert position.pnl_percentage(current_price) == Decimal("-10.00")

        # Edge case: zero entry price
        position.entry_price = Decimal("0")
        assert position.pnl_percentage(current_price) == Decimal("0")

    def test_position_types(self):
        # Long position
        long_position = Position(
            market_id="market_1", size=Decimal("10"), entry_price=Decimal("100.00")
        )
        assert long_position.is_long()
        assert not long_position.is_short()
        assert not long_position.is_flat()

        # Short position
        short_position = Position(
            market_id="market_1", size=Decimal("-10"), entry_price=Decimal("100.00")
        )
        assert not short_position.is_long()
        assert short_position.is_short()
        assert not short_position.is_flat()

        # Flat position
        flat_position = Position(
            market_id="market_1", size=Decimal("0"), entry_price=Decimal("100.00")
        )
        assert not flat_position.is_long()
        assert not flat_position.is_short()
        assert flat_position.is_flat()

    def test_position_serialization(self):
        position = Position(
            market_id="market_1",
            size=Decimal("-5.5"),
            entry_price=Decimal("100.50"),
            metadata={"strategy": "momentum"},
        )

        # Test to_dict
        position_dict = position.to_dict()
        expected_dict = {
            "market_id": "market_1",
            "size": "-5.5",
            "entry_price": "100.50",
            "metadata": {"strategy": "momentum"},
        }
        assert position_dict == expected_dict

        # Test from_dict
        reconstructed_position = Position.from_dict(position_dict)
        assert reconstructed_position.market_id == position.market_id
        assert reconstructed_position.size == position.size
        assert reconstructed_position.entry_price == position.entry_price
        assert reconstructed_position.metadata == position.metadata

    def test_position_json_serialization(self):
        position = Position(
            market_id="market_1", size=Decimal("10"), entry_price=Decimal("100.50")
        )

        # Test JSON round-trip
        json_str = position.to_json()
        reconstructed_position = Position.from_json(json_str)

        assert reconstructed_position.market_id == position.market_id
        assert reconstructed_position.size == position.size
        assert reconstructed_position.entry_price == position.entry_price
