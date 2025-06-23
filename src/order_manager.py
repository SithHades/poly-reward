from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import logging
import asyncio
from enum import Enum

from src.models import Order, Position, OrderSide, OrderStatus
from src.client import Client
from src.strategy import (
    PolymarketLiquidityStrategy,
)
from py_clob_client.clob_types import OrderType


class OrderLifecycleEvent(Enum):
    """Events in order lifecycle"""

    PLACED = "placed"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class OrderMetrics:
    """Track metrics for order performance"""

    total_orders_placed: int = 0
    total_orders_filled: int = 0
    total_orders_cancelled: int = 0
    total_volume_filled: Decimal = field(default=Decimal("0"))
    fill_rate: Decimal = field(default=Decimal("0"))
    avg_fill_time: timedelta = field(default=timedelta())

    def update_fill_rate(self):
        """Update fill rate calculation"""
        if self.total_orders_placed > 0:
            self.fill_rate = Decimal(self.total_orders_filled) / Decimal(
                self.total_orders_placed
            )


@dataclass
class ManagedOrder:
    """Extended order with management metadata"""

    order: Order
    placement_time: datetime
    last_status_check: datetime
    retry_count: int = 0
    hedge_pending: bool = False
    lifecycle_events: List[Tuple[datetime, OrderLifecycleEvent, str]] = field(
        default_factory=list
    )

    def add_lifecycle_event(self, event: OrderLifecycleEvent, details: str = ""):
        """Add lifecycle event with timestamp"""
        self.lifecycle_events.append((datetime.now(timezone.utc), event, details))


class OrderManager:
    """
    Complete order lifecycle management system for Polymarket liquidity provision

    Manages order placement, monitoring, cancellation, and hedging with proper state tracking.
    Integrates with PolymarketClient for execution and Strategy for order generation.
    """

    def __init__(
        self,
        client: Client,
        strategy: PolymarketLiquidityStrategy,
        max_active_orders: int = 50,
        status_check_interval: int = 30,
        enable_auto_hedging: bool = True,
    ):
        self.client = client
        self.strategy = strategy
        self.max_active_orders = max_active_orders
        self.status_check_interval = status_check_interval
        self.enable_auto_hedging = enable_auto_hedging

        self.logger = logging.getLogger("OrderManager")

        # Order tracking
        self.active_orders: Dict[str, ManagedOrder] = {}
        self.completed_orders: Dict[str, ManagedOrder] = {}
        self.order_metrics = OrderMetrics()

        # Position tracking
        self.positions: Dict[str, Position] = {}

        # State management
        self._running = False
        self._status_check_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the order manager background tasks"""
        if self._running:
            self.logger.warning("OrderManager already running")
            return

        self._running = True
        self.logger.info("Starting OrderManager")

        # Start background status monitoring
        self._status_check_task = asyncio.create_task(self._status_monitor())

    async def stop(self):
        """Stop the order manager and cleanup"""
        if not self._running:
            return

        self.logger.info("Stopping OrderManager")
        self._running = False

        if self._status_check_task:
            self._status_check_task.cancel()
            try:
                await self._status_check_task
            except asyncio.CancelledError:
                pass

        # Cancel all active orders
        await self.cancel_all_orders("shutdown")

    async def place_order(
        self,
        price: Decimal,
        size: Decimal,
        side: OrderSide,
        token_id: str,
        order_type: OrderType = OrderType.GTC,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Place a new order with validation and tracking

        Args:
            price: Order price
            size: Order size
            side: Buy or sell
            token_id: Market token ID
            order_type: Order type (default GTC)
            metadata: Additional order metadata

        Returns:
            Order ID if successful, None if failed
        """
        if len(self.active_orders) >= self.max_active_orders:
            self.logger.warning(
                f"Maximum active orders reached: {self.max_active_orders}"
            )
            return None

        # Validate order
        if not self._validate_order(price, size, side, token_id):
            return None

        try:
            # Place order through client
            self.logger.info(
                f"Placing order: {side.value} {size} at {price} for token {token_id}"
            )

            order_id = self.client.place_order(
                price=float(price),
                size=float(size),
                side=side.value,
                token_id=token_id,
                order_type=order_type,
            )

            if not order_id:
                self.logger.error(f"Order placement failed: {order_id}")
                return None

            order = Order(
                id=order_id,
                market_id=token_id,
                side=side,
                price=price,
                size=size,
                status=OrderStatus.PENDING,
                metadata=metadata or {},
            )

            # Create managed order
            managed_order = ManagedOrder(
                order=order,
                placement_time=datetime.now(timezone.utc),
                last_status_check=datetime.now(timezone.utc),
            )
            managed_order.add_lifecycle_event(
                OrderLifecycleEvent.PLACED, f"Placed {side.value} {size} at {price}"
            )

            # Track order
            self.active_orders[order_id] = managed_order
            self.order_metrics.total_orders_placed += 1

            self.logger.info(f"Order placed successfully: {order_id}")
            return order_id

        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None

    async def cancel_order(self, order_id: str, reason: str = "") -> bool:
        """
        Cancel a specific order

        Args:
            order_id: Order ID to cancel
            reason: Reason for cancellation

        Returns:
            True if cancellation successful
        """
        if order_id not in self.active_orders:
            self.logger.warning(f"Order {order_id} not found in active orders")
            return False

        managed_order = self.active_orders[order_id]

        try:
            self.logger.info(f"Cancelling order {order_id}: {reason}")

            # Cancel through client
            response = self.client.cancel_order(order_id)

            if response and "error" not in response:
                # Update order status
                managed_order.order.status = OrderStatus.CANCELLED
                managed_order.add_lifecycle_event(OrderLifecycleEvent.CANCELLED, reason)

                # Move to completed orders
                self.completed_orders[order_id] = managed_order
                del self.active_orders[order_id]

                self.order_metrics.total_orders_cancelled += 1
                self.order_metrics.update_fill_rate()

                self.logger.info(f"Order {order_id} cancelled successfully")
                return True
            else:
                self.logger.error(f"Failed to cancel order {order_id}: {response}")
                return False

        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def cancel_all_orders(self, reason: str = "") -> int:
        """
        Cancel all active orders

        Args:
            reason: Reason for mass cancellation

        Returns:
            Number of orders cancelled
        """
        order_ids = list(self.active_orders.keys())
        cancelled_count = 0

        self.logger.info(f"Cancelling {len(order_ids)} active orders: {reason}")

        for order_id in order_ids:
            if await self.cancel_order(order_id, reason):
                cancelled_count += 1

        self.logger.info(f"Cancelled {cancelled_count}/{len(order_ids)} orders")
        return cancelled_count

    async def update_order_status(self, order_id: str) -> bool:
        """
        Update status of a specific order

        Args:
            order_id: Order ID to update

        Returns:
            True if status updated successfully
        """
        if order_id not in self.active_orders:
            return False

        managed_order = self.active_orders[order_id]

        try:
            # Get current status from client
            status_response = self.client.get_order_status(order_id)

            if not status_response or "error" in status_response:
                return False

            # Parse status
            if self.client.paper_trading:
                # Paper trading simulation
                current_status = OrderStatus.OPEN
                filled_size = Decimal("0")
            else:
                # Real status from API
                status_str = status_response.get("status", "").lower()
                filled_size = Decimal(str(status_response.get("size_matched", "0")))

                if status_str == "matched":
                    current_status = OrderStatus.FILLED
                elif status_str == "cancelled":
                    current_status = OrderStatus.CANCELLED
                elif filled_size > 0 and filled_size < managed_order.order.size:
                    current_status = OrderStatus.PARTIALLY_FILLED
                elif status_str == "open":
                    current_status = OrderStatus.OPEN
                else:
                    current_status = OrderStatus.OPEN

            # Update order if status changed
            old_status = managed_order.order.status
            if (
                current_status != old_status
                or filled_size != managed_order.order.filled_size
            ):
                managed_order.order.status = current_status
                managed_order.order.filled_size = filled_size
                managed_order.last_status_check = datetime.now(timezone.utc)

                # Log status change
                if current_status != old_status:
                    event = (
                        OrderLifecycleEvent.FILLED
                        if current_status == OrderStatus.FILLED
                        else OrderLifecycleEvent.PARTIALLY_FILLED
                        if current_status == OrderStatus.PARTIALLY_FILLED
                        else OrderLifecycleEvent.CANCELLED
                    )

                    managed_order.add_lifecycle_event(
                        event, f"Status: {old_status.value} -> {current_status.value}"
                    )
                    self.logger.info(
                        f"Order {order_id} status changed: {old_status.value} -> {current_status.value}"
                    )

                # Handle filled orders
                if current_status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                    await self._handle_order_fill(managed_order)

                # Move completed orders
                if current_status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                    self.completed_orders[order_id] = managed_order
                    del self.active_orders[order_id]

                    if current_status == OrderStatus.FILLED:
                        self.order_metrics.total_orders_filled += 1
                        self.order_metrics.total_volume_filled += filled_size

                    self.order_metrics.update_fill_rate()

            return True

        except Exception as e:
            self.logger.error(f"Error updating order status for {order_id}: {e}")
            return False

    async def manage_order_lifecycle(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Complete order lifecycle management for a market

        Args:
            market_data: Market data including orderbooks and positions

        Returns:
            Summary of actions taken
        """
        summary = {
            "orders_placed": 0,
            "orders_cancelled": 0,
            "orders_updated": 0,
            "errors": [],
        }

        try:
            asset_id = market_data.get("asset_id")
            if not asset_id:
                summary["errors"].append("Missing asset_id in market_data")
                return summary

            # Update order statuses
            for order_id in list(self.active_orders.keys()):
                if await self.update_order_status(order_id):
                    summary["orders_updated"] += 1

            # Check for orders to cancel (risk management)
            market_orders = [
                mo
                for mo in self.active_orders.values()
                if mo.order.market_id == asset_id
            ]

            orders_to_cancel = self.strategy.should_cancel_orders(
                asset_id, [mo.order for mo in market_orders]
            )

            for order_id in orders_to_cancel:
                if await self.cancel_order(order_id, "strategy_risk_management"):
                    summary["orders_cancelled"] += 1

            # Generate new orders if conditions are favorable
            yes_orderbook = market_data.get("yes_orderbook")
            no_orderbook = market_data.get("no_orderbook")
            available_capital = market_data.get("available_capital", Decimal("1000"))

            if yes_orderbook and no_orderbook:
                new_orders = self.strategy.calculate_optimal_orders(
                    yes_orderbook, no_orderbook, self.positions, available_capital
                )

                for order_spec in new_orders:
                    order_id = await self.place_order(
                        price=order_spec["price"],
                        size=order_spec["size"],
                        side=order_spec["side"],
                        token_id=order_spec["asset_id"],
                        metadata={
                            "market_type": order_spec.get("market_type", ""),
                            "reason": order_spec.get("reason", ""),
                        },
                    )

                    if order_id:
                        summary["orders_placed"] += 1
                    else:
                        summary["errors"].append(f"Failed to place order: {order_spec}")

        except Exception as e:
            self.logger.error(f"Error in order lifecycle management: {e}")
            summary["errors"].append(str(e))

        return summary

    def get_order_status_summary(self) -> Dict[str, Any]:
        """Get summary of current order statuses"""
        return {
            "active_orders": len(self.active_orders),
            "completed_orders": len(self.completed_orders),
            "metrics": {
                "total_placed": self.order_metrics.total_orders_placed,
                "total_filled": self.order_metrics.total_orders_filled,
                "total_cancelled": self.order_metrics.total_orders_cancelled,
                "fill_rate": float(self.order_metrics.fill_rate),
                "total_volume": float(self.order_metrics.total_volume_filled),
            },
            "positions": {
                market_id: {
                    "size": float(pos.size),
                    "entry_price": float(pos.entry_price),
                }
                for market_id, pos in self.positions.items()
            },
        }

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get list of active orders"""
        return [
            {
                "id": managed_order.order.id,
                "market_id": managed_order.order.market_id,
                "side": managed_order.order.side.value,
                "price": float(managed_order.order.price),
                "size": float(managed_order.order.size),
                "filled_size": float(managed_order.order.filled_size),
                "status": managed_order.order.status.value,
                "age_minutes": (
                    datetime.now(timezone.utc) - managed_order.placement_time
                ).total_seconds()
                / 60,
                "metadata": managed_order.order.metadata,
            }
            for managed_order in self.active_orders.values()
        ]

    async def _status_monitor(self):
        """Background task to monitor order statuses"""
        while self._running:
            try:
                await asyncio.sleep(self.status_check_interval)

                if not self._running:
                    break

                # Update all active order statuses
                for order_id in list(self.active_orders.keys()):
                    await self.update_order_status(order_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in status monitor: {e}")

    async def _handle_order_fill(self, managed_order: ManagedOrder):
        """Handle order fill events"""
        order = managed_order.order

        # Update position
        self._update_position(order)

        # Generate hedge orders if enabled
        if self.enable_auto_hedging and not managed_order.hedge_pending:
            await self._generate_hedge_orders(managed_order)

    def _update_position(self, order: Order):
        """Update position tracking based on filled order"""
        market_id = order.market_id

        if market_id not in self.positions:
            self.positions[market_id] = Position(
                market_id=market_id, size=Decimal("0"), entry_price=order.price
            )

        position = self.positions[market_id]
        filled_size = order.filled_size

        if order.side == OrderSide.BUY:
            position.size += filled_size
        else:
            position.size -= filled_size

        # Update entry price (weighted average)
        if position.size != 0:
            position.entry_price = (
                order.price
            )  # Simplified - should be weighted average

    async def _generate_hedge_orders(self, managed_order: ManagedOrder):
        """Generate hedge orders for filled LP positions"""
        # TODO
        try:
            managed_order.hedge_pending = True

            # This would need actual orderbook data
            # For now, just log the hedge intent
            self.logger.info(
                f"Hedge order generation needed for {managed_order.order.id}"
            )

            # In full implementation:
            # 1. Get current orderbook data
            # 2. Calculate hedge orders using strategy.calculate_hedge_orders()
            # 3. Place hedge orders

        except Exception as e:
            self.logger.error(f"Error generating hedge orders: {e}")
        finally:
            managed_order.hedge_pending = False

    def _validate_order(
        self, price: Decimal, size: Decimal, side: OrderSide, token_id: str
    ) -> bool:
        """Validate order parameters"""
        if price <= 0:
            self.logger.error("Price must be positive")
            return False

        if size <= 0:
            self.logger.error("Size must be positive")
            return False

        if price > Decimal("0.99"):
            self.logger.error("Price too high (max 0.99)")
            return False

        if price < Decimal("0.01"):
            self.logger.error("Price too low (min 0.01)")
            return False

        if not token_id:
            self.logger.error("Token ID required")
            return False

        return True
