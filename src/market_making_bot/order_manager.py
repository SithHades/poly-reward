import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from src.core.models import BookSide, Token, OrderDetails, OrderArgsModel, OrderbookSnapshot
from src.polymarket_client import PolymarketClient
from .config import MarketMakingConfig


@dataclass
class ActiveOrder:
    """Represents an active market making order"""
    order_id: str
    market_id: str
    token_id: str
    side: BookSide
    price: float
    size: float
    created_at: datetime
    last_updated: datetime = field(default_factory=datetime.now)
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if order has expired based on TTL"""
        return (datetime.now() - self.created_at).total_seconds() > ttl_seconds
        
    def needs_refresh(self, current_best_price: float, tick_size: float, buffer_ticks: int) -> bool:
        """Check if order needs to be refreshed based on market conditions"""
        target_price_buffer = tick_size * buffer_ticks
        
        if self.side == BookSide.BUY:
            # For buy orders, we want to be close to but below the best bid
            target_price = current_best_price - target_price_buffer
            return abs(self.price - target_price) > tick_size
        else:
            # For sell orders, we want to be close to but above the best ask  
            target_price = current_best_price + target_price_buffer
            return abs(self.price - target_price) > tick_size


class OrderManager:
    """Manages order placement, tracking, and lifecycle for market making"""
    
    def __init__(self, config: MarketMakingConfig, client: PolymarketClient):
        self.config = config
        self.client = client
        self.logger = logging.getLogger(f"{__name__}.OrderManager")
        
        # Order tracking
        self.active_orders: Dict[str, ActiveOrder] = {}  # order_id -> ActiveOrder
        self.market_orders: Dict[str, List[str]] = {}  # market_id -> [order_ids]
        
        # Performance metrics
        self.orders_placed_today = 0
        self.orders_filled_today = 0
        self.orders_cancelled_today = 0
        
    def calculate_market_making_prices(self, 
                                     orderbook: OrderbookSnapshot, 
                                     tick_size: float,
                                     target_profit_margin: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate optimal bid and ask prices for market making"""
        
        best_bid = orderbook.best_bid()
        best_ask = orderbook.best_ask()
        midpoint = orderbook.midpoint
        
        if not best_bid or not best_ask:
            self.logger.warning("Incomplete orderbook - cannot calculate prices")
            return None, None
            
        # Calculate target prices with profit margin
        target_bid = midpoint - (target_profit_margin / 2)
        target_ask = midpoint + (target_profit_margin / 2)
        
        # Ensure we're competitive but profitable
        # Bid: Must be below midpoint but above best_bid - buffer
        competitive_bid = best_bid + (tick_size * self.config.tick_buffer_size)
        final_bid = min(target_bid, competitive_bid)
        
        # Ask: Must be above midpoint but below best_ask + buffer  
        competitive_ask = best_ask - (tick_size * self.config.tick_buffer_size)
        final_ask = max(target_ask, competitive_ask)
        
        # Round to valid tick sizes
        final_bid = round(final_bid / tick_size) * tick_size
        final_ask = round(final_ask / tick_size) * tick_size
        
        # Ensure minimum spread
        if (final_ask - final_bid) < self.config.min_spread_threshold:
            self.logger.debug("Spread too narrow for profitable market making")
            return None, None
            
        # Ensure prices are within valid range [0, 1]
        final_bid = max(0.001, min(0.999, final_bid))
        final_ask = max(0.001, min(0.999, final_ask))
        
        if final_bid >= final_ask:
            self.logger.debug("Invalid price relationship after adjustments")
            return None, None
            
        return final_bid, final_ask
        
    def place_market_making_orders(self,
                                  market_id: str,
                                  token_id: str, 
                                  orderbook: OrderbookSnapshot,
                                  position_size: float,
                                  tick_size: float) -> List[str]:
        """Place both bid and ask orders for market making"""
        
        bid_price, ask_price = self.calculate_market_making_prices(
            orderbook, tick_size, self.config.target_profit_margin
        )
        
        if not bid_price or not ask_price:
            self.logger.info(f"Skipping order placement for {market_id} - no profitable prices")
            return []
            
        placed_orders = []
        
        # Place buy order
        if not self.config.dry_run:
            try:
                buy_order_args = OrderArgsModel(
                    token_id=token_id,
                    price=bid_price,
                    size=position_size,
                    side=BookSide.BUY
                )
                buy_order = self.client.create_order(buy_order_args)
                buy_order_id = self.client.place_order(buy_order)
                
                if buy_order_id:
                    active_order = ActiveOrder(
                        order_id=buy_order_id,
                        market_id=market_id,
                        token_id=token_id,
                        side=BookSide.BUY,
                        price=bid_price,
                        size=position_size,
                        created_at=datetime.now()
                    )
                    
                    self.active_orders[buy_order_id] = active_order
                    if market_id not in self.market_orders:
                        self.market_orders[market_id] = []
                    self.market_orders[market_id].append(buy_order_id)
                    placed_orders.append(buy_order_id)
                    self.orders_placed_today += 1
                    
                    self.logger.info(f"Placed BUY order {buy_order_id}: {position_size} @ {bid_price}")
                    
            except Exception as e:
                self.logger.error(f"Failed to place buy order for {market_id}: {e}")
        else:
            self.logger.info(f"DRY RUN: Would place BUY order: {position_size} @ {bid_price}")
            
        # Place sell order
        if not self.config.dry_run:
            try:
                sell_order_args = OrderArgsModel(
                    token_id=token_id,
                    price=ask_price,
                    size=position_size,
                    side=BookSide.SELL
                )
                sell_order = self.client.create_order(sell_order_args)
                sell_order_id = self.client.place_order(sell_order)
                
                if sell_order_id:
                    active_order = ActiveOrder(
                        order_id=sell_order_id,
                        market_id=market_id,
                        token_id=token_id,
                        side=BookSide.SELL,
                        price=ask_price,
                        size=position_size,
                        created_at=datetime.now()
                    )
                    
                    self.active_orders[sell_order_id] = active_order
                    if market_id not in self.market_orders:
                        self.market_orders[market_id] = []
                    self.market_orders[market_id].append(sell_order_id)
                    placed_orders.append(sell_order_id)
                    self.orders_placed_today += 1
                    
                    self.logger.info(f"Placed SELL order {sell_order_id}: {position_size} @ {ask_price}")
                    
            except Exception as e:
                self.logger.error(f"Failed to place sell order for {market_id}: {e}")
        else:
            self.logger.info(f"DRY RUN: Would place SELL order: {position_size} @ {ask_price}")
            
        return placed_orders
        
    def refresh_market_orders(self,
                            market_id: str,
                            token_id: str,
                            orderbook: OrderbookSnapshot,
                            position_size: float,
                            tick_size: float) -> List[str]:
        """Cancel existing orders and place new ones for a market"""
        
        # Cancel existing orders for this market
        cancelled_orders = self.cancel_market_orders(market_id)
        self.logger.info(f"Cancelled {len(cancelled_orders)} orders for market {market_id}")
        
        # Place new orders
        new_orders = self.place_market_making_orders(
            market_id, token_id, orderbook, position_size, tick_size
        )
        
        return new_orders
        
    def cancel_market_orders(self, market_id: str) -> List[str]:
        """Cancel all orders for a specific market"""
        
        if market_id not in self.market_orders:
            return []
            
        order_ids = self.market_orders[market_id].copy()
        cancelled_orders = []
        
        if not self.config.dry_run and order_ids:
            try:
                cancelled, not_cancelled = self.client.cancel_orders(order_ids)
                cancelled_orders = cancelled
                self.orders_cancelled_today += len(cancelled)
                
                self.logger.info(f"Cancelled {len(cancelled)} orders, failed to cancel {len(not_cancelled)}")
                
            except Exception as e:
                self.logger.error(f"Failed to cancel orders for market {market_id}: {e}")
        else:
            self.logger.info(f"DRY RUN: Would cancel {len(order_ids)} orders for market {market_id}")
            cancelled_orders = order_ids
            
        # Remove cancelled orders from tracking
        for order_id in cancelled_orders:
            if order_id in self.active_orders:
                del self.active_orders[order_id]
                
        self.market_orders[market_id] = []
        return cancelled_orders
        
    def cancel_expired_orders(self) -> int:
        """Cancel all expired orders based on TTL"""
        
        expired_orders = []
        current_time = datetime.now()
        
        for order_id, order in self.active_orders.items():
            if order.is_expired(self.config.order_ttl_seconds):
                expired_orders.append(order_id)
                
        if expired_orders:
            self.logger.info(f"Found {len(expired_orders)} expired orders")
            
            if not self.config.dry_run:
                try:
                    cancelled, not_cancelled = self.client.cancel_orders(expired_orders)
                    self.orders_cancelled_today += len(cancelled)
                    
                    # Remove cancelled orders from tracking
                    for order_id in cancelled:
                        if order_id in self.active_orders:
                            market_id = self.active_orders[order_id].market_id
                            if market_id in self.market_orders and order_id in self.market_orders[market_id]:
                                self.market_orders[market_id].remove(order_id)
                            del self.active_orders[order_id]
                            
                except Exception as e:
                    self.logger.error(f"Failed to cancel expired orders: {e}")
                    return 0
            else:
                self.logger.info(f"DRY RUN: Would cancel {len(expired_orders)} expired orders")
                # In dry run, still remove from tracking
                for order_id in expired_orders:
                    if order_id in self.active_orders:
                        market_id = self.active_orders[order_id].market_id
                        if market_id in self.market_orders and order_id in self.market_orders[market_id]:
                            self.market_orders[market_id].remove(order_id)
                        del self.active_orders[order_id]
                        
        return len(expired_orders)
        
    def update_order_status(self, risk_manager=None):
        """Update status of all active orders and notify risk manager of fills"""
        
        if not self.active_orders:
            return []
            
        filled_orders = []
        
        try:
            # Get current open orders from exchange
            open_orders = self.client.get_orders()
            open_order_ids = {order.order_id for order in open_orders}
            
            # Find orders that are no longer open (filled or cancelled)
            completed_orders = []
            for order_id in list(self.active_orders.keys()):
                if order_id not in open_order_ids:
                    completed_orders.append(order_id)
                    
            # Process completed orders
            for order_id in completed_orders:
                if order_id in self.active_orders:
                    order = self.active_orders[order_id]
                    market_id = order.market_id
                    
                    # Check if order was filled (vs cancelled) by getting trade history
                    was_filled = self._check_if_order_filled(order_id, order)
                    
                    if was_filled and risk_manager:
                        # Notify risk manager of the fill
                        filled_orders.append(order)
                        self.logger.info(f"Order {order_id} FILLED: {order.side} {order.size} @ {order.price}")
                    else:
                        self.logger.info(f"Order {order_id} CANCELLED: {order.side} {order.size} @ {order.price}")
                    
                    # Remove from tracking
                    if market_id in self.market_orders and order_id in self.market_orders[market_id]:
                        self.market_orders[market_id].remove(order_id)
                        
                    del self.active_orders[order_id]
                    self.orders_filled_today += 1 if was_filled else 0
                    
        except Exception as e:
            self.logger.error(f"Failed to update order status: {e}")
            
        return filled_orders
        
    def _check_if_order_filled(self, order_id: str, order: ActiveOrder) -> bool:
        """Check if an order was filled by examining recent trades"""
        try:
            # Get recent trades for this token to see if our order was filled
            # This is a simplified check - in production you'd want more sophisticated trade matching
            last_price = self.client.get_last_trade_price(order.token_id)
            
            # If last trade price matches our order price (within tolerance), likely filled
            price_tolerance = 0.001
            if abs(last_price - order.price) <= price_tolerance:
                return True
                
            return False
        except Exception as e:
            self.logger.error(f"Failed to check fill status for order {order_id}: {e}")
            return False  # Conservative assumption
            
    def get_market_orders(self, market_id: str) -> List[ActiveOrder]:
        """Get all active orders for a specific market"""
        
        if market_id not in self.market_orders:
            return []
            
        return [self.active_orders[order_id] 
                for order_id in self.market_orders[market_id] 
                if order_id in self.active_orders]
                
    def has_active_orders(self, market_id: str) -> bool:
        """Check if there are active orders for a market"""
        
        return market_id in self.market_orders and len(self.market_orders[market_id]) > 0
        
    def cleanup_all_orders(self):
        """Emergency cleanup - cancel all active orders"""
        
        if not self.active_orders:
            self.logger.info("No active orders to cleanup")
            return
            
        all_order_ids = list(self.active_orders.keys())
        self.logger.warning(f"Emergency cleanup: cancelling {len(all_order_ids)} orders")
        
        if not self.config.dry_run:
            try:
                cancelled, not_cancelled = self.client.cancel_orders(all_order_ids)
                self.logger.info(f"Cleanup complete: cancelled {len(cancelled)} orders")
                
                if not_cancelled:
                    self.logger.warning(f"Failed to cancel {len(not_cancelled)} orders: {not_cancelled}")
                    
            except Exception as e:
                self.logger.error(f"Failed to cleanup orders: {e}")
        else:
            self.logger.info(f"DRY RUN: Would cancel {len(all_order_ids)} orders in cleanup")
            
        # Clear all tracking regardless
        self.active_orders.clear()
        self.market_orders.clear()
        
    def get_order_summary(self) -> Dict:
        """Get summary of order management metrics"""
        
        return {
            "active_orders": len(self.active_orders),
            "markets_with_orders": len(self.market_orders),
            "orders_placed_today": self.orders_placed_today,
            "orders_filled_today": self.orders_filled_today,
            "orders_cancelled_today": self.orders_cancelled_today,
            "fill_rate": self.orders_filled_today / max(1, self.orders_placed_today),
        }