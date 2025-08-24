import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.core.models import Market, SimplifiedMarket, OrderbookSnapshot, OrderbookLevel, MarketCondition
from src.core.constants import MARKETS
from src.polymarket_client import PolymarketClient
from src.parsing_utils import get_current_market_slug, get_next_market_slug, slug_to_datetime, get_market_end_time_from_slug
from .config import MarketMakingConfig
from .risk_manager import RiskManager
from .order_manager import OrderManager


class MarketMakingStrategy:
    """Core market making strategy for hourly Ethereum prediction markets"""
    
    def __init__(self, config: MarketMakingConfig, client: PolymarketClient, 
                 risk_manager: RiskManager, order_manager: OrderManager):
        self.config = config
        self.client = client
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.logger = logging.getLogger(f"{__name__}.MarketMakingStrategy")
        
        # Strategy state
        self.active_markets: Dict[str, Market] = {}
        # Initialize last_market_refresh to past time to trigger immediate market loading
        self.market_refresh_interval = 300  # Refresh market list every 5 minutes
        self.last_market_refresh = datetime.now() - timedelta(seconds=self.market_refresh_interval + 1)
        
        # Performance tracking
        self.markets_evaluated = 0
        self.orders_placed = 0
        self.profitable_spreads_found = 0
        
    async def find_active_markets(self) -> List[Market]:
        """Find active hourly prediction markets for the configured crypto"""
        
        try:
            # Get current and next market slugs
            current_slug = get_current_market_slug(self.config.crypto)
            next_slug = get_next_market_slug(self.config.crypto)
            
            self.logger.info(f"Looking for markets: {current_slug}, {next_slug}")
            
            active_markets = []
            
            # Try to get current market
            try:
                current_market = self.client.get_market_by_slug(current_slug)
                if current_market and self._is_market_tradeable(current_market):
                    active_markets.append(current_market)
                    self.logger.info(f"Found active current market: {current_market.market_slug}")
            except Exception as e:
                self.logger.warning(f"Could not fetch current market {current_slug}: {e}")
                
            # Try to get next market
            try:
                next_market = self.client.get_market_by_slug(next_slug)
                if next_market and self._is_market_tradeable(next_market):
                    active_markets.append(next_market)
                    self.logger.info(f"Found active next market: {next_market.market_slug}")
            except Exception as e:
                self.logger.warning(f"Could not fetch next market {next_slug}: {e}")
                
            self.logger.info(f"Total active markets found: {len(active_markets)}")
            return active_markets
            
        except Exception as e:
            self.logger.error(f"Failed to find active markets: {e}")
            return []
            
    def _is_hourly_eth_market(self, market: Market) -> bool:
        """Check if a market is an hourly Ethereum prediction market"""
        
        if not market.market_slug:
            return False
            
        # Check for ethereum hourly market patterns
        slug_patterns = [
            "ethereum-up-or-down",
            "eth-up-or-down", 
            "ethereum-price",
            "eth-price"
        ]
        
        slug_lower = market.market_slug.lower()
        contains_pattern = any(pattern in slug_lower for pattern in slug_patterns)
        
        # Check for time indicators (hourly markets typically have time stamps)
        time_indicators = ["am-et", "pm-et", "hourly"]
        contains_time = any(indicator in slug_lower for indicator in time_indicators)
        
        return contains_pattern and contains_time
        
    def _is_market_tradeable(self, market: Market) -> bool:
        """Check if a market is suitable for trading with detailed logging"""
        
        market_info = f"Market {market.market_slug}"
        
        # Basic market status checks
        if not market.active:
            self.logger.debug(f"{market_info} - Not active")
            return False
        
        if market.closed:
            self.logger.debug(f"{market_info} - Closed")
            return False
            
        if market.archived:
            self.logger.debug(f"{market_info} - Archived")
            return False
            
        if not market.accepting_orders:
            self.logger.debug(f"{market_info} - Not accepting orders")
            return False
            
        # Check if market has order book enabled (less strict - warn instead of block)
        if not market.enable_order_book:
            self.logger.warning(f"{market_info} - Order book not enabled, but attempting anyway")
            # Don't return False - try to trade anyway
            
        # Check tokens (more flexible - allow 2+ tokens, focusing on binary outcomes)
        if len(market.tokens) < 2:
            self.logger.debug(f"{market_info} - Has {len(market.tokens)} tokens, need at least 2")
            return False
        elif len(market.tokens) > 2:
            self.logger.info(f"{market_info} - Has {len(market.tokens)} tokens, will focus on first 2")
            
        # Calculate market end time from slug (ignore unreliable end_date_iso)
        market_end_time = get_market_end_time_from_slug(market.market_slug)
        if market_end_time:
            # Convert current time to ET for consistent comparison
            from src.parsing_utils import ET
            current_time_et = datetime.now(ET)
            
            time_to_expiry = market_end_time - current_time_et
            expiry_hours = time_to_expiry.total_seconds() / 3600
            
            # Debug logging
            self.logger.debug(f"{market_info} - Market end time (from slug): {market_end_time}")
            self.logger.debug(f"{market_info} - Current time ET: {current_time_et}")
            self.logger.debug(f"{market_info} - Time to expiry: {expiry_hours:.2f} hours")
            
            # Much more flexible time check - allow trading much closer to expiry
            min_minutes = max(5, self.config.min_time_to_expiry_hours * 60)  # At least 5 minutes
            
            if time_to_expiry.total_seconds() < min_minutes * 60:
                self.logger.debug(f"{market_info} - Too close to expiry: {expiry_hours:.2f} hours left (min: {min_minutes/60:.2f} hours)")
                return False
            else:
                self.logger.info(f"{market_info} - Time to expiry: {expiry_hours:.2f} hours")
        else:
            self.logger.warning(f"{market_info} - Could not parse end time from market slug: {market.market_slug}")
            
        self.logger.info(f"{market_info} - Passed all tradeability checks")
        return True
        
    async def evaluate_market_opportunity(self, market: Market) -> Optional[Tuple[dict, float, dict]]:
        """Evaluate if a market presents a good market making opportunity"""
        
        self.markets_evaluated += 1
        
        try:
            # Get market tokens - typically "Up" and "Down" for price prediction
            if len(market.tokens) != 2:
                self.logger.debug(f"Market {market.market_slug} has {len(market.tokens)} tokens, expected 2")
                return None
                
            # Work with both tokens for proper binary market making
            token_a = market.tokens[0]  # First token (e.g., "Up" or "Yes")
            token_b = market.tokens[1]  # Second token (e.g., "Down" or "No")
            
            # Get order books for both tokens
            order_book_a = self.client.get_order_book(token_a.token_id)
            order_book_b = self.client.get_order_book(token_b.token_id)
            
            # Convert to our OrderbookSnapshot format for both tokens
            def convert_orderbook(token_id, order_book):
                bids = [OrderbookLevel(price=float(bid.price), size=float(bid.size)) 
                       for bid in order_book.bids]
                asks = [OrderbookLevel(price=float(ask.price), size=float(ask.size)) 
                       for ask in order_book.asks]
                
                if not bids or not asks:
                    return None
                    
                # Calculate spread and midpoint
                best_bid = max(bid.price for bid in bids)  # Highest bid price
                best_ask = min(ask.price for ask in asks)  # Lowest ask price
                midpoint = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                
                return OrderbookSnapshot(
                    asset_id=token_id,
                    bids=bids,
                    asks=asks,
                    midpoint=midpoint,
                    spread=spread,
                    timestamp=datetime.now()
                )
            
            orderbook_a = convert_orderbook(token_a.token_id, order_book_a)
            orderbook_b = convert_orderbook(token_b.token_id, order_book_b)
            
            if not orderbook_a or not orderbook_b:
                self.logger.debug(f"Incomplete order book for {market.market_slug}")
                return None
            
            # Check binary relationship: prices should roughly sum to 1.0
            combined_midpoint = orderbook_a.midpoint + orderbook_b.midpoint
            if abs(combined_midpoint - 1.0) > 0.1:  # Allow some tolerance for market inefficiency
                self.logger.warning(f"Binary tokens don't sum to 1.0: {combined_midpoint:.4f} for {market.market_slug}")
            
            # Use the better spread of the two tokens for opportunity assessment
            combined_spread = min(orderbook_a.spread, orderbook_b.spread)
            combined_midpoint = (orderbook_a.midpoint + orderbook_b.midpoint) / 2
            
            self.logger.info(f"Market {market.market_slug} analysis:")
            self.logger.info(f"  Token A ({token_a.outcome}): mid={orderbook_a.midpoint:.4f}, spread={orderbook_a.spread:.4f}")
            self.logger.info(f"  Token B ({token_b.outcome}): mid={orderbook_b.midpoint:.4f}, spread={orderbook_b.spread:.4f}")
            self.logger.info(f"  Combined: mid={combined_midpoint:.4f}, spread={combined_spread:.4f}")
            self.logger.info(f"  Combined midpoint sum: {orderbook_a.midpoint + orderbook_b.midpoint:.4f}")
            
            # Update volatility metrics for both tokens
            self.risk_manager.update_volatility_metrics(f"{market.condition_id}_A", orderbook_a)
            self.risk_manager.update_volatility_metrics(f"{market.condition_id}_B", orderbook_b)
            
            # Check if spread is profitable
            if combined_spread < self.config.min_spread_threshold:
                self.logger.info(f"Combined spread too narrow for {market.market_slug}: {combined_spread:.4f} < {self.config.min_spread_threshold}")
                return None
                
            # Check risk conditions
            if not self.risk_manager.can_enter_position(market, self.config.base_position_size):
                self.logger.info(f"Risk manager rejected {market.market_slug}")
                return None
                
            # Check market condition (use token A as representative)
            market_condition = self.risk_manager.assess_market_condition(market.condition_id)
            if market_condition in [MarketCondition.VOLATILE, MarketCondition.UNAVAILABLE]:
                self.logger.info(f"Market condition unfavorable for {market.market_slug}: {market_condition}")
                return None
                
            self.profitable_spreads_found += 1
            self.logger.info(f"Found binary opportunity in {market.market_slug}: spread={combined_spread:.4f}, combined_midpoint={combined_midpoint:.4f}")
            self.logger.info(f"Token A ({token_a.outcome}): mid={orderbook_a.midpoint:.4f}, Token B ({token_b.outcome}): mid={orderbook_b.midpoint:.4f}")
            
            # Return both tokens and orderbooks
            tokens_info = {
                'token_a': {'token': token_a, 'orderbook': orderbook_a},
                'token_b': {'token': token_b, 'orderbook': orderbook_b}
            }
            orderbooks_info = {'orderbook_a': orderbook_a, 'orderbook_b': orderbook_b}
            
            return tokens_info, combined_midpoint, orderbooks_info
            
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to evaluate market {market.market_slug}: {e}\nStack trace:\n{traceback.format_exc()}")
            return None
            
    async def execute_market_making(self, market: Market, tokens_info: dict, orderbooks_info: dict):
        """Execute market making strategy for a binary market with both tokens"""
        
        try:
            # Calculate position size based on risk management
            available_balance = self._get_available_balance()
            position_size = self.risk_manager.get_position_size_for_market(market, available_balance)
            
            # Get tick size for both tokens (use the first token as reference)
            token_a = tokens_info['token_a']['token']
            tick_size = self.client.get_tick_size(token_a.token_id)
            if tick_size == 0:
                tick_size = market.minimum_tick_size or 0.001
                
            # Check if we already have orders in this market
            if self.order_manager.has_active_orders(market.condition_id):
                # Check if orders need refreshing
                existing_orders = self.order_manager.get_market_orders(market.condition_id)
                if existing_orders:
                    last_order = max(existing_orders, key=lambda x: x.created_at)
                    if self.risk_manager.should_refresh_orders(market.condition_id, last_order.created_at):
                        self.logger.info(f"Refreshing binary orders for {market.market_slug}")
                        orders = self.order_manager.refresh_binary_market_orders(
                            market.condition_id, tokens_info, orderbooks_info, position_size, tick_size
                        )
                        self.orders_placed += len(orders)
                    else:
                        self.logger.debug(f"Orders for {market.market_slug} don't need refreshing yet")
                else:
                    # No existing orders, place new ones
                    orders = self.order_manager.place_binary_market_making_orders(
                        market.condition_id, tokens_info, orderbooks_info, position_size, tick_size
                    )
                    self.orders_placed += len(orders)
            else:
                # No existing orders, place new ones
                orders = self.order_manager.place_binary_market_making_orders(
                    market.condition_id, tokens_info, orderbooks_info, position_size, tick_size
                )
                self.orders_placed += len(orders)
                
        except Exception as e:
            self.logger.error(f"Failed to execute binary market making for {market.market_slug}: {e}")
            
    async def run_strategy_cycle(self):
        """Run one complete cycle of the market making strategy"""
        
        try:
            self.logger.info("Starting market making strategy cycle")
            
            # Refresh market list if needed
            current_time = datetime.now()
            if (current_time - self.last_market_refresh).total_seconds() > self.market_refresh_interval:
                self.logger.info("Refreshing active markets list")
                markets = await self.find_active_markets()
                
                # Update active markets dict
                self.active_markets.clear()
                for market in markets:
                    self.active_markets[market.condition_id] = market
                    
                self.last_market_refresh = current_time
                self.logger.info(f"Refreshed markets: {len(self.active_markets)} active")
            
            # Cancel expired orders across all markets
            expired_count = self.order_manager.cancel_expired_orders()
            if expired_count > 0:
                self.logger.info(f"Cancelled {expired_count} expired orders")
                
            # Update order status and process any fills
            filled_orders = self.order_manager.update_order_status(self.risk_manager)
            
            # Process filled orders to update positions
            for filled_order in filled_orders:
                self.risk_manager.process_order_fill(filled_order, filled_order.market_id)
                
            # Sync positions with exchange periodically
            if len(filled_orders) > 0:
                self.risk_manager.sync_positions_with_exchange()
            
            # Process each active market
            for market_id, market in self.active_markets.items():
                try:
                    # Skip if market is too close to expiry (using slug-based timing)
                    market_end_time = get_market_end_time_from_slug(market.market_slug)
                    if market_end_time:
                        from src.parsing_utils import ET
                        current_time_et = datetime.now(ET)
                        time_to_expiry = market_end_time - current_time_et
                        
                        if time_to_expiry.total_seconds() < self.config.market_close_buffer_minutes * 60:
                            self.logger.info(f"Market {market.market_slug} too close to expiry, cancelling orders")
                            self.order_manager.cancel_market_orders(market_id)
                            continue
                    
                    # Evaluate market opportunity
                    opportunity = await self.evaluate_market_opportunity(market)
                    if opportunity:
                        tokens_info, combined_midpoint, orderbooks_info = opportunity
                        await self.execute_market_making(market, tokens_info, orderbooks_info)
                    else:
                        # If no opportunity but we have orders, consider cancelling them
                        if self.order_manager.has_active_orders(market_id):
                            self.logger.info(f"No opportunity in {market.market_slug}, but have active orders")
                            # Could implement logic to cancel orders if conditions changed
                            
                except Exception as e:
                    self.logger.error(f"Error processing market {market.market_slug}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in strategy cycle: {e}")
            
    def _get_available_balance(self) -> float:
        """Get available balance for trading"""
        try:
            # Get actual balance from exchange
            total_balance = self.client.get_collateral_balance()
            
            # Calculate total exposure from current risk metrics (local tracking)
            total_exposure = sum(
                metrics.current_exposure 
                for metrics in self.risk_manager.market_risk_metrics.values()
            )
            
            available = total_balance - total_exposure
            self.logger.debug(f"Balance calculation: total={total_balance:.2f}, exposure={total_exposure:.2f}, available={available:.2f}")
            
            return max(0, available)
            
        except Exception as e:
            self.logger.error(f"Failed to get available balance: {e}")
            # Conservative fallback
            return 100.0
            
    def get_strategy_metrics(self) -> Dict:
        """Get performance metrics for the strategy"""
        
        return {
            "active_markets": len(self.active_markets),
            "markets_evaluated": self.markets_evaluated,
            "profitable_spreads_found": self.profitable_spreads_found,
            "orders_placed": self.orders_placed,
            "opportunity_rate": self.profitable_spreads_found / max(1, self.markets_evaluated),
            "last_market_refresh": self.last_market_refresh,
        }