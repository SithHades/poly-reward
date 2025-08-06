import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.core.models import Market, SimplifiedMarket, OrderbookSnapshot, OrderbookLevel, MarketCondition
from src.core.constants import MARKETS
from src.polymarket_client import PolymarketClient
from src.parsing_utils import get_current_market_slug, get_next_market_slug, slug_to_datetime
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
        self.last_market_refresh = datetime.now()
        self.market_refresh_interval = 300  # Refresh market list every 5 minutes
        
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
                
            # Additionally, search for any other active hourly ETH markets
            try:
                all_markets = self.client.get_active_markets()
                hourly_eth_markets = [
                    market for market in all_markets 
                    if self._is_hourly_eth_market(market) and self._is_market_tradeable(market)
                    and market.condition_id not in [m.condition_id for m in active_markets]
                ]
                active_markets.extend(hourly_eth_markets)
                
                if hourly_eth_markets:
                    self.logger.info(f"Found {len(hourly_eth_markets)} additional hourly ETH markets")
                    
            except Exception as e:
                self.logger.error(f"Failed to search for additional markets: {e}")
                
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
        """Check if a market is suitable for trading"""
        
        # Basic market status checks
        if not market.active or market.closed or market.archived or not market.accepting_orders:
            return False
            
        # Check if market has order book enabled
        if not market.enable_order_book:
            return False
            
        # Check tokens (should be binary outcome: Up/Down, Yes/No)
        if len(market.tokens) != 2:
            return False
            
        # Check time to expiry
        if market.end_date_iso:
            time_to_expiry = market.end_date_iso - datetime.now(market.end_date_iso.tzinfo)
            if time_to_expiry.total_seconds() < self.config.min_time_to_expiry_hours * 3600:
                return False
                
        return True
        
    async def evaluate_market_opportunity(self, market: Market) -> Optional[Tuple[str, float, OrderbookSnapshot]]:
        """Evaluate if a market presents a good market making opportunity"""
        
        self.markets_evaluated += 1
        
        try:
            # Get market tokens - typically "Up" and "Down" for price prediction
            if len(market.tokens) != 2:
                self.logger.debug(f"Market {market.market_slug} has {len(market.tokens)} tokens, expected 2")
                return None
                
            # For simplicity, focus on the first token (could be "Up" or "Yes")
            token = market.tokens[0]
            token_id = token.token_id
            
            # Get order book
            order_book_summary = self.client.get_order_book(token_id)
            
            # Convert to our OrderbookSnapshot format
            bids = [OrderbookLevel(price=float(bid['price']), size=float(bid['size'])) 
                   for bid in order_book_summary.bids]
            asks = [OrderbookLevel(price=float(ask['price']), size=float(ask['size'])) 
                   for ask in order_book_summary.asks]
            
            if not bids or not asks:
                self.logger.debug(f"Incomplete order book for {market.market_slug}")
                return None
                
            # Calculate spread and midpoint
            best_bid = bids[0].price
            best_ask = asks[0].price
            midpoint = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            orderbook = OrderbookSnapshot(
                asset_id=token_id,
                bids=bids,
                asks=asks,
                midpoint=midpoint,
                spread=spread,
                timestamp=datetime.now()
            )
            
            # Update volatility metrics
            self.risk_manager.update_volatility_metrics(market.condition_id, orderbook)
            
            # Check if spread is profitable
            if spread < self.config.min_spread_threshold:
                self.logger.debug(f"Spread too narrow for {market.market_slug}: {spread:.4f}")
                return None
                
            # Check risk conditions
            if not self.risk_manager.can_enter_position(market, self.config.base_position_size):
                self.logger.debug(f"Risk manager rejected {market.market_slug}")
                return None
                
            # Check market condition
            market_condition = self.risk_manager.assess_market_condition(market.condition_id)
            if market_condition in [MarketCondition.VOLATILE, MarketCondition.UNAVAILABLE]:
                self.logger.debug(f"Market condition unfavorable for {market.market_slug}: {market_condition}")
                return None
                
            self.profitable_spreads_found += 1
            self.logger.info(f"Found opportunity in {market.market_slug}: spread={spread:.4f}, midpoint={midpoint:.4f}")
            
            return token_id, midpoint, orderbook
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate market {market.market_slug}: {e}")
            return None
            
    async def execute_market_making(self, market: Market, token_id: str, orderbook: OrderbookSnapshot):
        """Execute market making strategy for a specific market"""
        
        try:
            # Calculate position size based on risk management
            available_balance = self._get_available_balance()
            position_size = self.risk_manager.get_position_size_for_market(market, available_balance)
            
            # Get tick size for the market
            tick_size = self.client.get_tick_size(token_id)
            if tick_size == 0:
                tick_size = market.minimum_tick_size or 0.001
                
            # Check if we already have orders in this market
            if self.order_manager.has_active_orders(market.condition_id):
                # Check if orders need refreshing
                existing_orders = self.order_manager.get_market_orders(market.condition_id)
                if existing_orders:
                    last_order = max(existing_orders, key=lambda x: x.created_at)
                    if self.risk_manager.should_refresh_orders(market.condition_id, last_order.created_at):
                        self.logger.info(f"Refreshing orders for {market.market_slug}")
                        orders = self.order_manager.refresh_market_orders(
                            market.condition_id, token_id, orderbook, position_size, tick_size
                        )
                        self.orders_placed += len(orders)
                    else:
                        self.logger.debug(f"Orders for {market.market_slug} don't need refreshing yet")
                else:
                    # No existing orders, place new ones
                    orders = self.order_manager.place_market_making_orders(
                        market.condition_id, token_id, orderbook, position_size, tick_size
                    )
                    self.orders_placed += len(orders)
            else:
                # No existing orders, place new ones
                orders = self.order_manager.place_market_making_orders(
                    market.condition_id, token_id, orderbook, position_size, tick_size
                )
                self.orders_placed += len(orders)
                
        except Exception as e:
            self.logger.error(f"Failed to execute market making for {market.market_slug}: {e}")
            
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
                    # Skip if market is too close to expiry
                    if market.end_date_iso:
                        time_to_expiry = market.end_date_iso - datetime.now(market.end_date_iso.tzinfo)
                        if time_to_expiry.total_seconds() < self.config.market_close_buffer_minutes * 60:
                            self.logger.info(f"Market {market.market_slug} too close to expiry, cancelling orders")
                            self.order_manager.cancel_market_orders(market_id)
                            continue
                    
                    # Evaluate market opportunity
                    opportunity = await self.evaluate_market_opportunity(market)
                    if opportunity:
                        token_id, midpoint, orderbook = opportunity
                        await self.execute_market_making(market, token_id, orderbook)
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
            total_balance = self.client.get_balance()
            
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