import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from src.core.models import Market, Position, VolatilityMetrics, MarketCondition
from src.polymarket_client import PolymarketClient
from .config import MarketMakingConfig


@dataclass
class RiskMetrics:
    """Risk metrics for a specific market"""
    current_exposure: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    position_count: int = 0
    avg_entry_price: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


class RiskManager:
    """Manages risk across all market making activities"""
    
    def __init__(self, config: MarketMakingConfig, client: PolymarketClient):
        self.config = config
        self.client = client
        self.logger = logging.getLogger(f"{__name__}.RiskManager")
        
        # Risk tracking
        self.market_risk_metrics: Dict[str, RiskMetrics] = {}
        self.consecutive_losses = 0
        self.total_realized_pnl = 0.0
        self.daily_loss_limit = 100.0  # Daily loss limit in USD
        self.emergency_stop_active = False
        
        # Volatility tracking per market
        self.volatility_trackers: Dict[str, VolatilityMetrics] = {}
        
    def can_enter_position(self, market: Market, position_size: float) -> bool:
        """Check if we can enter a new position in the given market"""
        
        if self.emergency_stop_active:
            self.logger.warning("Emergency stop active - no new positions allowed")
            return False
            
        # Check individual market exposure
        market_risk = self.market_risk_metrics.get(market.condition_id, RiskMetrics())
        if market_risk.current_exposure + position_size > self.config.max_exposure_per_market:
            self.logger.info(f"Market exposure limit reached for {market.market_slug}")
            return False
            
        # Check total exposure across all markets
        total_exposure = sum(metrics.current_exposure for metrics in self.market_risk_metrics.values())
        if total_exposure + position_size > self.config.max_total_exposure:
            self.logger.info("Total exposure limit reached across all markets")
            return False
            
        # Check if market is too close to expiry
        if market.end_date_iso:
            time_to_expiry = market.end_date_iso - datetime.now(market.end_date_iso.tzinfo)
            if time_to_expiry.total_seconds() < self.config.min_time_to_expiry_hours * 3600:
                self.logger.info(f"Market {market.market_slug} too close to expiry")
                return False
                
        # Check volatility
        market_condition = self.assess_market_condition(market.condition_id)
        if market_condition == MarketCondition.VOLATILE:
            self.logger.info(f"Market {market.market_slug} too volatile for new positions")
            return False
            
        return True
        
    def assess_market_condition(self, market_id: str) -> MarketCondition:
        """Assess the current condition of a market"""
        
        volatility_tracker = self.volatility_trackers.get(market_id)
        if not volatility_tracker:
            return MarketCondition.UNAVAILABLE
            
        if volatility_tracker.is_volatile(
            midpoint_threshold=self.config.volatility_threshold,
            spread_threshold=self.config.volatility_threshold * 2
        ):
            return MarketCondition.VOLATILE
            
        # Additional logic could check order book depth, competition, etc.
        return MarketCondition.ATTRACTIVE
        
    def process_order_fill(self, filled_order, market_id: str):
        """Process a filled order and update position metrics"""
        
        if market_id not in self.market_risk_metrics:
            self.market_risk_metrics[market_id] = RiskMetrics()
            
        metrics = self.market_risk_metrics[market_id]
        
        # Update position based on order fill
        fill_size = filled_order.size if filled_order.side.upper() == "BUY" else -filled_order.size
        fill_value = fill_size * filled_order.price
        
        # Update average entry price using weighted average
        if metrics.position_count == 0:
            # First position
            metrics.avg_entry_price = filled_order.price
        else:
            # Update weighted average
            total_value = (metrics.current_exposure * metrics.avg_entry_price) + fill_value
            total_size = metrics.current_exposure + abs(fill_value)
            if total_size > 0:
                metrics.avg_entry_price = total_value / total_size
        
        # Update exposure and position count
        metrics.current_exposure += abs(fill_value)
        metrics.position_count += 1
        metrics.last_update = datetime.now()
        
        self.logger.info(f"Position updated for {market_id}: exposure=${metrics.current_exposure:.2f}, avg_price={metrics.avg_entry_price:.4f}")
        
        # Check for stop loss (will implement when we have current market prices)
        return False
        
    def update_position_metrics(self, market_id: str, position: Position):
        """Update risk metrics for a position (legacy method - kept for compatibility)"""
        
        if market_id not in self.market_risk_metrics:
            self.market_risk_metrics[market_id] = RiskMetrics()
            
        metrics = self.market_risk_metrics[market_id]
        
        # Update exposure (position size * current price)
        metrics.current_exposure = abs(position.size * (position.current_price or position.entry_price))
        
        # Calculate unrealized PnL
        if position.current_price and position.entry_price:
            metrics.unrealized_pnl = position.size * (position.current_price - position.entry_price)
            
        metrics.position_count = 1 if position.size != 0 else 0
        metrics.avg_entry_price = position.entry_price
        metrics.last_update = datetime.now()
        
        # Check for stop loss
        if metrics.unrealized_pnl < -self.config.stop_loss_percentage * metrics.current_exposure:
            self.logger.warning(f"Stop loss triggered for market {market_id}")
            return True  # Signal to close position
            
        return False
        
    def get_current_positions(self) -> List[Position]:
        """Get current positions from the exchange"""
        try:
            return self.client.get_positions()
        except Exception as e:
            self.logger.error(f"Failed to get current positions: {e}")
            return []
            
    def sync_positions_with_exchange(self):
        """Sync local position tracking with actual exchange positions"""
        try:
            exchange_positions = self.get_current_positions()
            
            # Update metrics based on actual positions
            for position in exchange_positions:
                if position.size != 0:  # Only track non-zero positions
                    self.update_position_metrics(position.market_id, position)
                    
            self.logger.info(f"Synced {len(exchange_positions)} positions with exchange")
            
        except Exception as e:
            self.logger.error(f"Failed to sync positions with exchange: {e}")
        
    def record_trade_result(self, market_id: str, pnl: float):
        """Record the result of a completed trade"""
        
        if market_id not in self.market_risk_metrics:
            self.market_risk_metrics[market_id] = RiskMetrics()
            
        metrics = self.market_risk_metrics[market_id]
        metrics.realized_pnl += pnl
        self.total_realized_pnl += pnl
        
        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        # Check for emergency stop conditions
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self.logger.error(f"Emergency stop triggered: {self.consecutive_losses} consecutive losses")
            self.emergency_stop_active = True
            
        if self.total_realized_pnl < -self.daily_loss_limit:
            self.logger.error(f"Daily loss limit reached: ${self.total_realized_pnl:.2f}")
            self.emergency_stop_active = True
            
    def update_volatility_metrics(self, market_id: str, orderbook_snapshot):
        """Update volatility metrics for a market"""
        
        if market_id not in self.volatility_trackers:
            self.volatility_trackers[market_id] = VolatilityMetrics()
            
        self.volatility_trackers[market_id].add_snapshot(orderbook_snapshot)
        
    def get_position_size_for_market(self, market: Market, available_balance: float) -> float:
        """Calculate appropriate position size for a market"""
        
        # Start with base position size
        position_size = self.config.base_position_size
        
        # Adjust based on available balance
        position_size = min(position_size, max(available_balance * 0.1, 5.0))  # Max 10% of balance per position
        
        # Adjust based on time to expiry (smaller positions for shorter-term markets)
        if market.end_date_iso:
            time_to_expiry_hours = (market.end_date_iso - datetime.now(market.end_date_iso.tzinfo)).total_seconds() / 3600
            if time_to_expiry_hours < 2:
                position_size *= 0.5  # Reduce position size for markets expiring soon
                
        # Adjust based on volatility
        market_condition = self.assess_market_condition(market.condition_id)
        if market_condition == MarketCondition.VOLATILE:
            position_size *= 0.7  # Reduce position size in volatile conditions
        elif market_condition == MarketCondition.COMPETITIVE:
            position_size *= 0.8  # Slightly reduce position size in competitive markets
            
        return min(position_size, self.config.max_position_size)
        
    def should_refresh_orders(self, market_id: str, last_refresh: datetime) -> bool:
        """Determine if orders should be refreshed based on market conditions"""
        
        # Always refresh after the configured interval
        if (datetime.now() - last_refresh).total_seconds() > self.config.order_refresh_interval:
            return True
            
        # Refresh immediately if market becomes volatile
        market_condition = self.assess_market_condition(market_id)
        if market_condition == MarketCondition.VOLATILE:
            return True
            
        return False
        
    def get_risk_summary(self) -> Dict:
        """Get a summary of current risk metrics"""
        
        total_exposure = sum(metrics.current_exposure for metrics in self.market_risk_metrics.values())
        total_unrealized_pnl = sum(metrics.unrealized_pnl for metrics in self.market_risk_metrics.values())
        active_positions = sum(1 for metrics in self.market_risk_metrics.values() if metrics.position_count > 0)
        
        return {
            "total_exposure": total_exposure,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_realized_pnl": self.total_realized_pnl,
            "active_positions": active_positions,
            "consecutive_losses": self.consecutive_losses,
            "emergency_stop_active": self.emergency_stop_active,
            "markets_tracked": len(self.market_risk_metrics)
        }