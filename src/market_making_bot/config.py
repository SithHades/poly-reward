from dataclasses import dataclass
from typing import Optional
from src.core.constants import MARKETS


@dataclass
class MarketMakingConfig:
    """Configuration for the market making bot"""
    
    # Market selection
    crypto: MARKETS = "ethereum"
    
    # Trading parameters
    base_position_size: float = 10.0  # Base position size in USD
    max_position_size: float = 100.0  # Maximum position size per market
    min_spread_threshold: float = 0.015  # Minimum spread to consider profitable (1.5%) - reduced for better opportunities
    target_profit_margin: float = 0.005  # Target profit margin per trade (0.5%)
    
    # Risk management
    max_exposure_per_market: float = 50.0  # Max exposure per individual market
    max_total_exposure: float = 200.0  # Max total exposure across all markets
    stop_loss_percentage: float = 0.05  # Stop loss at 5% loss
    volatility_threshold: float = 0.03  # Pause trading if volatility exceeds 3%
    
    # Order placement
    order_refresh_interval: int = 30  # Refresh orders every 30 seconds
    tick_buffer_size: int = 2  # Number of ticks away from best bid/ask
    order_ttl_seconds: int = 300  # Order time-to-live in seconds
    
    # Market timing
    market_close_buffer_minutes: int = 5  # Stop trading X minutes before market close
    min_time_to_expiry_hours: float = 0.083  # Don't trade markets expiring in less than 5min (more aggressive)
    
    # Performance monitoring
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    
    # Safety features
    dry_run: bool = True  # Start in dry run mode
    enable_emergency_stop: bool = True
    max_consecutive_losses: int = 5  # Emergency stop after 5 consecutive losses