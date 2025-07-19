#!/usr/bin/env python3
"""
Polymarket Hourly Trading Bot

This bot continuously monitors price action and places limit orders on Polymarket
based on model predictions at the 45-minute mark of each hour.

Features:
- Real-time data collection using CCXT
- ML model prediction at 45-minute mark
- Polymarket order book analysis
- Automated limit order placement based on confidence thresholds
- Risk management and position tracking
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import ccxt
import pandas as pd
from dataclasses import dataclass, field

from src.core.constants import MARKETS, TICKERS
from src.parsing_utils import (
    ET,
    create_slug_from_datetime,
    get_current_market_hour_et,
    get_current_market_slug,
    slug_to_datetime,
)
from src.eth_candle_predictor import EthCandlePredictor, PredictionResult
from src.polymarket_client import PolymarketClient
from src.core.models import Market, OrderArgsModel, BookSide, Position, OrderDetails


@dataclass
class OrderTracker:
    """Track individual orders for lifecycle management"""

    order_id: str
    market_slug: str
    token_id: str
    token_outcome: str
    order_price: float
    order_size: float
    order_time: datetime
    predicted_direction: str
    confidence: float
    market_close_time: datetime
    # Order status tracking
    status: str = "PENDING"  # PENDING, FILLED, PARTIALLY_FILLED, CANCELLED, EXPIRED
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    last_status_check: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage"""
        return {
            "order_id": self.order_id,
            "market_slug": self.market_slug,
            "token_outcome": self.token_outcome,
            "order_price": self.order_price,
            "order_size": self.order_size,
            "order_time": self.order_time.isoformat(),
            "predicted_direction": self.predicted_direction,
            "confidence": self.confidence,
            "market_close_time": self.market_close_time.isoformat(),
            "status": self.status,
            "filled_size": self.filled_size,
            "avg_fill_price": self.avg_fill_price,
            "last_status_check": self.last_status_check.isoformat() if self.last_status_check else None,
        }


@dataclass
class PositionTracker:
    """Track actual filled positions for P&L analysis"""

    market_id: str
    market_slug: str
    token_id: str
    token_outcome: str
    position_size: float
    avg_entry_price: float
    current_price: float
    entry_time: datetime
    predicted_direction: str
    confidence: float
    market_close_time: datetime
    # Position status
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    final_outcome: Optional[str] = None  # "win" or "loss"
    pnl: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    last_price_update: Optional[datetime] = None
    
    def calculate_unrealized_pnl(self) -> float:
        """Calculate current unrealized PnL"""
        if self.current_price is None:
            return 0.0
        
        # For binary prediction markets, PnL is based on probability difference
        # If we bought at avg_entry_price and current price is current_price
        unrealized = self.position_size * (self.current_price - self.avg_entry_price)
        return unrealized
    
    def update_current_price(self, new_price: float):
        """Update current price and recalculate unrealized PnL"""
        self.current_price = new_price
        self.unrealized_pnl = self.calculate_unrealized_pnl()
        self.last_price_update = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage"""
        return {
            "market_id": self.market_id,
            "market_slug": self.market_slug,
            "token_outcome": self.token_outcome,
            "position_size": self.position_size,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "entry_time": self.entry_time.isoformat(),
            "predicted_direction": self.predicted_direction,
            "confidence": self.confidence,
            "market_close_time": self.market_close_time.isoformat(),
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
            "final_outcome": self.final_outcome,
            "pnl": self.pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "last_price_update": self.last_price_update.isoformat() if self.last_price_update else None,
        }


@dataclass
class TradingConfig:
    """Configuration for the trading bot"""

    # Model settings
    model_type: str = "logistic"  # Best performing from your analysis
    confidence_threshold: float = 0.75  # Minimum confidence to place orders
    safety_factor: float = 0.95  # Safety factor for order price
    # Trading settings
    max_position_size: float = 5.0  # Max USD to risk per trade
    max_daily_trades: int = 48  # Maximum trades per day
    order_timeout_minutes: int = 30  # Cancel orders after X minutes

    # Market settings
    market_slug: MARKETS = "ethereum"  # Market slug from constants
    min_liquidity: float = 100.0  # Minimum market liquidity required
    max_markets_to_check: int = 5  # Max future markets to analyze

    # Data settings
    data_refresh_interval: int = 30  # Refresh data every X seconds
    prediction_window_start: int = 42  # Start checking at 42 minutes
    prediction_window_end: int = 48  # Stop checking at 48 minutes
    
    # Position tracking settings
    position_check_interval: int = 60  # Check positions every X seconds
    order_status_check_interval: int = 30  # Check order status every X seconds

    def __post_init__(self):
        pass


@dataclass
class TradingState:
    """Current state of the trading bot"""

    current_hour_start: Optional[datetime] = None
    prediction_made: bool = False
    daily_trade_count: int = 0
    last_data_refresh: Optional[datetime] = None
    last_position_check: Optional[datetime] = None
    last_order_status_check: Optional[datetime] = None

    # Order tracking (orders placed but may not be filled)
    active_orders: Dict[str, OrderTracker] = field(default_factory=dict)
    completed_orders: Dict[str, OrderTracker] = field(default_factory=dict)
    
    # Position tracking (actual filled positions)
    open_positions: Dict[str, PositionTracker] = field(default_factory=dict)
    resolved_positions: List[PositionTracker] = field(default_factory=list)

    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0


class PolymarketHourlyTradingBot:
    """
    Main trading bot that combines hourly candle prediction with Polymarket trading
    """

    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.state = TradingState()

        # Initialize logging
        self.logger = logging.getLogger("HourlyTradingBot")
        self.logger.setLevel(logging.INFO)

        # Initialize components
        self.exchange = ccxt.binance({"enableRateLimit": True})
        self.polymarket = PolymarketClient()
        self.predictor = EthCandlePredictor(model_type=self.config.model_type)

        # Data storage
        self.ohlcv_data = pd.DataFrame()
        self.markets: list[Market] = []

        self.logger.info("Polymarket Hourly Trading Bot initialized")

    async def initialize(self):
        """Initialize the bot by loading historical data and training the model"""
        self.logger.info("Initializing bot...")

        # Load historical data for model training
        await self.load_historical_data()

        # Train the prediction model
        self.train_model()

        # Load Polymarket markets
        await self.refresh_markets()

        # Reconcile existing positions
        await self.reconcile_positions()

        self.logger.info("Bot initialization complete")

    async def load_historical_data(self, days: int = 30):
        """Load historical {TICKER}/USDT data for model training"""
        self.logger.info(
            f"Loading {days} days of historical {TICKERS[self.config.market_slug]} data..."
        )

        symbol = TICKERS[self.config.market_slug]
        timeframe = "1m"
        since = self.exchange.parse8601(
            (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        )

        all_ohlcv = []
        limit = 1000

        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                if len(ohlcv) == 0:
                    break

                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1

                # Rate limiting
                await asyncio.sleep(self.exchange.rateLimit / 1000)

                if len(all_ohlcv) % 5000 == 0:
                    self.logger.info(f"Loaded {len(all_ohlcv)} candles...")

            except Exception as e:
                self.logger.error(f"Error loading data: {e}")
                break

        # Convert to DataFrame
        self.ohlcv_data = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        self.ohlcv_data["timestamp"] = pd.to_datetime(
            self.ohlcv_data["timestamp"], unit="ms", utc=True
        )
        self.ohlcv_data = self.ohlcv_data.set_index("timestamp").sort_index()
        self.ohlcv_data = self.ohlcv_data[
            ~self.ohlcv_data.index.duplicated(keep="last")
        ]

        self.logger.info(f"Loaded {len(self.ohlcv_data)} 1-minute candles")

    def train_model(self):
        """Train the prediction model"""
        self.logger.info("Training prediction model...")

        if len(self.ohlcv_data) < 1000:
            raise ValueError("Not enough historical data to train model")

        # Train the model
        performance = self.predictor.train(self.ohlcv_data, test_size=0.2)

        self.logger.info("Model trained successfully:")
        self.logger.info(f"  Accuracy: {performance.accuracy:.3f}")
        self.logger.info(f"  AUC: {performance.auc_score:.3f}")

    async def refresh_markets(self):
        """Refresh hourly prediction markets"""
        self.logger.info(f"Refreshing {self.config.market_slug} hourly markets...")

        try:
            # Use the existing client method to get newest future series markets
            current_market_hour = get_current_market_hour_et()
            markets: list[Market] = [
                self.polymarket.get_market_by_slug(
                    get_current_market_slug(self.config.market_slug)
                )
            ]
            for _ in range(1, self.config.max_markets_to_check):
                current_market_hour += timedelta(hours=1)
                next_market_slug = create_slug_from_datetime(
                    current_market_hour, self.config.market_slug
                )
                markets.append(self.polymarket.get_market_by_slug(next_market_slug))

            # Filter for markets that are still active and have enough liquidity
            active_markets: list[Market] = []

            order_books = self.polymarket.get_order_books(
                [market.tokens[0].token_id for market in markets]
            )
            for market in markets:
                order_book = order_books[market.tokens[0].token_id]
                if order_book.get_liquidity() < self.config.min_liquidity:
                    continue
                active_markets.append(market)

            self.markets = active_markets

            self.logger.info(f"Found {len(active_markets)} active markets")

        except Exception as e:
            self.logger.error(f"Error refreshing {self.config.market_slug} markets: {e}")

    async def reconcile_positions(self):
        """Reconcile our internal position tracking with actual positions from Polymarket"""
        self.logger.info("Reconciling positions with Polymarket...")
        
        try:
            # Get actual positions from Polymarket
            actual_positions = self.polymarket.get_positions_by_fuzzy_slug(self.config.market_slug)
            
            # Clear existing positions and rebuild from actual data
            self.state.open_positions.clear()
            
            if actual_positions:
                for position in actual_positions:
                    if position.size > 0:  # Only track positions with actual size
                        # NOTE: position.market_id is actually the token_id (populated from API's 'asset' field)
                        # Try to find corresponding market info
                        market = self._find_market_by_token_id(position.market_id)
                        if not market:
                            self.logger.warning(f"Could not find market info for position {position.market_id}")
                            continue
                        
                        # Create position tracker
                        position_tracker = PositionTracker(
                            market_id=market.condition_id,
                            market_slug=position.slug,
                            token_id=position.market_id,
                            token_outcome=self._get_token_outcome(market, position.market_id),
                            position_size=position.size,
                            avg_entry_price=position.entry_price,
                            current_price=position.current_price or position.entry_price,
                            entry_time=position.last_updated or datetime.now(timezone.utc),
                            predicted_direction="unknown",  # We don't know the original prediction
                            confidence=0.0,  # We don't know the original confidence
                            market_close_time=self._get_market_close_time(position.slug),
                        )
                        
                        # Update unrealized PnL
                        position_tracker.update_current_price(position.current_price or position.entry_price)
                        
                        self.state.open_positions[position.market_id] = position_tracker
                        
                        self.logger.info(
                            f"Reconciled position: {position.slug} - Size: {position.size}, "
                            f"Entry: ${position.entry_price:.3f}, Current: ${position.current_price:.3f}"
                        )
            
            # Update total unrealized PnL
            self.state.total_unrealized_pnl = sum(
                pos.unrealized_pnl or 0 for pos in self.state.open_positions.values()
            )
            
            self.logger.info(f"Reconciled {len(self.state.open_positions)} positions")
            
        except Exception as e:
            self.logger.error(f"Error reconciling positions: {e}")
    
    def _find_market_by_token_id(self, token_id: str) -> Optional[Market]:
        """Find market by token ID"""
        for market in self.markets:
            for token in market.tokens:
                if token.token_id == token_id:
                    return market
        return None
    
    def _get_token_outcome(self, market: Market, token_id: str) -> str:
        """Get token outcome name"""
        for token in market.tokens:
            if token.token_id == token_id:
                return token.outcome
        return "Unknown"
    
    def _get_market_close_time(self, market_slug: str) -> datetime:
        """Get market close time from slug"""
        try:
            market_time = slug_to_datetime(market_slug)
            return market_time.replace(tzinfo=ET) + timedelta(hours=1)
        except Exception:
            return datetime.now(timezone.utc) + timedelta(hours=1)

    async def refresh_data(self):
        """Refresh price data and market information"""
        if (
            self.state.last_data_refresh is None
            or (datetime.now(timezone.utc) - self.state.last_data_refresh).total_seconds()
            > self.config.data_refresh_interval
        ):
            try:
                # Refresh recent price data
                current_time = datetime.now(timezone.utc)
                symbol = TICKERS[self.config.market_slug]
                
                # Get data from last hour
                since = self.exchange.parse8601(
                    (current_time - timedelta(hours=1)).isoformat()
                )
                
                recent_ohlcv = self.exchange.fetch_ohlcv(symbol, "1m", since)
                
                if recent_ohlcv:
                    # Convert to DataFrame
                    recent_df = pd.DataFrame(
                        recent_ohlcv, 
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    recent_df["timestamp"] = pd.to_datetime(
                        recent_df["timestamp"], unit="ms", utc=True
                    )
                    recent_df = recent_df.set_index("timestamp").sort_index()
                    
                    # Update main data
                    self.ohlcv_data = pd.concat([self.ohlcv_data, recent_df])
                    self.ohlcv_data = self.ohlcv_data[
                        ~self.ohlcv_data.index.duplicated(keep="last")
                    ]
                    
                    # Keep only last 7 days to prevent memory issues
                    cutoff_time = current_time - timedelta(days=7)
                    self.ohlcv_data = self.ohlcv_data[self.ohlcv_data.index >= cutoff_time]
                
                self.state.last_data_refresh = current_time
                
            except Exception as e:
                self.logger.error(f"Error refreshing {self.config.market_slug} data: {e}")

    async def update_order_status(self):
        """Update status of active orders"""
        if not self.state.active_orders:
            return
            
        if (
            self.state.last_order_status_check is None
            or (datetime.now(timezone.utc) - self.state.last_order_status_check).total_seconds()
            > self.config.order_status_check_interval
        ):
            try:
                # Get current orders from Polymarket
                current_orders = self.polymarket.get_orders()
                current_order_ids = {order.order_id for order in current_orders}
                
                # Update status of our tracked orders
                orders_to_remove = []
                for order_id, order_tracker in self.state.active_orders.items():
                    # Find the current order
                    current_order = next(
                        (order for order in current_orders if order.order_id == order_id),
                        None
                    )
                    
                    if current_order:
                        # Update order status
                        order_tracker.status = current_order.status
                        order_tracker.filled_size = current_order.matched_size
                        order_tracker.last_status_check = datetime.now(timezone.utc)
                        
                        # If order is fully filled, calculate average fill price
                        if current_order.matched_size > 0:
                            order_tracker.avg_fill_price = current_order.price
                        
                        # If order is complete (filled or cancelled), move to completed
                        if current_order.status in ["FILLED", "CANCELLED", "EXPIRED"]:
                            orders_to_remove.append(order_id)
                            self.state.completed_orders[order_id] = order_tracker
                            
                    else:
                        # Order no longer exists, likely filled or cancelled
                        order_tracker.status = "UNKNOWN"
                        order_tracker.last_status_check = datetime.now(timezone.utc)
                        orders_to_remove.append(order_id)
                        self.state.completed_orders[order_id] = order_tracker
                
                # Remove completed orders from active tracking
                for order_id in orders_to_remove:
                    del self.state.active_orders[order_id]
                
                self.state.last_order_status_check = datetime.now(timezone.utc)
                
                if orders_to_remove:
                    self.logger.info(f"Updated status for {len(orders_to_remove)} orders")
                
            except Exception as e:
                self.logger.error(f"Error updating order status: {e}")

    async def update_position_status(self):
        """Update status and prices of current positions"""
        if not self.state.open_positions:
            return
            
        if (
            self.state.last_position_check is None
            or (datetime.now(timezone.utc) - self.state.last_position_check).total_seconds()
            > self.config.position_check_interval
        ):
            try:
                # Get current positions from Polymarket
                actual_positions = self.polymarket.get_positions_by_fuzzy_slug(self.config.market_slug)
                actual_position_dict = {pos.market_id: pos for pos in actual_positions}
                
                # Update our tracked positions
                positions_to_remove = []
                for market_id, position in self.state.open_positions.items():
                    actual_position = actual_position_dict.get(market_id)
                    
                    if actual_position and actual_position.size > 0:
                        # Update position info
                        position.position_size = actual_position.size
                        position.avg_entry_price = actual_position.entry_price
                        position.update_current_price(
                            actual_position.current_price or actual_position.entry_price
                        )
                    else:
                        # Position no longer exists or size is 0
                        positions_to_remove.append(market_id)
                
                # Remove closed positions
                for market_id in positions_to_remove:
                    closed_position = self.state.open_positions[market_id]
                    # Check if it should be resolved
                    if datetime.now(timezone.utc) > closed_position.market_close_time:
                        await self._resolve_position(closed_position)
                    del self.state.open_positions[market_id]
                
                # Update total unrealized PnL
                self.state.total_unrealized_pnl = sum(
                    pos.unrealized_pnl or 0 for pos in self.state.open_positions.values()
                )
                
                self.state.last_position_check = datetime.now(timezone.utc)
                
                if positions_to_remove:
                    self.logger.info(f"Updated {len(positions_to_remove)} position statuses")
                
            except Exception as e:
                self.logger.error(f"Error updating position status: {e}")

    def should_make_prediction(self) -> bool:
        """Check if we should make a prediction based on current time"""
        current_time = datetime.now(timezone.utc)
        current_hour_start = current_time.replace(minute=0, second=0, microsecond=0)

        # Check if we're in a new hour
        if self.state.current_hour_start != current_hour_start:
            self.state.current_hour_start = current_hour_start
            self.state.prediction_made = False

        # Check if we're in the prediction window
        minutes_elapsed = current_time.minute
        in_prediction_window = (
            self.config.prediction_window_start
            <= minutes_elapsed
            <= self.config.prediction_window_end
        )

        return in_prediction_window and not self.state.prediction_made

    async def make_prediction(self) -> Optional[PredictionResult]:
        """Make a prediction for the current hour"""
        try:
            current_hour_start = self.state.current_hour_start

            # Make prediction
            prediction = self.predictor.predict(
                self.ohlcv_data, pd.Timestamp(current_hour_start)
            )

            self.logger.info(f"Prediction for hour {current_hour_start}:")
            self.logger.info(f"  Direction: {prediction.predicted_direction}")
            self.logger.info(f"  Confidence: {prediction.confidence:.1%}")
            self.logger.info(f"  Prob Green: {prediction.probability_green:.1%}")
            self.logger.info(f"  Prob Red: {prediction.probability_red:.1%}")

            self.state.prediction_made = True
            return prediction

        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return None

    def find_matching_markets(
        self, prediction: PredictionResult
    ) -> List[tuple[Market, str]]:
        """Find markets that match the prediction direction"""
        matching_markets = []

        for market in self.markets:
            try:
                # Parse market slug to get the time
                market_time = slug_to_datetime(market.market_slug)
                current_hour_start = self.state.current_hour_start

                # Check if this market corresponds to the current prediction window
                if (
                    market_time.replace(tzinfo=timezone.utc).replace(minute=0, second=0, microsecond=0)
                    == current_hour_start
                ):
                    # Find the token that matches our prediction
                    for token in market.tokens:
                        if (
                            prediction.predicted_direction == "green"
                            and token.outcome.lower() in ["yes", "up", "green"]
                        ):
                            matching_markets.append((market, token.token_id))
                            break
                        elif (
                            prediction.predicted_direction == "red"
                            and token.outcome.lower() in ["no", "down", "red"]
                        ):
                            matching_markets.append((market, token.token_id))
                            break

            except Exception as e:
                self.logger.error(f"Error processing market {market.market_slug}: {e}")
                continue

        return matching_markets

    async def analyze_order_book(self, token_id: str) -> dict:
        """Analyze the order book for a specific token"""
        try:
            order_book = self.polymarket.get_order_book(token_id)

            if not order_book.bids or not order_book.asks:
                return {}

            # Get best bid and ask
            best_bid = float(order_book.bids[0].price) if order_book.bids else 0.0
            best_ask = float(order_book.asks[0].price) if order_book.asks else 1.0

            # Calculate midpoint
            midpoint = (best_bid + best_ask) / 2

            # Calculate total liquidity
            bid_liquidity = sum(
                float(bid.price) * float(bid.size) for bid in order_book.bids
            ) if order_book.bids else 0.0
            ask_liquidity = sum(
                float(ask.price) * float(ask.size) for ask in order_book.asks
            ) if order_book.asks else 0.0

            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "midpoint": midpoint,
                "bid_liquidity": bid_liquidity,
                "ask_liquidity": ask_liquidity,
                "spread": best_ask - best_bid,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing order book for {token_id}: {e}")
            return {}

    def calculate_position_size(self, confidence: float, market_odds: float) -> float:
        """Calculate position size based on confidence and market odds"""
        # Kelly criterion inspired sizing
        edge = confidence - market_odds

        if edge <= 0:
            return 0

        # Conservative Kelly fraction
        kelly_fraction = edge / (1 - market_odds)
        position_fraction = min(kelly_fraction * 0.25, 0.1)  # Max 10% of bankroll

        # Calculate position size with minimum of $5
        position_size = min(
            position_fraction * self.config.max_position_size,
            self.config.max_position_size * 0.5,
        )  # Max 50% of max size per trade

        return max(position_size, 5)  # Enforce minimum $5 position size

    async def place_limit_order(
        self,
        market: Market,
        token_id: str,
        prediction: PredictionResult,
        order_book_info: dict,
    ) -> Optional[str]:
        """Place a limit order based on prediction and order book analysis"""
        try:
            # Check confidence threshold
            if prediction.confidence < self.config.confidence_threshold:
                self.logger.info(
                    f"Confidence {prediction.confidence:.1%} below threshold {self.config.confidence_threshold:.1%}"
                )
                return None

            # Check daily trade limit
            if self.state.daily_trade_count >= self.config.max_daily_trades:
                self.logger.info("Daily trade limit reached")
                return None

            # Determine order side and price
            market_prob = order_book_info.get("midpoint", 0.5)

            side = BookSide.BUY
            target_price = min(
                order_book_info.get("best_ask", 1.0),
                prediction.confidence * self.config.safety_factor,
            )
            best_bid = order_book_info.get("best_bid", 0.0)
            if target_price <= best_bid:
                self.logger.info(
                    f"Target price {target_price:.3f} not better than best bid {best_bid:.3f}"
                )
                return None

            # Calculate position size
            position_size = self.calculate_position_size(
                prediction.confidence, market_prob
            )

            if position_size < 5:  # Minimum size 5 order
                self.logger.info(f"Position size ${position_size:.2f} too small")
                return None

            # Check if the order is too small
            if position_size * target_price < 1:  # Minimum size 5 order
                self.logger.info(f"Position volume ${position_size:.2f} too small")
                return None

            target_price = round(
                round(target_price / market.minimum_tick_size)
                * market.minimum_tick_size,
                3,
            )
            target_size = float(round(position_size))
            
            # Create and place order
            order_args = OrderArgsModel(
                token_id=token_id, price=target_price, size=target_size, side=side
            )

            signed_order = self.polymarket.create_order(order_args)
            order_id = self.polymarket.place_order(signed_order)

            if order_id:
                self.logger.info("Order placed successfully:")
                self.logger.info(f"  Order ID: {order_id}")
                self.logger.info(f"  Market: {market.question}")
                self.logger.info(f"  Side: {side}")
                self.logger.info(f"  Price: ${target_price:.3f}")
                self.logger.info(f"  Size: ${position_size:.2f}")

                self.state.daily_trade_count += 1

                market_close_time = slug_to_datetime(market.market_slug) + timedelta(
                    hours=1
                )
                market_close_time = market_close_time.replace(tzinfo=ET)

                # Create order tracker (not position tracker - position created when filled)
                order_tracker = OrderTracker(
                    order_id=order_id,
                    market_slug=market.market_slug,
                    token_id=token_id,
                    token_outcome=next(
                        (
                            token.outcome
                            for token in market.tokens
                            if token.token_id == token_id
                        ),
                        "Unknown",
                    ),
                    order_price=target_price,
                    order_size=target_size,
                    order_time=datetime.now(timezone.utc),
                    predicted_direction=prediction.predicted_direction,
                    confidence=prediction.confidence,
                    market_close_time=market_close_time,
                    status="PENDING",
                )

                # Add to active orders (not positions)
                self.state.active_orders[order_id] = order_tracker

                return order_id

        except Exception as e:
            self.logger.error(f"Error placing order: {e}")

        return None

    async def cancel_expired_orders(self):
        """Cancel orders that have been open too long"""
        if not self.state.active_orders:
            return

        try:
            current_time = datetime.now(timezone.utc)
            orders_to_cancel = []
            
            for order_id, order_tracker in self.state.active_orders.items():
                # Check if order is older than timeout
                if (current_time - order_tracker.order_time).total_seconds() > (
                    self.config.order_timeout_minutes * 60
                ):
                    orders_to_cancel.append(order_id)

            if orders_to_cancel:
                self.logger.info(f"Cancelling {len(orders_to_cancel)} expired orders")
                removed_orders, failed_to_cancel = self.polymarket.cancel_orders(
                    orders_to_cancel
                )
                self.logger.info(f"Removed {len(removed_orders)} orders")
                self.logger.info(f"Failed to cancel {len(failed_to_cancel)} orders")
                
                # Update status of cancelled orders
                for order_id in removed_orders:
                    if order_id in self.state.active_orders:
                        self.state.active_orders[order_id].status = "CANCELLED"

        except Exception as e:
            self.logger.error(f"Error cancelling expired orders: {e}")

    async def check_new_positions(self):
        """Check for new positions from filled orders"""
        try:
            # Get current positions from Polymarket
            current_positions = self.polymarket.get_positions_by_fuzzy_slug(self.config.market_slug)
            
            if current_positions:
                for position in current_positions:
                    if position.size > 0 and position.market_id not in self.state.open_positions:
                        # NOTE: position.market_id is actually the token_id (populated from API's 'asset' field)
                        # Check if we have a completed order for this position
                        corresponding_order = None
                        for order_tracker in self.state.completed_orders.values():
                            if (order_tracker.token_id == position.market_id and 
                                order_tracker.status == "FILLED"):
                                corresponding_order = order_tracker
                                break
                        
                        # Create position tracker
                        position_tracker = PositionTracker(
                            market_id=self._find_market_by_token_id(position.market_id).condition_id if self._find_market_by_token_id(position.market_id) else "unknown",
                            market_slug=position.slug,
                            token_id=position.market_id,  # position.market_id is actually token_id
                            token_outcome=self._get_token_outcome(
                                self._find_market_by_token_id(position.market_id),
                                position.market_id
                            ) if self._find_market_by_token_id(position.market_id) else "Unknown",
                            position_size=position.size,
                            avg_entry_price=position.entry_price,
                            current_price=position.current_price or position.entry_price,
                            entry_time=position.last_updated or datetime.now(timezone.utc),
                            predicted_direction=corresponding_order.predicted_direction if corresponding_order else "unknown",
                            confidence=corresponding_order.confidence if corresponding_order else 0.0,
                            market_close_time=self._get_market_close_time(position.slug),
                        )
                        
                        position_tracker.update_current_price(position.current_price or position.entry_price)
                        self.state.open_positions[position.market_id] = position_tracker
                        
                        self.logger.info(f"New position detected: {position.slug} - Size: {position.size}")
            
        except Exception as e:
            self.logger.error(f"Error checking new positions: {e}")

    async def track_resolved_positions(self):
        """Track resolved positions for P&L analysis"""
        if not self.state.open_positions:
            return

        current_time = datetime.now(timezone.utc)
        positions_to_resolve = []

        try:
            for market_id, position in self.state.open_positions.items():
                # Check if market should be resolved (closed + some buffer time)
                if current_time > position.market_close_time + timedelta(minutes=5):
                    positions_to_resolve.append(position)

            if not positions_to_resolve:
                return

            # Resolve positions
            for position in positions_to_resolve:
                # Try to determine if position won or lost
                resolved_position = await self._resolve_position(position)

                if resolved_position:
                    # Update position
                    resolved_position.resolved = True
                    resolved_position.resolution_time = current_time

                    # Add to resolved positions
                    self.state.resolved_positions.append(resolved_position)

                    # Update performance metrics
                    self.state.total_trades += 1
                    self.state.total_pnl += resolved_position.pnl or 0
                    self.state.daily_pnl += resolved_position.pnl or 0

                    if resolved_position.final_outcome == "win":
                        self.state.winning_trades += 1

                    # Log the result
                    self.logger.info(
                        f"Position resolved: {resolved_position.market_slug}"
                    )
                    self.logger.info(f"  Outcome: {resolved_position.final_outcome}")
                    self.logger.info(f"  P&L: ${resolved_position.pnl:.2f}")
                    self.logger.info(
                        f"  Win Rate: {self.state.winning_trades / self.state.total_trades:.1%}"
                    )

                    # Remove from open positions
                    del self.state.open_positions[position.market_id]

        except Exception as e:
            self.logger.error(f"Error tracking resolved positions: {e}")

    async def _resolve_position(
        self, position: PositionTracker
    ) -> Optional[PositionTracker]:
        """Resolve a single position by checking market outcome"""
        try:
            # Get the market to check resolution
            market_close_time = position.market_close_time

            # Get candle data around the market close time
            close_candles = self.ohlcv_data[
                (self.ohlcv_data.index >= market_close_time - timedelta(minutes=5))
                & (self.ohlcv_data.index <= market_close_time + timedelta(minutes=5))
            ]

            if close_candles.empty:
                return None

            # Determine actual outcome based on price movement
            hour_start = market_close_time - timedelta(hours=1)
            hour_start = hour_start.replace(minute=0, second=0, microsecond=0)

            start_candles = self.ohlcv_data[
                (self.ohlcv_data.index >= hour_start)
                & (self.ohlcv_data.index < hour_start + timedelta(minutes=5))
            ]

            if start_candles.empty:
                return None

            # Get opening and closing prices
            open_price = start_candles.iloc[0]["open"]
            close_price = close_candles.iloc[-1]["close"]

            # Determine actual direction
            actual_direction = "green" if close_price > open_price else "red"

            # Check if prediction was correct
            prediction_correct = position.predicted_direction == actual_direction

            # Calculate P&L based on actual position
            if prediction_correct:
                # Won the bet - position now worth $1 per share
                pnl = position.position_size * (1.0 - position.avg_entry_price)
                final_outcome = "win"
            else:
                # Lost the bet - position now worth $0 per share
                pnl = position.position_size * (0.0 - position.avg_entry_price)
                final_outcome = "loss"

            # Create resolved position
            resolved_position = PositionTracker(
                market_id=position.market_id,
                market_slug=position.market_slug,
                token_id=position.token_id,
                token_outcome=position.token_outcome,
                position_size=position.position_size,
                avg_entry_price=position.avg_entry_price,
                current_price=1.0 if prediction_correct else 0.0,
                entry_time=position.entry_time,
                predicted_direction=position.predicted_direction,
                confidence=position.confidence,
                market_close_time=position.market_close_time,
                resolved=True,
                resolution_time=datetime.now(timezone.utc),
                final_outcome=final_outcome,
                pnl=pnl,
            )

            return resolved_position

        except Exception as e:
            self.logger.error(f"Error resolving position {position.market_id}: {e}")
            return None

    def get_performance_summary(self) -> Dict:
        """Get current performance summary"""
        win_rate = (
            self.state.winning_trades / self.state.total_trades
            if self.state.total_trades > 0
            else 0
        )
        avg_pnl = (
            self.state.total_pnl / self.state.total_trades
            if self.state.total_trades > 0
            else 0
        )

        return {
            "total_trades": self.state.total_trades,
            "winning_trades": self.state.winning_trades,
            "win_rate": win_rate,
            "total_pnl": self.state.total_pnl,
            "daily_pnl": self.state.daily_pnl,
            "total_unrealized_pnl": self.state.total_unrealized_pnl,
            "average_pnl_per_trade": avg_pnl,
            "open_positions": len(self.state.open_positions),
            "active_orders": len(self.state.active_orders),
            "completed_orders": len(self.state.completed_orders),
        }

    def log_performance_summary(self):
        """Log current performance summary"""
        summary = self.get_performance_summary()
        self.logger.info("=== Performance Summary ===")
        self.logger.info(f"Total Trades: {summary['total_trades']}")
        self.logger.info(f"Win Rate: {summary['win_rate']:.1%}")
        self.logger.info(f"Daily P&L: ${summary['daily_pnl']:.2f}")
        self.logger.info(f"Total P&L: ${summary['total_pnl']:.2f}")
        self.logger.info(f"Unrealized P&L: ${summary['total_unrealized_pnl']:.2f}")
        self.logger.info(f"Avg P&L per Trade: ${summary['average_pnl_per_trade']:.2f}")
        self.logger.info(f"Open Positions: {summary['open_positions']}")
        self.logger.info(f"Active Orders: {summary['active_orders']}")
        self.logger.info(f"Completed Orders: {summary['completed_orders']}")
        self.logger.info("===========================")

    def log_daily_summary(self):
        """Log daily performance summary"""
        self.log_performance_summary()
        self.logger.info("=== End of Day Summary ===")

    def save_position_data(self):
        """Save position tracking data to JSON file"""
        try:
            import json
            from datetime import datetime

            data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "resolved_positions": [
                    pos.to_dict() for pos in self.state.resolved_positions
                ],
                "open_positions": [
                    pos.to_dict() for pos in self.state.open_positions.values()
                ],
                "active_orders": [
                    order.to_dict() for order in self.state.active_orders.values()
                ],
                "completed_orders": [
                    order.to_dict() for order in self.state.completed_orders.values()
                ],
                "performance_summary": self.get_performance_summary(),
            }

            filename = f"position_tracking_{datetime.now().strftime('%Y%m%d')}.json"
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Position data saved to {filename}")

        except Exception as e:
            self.logger.error(f"Error saving position data: {e}")

    async def run_trading_cycle(self):
        """Run one complete trading cycle"""
        # Refresh data
        await self.refresh_data()

        # Check if we should make a prediction
        if not self.should_make_prediction():
            # TODO we could make a prediction whether to exit earlier and not let the position expire
            return

        self.logger.info("Making prediction for current hour...")

        # Make prediction
        prediction = await self.make_prediction()
        if not prediction:
            return

        # Check confidence threshold
        if prediction.confidence < self.config.confidence_threshold:
            self.logger.info(
                f"Confidence {prediction.confidence:.1%} below threshold, skipping trading"
            )
            return

        # Find matching markets
        matching_markets = self.find_matching_markets(prediction)
        if not matching_markets:
            self.logger.info(
                f"No matching {self.config.market_slug} markets found for prediction"
            )
            return

        self.logger.info(
            f"Found {len(matching_markets)} matching {self.config.market_slug} markets"
        )

        # Analyze each market and potentially place orders
        for market, token_id in matching_markets:
            try:
                self.logger.info(f"Analyzing market: {market.question}")

                # Analyze order book
                order_book_info = await self.analyze_order_book(token_id)
                if not order_book_info:
                    continue

                # Check if market is liquid enough
                total_liquidity = order_book_info.get(
                    "bid_liquidity", 0
                ) + order_book_info.get("ask_liquidity", 0)
                if total_liquidity < self.config.min_liquidity:
                    self.logger.info(
                        f"Market liquidity ${total_liquidity:.2f} below minimum"
                    )
                    continue

                # Place order
                order_id = await self.place_limit_order(
                    market, token_id, prediction, order_book_info
                )
                if order_id:
                    self.logger.info(f"Successfully placed order {order_id}")
                    break  # Only place one order per prediction

            except Exception as e:
                self.logger.error(f"Error processing market {market.question}: {e}")
                continue

    async def run(self):
        """Main bot loop"""
        self.logger.info("Starting Polymarket Hourly Trading Bot...")

        # Initialize
        await self.initialize()

        # Main loop
        while True:
            try:
                # Run trading cycle
                await self.run_trading_cycle()

                # Update order and position status
                await self.update_order_status()
                await self.update_position_status()
                await self.check_new_positions()

                # Cancel expired orders
                await self.cancel_expired_orders()

                # Track resolved positions
                await self.track_resolved_positions()

                # Refresh markets every hour
                current_time = datetime.now(timezone.utc)
                if current_time.minute == 0:
                    await self.refresh_markets()

                    # Log performance summary every hour
                    if current_time.hour % 1 == 0:
                        self.log_performance_summary()

                # Reset daily counter at midnight UTC
                if current_time.hour == 0 and current_time.minute == 0:
                    self.log_daily_summary()
                    self.save_position_data()
                    self.state.daily_trade_count = 0
                    self.state.daily_pnl = 0.0
                    self.logger.info("Reset daily counters")

                # Sleep for configured interval
                await asyncio.sleep(self.config.data_refresh_interval)

            except KeyboardInterrupt:
                self.logger.info("Bot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("hourly_trading_bot.log"),
            logging.StreamHandler(),
        ],
    )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Create and run bot
    config = TradingConfig(
        confidence_threshold=0.72,  # Higher threshold for safer trading
        max_position_size=5.0,  # Max $5 per trade
    )

    bot = PolymarketHourlyTradingBot(config)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
