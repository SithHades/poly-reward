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

from constants import MARKETS, TICKERS
from parsing_utils import (
    ET,
    create_slug_from_datetime,
    get_current_market_hour_et,
    get_current_market_slug,
    slug_to_datetime,
)
from src.eth_candle_predictor import EthCandlePredictor, PredictionResult
from src.polymarket_client import PolymarketClient
from src.models import Market, OrderArgsModel, BookSide


@dataclass
class PositionTracker:
    """Track individual position for P&L analysis"""

    order_id: str
    market_question: str
    token_id: str
    token_outcome: str
    entry_price: float
    position_size: float
    entry_time: datetime
    predicted_direction: str
    confidence: float
    market_close_time: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    final_outcome: Optional[str] = None  # "win" or "loss"
    pnl: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage"""
        return {
            "order_id": self.order_id,
            "market_question": self.market_question,
            "token_outcome": self.token_outcome,
            "entry_price": self.entry_price,
            "position_size": self.position_size,
            "entry_time": self.entry_time.isoformat(),
            "predicted_direction": self.predicted_direction,
            "confidence": self.confidence,
            "market_close_time": self.market_close_time.isoformat(),
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat()
            if self.resolution_time
            else None,
            "final_outcome": self.final_outcome,
            "pnl": self.pnl,
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

    def __post_init__(self):
        pass


@dataclass
class TradingState:
    """Current state of the trading bot"""

    current_hour_start: Optional[datetime] = None
    prediction_made: bool = False
    orders_placed: List[str] = field(default_factory=list)
    daily_trade_count: int = 0
    last_data_refresh: Optional[datetime] = None

    # Position tracking
    open_positions: Dict[str, PositionTracker] = field(default_factory=dict)
    resolved_positions: List[PositionTracker] = field(default_factory=list)

    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0


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
            self.logger.info(
                f"Found {len(active_markets)} active {self.config.market_slug} hourly markets"
            )

            # Log market details for debugging
            for market in active_markets:
                self.logger.info(f"  Market: {market.question}")
                self.logger.info(
                    f"  End time: {market.end_date_iso.strftime('%Y-%m-%d %H:%M:%S') if market.end_date_iso else 'N/A'}"
                )
                self.logger.info(f"  Tokens: {list(market.tokens.keys())}")
                self.logger.info(
                    f"  Liquidity: ${order_books[market.tokens[0].token_id].get_liquidity():,.2f}"
                )

        except Exception as e:
            self.logger.error(
                f"Error refreshing {self.config.market_slug} markets: {e}"
            )
            self.markets = []

    async def refresh_data(self):
        """Refresh recent data"""
        current_time = datetime.now(timezone.utc)

        # Check if we need to refresh
        if (
            self.state.last_data_refresh
            and (current_time - self.state.last_data_refresh).total_seconds()
            < self.config.data_refresh_interval
        ):
            return

        try:
            # Get last 2 hours of data to ensure we have enough for current hour
            # Calculate since based on last data point we have
            if self.ohlcv_data is not None and not self.ohlcv_data.empty:
                # Get timestamp of last candle and subtract 5 minutes for safety
                last_timestamp = self.ohlcv_data.index[-1]
                since = self.exchange.parse8601(
                    (last_timestamp - timedelta(minutes=5)).isoformat()
                )
                # Calculate how many minutes we need plus safety margin
                minutes_needed = (
                    int((current_time - last_timestamp).total_seconds() / 60) + 5
                )
                limit = min(120, minutes_needed)  # Cap at 120 minutes
            else:
                # If no data, get last 2 hours
                since = self.exchange.parse8601(
                    (current_time - timedelta(hours=2)).isoformat()
                )
                limit = 120

            # Fetch data with overlap to ensure no missing minutes
            ohlcv = self.exchange.fetch_ohlcv(
                TICKERS[self.config.market_slug], "1m", since, limit
            )

            if ohlcv:
                # Convert to DataFrame
                new_data = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                new_data["timestamp"] = pd.to_datetime(
                    new_data["timestamp"], unit="ms", utc=True
                )
                new_data = new_data.set_index("timestamp").sort_index()

                # Merge with existing data
                self.ohlcv_data = pd.concat([self.ohlcv_data, new_data])
                self.ohlcv_data = self.ohlcv_data[
                    ~self.ohlcv_data.index.duplicated(keep="last")
                ]
                self.ohlcv_data = self.ohlcv_data.sort_index()

                # Keep only last 48 hours to manage memory
                cutoff = current_time - timedelta(hours=48)
                self.ohlcv_data = self.ohlcv_data[self.ohlcv_data.index > cutoff]

                self.state.last_data_refresh = current_time
                self.logger.debug(
                    f"Refreshed {self.config.market_slug} data: {len(new_data)} new candles"
                )

        except Exception as e:
            self.logger.error(f"Error refreshing {self.config.market_slug} data: {e}")

    def should_make_prediction(self) -> bool:
        """Check if we should make a prediction based on current time"""
        current_time = datetime.now(timezone.utc)
        current_hour_start = current_time.replace(minute=0, second=0, microsecond=0)

        # Check if we're in a new hour
        if self.state.current_hour_start != current_hour_start:
            self.state.current_hour_start = current_hour_start
            self.state.prediction_made = False
            self.state.orders_placed = []

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
        """Find markets that match our prediction and timing"""
        matching_markets = []
        current_time = datetime.now(timezone.utc)
        closes_in = (
            current_time.replace(
                hour=current_time.hour + 1, minute=0, second=0, microsecond=0
            )
            - current_time
        ).total_seconds() / 60

        # We want markets that close around the next hour boundary
        current_hour_slug = get_current_market_slug()

        for market in self.markets:
            # Check if market closes within reasonable time window (next 30-90 minutes)
            market_slug = market.market_slug

            if market_slug == current_hour_slug:  # Market closes in 30-90 minutes
                # Find the appropriate token based on prediction
                target_token_id = None

                if prediction.predicted_direction == "green":
                    # Look for "up", "green", "yes" tokens for green candle
                    for token in market.tokens:
                        outcome_lower = token.outcome.lower()
                        if any(
                            word in outcome_lower
                            for word in [
                                "up",
                                "green",
                                "yes",
                                "higher",
                                "above",
                                "rise",
                            ]
                        ):
                            target_token_id = token.token_id
                            break
                else:
                    # Look for "down", "red", "no" tokens for red candle
                    for token in market.tokens:
                        outcome_lower = token.outcome.lower()
                        if any(
                            word in outcome_lower
                            for word in ["down", "red", "no", "lower", "below", "fall"]
                        ):
                            target_token_id = token.token_id
                            break

                if target_token_id:
                    matching_markets.append((market, target_token_id))
                    self.logger.info(f"Found matching market: {market.question}")
                    self.logger.info(f"  Target token: {token.outcome}")
                    self.logger.info(f"  Closes in: {closes_in:.1f} minutes")

        return matching_markets

    async def analyze_order_book(self, token_id: str) -> dict:
        """Analyze order book for a specific token"""
        try:
            order_book = self.polymarket.get_order_book(token_id)

            # Calculate best bid/ask
            best_bid = (
                max([float(bid["price"]) for bid in order_book.bids])
                if order_book.bids
                else 0
            )
            best_ask = (
                min([float(ask["price"]) for ask in order_book.asks])
                if order_book.asks
                else 1
            )

            # Calculate spread
            spread = best_ask - best_bid if best_bid > 0 and best_ask < 1 else 0

            # Calculate total liquidity
            bid_liquidity = sum([float(bid["size"]) for bid in order_book.bids])
            ask_liquidity = sum([float(ask["size"]) for ask in order_book.asks])

            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "bid_liquidity": bid_liquidity,
                "ask_liquidity": ask_liquidity,
                "midpoint": (best_bid + best_ask) / 2
                if best_bid > 0 and best_ask < 1
                else 0.5,
                "order_book": order_book,
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

                self.state.orders_placed.append(order_id)
                self.state.daily_trade_count += 1

                market_close_time = slug_to_datetime(market.market_slug) + timedelta(
                    hours=1
                )
                market_close_time = market_close_time.replace(tzinfo=ET)

                # Create position tracker
                position_tracker = PositionTracker(
                    order_id=order_id,
                    market_question=market.question,
                    token_id=token_id,
                    token_outcome=next(
                        (
                            token.outcome
                            for token in market.tokens
                            if token.token_id == token_id
                        ),
                        "Unknown",
                    ),
                    entry_price=target_price,
                    position_size=target_size,
                    entry_time=datetime.now(timezone.utc),
                    predicted_direction=prediction.predicted_direction,
                    confidence=prediction.confidence,
                    market_close_time=market_close_time,
                )

                # Add to open positions
                self.state.open_positions[order_id] = position_tracker

                return order_id

        except Exception as e:
            self.logger.error(f"Error placing order: {e}")

        return None

    async def cancel_expired_orders(self):
        """Cancel orders that have been open too long"""
        if not self.state.orders_placed:
            return

        try:
            # Get current orders
            open_orders = self.polymarket.get_orders()
            current_time = datetime.now(timezone.utc)

            orders_to_cancel = []
            for order in open_orders:
                if order.order_id in self.state.orders_placed:
                    # Check if order is older than timeout
                    order_time = datetime.fromtimestamp(
                        order.created_at / 1000, tz=timezone.utc
                    )
                    if (current_time - order_time).total_seconds() > (
                        self.config.order_timeout_minutes * 60
                    ):
                        orders_to_cancel.append(order.order_id)

            if orders_to_cancel:
                self.logger.info(f"Cancelling {len(orders_to_cancel)} expired orders")
                removed_orders, failed_to_cancel = self.polymarket.cancel_orders(
                    orders_to_cancel
                )
                self.logger.info(f"Removed {len(removed_orders)} orders")
                self.logger.info(f"Failed to cancel {len(failed_to_cancel)} orders")
                # Remove from our tracking
                for order_id in removed_orders:
                    if order_id in self.state.orders_placed:
                        self.state.orders_placed.remove(order_id)

        except Exception as e:
            self.logger.error(f"Error cancelling expired orders: {e}")

    async def track_resolved_positions(self):
        """Track resolved positions for P&L analysis"""
        if not self.state.open_positions:
            return

        current_time = datetime.now(timezone.utc)
        positions_to_resolve = []

        try:
            for order_id, position in self.state.open_positions.items():
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
                        f"Position resolved: {resolved_position.market_question}"
                    )
                    self.logger.info(f"  Outcome: {resolved_position.final_outcome}")
                    self.logger.info(f"  P&L: ${resolved_position.pnl:.2f}")
                    self.logger.info(
                        f"  Win Rate: {self.state.winning_trades / self.state.total_trades:.1%}"
                    )

                    # Remove from open positions
                    del self.state.open_positions[position.order_id]

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

            # Calculate P&L
            if prediction_correct:
                # Won the bet - get back entry cost plus winnings
                pnl = (
                    position.position_size
                    * (1 - position.entry_price)
                    / position.entry_price
                )
                final_outcome = "win"
            else:
                # Lost the bet - lose the entry cost
                pnl = -position.position_size
                final_outcome = "loss"

            # Create resolved position
            resolved_position = PositionTracker(
                order_id=position.order_id,
                market_question=position.market_question,
                token_id=position.token_id,
                token_outcome=position.token_outcome,
                entry_price=position.entry_price,
                position_size=position.position_size,
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
            self.logger.error(f"Error resolving position {position.order_id}: {e}")
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
            "average_pnl_per_trade": avg_pnl,
            "open_positions": len(self.state.open_positions),
        }

    def log_daily_summary(self):
        """Log daily performance summary"""
        summary = self.get_performance_summary()
        self.logger.info("=== Daily Performance Summary ===")
        self.logger.info(f"Total Trades: {summary['total_trades']}")
        self.logger.info(f"Win Rate: {summary['win_rate']:.1%}")
        self.logger.info(f"Daily P&L: ${summary['daily_pnl']:.2f}")
        self.logger.info(f"Total P&L: ${summary['total_pnl']:.2f}")
        self.logger.info(f"Avg P&L per Trade: ${summary['average_pnl_per_trade']:.2f}")
        self.logger.info(f"Open Positions: {summary['open_positions']}")
        self.logger.info("================================")

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

                # Cancel expired orders
                await self.cancel_expired_orders()

                # Track resolved positions
                # TODO: review this
                # await self.track_resolved_positions()

                # Refresh markets every hour
                current_time = datetime.now(timezone.utc)
                if current_time.minute == 0:
                    await self.refresh_markets()

                    # Log performance summary every 6 hours
                    if current_time.hour % 1 == 0:
                        summary = self.get_performance_summary()
                        self.logger.info(
                            f"Performance Update: {summary['total_trades']} trades, "
                            f"{summary['win_rate']:.1%} win rate, "
                            f"${summary['daily_pnl']:.2f} daily P&L"
                        )

                # Reset daily counter at midnight UTC
                if current_time.hour == 0 and current_time.minute == 0:
                    self.log_daily_summary()
                    self.save_position_data()
                    self.state.daily_trade_count = 0
                    self.state.daily_pnl = 0.0
                    self.logger.info("Reset daily counters")

                # Sleep for 1 minute
                await asyncio.sleep(60)

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
