from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timezone, timedelta
import re

from src.models import (
    BookSide,
    MarketCondition,
    OrderDetails,
    OrderbookSnapshot,
    Position,
    VolatilityMetrics,
    OrderArgsModel
)
from src.strategy_base import BaseStrategy


@dataclass
class EthPredictionStrategyConfig:
    """Configuration for the ETH Prediction Market Making Strategy"""

    # Market Making Parameters
    target_spread: float = 0.02  # e.g., 0.02 for 49/51 cents
    order_size: float = 5  # Default order size
    min_order_size: float = 5  # Minimum order size

    # Prediction Bias
    prediction_bias_factor: float = 0.005  # How much to bias prices based on prediction

    # Risk Management
    max_position_size: float = 50  # Max shares per market
    max_total_exposure: float = 50  # Max total exposure across markets
    volatility_exit_threshold: float = 0.02  # 2% midpoint move triggers exit

    # FIFO management
    max_order_age_minutes: int = 45  # Cancel orders older than 45 minutes

    # Hedging
    enable_yes_no_hedging: bool = True
    hedge_ratio: float = 1.0  # 1:1 hedging ratio


class EthPredictionMarketMakingStrategy(BaseStrategy):
    """
    Strategy to capitalize on Polymarket's hourly ETH prediction markets
    by biasing market making based on ETH price direction prediction.
    """

    def __init__(self, config: EthPredictionStrategyConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger("EthPredictionMarketMakingStrategy")
        self.volatility_tracker: Dict[str, VolatilityMetrics] = {}
        self.eth_prediction_cache: Dict[str, Dict[str, Any]] = {}

    async def get_eth_price_prediction(self) -> Dict[str, Any]:
        """
        Fetches ETH/USDT price data and predicts the direction for the next hour.
        This is a simplified adaptation of scripts/ethusdt.py.
        """
        symbol = "ETH/USDT"
        timeframe = "1m"
        exchange = ccxt.binance()

        try:
            # Fetch recent 1-minute OHLCV data
            since = exchange.parse8601(
                (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
            )
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")

            # Get current live price
            ticker = exchange.fetch_ticker(symbol)
            last_price = ticker["last"]
            current_time = pd.to_datetime("now", utc=True)
            live_data_df = pd.DataFrame(
                [
                    {
                        "timestamp": current_time,
                        "open": last_price,  # Use last_price instead of 0 to avoid skewing calculations
                        "high": last_price,  # Use last_price instead of 0 
                        "low": last_price,   # Use last_price instead of 0
                        "close": last_price,
                        "volume": 0,
                    }
                ]
            )
            live_data_df = live_data_df.set_index("timestamp")

            combined_df = pd.concat([df, live_data_df]).sort_index(ascending=True)
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]

            # Resample to 1-hour frequency and calculate 45-min delta
            df_1h = combined_df["close"].resample("1h").ohlc().dropna()
            df_15m = combined_df["close"].resample("15min").last().dropna()

            # Ensure the timestamps align for 45-min mark
            # Filter df_1h to only include candles where 45-min mark is available in df_15m
            # This line was causing an issue, as df_1h.index + pd.Timedelta(minutes=45) might not be directly in df_15m.index
            # Instead, we should check if the latest 1h candle has a corresponding 45min mark in df_15m
            # df_1h = df_1h[df_1h.index + pd.Timedelta(minutes=45) <= df_15m.index[-1]]

            if df_1h.empty:
                self.logger.warning("Not enough 1-hour data for prediction.")
                return {"direction": "flat", "confidence": 0.5}

            latest_1h_candle_start = df_1h.index[-1]
            open_price = df_1h.loc[latest_1h_candle_start, "open"]

            # Find the 15-min price closest to 45 minutes into the latest 1-hour candle
            target_45min_timestamp = latest_1h_candle_start + pd.Timedelta(minutes=45)
            # Find the closest timestamp in df_15m.index to target_45min_timestamp
            idx = np.abs(df_15m.index - target_45min_timestamp).argmin()
            price_45 = df_15m.iloc[idx]

            delta_45 = price_45 - open_price

            direction_45 = "up" if delta_45 > 0 else "down" if delta_45 < 0 else "flat"

            # Simplified confidence based on magnitude of delta_45
            confidence = min(
                abs(delta_45) / (open_price * 0.01), 1.0
            )  # 1% move gives 1.0 confidence

            self.logger.info(
                f"ETH Prediction: {direction_45} with confidence {confidence:.2f}"
            )
            return {"direction": direction_45, "confidence": confidence}

        except Exception as e:
            print(e)
            self.logger.error(f"Error getting ETH price prediction: {e}")
            return {"direction": "flat", "confidence": 0.5}  # Default to flat if error

    def analyze_market_condition(
        self,
        yes_orderbook: OrderbookSnapshot,
        no_orderbook: OrderbookSnapshot,
        **kwargs: Any,
    ) -> MarketCondition:
        """
        Analyzes market conditions for the hourly ETH prediction market.
        """
        market: Optional[Any] = kwargs.get("market")
        if (
            not market
            or not market.is_50_50_outcome
            or "eth" not in market.market_slug.lower()
        ):
            return MarketCondition.UNAVAILABLE

        # Check if it's an hourly market (e.g., "eth-price-at-10am-et-june-26-2025")
        # This is a heuristic and might need refinement based on actual market slugs
        if not any(
            re.search(r"\d{1,2}(am|pm)-et", market_slug.lower())
            for market_slug in [market.market_slug]
        ):
            self.logger.debug(
                f"Market {market.market_slug} does not appear to be an hourly ETH market."
            )
            return MarketCondition.UNAVAILABLE

        # Check volatility
        if self._is_market_volatile(
            yes_orderbook.asset_id, yes_orderbook, no_orderbook
        ):
            return MarketCondition.VOLATILE

        # For this strategy, we assume it's always attractive if it's the right market type and not volatile
        return MarketCondition.ATTRACTIVE

    async def calculate_orders(
        self,
        yes_orderbook: OrderbookSnapshot,
        no_orderbook: OrderbookSnapshot,
        current_positions: Dict[str, Position],
        available_capital: float,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Calculates optimal orders based on ETH price prediction.
        """
        market_condition = self.analyze_market_condition(
            yes_orderbook, no_orderbook, **kwargs
        )

        if market_condition != MarketCondition.ATTRACTIVE:
            self.logger.info(
                f"Skipping orders for {yes_orderbook.asset_id}: {market_condition.value}"
            )
            return []

        orders = []
        eth_prediction = await self.get_eth_price_prediction()
        prediction_direction = eth_prediction["direction"]
        prediction_confidence = eth_prediction["confidence"]

        # Determine bias based on prediction
        yes_bias = 0.0
        no_bias = 0.0
        if prediction_direction == "up":
            yes_bias = self.config.prediction_bias_factor * prediction_confidence
            no_bias = -self.config.prediction_bias_factor * prediction_confidence
        elif prediction_direction == "down":
            yes_bias = -self.config.prediction_bias_factor * prediction_confidence
            no_bias = self.config.prediction_bias_factor * prediction_confidence

        # Calculate target prices based on 49/51 cent inefficiency and prediction bias
        # For YES market
        yes_buy_price = round(0.49 + yes_bias, 3)
        yes_sell_price = round(0.51 + yes_bias, 3)

        # For NO market
        no_buy_price = round(0.49 + no_bias, 3)
        no_sell_price = round(0.51 + no_bias, 3)

        # Ensure prices are within valid range [0.01, 0.99]
        yes_buy_price = max(0.01, min(0.99, yes_buy_price))
        yes_sell_price = max(0.01, min(0.99, yes_sell_price))
        no_buy_price = max(0.01, min(0.99, no_buy_price))
        no_sell_price = max(0.01, min(0.99, no_sell_price))

        # Place orders for YES market
        orders.append(
            OrderArgsModel(
                side=BookSide.BUY,
                price=yes_buy_price,
                size=self.config.order_size,
                token_id=yes_orderbook.asset_id,
            )
        )
        orders.append(
            OrderArgsModel(
                side=BookSide.SELL,
                price=yes_sell_price,
                size=self.config.order_size,
                token_id=yes_orderbook.asset_id,
            )
        )

        # Place orders for NO market
        orders.append(
            OrderArgsModel(
                side=BookSide.BUY,
                price=no_buy_price,
                size=self.config.order_size,
                token_id=no_orderbook.asset_id,
            )
        )
        orders.append(
            OrderArgsModel(
                side=BookSide.SELL,
                price=no_sell_price,
                size=self.config.order_size,
                token_id=no_orderbook.asset_id,
            )
        )

        return orders

    def should_cancel_orders(
        self, asset_id: str, current_orders: List[OrderDetails]
    ) -> List[str]:
        """
        Determine which orders should be cancelled based on age or volatility.
        """
        orders_to_cancel = []
        now = datetime.now(timezone.utc)

        for order in current_orders:
            order_age = now - datetime.fromtimestamp(order.created_at, tz=timezone.utc)

            # Cancel old orders (FIFO management)
            if order_age > timedelta(minutes=self.config.max_order_age_minutes):
                orders_to_cancel.append(order.order_id)
                self.logger.info(
                    f"Cancelling old order {order.order_id}: {order_age.total_seconds() / 60:.1f} minutes old"
                )
                continue

            # Cancel if market became volatile
            if asset_id in self.volatility_tracker and self.volatility_tracker[
                asset_id
            ].is_volatile(self.config.volatility_exit_threshold):
                orders_to_cancel.append(order.order_id)
                self.logger.info(f"Cancelling order {order.order_id} due to volatility")
                continue

        return orders_to_cancel

    def on_fill(
        self,
        fill_event: OrderDetails,
        yes_orderbook: OrderbookSnapshot,
        no_orderbook: OrderbookSnapshot,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Handles order fill events, primarily for hedging.
        Since we can split USDT into YES/NO shares, the primary hedging is done by maintaining a balanced inventory.
        """
        hedge_orders = []

        # Determine the asset ID of the opposite token
        if fill_event.asset_id == yes_orderbook.asset_id:
            # Filled on YES, so we need to hedge with NO
            hedge_token_id = no_orderbook.asset_id
            # If we bought YES, we need to sell NO to hedge
            # If we sold YES, we need to buy NO to hedge (to cover our short position)
            hedge_side = BookSide.SELL if fill_event.side == BookSide.BUY else BookSide.BUY
        elif fill_event.asset_id == no_orderbook.asset_id:
            # Filled on NO, so we need to hedge with YES
            hedge_token_id = yes_orderbook.asset_id
            # If we bought NO, we need to sell YES to hedge
            # If we sold NO, we need to buy YES to hedge (to cover our short position)
            hedge_side = BookSide.SELL if fill_event.side == BookSide.BUY else BookSide.BUY
        else:
            self.logger.error(f"Filled order {fill_event.order_id} has an unknown asset_id: {fill_event.asset_id}")
            return hedge_orders

        # The price for the hedge order should be based on the 1 - filled_price relationship
        hedge_price = 1.0 - fill_event.price
        hedge_size = fill_event.original_size # Hedge with the same size as the filled order

        # Ensure hedge price is within valid range
        hedge_price = max(0.01, min(0.99, hedge_price))

        hedge_orders.append(
            OrderArgsModel(
                side=hedge_side,
                price=hedge_price,
                size=hedge_size,
                token_id=hedge_token_id,
            )
        )

        self.logger.info(f"Generated hedge order for fill {fill_event.order_id}: {hedge_side.value} {hedge_size} of {hedge_token_id} at {hedge_price}")

        return hedge_orders

    def update_volatility_metrics(self, asset_id: str, orderbook: OrderbookSnapshot):
        """Update volatility tracking for a market"""
        if asset_id not in self.volatility_tracker:
            self.volatility_tracker[asset_id] = VolatilityMetrics()

        metrics = self.volatility_tracker[asset_id]
        metrics.add_snapshot(orderbook)

    def _is_market_volatile(
        self,
        asset_id: str,
        yes_orderbook: OrderbookSnapshot,
        no_orderbook: OrderbookSnapshot,
    ) -> bool:
        """Check if market is currently volatile"""
        if asset_id not in self.volatility_tracker:
            return False

        return self.volatility_tracker[asset_id].is_volatile(
            self.config.volatility_exit_threshold
        )

    @property
    def name(self) -> str:
        return "EthPredictionMarketMakingStrategy"
