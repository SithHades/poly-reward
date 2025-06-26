from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
import logging
from src.models import (
    BookSide,
    MarketCondition,
    OrderDetails,
    OrderbookSnapshot,
    Position,
    VolatilityMetrics,
)
from src.strategy_base import BaseStrategy


@dataclass
class LiquidityProvisionConfig:
    """Configuration for liquidity provision strategy"""

    # Core LP parameters
    max_spread_from_midpoint: float = 0.03  # 3c max distance from midpoint
    min_order_size: float = 100  # Minimum shares for rewards
    optimal_distance_from_midpoint: float = 0.015  # 1.5c for safety

    # Risk management
    max_position_size: float = 1000  # Max shares per market
    max_total_exposure: float = 5000  # Max total exposure across markets
    volatility_exit_threshold: float = 0.02  # 2% midpoint move triggers exit

    # Competition assessment
    min_reward_share_threshold: float = 0.1  # 10% minimum reward share
    max_competition_density: float = 0.8  # 80% of spread already filled

    # FIFO management
    order_refresh_interval_minutes: int = 30  # Refresh orders every 30 minutes
    max_order_age_minutes: int = 45  # Cancel orders older than 45 minutes

    # Hedging
    enable_yes_no_hedging: bool = True
    hedge_ratio: float = 1.0  # 1:1 hedging ratio


class PolymarketLiquidityStrategy(BaseStrategy):
    """
    Advanced liquidity provision strategy for Polymarket

    Focuses on earning liquidity rewards while minimizing fill risk through:
    - Strategic positioning within 3c of midpoint
    - Competition analysis to avoid crowded markets
    - Volatility detection for risk management
    - YES/NO market hedging
    - FIFO queue management
    """

    def __init__(self, config: LiquidityProvisionConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger("PolymarketLiquidityStrategy")
        self.volatility_tracker: Dict[str, VolatilityMetrics] = {}
        self.order_timestamps: Dict[str, datetime] = {}

    def analyze_market_condition(
        self,
        yes_orderbook: OrderbookSnapshot,
        no_orderbook: OrderbookSnapshot,
        **kwargs: Any,
    ) -> MarketCondition:
        """
        Analyze market conditions to determine if suitable for liquidity provision

        Args:
            yes_orderbook: Orderbook for YES market
            no_orderbook: Orderbook for NO market

        Returns:
            MarketCondition indicating strategy recommendation
        """
        try:
            # Check volatility first
            if self._is_market_volatile(
                yes_orderbook.asset_id, yes_orderbook, no_orderbook
            ):
                return MarketCondition.VOLATILE

            # Analyze competition in the spread
            competition_density = self._calculate_competition_density(
                yes_orderbook, no_orderbook
            )
            if competition_density > self.config.max_competition_density:
                self.logger.info(
                    f"Market {yes_orderbook.asset_id} too competitive: {competition_density:.2%} density"
                )
                return MarketCondition.COMPETITIVE

            # Estimate potential reward share
            potential_reward_share = self._estimate_reward_share(
                yes_orderbook, no_orderbook
            )
            if potential_reward_share < self.config.min_reward_share_threshold:
                self.logger.info(
                    f"Market {yes_orderbook.asset_id} low reward share: {potential_reward_share:.2%}"
                )
                return MarketCondition.COMPETITIVE

            # Market looks attractive for LP
            self.logger.info(
                f"Market {yes_orderbook.asset_id} attractive: {competition_density:.2%} density, {potential_reward_share:.2%} reward share"
            )
            return MarketCondition.ATTRACTIVE

        except Exception as e:
            self.logger.error(f"Error analyzing market condition: {e}")
            return MarketCondition.UNAVAILABLE

    def calculate_orders(
        self,
        yes_orderbook: OrderbookSnapshot,
        no_orderbook: OrderbookSnapshot,
        current_positions: Dict[str, Position],
        available_capital: float,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Calculate optimal liquidity provision orders

        Args:
            yes_orderbook: YES market orderbook
            no_orderbook: NO market orderbook
            current_positions: Current positions by asset_id
            available_capital: Available capital for new orders

        Returns:
            List of order specifications
        """
        market_condition = self.analyze_market_condition(yes_orderbook, no_orderbook)

        if market_condition != MarketCondition.ATTRACTIVE:
            self.logger.info(
                f"Skipping orders for {yes_orderbook.asset_id}: {market_condition.value}"
            )
            return []

        orders = []

        # Calculate position sizes based on available capital and risk limits
        max_order_size = min(
            available_capital // 4,  # Use max 25% of available capital per market
            self.config.max_position_size,
        )

        if max_order_size < self.config.min_order_size:
            self.logger.warning(
                f"Insufficient capital for minimum order size: {max_order_size} < {self.config.min_order_size}"
            )
            return []

        # Generate YES market orders
        yes_orders = self._generate_market_orders(
            yes_orderbook,
            True,
            max_order_size,
            current_positions.get(yes_orderbook.asset_id),
        )
        orders.extend(yes_orders)

        # Generate NO market orders (hedging)
        if self.config.enable_yes_no_hedging:
            no_orders = self._generate_market_orders(
                no_orderbook,
                False,
                max_order_size,
                current_positions.get(no_orderbook.asset_id),
            )
            orders.extend(no_orders)

        return orders

    def should_cancel_orders(
        self, asset_id: str, current_orders: List[OrderDetails]
    ) -> List[str]:
        """
        Determine which orders should be cancelled based on age, volatility, or market conditions

        Args:
            asset_id: Market asset ID
            current_orders: List of current orders

        Returns:
            List of order IDs to cancel
        """
        orders_to_cancel = []
        now = datetime.now(timezone.utc)

        for order in current_orders:
            order_age = now - order.created_at

            # Cancel old orders (FIFO management)
            if order_age > timedelta(minutes=self.config.max_order_age_minutes):
                orders_to_cancel.append(order.order_id)
                self.logger.info(
                    f"Cancelling old order {order.order_id}: {order_age.total_seconds() / 60:.1f} minutes old"
                )
                continue

            # Cancel if market became volatile
            if (
                asset_id in self.volatility_tracker
                and self.volatility_tracker[asset_id].is_volatile()
            ):
                orders_to_cancel.append(order.order_id)
                self.logger.info(f"Cancelling order {order.order_id} due to volatility")
                continue

        return orders_to_cancel

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

    def _calculate_competition_density(
        self, yes_orderbook: OrderbookSnapshot, no_orderbook: OrderbookSnapshot
    ) -> float:
        """
        Calculate how much of the profitable spread is already occupied by other LPs

        Returns:
            float between 0 and 1 representing density of competition
        """
        # Define the 3c range around midpoint where LP rewards are earned
        yes_mid = yes_orderbook.midpoint
        spread_range = (
            yes_mid - self.config.max_spread_from_midpoint,
            yes_mid + self.config.max_spread_from_midpoint,
        )

        # Calculate existing volume in the range
        yes_volume_in_range = yes_orderbook.total_bid_volume_in_range(
            spread_range
        ) + yes_orderbook.total_ask_volume_in_range(spread_range)

        no_volume_in_range = no_orderbook.total_bid_volume_in_range(
            spread_range
        ) + no_orderbook.total_ask_volume_in_range(spread_range)

        total_volume_in_range = yes_volume_in_range + no_volume_in_range

        # Estimate total "capacity" of the range (simplified heuristic)
        estimated_capacity = (
            self.config.min_order_size * 10
        )  # Assume 10x min order size as capacity

        return min(total_volume_in_range / estimated_capacity, 1.0)

    def _estimate_reward_share(
        self, yes_orderbook: OrderbookSnapshot, no_orderbook: OrderbookSnapshot
    ) -> float:
        """
        Estimate potential reward share based on our planned order size vs existing competition

        Returns:
            float between 0 and 1 representing estimated reward share
        """
        # Calculate existing competition
        competition_density = self._calculate_competition_density(
            yes_orderbook, no_orderbook
        )

        # Simple heuristic: our share inversely related to competition density
        estimated_share = (1.0 - competition_density) * 0.5  # Max 50% in best case

        return max(estimated_share, 0.01)  # Minimum 1%

    def _generate_market_orders(
        self,
        orderbook: OrderbookSnapshot,
        is_yes_market: bool,
        max_size: float,
        current_position: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate orders for a specific market (YES or NO)"""
        orders = []

        # Calculate safe distance from midpoint to avoid fills
        target_distance = self.config.optimal_distance_from_midpoint
        midpoint = orderbook.midpoint

        # Generate bid order (buying at below midpoint)
        bid_price = midpoint - target_distance
        if bid_price > 0.01:  # Minimum price check
            orders.append(
                {
                    "side": BookSide.BUY,
                    "price": bid_price,
                    "size": self.config.min_order_size,
                    "market_type": "YES" if is_yes_market else "NO",
                    "asset_id": orderbook.asset_id,
                }
            )

        # Generate ask order (selling at above midpoint)
        ask_price = midpoint + target_distance
        if ask_price < 0.99:  # Maximum price check
            orders.append(
                {
                    "side": BookSide.SELL,
                    "price": ask_price,
                    "size": self.config.min_order_size,
                    "market_type": "YES" if is_yes_market else "NO",
                    "asset_id": orderbook.asset_id,
                }
            )

        return orders

    def calculate_hedge_orders(
        self,
        filled_order: OrderDetails,
        yes_orderbook: OrderbookSnapshot,
        no_orderbook: OrderbookSnapshot,
    ) -> list[dict[str, Any]]:
        """
        Calculate hedging orders when a liquidity provision order gets filled

        This implements the YES bid at p = NO ask at (1-p) relationship
        """
        hedge_orders = []

        if not self.config.enable_yes_no_hedging:
            return hedge_orders

        yes_asset_id = yes_orderbook.asset_id
        no_asset_id = no_orderbook.asset_id

        try:
            filled_price = filled_order.price
            filled_size = filled_order.matched_size

            # Calculate hedge price using YES/NO relationship: p_yes + p_no = 1
            if filled_order.asset_id == yes_asset_id:
                # Original order was YES, hedge with NO
                hedge_price = 1.0 - filled_price
                hedge_market_type = "NO"
                target_orderbook = no_orderbook
            elif filled_order.asset_id == no_asset_id:
                # Original order was NO, hedge with YES
                hedge_price = 1.0 - filled_price
                hedge_market_type = "YES"
                target_orderbook = yes_orderbook
            else:
                self.logger.error(
                    f"Filled order {filled_order.order_id} has invalid asset_id: {filled_order.asset_id}"
                )
                return hedge_orders

            # Determine hedge side (opposite of original)
            hedge_side = (
                BookSide.SELL if filled_order.side == BookSide.BUY else BookSide.BUY
            )

            # Size the hedge order
            hedge_size = filled_size * self.config.hedge_ratio

            hedge_orders.append(
                {
                    "side": hedge_side,
                    "price": hedge_price,
                    "size": hedge_size,
                    "market_type": hedge_market_type,
                    "asset_id": target_orderbook.asset_id,
                    "reason": f"hedge_for_{filled_order.order_id}",
                }
            )

            self.logger.info(
                f"Generated hedge order: {hedge_side.value} {hedge_size} at {hedge_price} in {hedge_market_type} market"
            )

        except Exception as e:
            self.logger.error(f"Error calculating hedge orders: {e}")

        return hedge_orders

    def on_fill(
        self,
        fill_event: OrderDetails,
        yes_orderbook: OrderbookSnapshot,
        no_orderbook: OrderbookSnapshot,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        # Use calculate_hedge_orders logic here
        return self.calculate_hedge_orders(fill_event, yes_orderbook, no_orderbook)

    @property
    def name(self) -> str:
        return "PolymarketLiquidityStrategy"


class SimpleMarketMakingStrategy(BaseStrategy):
    """
    Simple always-on market making strategy for demonstration/template purposes.
    Places orders at a fixed spread from the midpoint, no hedging or advanced logic.
    """

    def __init__(self, spread: float = 0.02, order_size: float = 100):
        self._spread = spread
        self._order_size = order_size
        self.logger = logging.getLogger("SimpleMarketMakingStrategy")

    def analyze_market_condition(
        self,
        yes_orderbook: OrderbookSnapshot,
        no_orderbook: OrderbookSnapshot,
        **kwargs: Any,
    ) -> str:
        return "always_on"

    def calculate_orders(
        self,
        yes_orderbook: OrderbookSnapshot,
        no_orderbook: OrderbookSnapshot,
        current_positions: Dict[str, Position],
        available_capital: float,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        orders = []
        midpoint = yes_orderbook.midpoint
        # Place a buy below and a sell above midpoint for YES market
        orders.append(
            {
                "side": BookSide.BUY,
                "price": max(midpoint - self._spread, 0.01),
                "size": self._order_size,
                "market_type": "YES",
                "asset_id": yes_orderbook.asset_id,
            }
        )
        orders.append(
            {
                "side": BookSide.SELL,
                "price": min(midpoint + self._spread, 0.99),
                "size": self._order_size,
                "market_type": "YES",
                "asset_id": yes_orderbook.asset_id,
            }
        )
        # Repeat for NO market
        midpoint_no = no_orderbook.midpoint
        orders.append(
            {
                "side": BookSide.BUY,
                "price": max(midpoint_no - self._spread, 0.01),
                "size": self._order_size,
                "market_type": "NO",
                "asset_id": no_orderbook.asset_id,
            }
        )
        orders.append(
            {
                "side": BookSide.SELL,
                "price": min(midpoint_no + self._spread, 0.99),
                "size": self._order_size,
                "market_type": "NO",
                "asset_id": no_orderbook.asset_id,
            }
        )
        return orders

    def on_fill(
        self,
        fill_event: OrderDetails,
        yes_orderbook: OrderbookSnapshot,
        no_orderbook: OrderbookSnapshot,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        # No hedging or follow-up for this simple strategy
        return []

    @property
    def name(self) -> str:
        return "SimpleMarketMakingStrategy"
