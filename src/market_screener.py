from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

from src.models import Market, Token
from src.client import Client
from src.strategy import OrderbookSnapshot, OrderbookLevel


@dataclass
class MarketOpportunity:
    """Represents a liquidity provision opportunity in a market"""

    market: Market
    yes_token: Token
    no_token: Token
    reward_score: float
    competition_density: float
    estimated_daily_rewards: float
    min_capital_required: float
    max_spread_allowed: float
    recommended_position_size: float
    risk_level: str  # "low", "medium", "high"

    def to_dict(self) -> Dict[str, any]:
        return {
            "question_id": self.market.question_id,
            "question": self.market.question,
            "reward_score": float(self.reward_score),
            "competition_density": float(self.competition_density),
            "estimated_daily_rewards": float(self.estimated_daily_rewards),
            "min_capital_required": float(self.min_capital_required),
            "max_spread_allowed": float(self.max_spread_allowed),
            "recommended_position_size": float(self.recommended_position_size),
            "risk_level": self.risk_level,
            "yes_token_id": self.yes_token.token_id,
            "no_token_id": self.no_token.token_id,
        }


@dataclass
class ScreeningCriteria:
    """Criteria for screening profitable markets"""

    min_daily_rewards: float = 50  # Minimum $50 daily rewards
    min_reward_rate: float = 10  # Minimum 10% APR
    max_min_order_size: float = 500  # Max $500 minimum order
    max_competition_density: float = 0.6  # Max 60% competition
    min_spread_budget: float = 0.02  # Min 2c spread allowed
    max_risk_level: str = "medium"  # Maximum acceptable risk level
    require_active_orders: bool = True  # Only markets accepting orders
    exclude_archived: bool = True  # Exclude archived markets


class PolymarketScreener:
    """
    Advanced market screening system for identifying optimal liquidity provision opportunities.

    Screens markets based on:
    - Reward rates and total daily rewards
    - Competition density in spread
    - Capital requirements vs returns
    - Risk assessment
    """

    def __init__(self, client: Client, criteria: Optional[ScreeningCriteria] = None):
        self.client = client
        self.criteria = criteria or ScreeningCriteria()
        self.logger = logging.getLogger("PolymarketScreener")

    def find_opportunities(self, max_markets: int = 100) -> List[MarketOpportunity]:
        """
        Find the best liquidity provision opportunities across all markets.

        Args:
            max_markets: Maximum number of markets to evaluate

        Returns:
            List of MarketOpportunity objects sorted by reward score
        """
        self.logger.info(f"Screening up to {max_markets} markets for LP opportunities")

        opportunities = []
        markets_evaluated = 0

        # Get sampling markets directly from client (client handles pagination)
        markets = self.client.get_sampling_markets()

        for market in markets[:max_markets]:
            try:
                # Quick filtering based on basic criteria
                if not self._passes_basic_criteria(market):
                    continue

                # Detailed analysis for promising markets
                opportunity = self._analyze_market_opportunity(market)

                if opportunity and self._meets_screening_criteria(opportunity):
                    opportunities.append(opportunity)
                    self.logger.info(
                        f"Found opportunity: {market.question[:50]}... Score: {opportunity.reward_score}"
                    )

                markets_evaluated += 1

            except Exception as e:
                self.logger.error(
                    f"Error analyzing market {getattr(market, 'question_id', 'unknown')}: {e}"
                )
                continue

        # Sort by reward score (descending)
        opportunities.sort(key=lambda x: x.reward_score, reverse=True)

        self.logger.info(
            f"Found {len(opportunities)} qualifying opportunities from {markets_evaluated} markets"
        )
        return opportunities

    def analyze_specific_market(self, question_id: str) -> Optional[MarketOpportunity]:
        """
        Perform detailed analysis of a specific market.

        Args:
            question_id: Market question ID to analyze

        Returns:
            MarketOpportunity if market is suitable, None otherwise
        """
        try:
            # Try to get the specific market directly
            market = self.client.get_market(question_id)
            if not market:
                self.logger.warning(f"Market {question_id} not found")
                return None

            # Quick filtering based on basic criteria
            if not self._passes_basic_criteria(market):
                self.logger.info(f"Market {question_id} does not meet basic criteria")
                return None

            # Detailed analysis
            opportunity = self._analyze_market_opportunity(market)

            if opportunity and self._meets_screening_criteria(opportunity):
                return opportunity
            else:
                self.logger.info(f"Market {question_id} not suitable for LP")
                return None

        except Exception as e:
            self.logger.error(f"Error analyzing market {question_id}: {e}")
            return None

    def get_market_orderbooks(
        self, market: Market
    ) -> Tuple[Optional[OrderbookSnapshot], Optional[OrderbookSnapshot]]:
        """
        Get orderbook snapshots for both YES and NO tokens of a market.

        Args:
            market: Market object to get orderbooks for

        Returns:
            Tuple of (yes_orderbook, no_orderbook)
        """
        try:
            if len(market.tokens) != 2:
                self.logger.error(
                    f"Market {market.question_id} does not have exactly 2 tokens"
                )
                return None, None
            yes_token = market.tokens[0]
            no_token = market.tokens[1]

            if not yes_token or not no_token:
                self.logger.error(f"Market {market.question_id} missing YES/NO tokens")
                return None, None

            # Get orderbooks
            yes_orderbook_data = self.client.get_order_book(yes_token.token_id)
            no_orderbook_data = self.client.get_order_book(no_token.token_id)

            # Convert to OrderbookSnapshot objects
            yes_orderbook = self._convert_to_orderbook_snapshot(
                yes_token.token_id, yes_orderbook_data, yes_token.price
            )
            no_orderbook = self._convert_to_orderbook_snapshot(
                no_token.token_id, no_orderbook_data, no_token.price
            )

            return yes_orderbook, no_orderbook

        except Exception as e:
            self.logger.error(
                f"Error fetching orderbooks for market {market.question_id}: {e}"
            )
            return None, None

    def _passes_basic_criteria(self, market: Market) -> bool:
        """Quick filtering based on basic criteria"""
        try:
            # Check if market is active and accepting orders
            if self.criteria.require_active_orders and not market.accepting_orders:
                return False

            # Check if market is archived
            if self.criteria.exclude_archived and market.archived:
                return False

            # Check minimum order size
            if market.minimum_order_size > float(self.criteria.max_min_order_size):
                return False

            # Check if rewards exist
            if not market.rewards:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in basic criteria check: {e}")
            return False

    def _analyze_market_opportunity(
        self, market: Market
    ) -> Optional[MarketOpportunity]:
        """Perform detailed analysis of a market opportunity"""
        try:
            rewards = market.rewards
            tokens = market.tokens

            if not rewards or not tokens:
                return None

            if len(tokens) != 2:
                self.logger.error(
                    f"Market {market.question_id} does not have exactly 2 tokens"
                )
                return None
            yes_token = tokens[0]
            no_token = tokens[1]

            if not yes_token or not no_token:
                return None

            # Calculate reward metrics
            total_daily_rewards = rewards.rewards_daily_rate
            min_size_for_rewards = rewards.min_size
            max_scoring_spread = rewards.max_spread

            # Get orderbook data for competition analysis
            yes_orderbook, no_orderbook = self.get_market_orderbooks(market)

            # Calculate competition density (simplified)
            competition_density = self._estimate_competition_density(
                yes_orderbook, no_orderbook, max_scoring_spread
            )

            # Calculate reward score (combination of rewards, competition, capital efficiency)
            reward_score = self._calculate_reward_score(
                total_daily_rewards,
                min_size_for_rewards,
                competition_density,
                max_scoring_spread,
            )

            # Estimate capital requirements
            min_capital_required = max(
                min_size_for_rewards * 2,  # YES + NO positions
                market.minimum_order_size * 2,
            )

            # Risk assessment
            risk_level = self._assess_risk_level(
                market, competition_density, max_scoring_spread
            )

            # Create opportunity object
            opportunity = MarketOpportunity(
                market=market,
                yes_token=yes_token,
                no_token=no_token,
                reward_score=reward_score,
                competition_density=competition_density,
                estimated_daily_rewards=total_daily_rewards,
                min_capital_required=min_capital_required,
                max_spread_allowed=max_scoring_spread,
                recommended_position_size=min_capital_required,
                risk_level=risk_level,
            )

            return opportunity

        except Exception as e:
            self.logger.error(f"Error analyzing market opportunity: {e}")
            return None

    def _estimate_competition_density(
        self,
        yes_orderbook: Optional[OrderbookSnapshot],
        no_orderbook: Optional[OrderbookSnapshot],
        max_scoring_spread: float,
    ) -> float:
        """Estimate competition density in the reward-earning spread"""
        if not yes_orderbook or not no_orderbook:
            return 0.5  # Default moderate competition

        try:
            # Calculate volume in the reward spread around midpoint
            spread_range = max_scoring_spread
            yes_mid = yes_orderbook.midpoint

            # Volume in reward range
            yes_volume = yes_orderbook.total_bid_volume_in_range(
                (yes_mid - spread_range, yes_mid + spread_range)
            ) + yes_orderbook.total_ask_volume_in_range(
                (yes_mid - spread_range, yes_mid + spread_range)
            )

            no_volume = no_orderbook.total_bid_volume_in_range(
                (yes_mid - spread_range, yes_mid + spread_range)
            ) + no_orderbook.total_ask_volume_in_range(
                (yes_mid - spread_range, yes_mid + spread_range)
            )

            total_volume = yes_volume + no_volume

            # Estimate capacity (simplified heuristic)
            estimated_capacity = 1000  # $1000 typical capacity

            return min(total_volume / estimated_capacity, 1.0)

        except Exception as e:
            self.logger.error(f"Error estimating competition density: {e}")
            return 0.5

    def _calculate_reward_score(
        self,
        total_daily_rewards: float,
        min_size: float,
        competition_density: float,
        max_spread: float,
    ) -> float:
        """Calculate overall reward score for ranking opportunities"""
        try:
            # Base score from daily rewards
            reward_component = total_daily_rewards

            # Penalty for high capital requirements
            capital_efficiency = max(
                0.1,
                1000 / max(min_size, 1),
            )

            # Penalty for high competition
            competition_factor = 1.0 - competition_density

            # Bonus for larger spreads (more room to avoid fills)
            spread_bonus = max_spread * 10

            # Combined score
            score = (
                reward_component * capital_efficiency * competition_factor
                + spread_bonus
            )

            return max(score, 0)

        except Exception as e:
            self.logger.error(f"Error calculating reward score: {e}")
            return 0

    def _assess_risk_level(
        self, market, competition_density: float, max_spread: float
    ) -> str:
        """Assess risk level of the market opportunity"""
        try:
            risk_factors = 0

            # High competition increases risk
            if competition_density > 0.7:
                risk_factors += 2
            elif competition_density > 0.4:
                risk_factors += 1

            # Small spread increases fill risk
            if max_spread < 0.02:
                risk_factors += 2
            elif max_spread < 0.03:
                risk_factors += 1

            # Market closing soon increases risk
            # Would need to parse end_date_iso and compare with current time
            # For now, we don't use the market parameter but keep it for future use
            _ = market  # Suppress unused variable warning

            if risk_factors >= 3:
                return "high"
            elif risk_factors >= 1:
                return "medium"
            else:
                return "low"

        except Exception:
            return "medium"

    def _meets_screening_criteria(self, opportunity: MarketOpportunity) -> bool:
        """Check if opportunity meets all screening criteria"""
        try:
            # Check minimum daily rewards
            if opportunity.estimated_daily_rewards < self.criteria.min_daily_rewards:
                return False

            # Check maximum competition
            if opportunity.competition_density > self.criteria.max_competition_density:
                return False

            # Check minimum spread
            if opportunity.max_spread_allowed < self.criteria.min_spread_budget:
                return False

            # Check risk level
            risk_levels = {"low": 0, "medium": 1, "high": 2}
            if risk_levels.get(opportunity.risk_level, 2) > risk_levels.get(
                self.criteria.max_risk_level, 1
            ):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking screening criteria: {e}")
            return False

    def _convert_to_orderbook_snapshot(
        self, asset_id: str, orderbook_data, current_price: float
    ) -> OrderbookSnapshot:
        """Convert client OrderBookSummary to OrderbookSnapshot"""
        try:
            # Parse bids from OrderBookSummary format
            bids = []
            for bid_data in orderbook_data.bids:
                price = float(bid_data.get("price", 0))
                size = float(bid_data.get("size", 0))
                bids.append(OrderbookLevel(price, size))

            # Parse asks from OrderBookSummary format
            asks = []
            for ask_data in orderbook_data.asks:
                price = float(ask_data.get("price", 0))
                size = float(ask_data.get("size", 0))
                asks.append(OrderbookLevel(price, size))

            # Calculate midpoint and spread
            if bids and asks:
                best_bid = bids[0].price
                best_ask = asks[0].price
                midpoint = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
            else:
                midpoint = float(current_price)
                spread = 0.02  # Default 2c spread

            return OrderbookSnapshot(
                asset_id=asset_id,
                bids=bids,
                asks=asks,
                midpoint=midpoint,
                spread=spread,
            )

        except Exception as e:
            self.logger.error(f"Error converting orderbook data: {e}")
            # Return empty orderbook
            return OrderbookSnapshot(
                asset_id=asset_id,
                bids=[],
                asks=[],
                midpoint=current_price,
                spread=0.02,
            )
