from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
import logging

from src.api_models import Market, Token
from src.polymarket_client import PolymarketClient
from src.strategy import OrderbookSnapshot, OrderbookLevel


@dataclass
class MarketOpportunity:
    """Represents a liquidity provision opportunity in a market"""
    market: Market
    yes_token: Token
    no_token: Token
    reward_score: Decimal
    competition_density: Decimal
    estimated_daily_rewards: Decimal
    min_capital_required: Decimal
    max_spread_allowed: Decimal
    recommended_position_size: Decimal
    risk_level: str  # "low", "medium", "high"
    
    def to_dict(self) -> Dict[str, any]:
        # Handle both dict and object style tokens
        yes_token_id = (self.yes_token.get('token_id') if isinstance(self.yes_token, dict) 
                       else getattr(self.yes_token, 'token_id', 'unknown'))
        no_token_id = (self.no_token.get('token_id') if isinstance(self.no_token, dict)
                      else getattr(self.no_token, 'token_id', 'unknown'))
        
        return {
            "question_id": getattr(self.market, 'question_id', 'unknown'),
            "question": getattr(self.market, 'question', 'Unknown question'),
            "reward_score": float(self.reward_score),
            "competition_density": float(self.competition_density),
            "estimated_daily_rewards": float(self.estimated_daily_rewards),
            "min_capital_required": float(self.min_capital_required),
            "max_spread_allowed": float(self.max_spread_allowed),
            "recommended_position_size": float(self.recommended_position_size),
            "risk_level": self.risk_level,
            "yes_token_id": yes_token_id,
            "no_token_id": no_token_id
        }


@dataclass
class ScreeningCriteria:
    """Criteria for screening profitable markets"""
    min_daily_rewards: Decimal = Decimal("50")  # Minimum $50 daily rewards
    min_reward_rate: Decimal = Decimal("10")    # Minimum 10% APR
    max_min_order_size: Decimal = Decimal("500")  # Max $500 minimum order
    max_competition_density: Decimal = Decimal("0.6")  # Max 60% competition
    min_spread_budget: Decimal = Decimal("0.02")  # Min 2c spread allowed
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
    
    def __init__(self, client: PolymarketClient, criteria: Optional[ScreeningCriteria] = None):
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
        next_cursor = None
        markets_evaluated = 0
        
        while markets_evaluated < max_markets:
            # Fetch batch of markets
            batch_size = min(100, max_markets - markets_evaluated)
            response = self.client.get_sampling_markets(next_cursor, batch_size)
            
            if not response.get("data"):
                break
                
            markets = response["data"]
            
            for market_data in markets:
                if markets_evaluated >= max_markets:
                    break
                    
                try:
                    # Parse market data into API model
                    market = self._parse_market_data(market_data)
                    
                    # Quick filtering based on basic criteria
                    if not self._passes_basic_criteria(market):
                        continue
                    
                    # Detailed analysis for promising markets
                    opportunity = self._analyze_market_opportunity(market)
                    
                    if opportunity and self._meets_screening_criteria(opportunity):
                        opportunities.append(opportunity)
                        self.logger.info(f"Found opportunity: {market.question[:50]}... Score: {opportunity.reward_score}")
                    
                    markets_evaluated += 1
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing market {market_data.get('question_id', 'unknown')}: {e}")
                    continue
            
            # Check for more markets
            next_cursor = response.get("next_cursor")
            if not next_cursor:
                break
        
        # Sort by reward score (descending)
        opportunities.sort(key=lambda x: x.reward_score, reverse=True)
        
        self.logger.info(f"Found {len(opportunities)} qualifying opportunities from {markets_evaluated} markets")
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
            # For now, we'll search through sampling markets
            # In a real implementation, there might be a direct market lookup
            markets = self.find_opportunities(max_markets=1000)
            for opportunity in markets:
                if opportunity.market.question_id == question_id:
                    return opportunity
            
            self.logger.warning(f"Market {question_id} not found or not suitable for LP")
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing market {question_id}: {e}")
            return None
    
    def get_market_orderbooks(self, market: Market) -> Tuple[Optional[OrderbookSnapshot], Optional[OrderbookSnapshot]]:
        """
        Get orderbook snapshots for both YES and NO tokens of a market.
        
        Args:
            market: Market object to get orderbooks for
            
        Returns:
            Tuple of (yes_orderbook, no_orderbook)
        """
        try:
            yes_token = None
            no_token = None
            
            for token in market.tokens:
                if token.outcome.lower() == "yes":
                    yes_token = token
                elif token.outcome.lower() == "no":
                    no_token = token
            
            if not yes_token or not no_token:
                self.logger.error(f"Market {market.question_id} missing YES/NO tokens")
                return None, None
            
            # Get orderbooks
            yes_orderbook_data = self.client.get_orderbook(yes_token.token_id)
            no_orderbook_data = self.client.get_orderbook(no_token.token_id)
            
            # Convert to OrderbookSnapshot objects
            yes_orderbook = self._convert_to_orderbook_snapshot(
                yes_token.token_id, yes_orderbook_data, yes_token.price
            )
            no_orderbook = self._convert_to_orderbook_snapshot(
                no_token.token_id, no_orderbook_data, no_token.price
            )
            
            return yes_orderbook, no_orderbook
            
        except Exception as e:
            self.logger.error(f"Error fetching orderbooks for market {market.question_id}: {e}")
            return None, None
    
    def _parse_market_data(self, market_data: Dict) -> Market:
        """Parse API response data into Market object"""
        # Parse rewards
        rewards_data = market_data.get("rewards", {})
        reward_rates = []
        
        for rate_data in rewards_data.get("rates", []):
            reward_rates.append({
                "asset_address": rate_data.get("asset_address", ""),
                "rewards_daily_rate": float(rate_data.get("rewards_daily_rate", 0))
            })
        
        rewards = {
            "rates": reward_rates,
            "min_size": float(rewards_data.get("min_size", 0)),
            "max_spread": float(rewards_data.get("max_spread", 0.1))
        }
        
        # Parse tokens
        tokens = []
        for token_data in market_data.get("tokens", []):
            tokens.append({
                "token_id": token_data.get("token_id", ""),
                "outcome": token_data.get("outcome", ""),
                "price": float(token_data.get("price", 0.5)),
                "winner": bool(token_data.get("winner", False))
            })
        
        # Create Market object (simplified - in real implementation would use proper dataclass)
        market = type('Market', (), {})()
        for key, value in market_data.items():
            if key == "rewards":
                setattr(market, key, rewards)
            elif key == "tokens":
                setattr(market, key, tokens)
            else:
                setattr(market, key, value)
        
        return market
    
    def _passes_basic_criteria(self, market) -> bool:
        """Quick filtering based on basic criteria"""
        try:
            # Check if market is active and accepting orders
            if self.criteria.require_active_orders and not getattr(market, 'accepting_orders', False):
                return False
            
            # Check if market is archived
            if self.criteria.exclude_archived and getattr(market, 'archived', False):
                return False
            
            # Check minimum order size
            min_order_size = getattr(market, 'minimum_order_size', float('inf'))
            if min_order_size > float(self.criteria.max_min_order_size):
                return False
            
            # Check if rewards exist
            rewards = getattr(market, 'rewards', {})
            if not rewards or not rewards.get('rates'):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in basic criteria check: {e}")
            return False
    
    def _analyze_market_opportunity(self, market) -> Optional[MarketOpportunity]:
        """Perform detailed analysis of a market opportunity"""
        try:
            rewards = getattr(market, 'rewards', {})
            tokens = getattr(market, 'tokens', [])
            
            if not rewards or not tokens:
                return None
            
            # Find YES and NO tokens
            yes_token = None
            no_token = None
            
            for token in tokens:
                if isinstance(token, dict):
                    if token.get('outcome', '').lower() == 'yes':
                        yes_token = token
                    elif token.get('outcome', '').lower() == 'no':
                        no_token = token
                else:
                    # Handle object-style access
                    if getattr(token, 'outcome', '').lower() == 'yes':
                        yes_token = token
                    elif getattr(token, 'outcome', '').lower() == 'no':
                        no_token = token
            
            if not yes_token or not no_token:
                return None
            
            # Calculate reward metrics
            total_daily_rewards = sum(
                rate.get('rewards_daily_rate', 0) if isinstance(rate, dict) 
                else getattr(rate, 'rewards_daily_rate', 0)
                for rate in rewards.get('rates', [])
            )
            
            min_size_for_rewards = rewards.get('min_size', 0)
            max_spread = rewards.get('max_spread', 0.1)
            
            # Get orderbook data for competition analysis
            yes_orderbook, no_orderbook = self.get_market_orderbooks(market)
            
            # Calculate competition density (simplified)
            competition_density = self._estimate_competition_density(yes_orderbook, no_orderbook, max_spread)
            
            # Calculate reward score (combination of rewards, competition, capital efficiency)
            reward_score = self._calculate_reward_score(
                total_daily_rewards, 
                min_size_for_rewards, 
                competition_density,
                max_spread
            )
            
            # Estimate capital requirements
            min_capital_required = max(
                Decimal(str(min_size_for_rewards * 2)),  # YES + NO positions
                Decimal(str(getattr(market, 'minimum_order_size', 0) * 2))
            )
            
            # Risk assessment
            risk_level = self._assess_risk_level(market, competition_density, max_spread)
            
            # Create opportunity object
            opportunity = MarketOpportunity(
                market=market,
                yes_token=yes_token,
                no_token=no_token,
                reward_score=reward_score,
                competition_density=competition_density,
                estimated_daily_rewards=Decimal(str(total_daily_rewards)),
                min_capital_required=min_capital_required,
                max_spread_allowed=Decimal(str(max_spread)),
                recommended_position_size=min_capital_required,
                risk_level=risk_level
            )
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Error analyzing market opportunity: {e}")
            return None
    
    def _estimate_competition_density(self, yes_orderbook: Optional[OrderbookSnapshot], 
                                    no_orderbook: Optional[OrderbookSnapshot], 
                                    max_spread: float) -> Decimal:
        """Estimate competition density in the reward-earning spread"""
        if not yes_orderbook or not no_orderbook:
            return Decimal("0.5")  # Default moderate competition
        
        try:
            # Calculate volume in the reward spread around midpoint
            spread_range = Decimal(str(max_spread))
            yes_mid = yes_orderbook.midpoint
            
            # Volume in reward range
            yes_volume = (
                yes_orderbook.total_bid_volume_in_range((yes_mid - spread_range, yes_mid + spread_range)) +
                yes_orderbook.total_ask_volume_in_range((yes_mid - spread_range, yes_mid + spread_range))
            )
            
            no_volume = (
                no_orderbook.total_bid_volume_in_range((yes_mid - spread_range, yes_mid + spread_range)) +
                no_orderbook.total_ask_volume_in_range((yes_mid - spread_range, yes_mid + spread_range))
            )
            
            total_volume = yes_volume + no_volume
            
            # Estimate capacity (simplified heuristic)
            estimated_capacity = Decimal("1000")  # $1000 typical capacity
            
            return min(total_volume / estimated_capacity, Decimal("1.0"))
            
        except Exception as e:
            self.logger.error(f"Error estimating competition density: {e}")
            return Decimal("0.5")
    
    def _calculate_reward_score(self, total_daily_rewards: float, min_size: float, 
                              competition_density: Decimal, max_spread: float) -> Decimal:
        """Calculate overall reward score for ranking opportunities"""
        try:
            # Base score from daily rewards
            reward_component = Decimal(str(total_daily_rewards))
            
            # Penalty for high capital requirements
            capital_efficiency = max(Decimal("0.1"), Decimal("1000") / max(Decimal(str(min_size)), Decimal("1")))
            
            # Penalty for high competition
            competition_factor = Decimal("1.0") - competition_density
            
            # Bonus for larger spreads (more room to avoid fills)
            spread_bonus = Decimal(str(max_spread)) * Decimal("10")
            
            # Combined score
            score = reward_component * capital_efficiency * competition_factor + spread_bonus
            
            return max(score, Decimal("0"))
            
        except Exception as e:
            self.logger.error(f"Error calculating reward score: {e}")
            return Decimal("0")
    
    def _assess_risk_level(self, market, competition_density: Decimal, max_spread: float) -> str:
        """Assess risk level of the market opportunity"""
        try:
            risk_factors = 0
            
            # High competition increases risk
            if competition_density > Decimal("0.7"):
                risk_factors += 2
            elif competition_density > Decimal("0.4"):
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
            if risk_levels.get(opportunity.risk_level, 2) > risk_levels.get(self.criteria.max_risk_level, 1):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking screening criteria: {e}")
            return False
    
    def _convert_to_orderbook_snapshot(self, asset_id: str, orderbook_data: Dict, 
                                     current_price: float) -> OrderbookSnapshot:
        """Convert API orderbook data to OrderbookSnapshot"""
        try:
            # Parse bids
            bids = []
            for bid_data in orderbook_data.get("bids", []):
                price = Decimal(str(bid_data.get("price", 0)))
                size = Decimal(str(bid_data.get("size", 0)))
                bids.append(OrderbookLevel(price, size))
            
            # Parse asks
            asks = []
            for ask_data in orderbook_data.get("asks", []):
                price = Decimal(str(ask_data.get("price", 0)))
                size = Decimal(str(ask_data.get("size", 0)))
                asks.append(OrderbookLevel(price, size))
            
            # Calculate midpoint and spread
            if bids and asks:
                best_bid = bids[0].price
                best_ask = asks[0].price
                midpoint = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
            else:
                midpoint = Decimal(str(current_price))
                spread = Decimal("0.02")  # Default 2c spread
            
            return OrderbookSnapshot(
                asset_id=asset_id,
                bids=bids,
                asks=asks,
                midpoint=midpoint,
                spread=spread
            )
            
        except Exception as e:
            self.logger.error(f"Error converting orderbook data: {e}")
            # Return empty orderbook
            return OrderbookSnapshot(
                asset_id=asset_id,
                bids=[],
                asks=[],
                midpoint=Decimal(str(current_price)),
                spread=Decimal("0.02")
            )