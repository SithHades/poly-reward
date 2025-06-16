from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class RewardRate:
    asset_address: str  # ERC-20 token address for rewards
    rewards_daily_rate: float  # Daily reward rate

@dataclass
class Rewards:
    rates: List[RewardRate]
    min_size: float  # Minimum market size to activate rewards
    max_spread: float  # Max allowed spread for reward eligibility

@dataclass
class Token:
    token_id: str  # Unique token ID for outcome
    outcome: str  # "Yes" or "No"
    price: float  # Current price
    winner: bool  # Outcome resolution

@dataclass
class Market:
    enable_order_book: bool
    active: bool
    closed: bool
    archived: bool
    accepting_orders: bool
    accepting_order_timestamp: str  # ISO timestamp
    minimum_order_size: float
    minimum_tick_size: float
    condition_id: str  # Condition hash
    question_id: str  # Unique market identifier
    question: str  # Market question
    description: str  # Market rules and resolution conditions
    market_slug: str  # URL-friendly name
    end_date_iso: str  # Market close time
    game_start_time: Optional[str]  # Null or datetime
    seconds_delay: int
    fpmm: str  # Usually empty, could refer to pricing mechanism
    maker_base_fee: float
    taker_base_fee: float
    notifications_enabled: bool
    neg_risk: bool
    neg_risk_market_id: str
    neg_risk_request_id: str
    icon: str  # URL to image
    image: str  # Duplicate of icon
    rewards: Rewards
    is_50_50_outcome: bool
    tokens: List[Token]  # Yes/No outcome tokens
    tags: List[str]  # Market category tags