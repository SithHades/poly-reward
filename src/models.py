from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

Token = str  # string identifier for a token, mostly representing a market side like "Yes", or "Up"

Midpoint = float  # the midpoint price of a market side.
Spread = float


class OrderStatus(str, Enum):
    LIVE = "LIVE"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class BookSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


SidePriceMap = dict[BookSide, float]  # mapping from BookSide to price for a given token
PricesResponse = dict[Token, SidePriceMap]  # mapping from token_id to SidePriceMap


class OrderArgsModel(BaseModel):
    token_id: Annotated[Token, Field(description="TokenID of the order")]
    price: Annotated[
        float,
        Field(description="Price used to create the order", default_factory=float),
    ]
    size: Annotated[
        float, Field(description="Size of the order", default_factory=float)
    ]
    side: Annotated[BookSide, Field(description="Side of the order (buy/sell)")]


class OrderDetails(BaseModel):
    order_id: Annotated[str, Field(alias="id", description="Id of the order")]
    status: Annotated[OrderStatus, Field(description="Status of the order")]
    owner: Annotated[str, Field(description="Owner UUID")]
    maker_address: Annotated[str, Field(description="Ethereum address of the maker")]
    market_id: Annotated[
        str,
        Field(alias="market", description="Market address where the order is placed"),
    ]
    asset_id: Annotated[
        Token, Field(description="Unique asset identifier, represents the side/token")
    ]
    side: Annotated[BookSide, Field(description="Side of the order (BUY or SELL)")]
    original_size: Annotated[
        int, Field(gt=0, description="Total size of the order in shares")
    ]
    matched_size: Annotated[
        int, Field(alias="size_matched", ge=0, description="Matched size in shares")
    ]
    price: Annotated[float, Field(description="Order price in float format")]
    expiration: Annotated[
        int, Field(description="Expiration time in UNIX timestamp, 0 if no expiry")
    ] = 0
    order_type: Annotated[str, Field(description="Type of order (e.g., GTC)")]
    created_at: Annotated[int, Field(description="Timestamp of order creation")]
    associate_trades: Annotated[
        list, Field(description="List of associated trades")
    ] = []

    @field_validator("price", mode="before")
    def parse_price(cls, v):
        return float(v)

    @field_validator("original_size", mode="before")
    def parse_original_size(cls, v):
        return int(v)

    class Config:
        validate_by_name = True


class TokenInfo(BaseModel):
    token_id: str  # equals asset_id in most cases.
    outcome: str  # the outcome of the token, e.g., "Yes", "No", "Up", "Down"
    price: float
    winner: Optional[bool] = None


class Rewards(BaseModel):
    rewards_daily_rate: float
    min_size: float
    max_spread: float


class Market(BaseModel):
    enable_order_book: bool
    active: bool
    closed: bool
    archived: bool
    accepting_orders: bool
    minimum_order_size: float
    minimum_tick_size: float
    condition_id: str  # ID of the market
    question_id: str
    question: str
    description: str
    market_slug: str
    end_date_iso: Optional[datetime] = None
    maker_base_fee: float
    taker_base_fee: float
    notifications_enabled: bool
    neg_risk: bool
    neg_risk_market_id: str
    neg_risk_request_id: str
    rewards: Optional[Rewards] = None
    is_50_50_outcome: bool
    tokens: list[TokenInfo]
    tags: Optional[list[str]] = None


class SimplifiedMarket(BaseModel):
    condition_id: str
    rewards: Optional[Rewards] = None
    tokens: list[TokenInfo]
    active: bool
    closed: bool
    archived: bool
    accepting_orders: bool
    slug: Annotated[str, Field(description="Slug of the market")] = ""


class Position(BaseModel):
    market_id: str
    size: int
    entry_price: float
    current_price: float | None = None
    last_updated: datetime | None = None


class MarketCondition(Enum):
    """Market conditions for strategy decision making"""

    ATTRACTIVE = "attractive"  # Low competition, good for LP
    COMPETITIVE = "competitive"  # High competition, low reward share
    VOLATILE = "volatile"  # High volatility, risk of fills
    UNAVAILABLE = "unavailable"  # Outside parameters or errors


@dataclass
class OrderbookLevel:
    """Represents a single level in the orderbook"""

    price: float
    size: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OrderbookSnapshot:
    """Complete orderbook snapshot for a market"""

    asset_id: str
    bids: list[OrderbookLevel]
    asks: list[OrderbookLevel]
    midpoint: float
    spread: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    def total_bid_volume_in_range(self, price_range: Tuple[float, float]) -> float:
        """Calculate total bid volume within price range"""
        min_price, max_price = price_range
        return sum(
            level.size for level in self.bids if min_price <= level.price <= max_price
        )

    def total_ask_volume_in_range(self, price_range: tuple[float, float]) -> float:
        """Calculate total ask volume within price range"""
        min_price, max_price = price_range
        return sum(
            level.size for level in self.asks if min_price <= level.price <= max_price
        )


@dataclass
class VolatilityMetrics:
    """Tracks market volatility indicators"""

    midpoint_changes: list[float] = field(default_factory=list)
    spread_changes: list[float] = field(default_factory=list)
    volume_spikes: list[float] = field(default_factory=list)
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    window: int = 50
    snapshots: list["OrderbookSnapshot"] = field(
        default_factory=list
    )  # Store recent snapshots

    def add_snapshot(self, snapshot: OrderbookSnapshot):
        """Add a new orderbook snapshot and update volatility metrics"""
        if self.snapshots:
            prev = self.snapshots[-1]
            # Calculate midpoint change
            midpoint_change = snapshot.midpoint - prev.midpoint
            self.add_midpoint_change(midpoint_change)
            # Calculate spread change
            spread_change = snapshot.spread - prev.spread
            self.add_spread_change(spread_change)
            # Optionally, detect volume spikes (example: total volume difference)
            prev_bid_vol = sum(b.size for b in prev.bids)
            prev_ask_vol = sum(a.size for a in prev.asks)
            curr_bid_vol = sum(b.size for b in snapshot.bids)
            curr_ask_vol = sum(a.size for a in snapshot.asks)
            volume_spike = abs(
                (curr_bid_vol + curr_ask_vol) - (prev_bid_vol + prev_ask_vol)
            )
            self.add_volume_spike(volume_spike)
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.window:
            self.snapshots.pop(0)
        self.last_update = datetime.now(timezone.utc)

    def add_midpoint_change(self, change: float):
        """Add midpoint change and maintain time window"""
        self._add_to_window(self.midpoint_changes, change)

    def add_spread_change(self, change: float):
        """Add spread change and maintain time window"""
        self._add_to_window(self.spread_changes, change)

    def add_volume_spike(self, volume: float):
        """Add volume spike and maintain time window"""
        self._add_to_window(self.volume_spikes, volume)

    def _add_to_window(self, data_list: list[float], value: float):
        """Add value and maintain time window (simplified - in reality would track timestamps)"""
        data_list.append(value)
        if len(data_list) > 50:  # Keep last 50 data points as proxy for time window
            data_list.pop(0)
        self.last_update = datetime.now(timezone.utc)

    def is_volatile(
        self,
        midpoint_threshold: float = 0.02,
        spread_threshold: float = 0.05,
    ) -> bool:
        """Determine if market is currently volatile"""
        if not self.midpoint_changes and not self.spread_changes:
            return False

        # Check for recent significant midpoint moves
        if self.midpoint_changes:
            recent_midpoint_data = (
                self.midpoint_changes[-10:]
                if len(self.midpoint_changes) >= 10
                else self.midpoint_changes
            )
            recent_midpoint_volatility = max(
                abs(change) for change in recent_midpoint_data
            )
        else:
            recent_midpoint_volatility = 0

        # Check for recent spread expansion
        if self.spread_changes:
            recent_spread_data = (
                self.spread_changes[-10:]
                if len(self.spread_changes) >= 10
                else self.spread_changes
            )
            recent_spread_volatility = max(abs(change) for change in recent_spread_data)
        else:
            recent_spread_volatility = 0

        return (
            recent_midpoint_volatility > midpoint_threshold
            or recent_spread_volatility > spread_threshold
        )
