from enum import Enum
from typing import Annotated, Optional

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


SidePriceMap = dict[
    BookSide, float
]  # mapping from BookSide to price for a given token
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
    end_date_iso: Optional[str] = None
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
