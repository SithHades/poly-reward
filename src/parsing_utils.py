import re
from typing import Literal
from constants import MARKETS
from src.models import Market, Rewards, SimplifiedMarket, TokenInfo
from dateutil import tz
from dateutil.parser import parse as parse_date
from datetime import datetime, timedelta, tzinfo


# Timezone definitions
UTC = tz.UTC
ET = tz.gettz("America/New_York")
CEST = tz.gettz("Europe/Berlin")  # Use Berlin as it's more reliable for CEST


def map_market(raw: dict) -> Market:
    try:
        # Map tokens
        tokens = [
            TokenInfo(
                token_id=token["token_id"],
                outcome=token["outcome"],
                price=token["price"],
            )
            for token in raw.get("tokens", [])
        ]
        # Map rewards (if present)
        rewards_data = raw.get("rewards")
        rewards = None
        if rewards_data and "rates" in rewards_data and rewards_data["rates"]:
            rewards = Rewards(
                rewards_daily_rate=rewards_data["rates"][0]["rewards_daily_rate"],
                min_size=rewards_data.get("min_size", 0),
                max_spread=rewards_data.get("max_spread", 0),
            )
        # Build Market
        return Market(
            enable_order_book=raw.get("enable_order_book", False),
            active=raw.get("active", False),
            closed=raw.get("closed", False),
            archived=raw.get("archived", False),
            accepting_orders=raw.get("accepting_orders", False),
            minimum_order_size=raw.get("minimum_order_size", 0),
            minimum_tick_size=raw.get("minimum_tick_size", 0),
            condition_id=raw.get("condition_id", ""),
            question_id=raw.get("question_id", ""),
            question=raw.get("question", ""),
            description=raw.get("description", ""),
            market_slug=raw.get("market_slug", ""),
            end_date_iso=raw.get("end_date_iso", ""),
            maker_base_fee=raw.get("maker_base_fee", 0),
            taker_base_fee=raw.get("taker_base_fee", 0),
            notifications_enabled=raw.get("notifications_enabled", False),
            neg_risk=raw.get("neg_risk", False),
            neg_risk_market_id=raw.get("neg_risk_market_id", ""),
            neg_risk_request_id=raw.get("neg_risk_request_id", ""),
            rewards=rewards,
            is_50_50_outcome=raw.get("is_50_50_outcome", False),
            tokens=tokens,
            tags=raw.get("tags", []),
        )
    except KeyError as e:
        raise ValueError(f"Missing required field in raw data: {e}") from e


def map_simplified_market(raw: dict) -> SimplifiedMarket:
    try:
        # Map tokens
        tokens = []
        for token in raw.get("tokens", []):
            if not all(
                hasattr(token, field) for field in ["token_id", "outcome", "price"]
            ):
                raise ValueError(f"Missing required field in token: {token}")
            if not all(
                token.get(field) is not None
                for field in ["token_id", "outcome", "price"]
            ):
                raise ValueError(f"Missing required field in token: {token}")
            tokens.append(
                TokenInfo(
                    token_id=token["token_id"],
                    outcome=token["outcome"],
                    price=token["price"],
                )
            )

        # Map rewards (if present)
        rewards_data = raw.get("rewards")
        rewards = None
        if rewards_data and "rates" in rewards_data and rewards_data["rates"]:
            rewards = Rewards(
                rewards_daily_rate=rewards_data["rates"][0]["rewards_daily_rate"],
                min_size=rewards_data.get("min_size", 0),
                max_spread=rewards_data.get("max_spread", 0),
            )
        # Build SimplifiedMarket
        return SimplifiedMarket(
            condition_id=raw.get("condition_id", ""),
            rewards=rewards,
            tokens=tokens,
            active=raw.get("active", False),
            closed=raw.get("closed", False),
            archived=raw.get("archived", False),
            accepting_orders=raw.get("accepting_orders", False),
        )
    except KeyError as e:
        raise ValueError(f"Missing required field in raw data: {e}") from e


def transform_gamma_market_to_simplified(market: dict) -> SimplifiedMarket:
    # Parse tokens
    outcomes = eval(market.get("outcomes", "[]"))
    prices = eval(market.get("outcomePrices", "[]"))
    clob_ids = eval(market.get("clobTokenIds", "[]"))
    tokens = [
        TokenInfo(
            token_id=clob_ids[i] if i < len(clob_ids) else "unknown",
            outcome=outcomes[i],
            price=float(prices[i]),
        )
        for i in range(min(len(outcomes), len(prices)))
    ]

    # Parse rewards
    rewards = (
        Rewards(
            rewards_daily_rate=float(market.get("rewardsDailyRate", 0)),
            min_size=float(market.get("rewardsMinSize", 0)),
            max_spread=float(market.get("rewardsMaxSpread", 0)),
        )
        if any(
            k in market
            for k in ["rewardsMinSize", "rewardsMaxSpread", "rewardsDailyRate"]
        )
        else None
    )

    slug = market.get("slug", "")

    return SimplifiedMarket(
        condition_id=market["conditionId"],
        rewards=rewards,
        tokens=tokens,
        active=market.get("active", False),
        closed=market.get("closed", False),
        archived=market.get("archived", False),
        accepting_orders=market.get("acceptingOrders", False),
        slug=slug,
    )


def extract_datetime_from_slug(slug):
    match = re.search(r"([a-z]+)-(\d+)-(\d+)(am|pm)-et", slug)
    if not match:
        return None

    month_str, day_str, hour_str, meridiem = match.groups()
    try:
        # Build a date string like "June 22 6am"
        date_str = f"{month_str} {day_str} {hour_str}{meridiem}"
        # Parse with current year, assume ET timezone
        dt = parse_date(date_str + f" {datetime.now().year}", fuzzy=True).replace(
            tzinfo=ET
        )
        return dt
    except Exception:
        return None


def create_slug_from_datetime(dt: datetime, slug: Literal["ethereum", "bitcoin", "solana", "xrp"]="ethereum") -> str:
    """
    Create a slug for an ETH hourly prediction market of the form:
    ethereum-up-or-down-july-7-5am-et
    :param dt: The datetime to create the slug for.
    :param slug: The cryptocurrency type for the slug.
    :return: The slug.
    """
    # Convert to 12-hour format without leading zero
    hour_12 = dt.strftime('%I').lstrip('0') or '12'  # Handle midnight case
    month = dt.strftime('%B').lower()
    am_pm = dt.strftime('%p').lower()
    
    return f"{slug}-up-or-down-{month}-{dt.day}-{hour_12}{am_pm}-et"


def convert_to_et(dt: datetime, source_tz: tzinfo = None) -> datetime:
    """Convert datetime from source timezone to ET"""
    if dt.tzinfo is None:
        if source_tz is None:
            raise ValueError("source_tz must be provided for naive datetime")
        dt = dt.replace(tzinfo=source_tz)
    return dt.astimezone(ET)


def get_current_market_hour_et() -> datetime:
    """Get the current market hour in ET (rounded down to the hour)"""
    now_et = datetime.now(ET)
    return now_et.replace(minute=0, second=0, microsecond=0)


def get_next_market_hour_et() -> datetime:
    """Get the next market hour in ET"""
    current_hour = get_current_market_hour_et()
    return current_hour + timedelta(hours=1)


def get_current_market_slug(crypto: MARKETS = "ethereum") -> str:
    """Get the slug for the current market hour"""
    current_hour_et = get_current_market_hour_et()
    return create_slug_from_datetime(current_hour_et, crypto)


def get_next_market_slug(crypto: MARKETS = "ethereum") -> str:
    """Get the slug for the next market hour"""
    next_hour_et = get_next_market_hour_et()
    return create_slug_from_datetime(next_hour_et, crypto)


def convert_utc_to_market_slug(utc_dt: datetime, crypto: MARKETS = "ethereum") -> str:
    """Convert UTC datetime to market slug"""
    et_dt = convert_to_et(utc_dt, UTC)
    return create_slug_from_datetime(et_dt, crypto)


def convert_cest_to_market_slug(cest_dt: datetime, crypto: MARKETS = "ethereum") -> str:
    """Convert CEST datetime to market slug"""
    et_dt = convert_to_et(cest_dt, CEST)
    return create_slug_from_datetime(et_dt, crypto)
