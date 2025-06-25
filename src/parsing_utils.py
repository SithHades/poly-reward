import re
from src.models import Market, Rewards, SimplifiedMarket, TokenInfo
from dateutil import tz
from dateutil.parser import parse as parse_date
from datetime import datetime


ET_TZ = tz.gettz("America/New_York")


def map_market(raw: dict) -> Market:
    try:
        # Map tokens
        tokens = [TokenInfo(token_id=token["token_id"], outcome=token["outcome"], price=token["price"]) for token in raw.get("tokens", [])]
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
            if not all(hasattr(token, field) for field in ["token_id", "outcome", "price"]):
                raise ValueError(f"Missing required field in token: {token}")
            if not all(token.get(field) is not None for field in ["token_id", "outcome", "price"]):
                raise ValueError(f"Missing required field in token: {token}")
            tokens.append(TokenInfo(token_id=token["token_id"], outcome=token["outcome"], price=token["price"]))

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

    return SimplifiedMarket(
        condition_id=market["conditionId"],
        rewards=rewards,
        tokens=tokens,
        active=market.get("active", False),
        closed=market.get("closed", False),
        archived=market.get("archived", False),
        accepting_orders=market.get("acceptingOrders", False),
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
            tzinfo=ET_TZ
        )
        return dt
    except Exception:
        return None
