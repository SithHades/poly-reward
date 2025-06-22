from src.models import Market, Rewards, SimplifiedMarket, TokenInfo


def map_market(raw: dict) -> Market:
    try:
        # Map tokens
        tokens = [TokenInfo(**token) for token in raw.get("tokens", [])]
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
        tokens = [TokenInfo(**token) for token in raw.get("tokens", [])]
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
