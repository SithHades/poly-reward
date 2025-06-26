import logging
import os
from typing import Optional, List, Dict, Any
import requests
from datetime import datetime, timezone

from src.models import (
    BookSide,
    Market,
    OrderArgsModel,
    OrderDetails,
    OrderbookLevel,
    OrderbookSnapshot,
    TokenInfo,
    Token,
    Rewards,
)

LIMITLESS_API_URL = os.getenv("LIMITLESS_API_URL", "https://api.limitless.exchange")

class LimitlessClient:
    def __init__(self, api_url: Optional[str] = None, session: Optional[requests.Session] = None):
        self.api_url = api_url or LIMITLESS_API_URL
        self.session = session or requests.Session()
        self.logger = logging.getLogger("LimitlessClient")

    def get_markets(self) -> List[Market]:
        """Fetch all active markets from Limitless Exchange."""
        url = f"{self.api_url}/markets/active"
        self.logger.info(f"Fetching active markets from {url}")
        resp = self.session.get(url)
        resp.raise_for_status()
        data = resp.json()
        return [self._map_market(m) for m in data.get("markets", data)]

    def get_market(self, address_or_slug: str) -> Optional[Market]:
        """Fetch a single market by address or slug."""
        url = f"{self.api_url}/markets/{address_or_slug}"
        self.logger.info(f"Fetching market {address_or_slug} from {url}")
        resp = self.session.get(url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return self._map_market(data)

    def get_orderbook(self, slug: str) -> OrderbookSnapshot:
        """Fetch the orderbook for a given market slug."""
        url = f"{self.api_url}/markets/{slug}/orderbook"
        self.logger.info(f"Fetching orderbook for {slug} from {url}")
        resp = self.session.get(url)
        resp.raise_for_status()
        data = resp.json()
        return self._map_orderbook(slug, data)

    def get_user_orders(self, slug: str) -> List[OrderDetails]:
        """Fetch all user orders for a given market slug."""
        url = f"{self.api_url}/markets/{slug}/user-orders"
        self.logger.info(f"Fetching user orders for market {slug} from {url}")
        resp = self.session.get(url)
        resp.raise_for_status()
        data = resp.json()
        # Assume data is a list of order dicts
        return [self._map_order_details(o) for o in data]

    def create_order(self, order_args: OrderArgsModel, owner_id: int, order_type: str, market_slug: str) -> Dict[str, Any]:
        """Create and post a new order to Limitless Exchange."""
        url = f"{self.api_url}/orders"
        self.logger.info(f"Creating order at {url} for market {market_slug}")
        payload = {
            "order": self._order_args_to_api(order_args),
            "ownerId": owner_id,
            "orderType": order_type,
            "marketSlug": market_slug,
        }
        resp = self.session.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a single order by order ID."""
        url = f"{self.api_url}/orders/{order_id}"
        self.logger.info(f"Cancelling order {order_id} at {url}")
        resp = self.session.delete(url)
        resp.raise_for_status()
        return resp.json()

    def cancel_orders_batch(self, order_ids: List[str]) -> Dict[str, Any]:
        """Cancel multiple orders in a batch."""
        url = f"{self.api_url}/orders/cancel-batch"
        self.logger.info(f"Cancelling batch orders at {url}")
        payload = {"orderIds": order_ids}
        resp = self.session.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def cancel_all_orders(self, market_slug: str) -> Dict[str, Any]:
        """Cancel all orders for a specific market."""
        url = f"{self.api_url}/orders/all/{market_slug}"
        self.logger.info(f"Cancelling all orders for market {market_slug} at {url}")
        resp = self.session.delete(url)
        resp.raise_for_status()
        return resp.json()

    # --- Mapping helpers ---
    def _map_market(self, raw: dict) -> Market:
        """Map Limitless API market dict to shared Market model."""
        tokens = []
        for t in raw.get("tokens", []):
            tokens.append(
                TokenInfo(
                    token_id=t.get("token_id", t.get("id", "")),
                    outcome=t.get("outcome", ""),
                    price=float(t.get("price", 0)),
                )
            )
        rewards = None
        rewards_data = raw.get("rewards")
        if rewards_data:
            rewards = Rewards(
                rewards_daily_rate=float(rewards_data.get("todaysRewards", 0)),
                min_size=float(rewards_data.get("min_size", 0)),
                max_spread=float(rewards_data.get("max_spread", 0)),
            )
        return Market(
            enable_order_book=raw.get("enable_order_book", True),
            active=raw.get("active", not raw.get("closed", False)),
            closed=raw.get("closed", False),
            archived=raw.get("archived", False),
            accepting_orders=raw.get("accepting_orders", True),
            minimum_order_size=raw.get("minimum_order_size", 0),
            minimum_tick_size=raw.get("minimum_tick_size", 0),
            condition_id=raw.get("condition_id", raw.get("id", "")),
            question_id=raw.get("question_id", ""),
            question=raw.get("title", raw.get("question", "")),
            description=raw.get("description", ""),
            market_slug=raw.get("slug", ""),
            end_date_iso=raw.get("deadline", None),
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

    def _map_orderbook(self, asset_id: str, data: dict) -> OrderbookSnapshot:
        """Map Limitless API orderbook to OrderbookSnapshot."""
        bids = [OrderbookLevel(price=float(b["price"]), size=float(b["size"])) for b in data.get("bids", [])]
        asks = [OrderbookLevel(price=float(a["price"]), size=float(a["size"])) for a in data.get("asks", [])]
        if bids and asks:
            best_bid = bids[0].price
            best_ask = asks[0].price
            midpoint = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
        else:
            midpoint = 0.5
            spread = 0.0
        return OrderbookSnapshot(
            asset_id=asset_id,
            bids=bids,
            asks=asks,
            midpoint=midpoint,
            spread=spread,
            timestamp=datetime.now(timezone.utc),
        )

    def _order_args_to_api(self, order_args: OrderArgsModel) -> dict:
        """Convert OrderArgsModel to Limitless API order dict."""
        return {
            "tokenId": order_args.token_id,
            "price": order_args.price,
            "size": order_args.size,
            "side": 0 if order_args.side == BookSide.BUY else 1,  # 0=BUY, 1=SELL
        }

    def _map_order_details(self, raw: dict) -> OrderDetails:
        """Map Limitless API order dict to shared OrderDetails model."""
        return OrderDetails(
            order_id=raw.get("id", raw.get("order_id", "")),
            status=raw.get("status", "LIVE"),
            owner=raw.get("owner", ""),
            maker_address=raw.get("maker", ""),
            market_id=raw.get("market", raw.get("market_id", "")),
            asset_id=raw.get("token_id", raw.get("asset_id", "")),
            side=BookSide.BUY if raw.get("side", 0) == 0 else BookSide.SELL,
            original_size=int(raw.get("original_size", raw.get("size", 0))),
            matched_size=int(raw.get("matched_size", raw.get("size_matched", 0))),
            price=float(raw.get("price", 0)),
            expiration=int(raw.get("expiration", 0)),
            order_type=raw.get("order_type", "GTC"),
            created_at=int(raw.get("created_at", datetime.now(timezone.utc).timestamp())),
            associate_trades=raw.get("associate_trades", []),
        ) 