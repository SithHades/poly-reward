import logging
import os
from typing import Optional, List, Dict, Any
import requests
from datetime import datetime, timezone

try:
    from eth_account import Account
    from eth_account.messages import encode_defunct
except ImportError:
    raise ImportError(
        "eth_account is required for signing. Install with 'uv pip install eth-account'."
    )

from src.models import (
    BookSide,
    Market,
    OrderArgsModel,
    OrderDetails,
    OrderbookLevel,
    OrderbookSnapshot,
    TokenInfo,
    Rewards,
)

LIMITLESS_API_URL = os.getenv("LIMITLESS_API_URL", "https://api.limitless.exchange")
LIMITLESS_WALLET = os.getenv("LIMITLESS_WALLET")
LIMITLESS_PK = os.getenv("LIMITLESS_PK")


class LimitlessClient:
    def __init__(
        self, api_url: Optional[str] = None, session: Optional[requests.Session] = None
    ):
        self.api_url = api_url or LIMITLESS_API_URL
        self.session = session or requests.Session()
        self.logger = logging.getLogger("LimitlessClient")
        self.wallet = LIMITLESS_WALLET
        self.private_key = LIMITLESS_PK
        if not self.wallet or not self.private_key:
            self.logger.warning(
                "LIMITLESS_WALLET and LIMITLESS_PK must be set for authentication."
            )

    def login(
        self,
        client: str = "eoa",
        smart_wallet: Optional[str] = None,
        referral: Optional[str] = None,
    ) -> bool:
        """
        Authenticate with the Limitless API using wallet signature.
        - Fetches a signing message from /auth/signing-message
        - Signs it with the private key
        - Posts to /auth/login with required headers and body
        - Stores session cookie for future requests
        Returns True if login is successful, False otherwise.
        """
        if not self.wallet or not self.private_key:
            raise ValueError(
                "LIMITLESS_WALLET and LIMITLESS_PK must be set in environment."
            )
        # 1. Get signing message
        url = f"{self.api_url}/auth/signing-message"
        self.logger.info(f"Fetching signing message from {url}")
        resp = self.session.get(url)
        resp.raise_for_status()
        signing_message = resp.json()
        if not isinstance(signing_message, str):
            raise ValueError("Signing message response is not a string.")
        # 2. Sign the message
        acct = Account.from_key(self.private_key)
        message = encode_defunct(text=signing_message)
        signature = Account.sign_message(
            message, private_key=self.private_key
        ).signature.hex()
        # 3. Prepare headers and body
        headers = {
            "x-account": self.wallet,
            "x-signing-message": signing_message,
            "x-signature": signature,
        }
        body = {
            "client": client,
        }
        if smart_wallet:
            body["smartWallet"] = smart_wallet
        if referral:
            body["r"] = referral
        # 4. Post to /auth/login
        login_url = f"{self.api_url}/auth/login"
        self.logger.info(f"Logging in at {login_url}")
        resp = self.session.post(login_url, headers=headers, json=body)
        if resp.status_code == 200:
            self.logger.info("Login successful. Session cookie stored.")
            return True
        else:
            self.logger.error(f"Login failed: {resp.status_code} {resp.text}")
            return False

    def verify_auth(self) -> bool:
        """
        Verify if the current session is authenticated (session cookie is valid).
        Returns True if authenticated, False otherwise.
        """
        url = f"{self.api_url}/auth/verify-auth"
        self.logger.info(f"Verifying authentication at {url}")
        resp = self.session.get(url)
        if resp.status_code == 200:
            self.logger.info("Session is authenticated.")
            return True
        else:
            self.logger.warning(f"Session not authenticated: {resp.status_code}")
            return False

    def logout(self) -> bool:
        """
        Logout and clear the session cookie.
        Returns True if logout is successful.
        """
        url = f"{self.api_url}/auth/logout"
        self.logger.info(f"Logging out at {url}")
        resp = self.session.post(url)
        if resp.status_code == 200:
            self.logger.info("Logout successful.")
            self.session.cookies.clear()
            return True
        else:
            self.logger.warning(f"Logout failed: {resp.status_code}")
            return False

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

    def create_order(
        self,
        order_args: OrderArgsModel,
        owner_id: int,
        order_type: str,
        market_slug: str,
    ) -> Dict[str, Any]:
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
        bids = [
            OrderbookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderbookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in data.get("asks", [])
        ]
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
            created_at=int(
                raw.get("created_at", datetime.now(timezone.utc).timestamp())
            ),
            associate_trades=raw.get("associate_trades", []),
        )
