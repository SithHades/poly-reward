import os
import time
import logging
from typing import Any, Optional
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.constants import POLYGON

class PolymarketClient:
    def __init__(
        self,
        host: str = "https://clob.polymarket.com",
        key: Optional[str] = None,
        chain_id: int = POLYGON,
        signature_type: Optional[int] = None,
        funder: Optional[str] = None,
        paper_trading: bool = False,
        rate_limit_per_sec: int = 5,
    ):
        self.logger = logging.getLogger("PolymarketClient")
        self.paper_trading = paper_trading
        self.rate_limit_per_sec = rate_limit_per_sec
        self.last_request_time = 0.0
        self.client = ClobClient(
            host,
            key=key or os.getenv("PK"),
            chain_id=chain_id,
            signature_type=signature_type,
            funder=funder,
        )
        self.client.set_api_creds(self.client.create_or_derive_api_creds())

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self.last_request_time
        min_interval = 1.0 / self.rate_limit_per_sec
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def _with_backoff(self, func, *args, **kwargs):
        max_retries = 5
        delay = 1.0
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in {func.__name__}: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2

    def get_market_data(self, market_id: str) -> Any:
        """Fetch market data for a given market_id."""
        self.logger.info(f"Fetching market data for {market_id}")
        # Placeholder: Replace with actual API call
        return self._with_backoff(self.client.get_market, market_id)

    def place_order(self, price: float, size: float, side: str, token_id: str, order_type: OrderType = OrderType.GTC) -> Any:
        """Place an order on the market."""
        self.logger.info(f"Placing order: price={price}, size={size}, side={side}, token_id={token_id}")
        if self.paper_trading:
            self.logger.info("Paper trading mode: order not sent.")
            return {"status": "paper", "price": price, "size": size, "side": side, "token_id": token_id}
        order_args = OrderArgs(
            price=price,
            size=size,
            side=BUY if side.lower() == "buy" else SELL,
            token_id=token_id,
        )
        return self._with_backoff(self.client.create_and_post_order, order_args, order_type)

    def cancel_order(self, order_id: str) -> Any:
        """Cancel an order by order_id."""
        self.logger.info(f"Cancelling order {order_id}")
        if self.paper_trading:
            self.logger.info("Paper trading mode: cancel not sent.")
            return {"status": "paper", "order_id": order_id}
        return self._with_backoff(self.client.cancel_order, order_id)

    def get_order_status(self, order_id: str) -> Any:
        """Get the status of an order."""
        self.logger.info(f"Getting status for order {order_id}")
        return self._with_backoff(self.client.get_order, order_id)

    def get_positions(self, user_address: Optional[str] = None) -> Any:
        """Get current positions for the user."""
        self.logger.info(f"Getting positions for user {user_address or 'default'}")
        # Placeholder: Replace with actual API call if available
        return self._with_backoff(self.client.get_positions, user_address) if hasattr(self.client, "get_positions") else None 