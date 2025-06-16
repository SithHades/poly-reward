import os
import time
import logging
from typing import Any, Optional, Dict, List
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BookParams, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.constants import POLYGON
from src.api_models import Market, Rewards, RewardRate, Token

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

    def get_market_data(self, condition_id: str) -> Any:
        """Fetch market data for a given condition_id."""
        self.logger.info(f"Fetching market data for {condition_id}")
        return self._with_backoff(self.client.get_market, condition_id)

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
    
    def get_sampling_markets(self, next_cursor: Optional[str] = None, limit: int = 500) -> Dict[str, Any]:
        """
        Fetch sampling markets with reward information for liquidity provision screening.
        
        Args:
            next_cursor: Pagination cursor for next batch of markets
            limit: Maximum number of markets to return (default: 100)
            
        Returns:
            Dict containing:
                - data: List of Market objects with reward information
                - next_cursor: Cursor for next batch
                - limit: Number of markets requested
                - count: Number of markets returned
        """
        self.logger.info(f"Fetching sampling markets with cursor: {next_cursor}, limit: {limit}")
        
        if self.paper_trading:
            # Return mock data for paper trading
            return self._get_mock_sampling_markets(limit)
        
        try:
            # Check if the client has the sampling markets method
            if hasattr(self.client, 'get_sampling_markets'):
                params = dict()
                if next_cursor:
                    params['cursor'] = next_cursor
                return self._with_backoff(self.client.get_sampling_markets, **params)
            else:
                # Fallback: Use get_markets and filter for reward-eligible markets
                self.logger.warning("get_sampling_markets not available, using fallback method")
                return self._get_markets_fallback(next_cursor, limit)
                
        except Exception as e:
            self.logger.error(f"Error fetching sampling markets: {e}")
            # Return empty result structure on error
            return {
                "data": [],
                "next_cursor": next_cursor,
                "limit": limit,
                "count": 0
            }
    
    def sort_orderbooks(self, orderbooks: List[Dict[str, Any]] | Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Sort orderbooks by price.
        
        Args:
            orderbooks: List of orderbooks to sort
        """
        rlist = True
        if isinstance(orderbooks, dict):
            orderbooks = [orderbooks]
            rlist = False
        for entry in orderbooks:
            entry['bids'] = sorted(
                entry['bids'], key=lambda x: float(x['price']), reverse=True
            )
            entry['asks'] = sorted(
                entry['asks'], key=lambda x: float(x['price'])
            )
        return orderbooks if rlist else orderbooks[0]
    
    def get_orderbooks(self, token_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get orderbooks for a list of token IDs.
        
        Args:
            token_ids: List of token IDs to get orderbooks for
        """
        return self.sort_orderbooks(self._with_backoff(self.client.get_order_books, [BookParams(token_id=token_id) for token_id in token_ids]))

    def get_orderbook(self, token_id: str) -> Dict[str, Any]:
        """
        Get orderbook for a specific token.
        
        Args:
            token_id: The token ID to get orderbook for
            
        Returns:
            Dict containing bid/ask levels with prices and sizes
        """
        self.logger.info(f"Fetching orderbook for token {token_id}")
        
        if self.paper_trading:
            return self.sort_orderbooks(self._get_mock_orderbook(token_id))
        
        try:
            return self.sort_orderbooks(self._with_backoff(self.client.get_order_book, token_id))
        except Exception as e:
            self.logger.error(f"Error fetching orderbook for {token_id}: {e}")
            return {"bids": [], "asks": []}
    
    def calculate_midpoint(self, token_id: str) -> float:
        """
        Calculate current midpoint price for a token.
        
        Args:
            token_id: The token ID to get midpoint for
            
        Returns:
            Current midpoint price
        """
        try:
            orderbook = self.get_orderbook(token_id)
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            if not bids or not asks:
                self.logger.warning(f"Incomplete orderbook for {token_id}")
                if bids:
                    return float(bids[0].get("price", 0))
                elif asks:
                    return float(asks[0].get("price", 1))
                return 0.5  # Default midpoint
            
            best_bid = float(bids[0].get("price", 0))
            best_ask = float(asks[0].get("price", 1))
            
            return (best_bid + best_ask) / 2.0
            
        except Exception as e:
            self.logger.error(f"Error calculating midpoint for {token_id}: {e}")
            return 0.5
    
    def get_midpoint(self, token_id: str) -> float:
        """
        Get current midpoint price for a token.
        
        Args:
            token_id: The token ID to get midpoint for
        
        Returns:
            Current midpoint price
        """
        return self.client.get_midpoint(token_id)
            
    def get_midpoints(self, token_ids: List[str]) -> Dict[str, float]:
        """
        Get midpoints for a list of token IDs.
        
        Args:
            token_ids: List of token IDs to get midpoints for
        
        Returns:
            Dict mapping token_id to midpoint price
        """
        return self.client.get_midpoints(params=[BookParams(token_id=token_id) for token_id in token_ids])
    
    def _get_mock_sampling_markets(self, limit: int) -> Dict[str, Any]:
        """Generate mock sampling markets data for testing"""
        mock_markets = []
        
        for i in range(min(limit, 3)):  # Generate 3 mock markets
            mock_markets.append({
                "enable_order_book": True,
                "active": True,
                "closed": False,
                "archived": False,
                "accepting_orders": True,
                "accepting_order_timestamp": "2024-01-01T00:00:00Z",
                "minimum_order_size": 10.0 + i * 5,  # Varying min sizes
                "minimum_tick_size": 0.01,
                "condition_id": f"condition_{i}",
                "question_id": f"question_{i}",
                "question": f"Mock Market Question {i}?",
                "description": f"Mock market {i} for testing",
                "market_slug": f"mock-market-{i}",
                "end_date_iso": "2024-12-31T23:59:59Z",
                "game_start_time": None,
                "seconds_delay": 0,
                "fpmm": "",
                "maker_base_fee": 0.0,
                "taker_base_fee": 0.01,
                "notifications_enabled": True,
                "neg_risk": False,
                "neg_risk_market_id": "",
                "neg_risk_request_id": "",
                "icon": "https://example.com/icon.png",
                "image": "https://example.com/image.png",
                "rewards": {
                    "rates": [
                        {
                            "asset_address": "0x1234567890123456789012345678901234567890",
                            "rewards_daily_rate": 100.0 + i * 50  # Varying reward rates
                        }
                    ],
                    "min_size": 50.0 + i * 25,  # Varying minimum sizes for rewards
                    "max_spread": 0.03 - i * 0.005  # Varying max spreads
                },
                "is_50_50_outcome": True,
                "tokens": [
                    {
                        "token_id": f"token_yes_{i}",
                        "outcome": "Yes",
                        "price": 0.5 + i * 0.1,
                        "winner": False
                    },
                    {
                        "token_id": f"token_no_{i}",
                        "outcome": "No", 
                        "price": 0.5 - i * 0.1,
                        "winner": False
                    }
                ],
                "tags": ["test", "mock", f"category_{i}"]
            })
        
        return {
            "data": mock_markets,
            "next_cursor": None,
            "limit": limit,
            "count": len(mock_markets)
        }
    
    def _get_mock_orderbook(self, token_id: str) -> Dict[str, Any]:
        """Generate mock orderbook data for testing"""
        return {
            "bids": [
                {"price": "0.52", "size": "100"},
                {"price": "0.51", "size": "200"},
                {"price": "0.50", "size": "300"}
            ],
            "asks": [
                {"price": "0.54", "size": "150"},
                {"price": "0.55", "size": "250"},
                {"price": "0.56", "size": "350"}
            ]
        }
    
    def _get_markets_fallback(self, next_cursor: Optional[str], limit: int) -> Dict[str, Any]:
        """Fallback method when get_sampling_markets is not available"""
        self.logger.info("Using fallback market fetching method")
        
        try:
            # Try to get markets using available methods
            if hasattr(self.client, 'get_markets'):
                markets_data = self._with_backoff(self.client.get_markets)
                # Filter and format for reward-eligible markets
                # This would need to be implemented based on available market data
                return {
                    "data": markets_data.data[:limit] if hasattr(markets_data, 'data') and isinstance(markets_data.data, list) else [],
                    "next_cursor": None,
                    "limit": limit,
                    "count": min(len(markets_data) if isinstance(markets_data, list) else 0, limit)
                }
            else:
                self.logger.warning("No market fetching method available")
                return {
                    "data": [],
                    "next_cursor": next_cursor,
                    "limit": limit,
                    "count": 0
                }
        except Exception as e:
            self.logger.error(f"Fallback market fetching failed: {e}")
            return {
                "data": [],
                "next_cursor": next_cursor,
                "limit": limit,
                "count": 0
            }
