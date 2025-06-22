from decimal import Decimal
import logging
import os
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    BookParams,
    OrderBookSummary,
    OrderScoringParams,
    OrdersScoringParams,
    OpenOrderParams,
    OrderArgs,
    OrderType,
    PostOrdersArgs,
    SignedOrder,
    TradeParams,
)
from py_clob_client.constants import POLYGON
from py_clob_client.exceptions import PolyApiException
from py_clob_client.order_builder.builder import BUY, SELL
from ratelimit import limits, sleep_and_retry

from src.parsing_utils import map_market, map_simplified_market
from src.constants import DEFAULT_MIDPOINT, DEFAULT_SPREAD, DEFAULT_TICK_SIZE
from src.models import (
    BookSide,
    Market,
    Midpoint,
    OrderArgsModel,
    OrderDetails,
    PricesResponse,
    SimplifiedMarket,
    Spread,
    Token,
)


def rate_limited(calls, period):
    def decorator(func):
        @sleep_and_retry
        @limits(calls=calls, period=period)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


class Client:
    def __init__(
        self,
        host: str = "https://clob.polymarket.com",
        key: Optional[str] = None,
        address: Optional[str] = None,
    ):
        self.logger = logging.getLogger("Client")
        client = ClobClient(
            host,
            key=key or os.getenv("PK"),
            chain_id=POLYGON,
            signature_type=1,
            funder=address or os.getenv("BROWSER_ADDRESS"),
        )

        rate_limits = {
            "get_order_book": (50, 10),
            "get_order_books": (50, 10),
            "get_price": (100, 10),
            "get_prices": (100, 10),
            "get_sampling_markets": (50, 10),
            "get_sampling_simplified_markets": (50, 10),
            "get_markets": (50, 10),
            "get_simplified_markets": (50, 10),
            "post_order": (500, 10),
            "post_orders": (500, 10),
            "cancel_order": (500, 10),
            "cancel_orders": (500, 10),
            "cancel_all": (500, 10),
        }

        # Apply rate limits to client methods
        for method_name, (calls, period) in rate_limits.items():
            original_method = getattr(client, method_name)
            wrapped_method = rate_limited(calls, period)(original_method)
            setattr(client, method_name, wrapped_method)

        self.client = client

        self.client.set_api_creds(self.client.create_or_derive_api_creds())

    def get_midpoint(self, token: Token) -> Midpoint:
        """
        Get the midpoint for a given token.
        :param token: The token for which to get the midpoint.
        :return: The midpoint price as a Decimal. If an error occurs, returns Decimal("0").
        """
        self.logger.info(f"Getting midpoint for token: {token}")
        midpoint_data = self.client.get_midpoint(token)
        try:
            midpoint = Decimal(midpoint_data["mid"])
            return midpoint
        except (KeyError, ValueError) as e:
            self.logger.error(f"Error getting midpoint for token {token}: {e}")
            return Decimal(DEFAULT_MIDPOINT)

    def get_midpoints(self, tokens: list[Token]) -> dict[Token, Midpoint]:
        """
        Get midpoints for a list of tokens.
        :param tokens: List of tokens to get midpoints for.
        :return: Dictionary mapping tokens to their midpoints.
        """
        self.logger.info(f"Getting midpoints for tokens: {tokens}")
        midpoints = self.client.get_midpoints([BookParams(token) for token in tokens])
        for token, midpoint in midpoints.items():
            try:
                midpoints[token] = Decimal(midpoint)
            except (KeyError, ValueError) as e:
                self.logger.error(f"Error getting midpoint for token {token}: {e}")
                midpoints[token] = Decimal(DEFAULT_MIDPOINT)
        return midpoints

    def get_price(self, token: Token, side: BookSide) -> Decimal:
        """
        Get the price for a given token and side.
        :param token: The token for which to get the price.
        :param side: The side of the order ('BUY' or 'SELL').
        :return: The price as a Decimal. If an error occurs, returns Decimal("0").
        """
        self.logger.info(f"Getting price for token: {token}, side: {side}")
        try:
            price = self.client.get_price(token, side)
            return Decimal(price["price"])
        except Exception as e:
            self.logger.error(
                f"Error getting price for token {token}, side {side}: {e}"
            )
            return Decimal(DEFAULT_MIDPOINT)

    def get_prices(self, params: list[BookParams]) -> PricesResponse:
        """
        Get prices for a given list of BookParams objects.
        :param params: The list of BookParams objects containing token and side.
        :return: A dictionary of prices for the specified token and side.
        """
        self.logger.info(f"Getting prices for params: {params}")
        try:
            prices = self.client.get_prices(params)
            return {
                token: {
                    BookSide(str(side).lower()): Decimal(price)
                    for side, price in side_prices.items()
                }
                for token, side_prices in prices.items()
            }
        except Exception as e:
            self.logger.error(f"Error getting prices for params {params}: {e}")
            return {}

    def get_spread(self, token: Token) -> Spread:
        """
        Get the spread for a given token.
        :param token: The token for which to get the spread.
        :return: The spread as a Decimal. If an error occurs, returns Decimal("0").
        """
        self.logger.info(f"Getting spread for token: {token}")
        try:
            spread = self.client.get_spread(token)
            return Decimal(spread)
        except Exception as e:
            self.logger.error(f"Error getting spread for token {token}: {e}")
            return Decimal(DEFAULT_SPREAD)

    def get_spreads(self, tokens: list[Token]) -> dict[Token, Spread]:
        """
        Get spreads for a list of tokens.
        :param tokens: List of tokens to get spreads for.
        :return: Dictionary mapping tokens to their spreads.
        """
        self.logger.info(f"Getting spreads for tokens: {tokens}")
        spreads_data = self.client.get_spreads([BookParams(token) for token in tokens])
        spreads = {}
        for token, spread in spreads_data.items():
            try:
                spreads[token] = Decimal(spread)
            except (KeyError, ValueError) as e:
                self.logger.error(f"Error getting spread for token {token}: {e}")
                spreads[token] = Decimal(DEFAULT_SPREAD)
        return spreads

    def get_tick_size(self, token: Token) -> Decimal:
        """
        Get the tick size for a given token.
        :param token: The token for which to get the tick size.
        :return: The tick size as a Decimal. If an error occurs, returns Decimal("0").
        """
        self.logger.info(f"Getting tick size for token: {token}")
        try:
            tick_size = self.client.get_tick_size(token)
            return Decimal(tick_size)
        except Exception as e:
            self.logger.error(f"Error getting tick size for token {token}: {e}")
            return Decimal(DEFAULT_TICK_SIZE)

    def get_neg_risk(self, token: Token) -> bool:
        """
        Get the negative risk status for a given token.
        :param token: The token for which to get the negative risk status.
        :return: True if negative risk is enabled, False otherwise. If an error occurs, returns False.
        """
        self.logger.info(f"Getting negative risk status for token: {token}")
        try:
            neg_risk = self.client.get_neg_risk(token)
            return neg_risk
        except Exception as e:
            self.logger.error(f"Error getting negative risk for token {token}: {e}")
            return True

    def create_order(self, order_args: OrderArgsModel) -> SignedOrder:
        """
        Create an order with the given parameters.
        :param order_args: The arguments for creating the order.
        :return: The created order.
        """
        self.logger.info(f"Creating order with args: {order_args}")
        try:
            order_args = OrderArgs(
                token_id=order_args.token_id,
                price=float(order_args.price),
                size=float(order_args.size),
                side=BUY if order_args.side == BookSide.BUY else SELL,
            )
            order = self.client.create_order(order_args)
            return order
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            raise

    def place_order(
        self, order: SignedOrder, order_type: Optional[OrderType] = "GTC"
    ) -> str:
        """
        Place an order on the CLOB.
        :param order: The signed order to place.
        :return: The order ID if the order was placed successfully, otherwise an empty string.
        :raises PolyApiException: If there is an error with the API.
        """
        self.logger.info(f"Placing order: {order}")
        try:
            result = self.client.post_order(order)
            order_id = result.get("orderID", None)
            status = result.get("status", None)
            error_msg = result.get("errorMsg", None)
            if order_id:
                self.logger.info(
                    f"Order placed successfully: {order_id}, status: {status}"
                )
                return order_id
            else:
                raise Exception(f"Failed to place order: {error_msg}")
        except PolyApiException as e:
            self.logger.error(f"PolyApiException while placing order: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise e

    def place_orders(self, orders: list[PostOrdersArgs]) -> tuple[list[str], list[str]]:
        """
        Place multiple orders on the CLOB.
        :param orders: List of signed orders to place.
        :return: List of order IDs for successfully placed orders and a list of indexes of failed orders.
        """
        self.logger.info(f"Placing {len(orders)} orders: {orders}")
        try:
            results = self.client.post_orders(orders)
            order_ids = []
            failed_indexes = []
            for i, result in enumerate(results):
                order_id = result.get("orderID", None)
                status = result.get("status", None)
                error_msg = result.get("errorMsg", None)
                if order_id:
                    self.logger.info(
                        f"Order {i} placed successfully: {order_id}, status: {status}"
                    )
                    order_ids.append(order_id)
                else:
                    self.logger.error(f"Failed to place order {i}: {error_msg}")
                    failed_indexes.append(i)
            return order_ids, failed_indexes
        except Exception as e:
            self.logger.error(f"Error placing multiple orders: {e}")
            return [], []

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order by its ID.
        :param order_id: The ID of the order to cancel.
        :return: True if the order was canceled successfully, False otherwise.
        """
        self.logger.info(f"Cancelling order with ID: {order_id}")
        try:
            result = self.client.cancel(order_id)
            not_canceled = result.get("not_canceled", [])
            canceled = result.get("canceled", [])
            return order_id in canceled and order_id not in not_canceled
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def cancel_orders(self, order_ids: list[str]) -> tuple[list[str], list[str]]:
        """
        Cancel multiple orders by their IDs.
        :param order_ids: List of order IDs to cancel.
        :return: List of successfully canceled order IDs and a list of order IDs that were not canceled.
        """
        self.logger.info(f"Cancelling orders: {order_ids}")
        try:
            result = self.client.cancel_orders(order_ids)
            canceled = result.get("canceled", [])
            not_canceled = result.get("not_canceled", [])
            return canceled, not_canceled
        except Exception as e:
            self.logger.error(f"Error cancelling multiple orders: {e}")
            return [], []

    def cancel_all(self) -> list[str]:
        """
        Cancel all open orders.
        :return: List of order IDs that were not successfully canceled.
        """
        self.logger.info("Cancelling all open orders")
        try:
            result = self.client.cancel_all()
            not_canceled = result.get("not_canceled", [])
            return not_canceled
        except Exception as e:
            self.logger.error(f"Error cancelling all orders: {e}")
            return []

    def get_orders(
        self, params: Optional[OpenOrderParams] = None
    ) -> list[OrderDetails]:
        """
        Get all open orders.
        :return: List of OrderDetails for all open orders.
        """
        self.logger.info("Getting all open orders")
        try:
            raw_orders = self.client.get_orders(params)
            return [OrderDetails(**order) for order in raw_orders]
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            return []

    def get_order_book(self, token: Token) -> OrderBookSummary:
        """
        Get the order book for a given token.
        :param token: The token for which to get the order book.
        :return: An OrderBookSummary object containing the order book details.
        """
        self.logger.info(f"Getting order book for token: {token}")
        try:
            return self.client.get_order_book(token)
        except Exception as e:
            self.logger.error(f"Error getting order book for token {token}: {e}")
            return OrderBookSummary(
                market="", asset_id=token, bids=[], asks=[], timestamp="", hash=""
            )

    def get_order_books(self, tokens: list[Token]) -> dict[Token, OrderBookSummary]:
        """
        Get order books for a list of tokens.
        :param tokens: List of tokens to get order books for.
        :return: Dictionary mapping tokens to their OrderBookSummary.
        """
        self.logger.info(f"Getting order books for tokens: {tokens}")
        try:
            order_books = self.client.get_order_books(
                [BookParams(token) for token in tokens]
            )
            return {book.asset_id: OrderBookSummary(**book) for book in order_books}
        except Exception as e:
            self.logger.error(f"Error getting order books for tokens {tokens}: {e}")
            return {}

    def get_trades(self, params: TradeParams = None):
        # TODO Implement if really needed
        return self.client.get_trades(params)

    def get_last_trade_price(self, token: Token) -> Decimal:
        """
        Get the last trade price for a given token.
        :param token: The token for which to get the last trade price.
        :return: The last trade price as a Decimal. If an error occurs, returns Decimal("0").
        """
        self.logger.info(f"Getting last trade price for token: {token}")
        try:
            last_trade_price = self.client.get_last_trade_price(token)
            return Decimal(last_trade_price.get("price", "0"))
        except Exception as e:
            self.logger.error(f"Error getting last trade price for token {token}: {e}")
            return Decimal("0")

    def get_last_trade_prices(self, tokens: list[Token]) -> dict[Token, Decimal]:
        """
        Get last trade prices for a list of tokens.
        :param tokens: List of tokens to get last trade prices for.
        :return: Dictionary mapping tokens to their last trade prices.
        """
        self.logger.info(f"Getting last trade prices for tokens: {tokens}")
        try:
            last_trade_prices = self.client.get_last_trade_prices(
                [BookParams(token) for token in tokens]
            )
            return {
                token: Decimal(price.get("price", "0"))
                for token, price in last_trade_prices.items()
            }
        except Exception as e:
            self.logger.error(
                f"Error getting last trade prices for tokens {tokens}: {e}"
            )
            return {token: Decimal("0") for token in tokens}

    def is_order_scoring(self, order_id: str):
        """
        Check if order scoring is enabled for a given order ID.
        :param order_id: The order ID to check.
        :return: True if order scoring is enabled, False otherwise.
        """
        self.logger.info(
            f"Checking if order scoring is enabled for order ID: {order_id}"
        )
        try:
            return bool(
                self.client.is_order_scoring(OrderScoringParams(orderId=order_id)).get(
                    "scoring", False
                )
            )
        except Exception as e:
            self.logger.error(
                f"Error checking order scoring for order ID {order_id}: {e}"
            )
            return False

    def are_orders_scoring(self, order_ids: list[str]) -> dict[str, bool]:
        """
        Check if order scoring is enabled for multiple order IDs.
        :param order_ids: List of order IDs to check.
        :return: Dictionary mapping order IDs to their scoring status.
        """
        self.logger.info(
            f"Checking if orders scoring is enabled for order IDs: {order_ids}"
        )
        try:
            return self.client.are_orders_scoring(
                OrdersScoringParams(orderIds=order_ids)
            )
        except Exception as e:
            self.logger.error(
                f"Error checking orders scoring for order IDs {order_ids}: {e}"
            )
            return {order_id: False for order_id in order_ids}

    def get_sampling_markets(self) -> list[Market]:
        """
        Get sampling markets.
        :return: List of sampling markets.
        """
        self.logger.info("Getting sampling markets")
        try:
            data = self.client.get_sampling_markets()
            next_cursor = data.get("next_cursor")
            raw_markets = data.get("data", [])
            while data and data.get("next_cursor") != "LTE=":
                data = self.client.get_sampling_markets(next_cursor=next_cursor)
                next_cursor = data.get("next_cursor")
                raw_markets.extend(data.get("data", []))
            markets = []
            for market in raw_markets:
                try:
                    markets.append(map_market(market))
                except ValueError as e:
                    self.logger.error(f"Error mapping market {market}: {e}")
            return markets
        except Exception as e:
            self.logger.error(f"Unhandled error getting sampling markets: {e}")
            if "markets" in locals():
                return markets
            else:
                return []

    def get_sampling_simplified_markets(self) -> list[SimplifiedMarket]:
        """
        Get simplified sampling markets (markets with rewards enabled).
        :return: List of simplified sampling markets.
        """
        self.logger.info("Getting simplified sampling markets")
        try:
            data = self.client.get_sampling_simplified_markets()
            next_cursor = data.get("next_cursor")
            raw_markets = data.get("data", [])
            while data and data.get("next_cursor") != "LTE=":
                data = self.client.get_sampling_simplified_markets(
                    next_cursor=next_cursor
                )
                next_cursor = data.get("next_cursor")
                raw_markets.extend(data.get("data", []))
            markets = []
            for market in raw_markets:
                try:
                    markets.append(map_simplified_market(market))
                except ValueError as e:
                    self.logger.error(f"Error mapping simplified market {market}: {e}")
            return markets
        except Exception as e:
            self.logger.error(
                f"Unhandled error getting simplified sampling markets: {e}"
            )
            if "markets" in locals():
                return markets
            else:
                return []

    def get_markets(self) -> list[Market]:
        """
        Get all markets.
        :return: List of all markets.
        """
        self.logger.info("Getting all markets")
        try:
            data = self.client.get_markets()
            next_cursor = data.get("next_cursor")
            raw_markets = data.get("data", [])
            while data and data.get("next_cursor") != "LTE=":
                data = self.client.get_markets(next_cursor=next_cursor)
                next_cursor = data.get("next_cursor")
                raw_markets.extend(data.get("data", []))
            markets = []
            for market in raw_markets:
                try:
                    markets.append(map_market(market))
                except ValueError as e:
                    self.logger.error(f"Error mapping market {market}: {e}")
            return markets
        except Exception as e:
            self.logger.error(f"Unhandled error getting markets: {e}")
            if "markets" in locals():
                return markets
            else:
                return []

    def get_simplified_markets(self) -> list[SimplifiedMarket]:
        """
        Get simplified markets (markets with rewards enabled).
        :return: List of simplified markets.
        """
        self.logger.info("Getting simplified markets")
        try:
            data = self.client.get_simplified_markets()
            next_cursor = data.get("next_cursor")
            raw_markets = data.get("data", [])
            while data and data.get("next_cursor") != "LTE=":
                data = self.client.get_simplified_markets(next_cursor=next_cursor)
                next_cursor = data.get("next_cursor")
                raw_markets.extend(data.get("data", []))
            markets = []
            for market in raw_markets:
                try:
                    markets.append(map_market(market))
                except ValueError as e:
                    self.logger.error(f"Error mapping simplified market {market}: {e}")
            return markets
        except Exception as e:
            self.logger.error(f"Unhandled error getting simplified markets: {e}")
            if "markets" in locals():
                return markets
            else:
                return []

    def get_market(self, market_id: str) -> Market | None:
        """
        Get a specific market by its ID.
        :param market_id: The ID of the market to retrieve.
        :return: The Market object corresponding to the given ID.
        """
        self.logger.info(f"Getting market with ID: {market_id}")
        try:
            raw_market = self.client.get_market(market_id)
            return map_market(raw_market)
        except Exception as e:
            self.logger.error(f"Error getting market {market_id}: {e}")
            return None

    def get_simplified_market(self, market_id: str) -> SimplifiedMarket | None:
        """
        Get a simplified market by its ID.
        :param market_id: The ID of the market to retrieve.
        :return: The SimplifiedMarket object corresponding to the given ID.
        """
        self.logger.info(f"Getting simplified market with ID: {market_id}")
        try:
            raw_market = self.client.get_market(market_id)
            return map_simplified_market(raw_market)
        except Exception as e:
            self.logger.error(f"Error getting simplified market {market_id}: {e}")
            return None

    def get_active_markets(self) -> list[Market]:
        """
        Get all active markets.
        :return: List of active markets.
        """
        self.logger.info("Getting all active markets")
        try:
            data = self.client.get_active_markets()
            next_cursor = data.get("next_cursor")
            raw_markets = data.get("data", [])
            while data and data.get("next_cursor") != "LTE=":
                data = self.client.get_active_markets(next_cursor=next_cursor)
                next_cursor = data.get("next_cursor")
                raw_markets.extend(data.get("data", []))
            markets = []
            for market in raw_markets:
                try:
                    markets.append(map_market(market))
                except ValueError as e:
                    self.logger.error(f"Error mapping active market {market}: {e}")
            return markets
        except Exception as e:
            self.logger.error(f"Unhandled error getting active markets: {e}")
            if "markets" in locals():
                return markets
            else:
                return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = Client()
