from abc import ABC, abstractmethod
from typing import Any, Dict, List
from src.models import OrderDetails, OrderbookSnapshot, Position


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """

    @abstractmethod
    def analyze_market_condition(
        self,
        yes_orderbook: OrderbookSnapshot,
        no_orderbook: OrderbookSnapshot,
        **kwargs: Any,
    ) -> Any:
        """
        Analyze market conditions and return a signal or recommendation.
        """
        pass

    @abstractmethod
    def calculate_orders(
        self,
        yes_orderbook: OrderbookSnapshot,
        no_orderbook: OrderbookSnapshot,
        current_positions: Dict[str, Position],
        available_capital: float,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Given market data, positions, and available capital, return a list of order instructions.
        """
        pass

    @abstractmethod
    def on_fill(
        self,
        fill_event: OrderDetails,
        yes_orderbook: OrderbookSnapshot,
        no_orderbook: OrderbookSnapshot,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        React to an order fill event (e.g., for hedging or rebalancing).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the strategy.
        """
        pass
