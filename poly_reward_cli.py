import asyncio
from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    Static,
    ListView,
    ListItem,
    Label,
    Input,
    RichLog,
)

from textual.screen import Screen

import importlib
import sys
import os
import inspect
from typing import Type

from src.eth_prediction_strategy import EthPredictionMarketMakingStrategy, EthPredictionStrategyConfig

# Dynamic import helpers
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Discover strategies
strategy_mod = importlib.import_module("strategy")
eth_prediction_strategy_mod = importlib.import_module("eth_prediction_strategy")
BaseStrategy = getattr(importlib.import_module("strategy_base"), "BaseStrategy")

strategies = [
    cls
    for name, cls in inspect.getmembers(strategy_mod, inspect.isclass)
    if issubclass(cls, BaseStrategy) and cls is not BaseStrategy
] + [
    cls
    for name, cls in inspect.getmembers(eth_prediction_strategy_mod, inspect.isclass)
    if issubclass(cls, BaseStrategy) and cls is not BaseStrategy
]

# Discover screeners
market_screener_mod = importlib.import_module("market_screener")
MarketScreener = getattr(market_screener_mod, "MarketScreener")
screeners = [MarketScreener]  # Extendable for more screeners

# Import Client
client_mod = importlib.import_module("polymarket_client")
Client = getattr(client_mod, "PolymarketClient")

# Singleton Client instance
client_instance = Client()


def get_strategy_config(strategy_cls: Type) -> any:
    """
    Returns the configuration object for a given strategy class.
    This can be expanded to load configs from a file or a UI.
    """
    if strategy_cls.__name__ == "EthPredictionMarketMakingStrategy":
        return EthPredictionStrategyConfig()
    # For other strategies, we can return a default config or None
    # if they don't require a specific configuration.
    return None


class MainMenu(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            """
[bold]Poly Reward Terminal UI[/bold]
Select an option:
""",
            id="title",
        )
        yield ListView(
            ListItem(Label("Run a Strategy")),
            ListItem(Label("Run a Market Screener")),
            ListItem(Label("Run ETH Prediction Strategy")),
            ListItem(Label("Get ETH Price Prediction")),
            ListItem(Label("Manage Order Managers")),
            ListItem(Label("Client Actions")),
            ListItem(Label("Settings/Config")),
            ListItem(Label("Exit")),
            id="main-menu-list",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.control.index
        if idx == 0:
            self.app.push_screen(StrategyMenu())
        elif idx == 1:
            self.app.push_screen(ScreenerMenu())
        elif idx == 2:
            self.app.push_screen(RunEthPredictionStrategyScreen())
        elif idx == 3:
            self.app.push_screen(GetEthPredictionScreen())
        elif idx == 4:
            self.app.push_screen(OrderManagerMenu())
        elif idx == 5:
            self.app.push_screen(ClientActionsMenu())
        elif idx == 6:
            self.app.push_screen(SettingsMenu())
        elif idx == 7:
            self.app.exit()


class StrategyMenu(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[bold]Available Strategies[/bold]", id="title")
        yield ListView(
            *[ListItem(Label(cls.__name__)) for cls in strategies],
            ListItem(Label("Back")),
            id="strategy-list",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.control.index
        if idx < len(strategies):
            self.app.push_screen(
                PlaceholderScreen(f"Selected strategy: {strategies[idx].__name__}")
            )
        else:
            self.app.pop_screen()


class ScreenerMenu(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[bold]Available Market Screeners[/bold]", id="title")
        yield ListView(
            *[ListItem(Label(cls.__name__)) for cls in screeners],
            ListItem(Label("Back")),
            id="screener-list",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.control.index
        if idx < len(screeners):
            # Assuming only one screener for now, directly go to strategy selection
            self.app.push_screen(
                ScreenerStrategySelectionScreen(screener_cls=screeners[idx])
            )
        else:
            self.app.pop_screen()


class ScreenerStrategySelectionScreen(Screen):
    def __init__(self, screener_cls: Type):
        super().__init__()
        self.screener_cls = screener_cls

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            f"[bold]Select Strategy for {self.screener_cls.__name__}[/bold]", id="title"
        )
        yield ListView(
            *[ListItem(Label(cls.__name__)) for cls in strategies],
            ListItem(Label("Back")),
            id="strategy-selection-list",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.control.index
        if idx < len(strategies):
            selected_strategy_cls = strategies[idx]
            self.app.push_screen(
                RunScreenerScreen(
                    screener_cls=self.screener_cls, strategy_cls=selected_strategy_cls
                )
            )
        else:
            self.app.pop_screen()


class RunScreenerScreen(Screen):
    def __init__(self, screener_cls: Type, strategy_cls: Type):
        super().__init__()
        self.screener_cls = screener_cls
        self.strategy_cls = strategy_cls

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            f"[bold]Running {self.screener_cls.__name__} with {self.strategy_cls.__name__}[/bold]",
            id="title",
        )
        self.textlog = RichLog(
            highlight=True, markup=True, wrap=True, id="screener-log"
        )
        yield self.textlog
        yield ListView(ListItem(Label("Back")), id="back-list")
        yield Footer()

    async def on_mount(self) -> None:
        self.textlog.write("Initializing screener and strategy...")
        try:
            # Get the appropriate config for the selected strategy
            strategy_config = get_strategy_config(self.strategy_cls)

            # Instantiate the strategy with its config
            # Note: If a strategy has no specific config, it should handle None gracefully.
            strategy_instance = self.strategy_cls(config=strategy_config)
            screener_instance = self.screener_cls(
                client=client_instance, strategy=strategy_instance
            )

            self.textlog.write("Finding attractive markets...")
            attractive_markets = await screener_instance.find_opportunities()

            if not attractive_markets:
                self.textlog.write("[red]No attractive markets found.[/red]")
            else:
                self.textlog.write("[green]Attractive Markets:[/green]")
                for market in attractive_markets:
                    self.textlog.write(f"- {market.market_slug} ({market.question})")
        except Exception as e:
            self.textlog.write(f"[red]Error: {e}[/red]")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.app.pop_screen()


class RunEthPredictionStrategyScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[bold]Running ETH Prediction Strategy[/bold]", id="title")
        self.textlog = RichLog(highlight=True, markup=True, wrap=True, id="strategy-log")
        yield self.textlog
        yield ListView(ListItem(Label("Back")), id="back-list")
        yield Footer()

    async def on_mount(self) -> None:
        self.textlog.write("Initializing ETH Prediction Strategy...")
        try:
            strategy_config = EthPredictionStrategyConfig()
            strategy_instance: EthPredictionMarketMakingStrategy = eth_prediction_strategy_mod.EthPredictionMarketMakingStrategy(config=strategy_config)

            # For demonstration, we'll use a dummy market and orderbooks
            # In a real scenario, you'd fetch these from PolymarketClient
            from src.models import Market, OrderbookSnapshot, OrderbookLevel, TokenInfo

            # Create a dummy market that matches the criteria in analyze_market_condition
            dummy_market = Market(
                enable_order_book=True,
                active=True,
                closed=False,
                archived=False,
                accepting_orders=True,
                minimum_order_size=10,
                minimum_tick_size=0.001,
                condition_id="dummy-eth-market",
                question_id="dummy-eth-question",
                question="Will ETH price be up at 10am ET June 27 2025?",
                description="Dummy ETH hourly prediction market",
                market_slug="eth-price-at-10am-et-june-27-2025",
                end_date_iso="2025-06-27T10:00:00Z",
                maker_base_fee=0,
                taker_base_fee=0,
                notifications_enabled=False,
                neg_risk=False,
                neg_risk_market_id="",
                neg_risk_request_id="",
                rewards=None,
                is_50_50_outcome=True,
                tokens=[
                    TokenInfo(token_id="yes-token-id", outcome="Yes", price=0.5),
                    TokenInfo(token_id="no-token-id", outcome="No", price=0.5)
                ],
                tags=[]
            )

            # Create dummy orderbooks
            yes_orderbook = OrderbookSnapshot(
                asset_id="yes-token-id",
                bids=[OrderbookLevel(price=0.49, size=100)],
                asks=[OrderbookLevel(price=0.51, size=100)],
                midpoint=0.5,
                spread=0.02
            )
            no_orderbook = OrderbookSnapshot(
                asset_id="no-token-id",
                bids=[OrderbookLevel(price=0.49, size=100)],
                asks=[OrderbookLevel(price=0.51, size=100)],
                midpoint=0.5,
                spread=0.02
            )

            self.textlog.write("Calculating orders...")
            # Pass an empty dictionary for current_positions and 1000 for available_capital
            orders = await strategy_instance.calculate_orders(
                yes_orderbook, no_orderbook, {}, 1000, market=dummy_market
            )

            if not orders:
                self.textlog.write("[red]No orders calculated.[/red]")
            else:
                self.textlog.write("[green]Calculated Orders:[/green]")
                for order in orders:
                    self.textlog.write(f"- {order.side.value} {order.size} of {order.token_id} at {order.price}")
        except Exception as e:
            self.textlog.write(f"[red]Error: {e}[/red]")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.app.pop_screen()


class GetEthPredictionScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[bold]ETH Price Prediction[/bold]", id="title")
        self.textlog = RichLog(highlight=True, markup=True, wrap=True, id="prediction-log")
        yield self.textlog
        yield ListView(ListItem(Label("Back")), id="back-list")
        yield Footer()

    async def on_mount(self) -> None:
        self.textlog.write("Fetching ETH price prediction...")
        try:
            strategy_config = EthPredictionStrategyConfig()
            strategy_instance: EthPredictionMarketMakingStrategy = eth_prediction_strategy_mod.EthPredictionMarketMakingStrategy(config=strategy_config)
            prediction = await strategy_instance.get_eth_price_prediction()
            self.textlog.write(f"[green]Prediction: {prediction['direction'].upper()} with confidence {prediction['confidence']:.2f}[/green]")
        except Exception as e:
            self.textlog.write(f"[red]Error: {e}[/red]")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.app.pop_screen()


class OrderManagerMenu(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[bold]Order Manager (Placeholder)[/bold]", id="title")
        yield ListView(ListItem(Label("Back")), id="order-manager-list")
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.app.pop_screen()


class ClientActionsMenu(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[bold]Client Actions[/bold]", id="title")
        yield ListView(
            ListItem(Label("List all open orders")),
            ListItem(Label("Cancel an order by ID")),
            ListItem(Label("Get order book for a token")),
            ListItem(Label("List all markets")),
            ListItem(Label("Get market details by market id")),
            ListItem(Label("Back")),
            id="client-actions-list",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.control.index
        if idx == 0:
            self.app.push_screen(ClientListOpenOrdersScreen())
        elif idx == 1:
            self.app.push_screen(ClientCancelOrderScreen())
        elif idx == 2:
            self.app.push_screen(ClientOrderBookScreen())
        elif idx == 3:
            self.app.push_screen(ClientListMarketsScreen())
        elif idx == 4:
            self.app.push_screen(ClientMarketDetailsScreen())
        elif idx == 5:
            self.app.pop_screen()


class ClientListOpenOrdersScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[bold]All Open Orders[/bold]", id="title")
        self.textlog = RichLog(highlight=True, markup=True, wrap=True, id="orders-log")
        yield self.textlog
        yield ListView(ListItem(Label("Back")), id="back-list")
        yield Footer()

    async def on_mount(self) -> None:
        self.textlog.write("Loading open orders...")
        try:
            orders = await asyncio.to_thread(client_instance.get_orders)
            if not orders:
                self.textlog.write("[red]No open orders found.[/red]")
            else:
                for order in orders:
                    self.textlog.write(str(order))
        except Exception as e:
            self.textlog.write(f"[red]Error: {e}[/red]")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.app.pop_screen()


class ClientCancelOrderScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[bold]Cancel Order by ID[/bold]", id="title")
        self.input = Input(placeholder="Enter order ID", id="order-id-input")
        yield self.input
        self.textlog = RichLog(highlight=True, markup=True, wrap=True, id="cancel-log")
        yield self.textlog
        yield ListView(ListItem(Label("Back")), id="back-list")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        order_id = event.value.strip()
        if not order_id:
            self.textlog.write("[red]Order ID required.[/red]")
            return
        self.textlog.write(f"Cancelling order {order_id}...")

        async def do_cancel():
            try:
                result = await asyncio.to_thread(client_instance.cancel_order, order_id)
                if result:
                    self.textlog.write(f"[green]Order {order_id} cancelled.[/green]")
                else:
                    self.textlog.write(f"[red]Failed to cancel order {order_id}.[/red]")
            except Exception as e:
                self.textlog.write(f"[red]Error: {e}[/red]")

        asyncio.create_task(do_cancel())

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.app.pop_screen()


class ClientOrderBookScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[bold]Get Order Book for Token[/bold]", id="title")
        self.input = Input(placeholder="Enter token ID", id="token-id-input")
        yield self.input
        self.textlog = RichLog(
            highlight=True, markup=True, wrap=True, id="orderbook-log"
        )
        yield self.textlog
        yield ListView(ListItem(Label("Back")), id="back-list")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        token_id = event.value.strip()
        if not token_id:
            self.textlog.write("[red]Token ID required.[/red]")
            return
        self.textlog.write(f"Getting order book for token {token_id}...")

        async def do_get():
            try:
                # Token is just a string here; adapt if Token is a class
                orderbook = await asyncio.to_thread(
                    client_instance.get_order_book, token_id
                )
                self.textlog.write(str(orderbook))
            except Exception as e:
                self.textlog.write(f"[red]Error: {e}[/red]")

        asyncio.create_task(do_get())

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.app.pop_screen()


class ClientListMarketsScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[bold]All Markets[/bold]", id="title")
        self.textlog = RichLog(highlight=True, markup=True, wrap=True, id="markets-log")
        yield self.textlog
        yield ListView(ListItem(Label("Back")), id="back-list")
        yield Footer()

    async def on_mount(self) -> None:
        self.textlog.write("Loading all markets...")
        try:
            markets = await asyncio.to_thread(client_instance.get_markets)
            if not markets:
                self.textlog.write("[red]No markets found.[/red]")
            else:
                for market in markets:
                    self.textlog.write(str(market))
        except Exception as e:
            self.textlog.write(f"[red]Error: {e}[/red]")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.app.pop_screen()


class ClientMarketDetailsScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[bold]Get Market Details by ID[/bold]", id="title")
        self.input = Input(placeholder="Enter market ID", id="market-id-input")
        yield self.input
        self.textlog = RichLog(
            highlight=True, markup=True, wrap=True, id="marketdetails-log"
        )
        yield self.textlog
        yield ListView(ListItem(Label("Back")), id="back-list")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        market_id = event.value.strip()
        if not market_id:
            self.textlog.write("[red]Market ID required.[/red]")
            return
        self.textlog.write(f"Getting market details for {market_id}...")

        async def do_get():
            try:
                market = await asyncio.to_thread(client_instance.get_market, market_id)
                self.textlog.write(str(market))
            except Exception as e:
                self.textlog.write(f"[red]Error: {e}[/red]")

        asyncio.create_task(do_get())

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.app.pop_screen()


class SettingsMenu(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[bold]Settings/Config (Placeholder)[/bold]", id="title")
        yield ListView(ListItem(Label("Back")), id="settings-list")
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.app.pop_screen()


class PlaceholderScreen(Screen):
    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"[bold]{self.message}[/bold]", id="placeholder")
        yield ListView(ListItem(Label("Back")), id="placeholder-list")
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.app.pop_screen()


class PolyRewardApp(App):
    CSS_PATH = None
    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    async def on_mount(self) -> None:
        await self.push_screen(MainMenu())

def main():
    PolyRewardApp().run()


if __name__ == "__main__":
    main()