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

# Dynamic import helpers
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Discover strategies
strategy_mod = importlib.import_module("strategy")
BaseStrategy = getattr(importlib.import_module("strategy_base"), "BaseStrategy")

strategies = [
    cls
    for name, cls in inspect.getmembers(strategy_mod, inspect.isclass)
    if issubclass(cls, BaseStrategy) and cls is not BaseStrategy
]

# Discover screeners
market_screener_mod = importlib.import_module("market_screener")
PolymarketScreener = getattr(market_screener_mod, "PolymarketScreener")
screeners = [PolymarketScreener]  # Extendable for more screeners

# Import Client
client_mod = importlib.import_module("client")
Client = getattr(client_mod, "Client")

# Singleton Client instance
client_instance = Client()


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
            self.app.push_screen(OrderManagerMenu())
        elif idx == 3:
            self.app.push_screen(ClientActionsMenu())
        elif idx == 4:
            self.app.push_screen(SettingsMenu())
        elif idx == 5:
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
            self.app.push_screen(
                PlaceholderScreen(f"Selected screener: {screeners[idx].__name__}")
            )
        else:
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
