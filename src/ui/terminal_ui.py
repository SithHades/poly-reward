from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
import time

console = Console()

def display_edge_dashboard(strategy_result):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=2),
        Layout(name="orders", ratio=1)
    )
    # Header
    market = strategy_result["market"]
    model_prob = strategy_result["model_prob"]
    layout["header"].update(Panel(f"[bold]Market:[/bold] {market['name']} | [bold]Model P(UP):[/bold] {model_prob:.2%}", title="ETH Hourly Market"))
    # Order Book Table
    order_book = strategy_result["order_book"]
    ob_table = Table(title="Order Book", show_header=True, header_style="bold magenta")
    ob_table.add_column("Side")
    ob_table.add_column("Price", justify="right")
    ob_table.add_column("Size", justify="right")
    for side in ["yes", "no"]:
        for price, size in order_book[side]:
            ob_table.add_row(side.upper(), f"{price:.2f}", str(size))
    layout["main"].update(ob_table)
    # Orders Table
    orders = strategy_result["orders"]
    orders_table = Table(title="Opportunities", show_header=True, header_style="bold green")
    orders_table.add_column("Action")
    orders_table.add_column("Price", justify="right")
    orders_table.add_column("Size", justify="right")
    orders_table.add_column("Edge (%)", justify="right")
    for order in orders:
        orders_table.add_row(order['side'], f"{order['price']:.2f}", str(order['size']), f"{order['edge']:.2f}")
    layout["orders"].update(orders_table)
    console.print(layout)

if __name__ == "__main__":
    # Demo with mock data
    mock_result = {
        "market": {"name": "ETH Hourly 2025-06-10 13:00"},
        "model_prob": 0.62,
        "order_book": {
            "yes": [(0.51, 100), (0.52, 50)],
            "no": [(0.49, 100), (0.48, 50)]
        },
        "orders": [
            {"side": "buy_yes", "price": 0.51, "size": 100, "edge": 11.0},
            {"side": "buy_no", "price": 0.48, "size": 50, "edge": 10.0}
        ]
    }
    with Live(refresh_per_second=1) as live:
        for _ in range(5):
            live.update(Panel("[bold cyan]Polymarket Edge Dashboard Demo[/bold cyan]"))
            display_edge_dashboard(mock_result)
            time.sleep(1) 