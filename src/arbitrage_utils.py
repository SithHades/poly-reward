from src.core.models import OrderbookSnapshot


def find_arbitrage_opportunities(
    yes_orderbook: OrderbookSnapshot,
    no_orderbook: OrderbookSnapshot,
    max_total_cost: float = 1.0,
    max_levels: int = 5,
) -> list[dict[str, float]]:
    """
    Find arbitrage opportunities between two orderbooks (YES/NO or UP/DOWN).
    Looks for pairs of (ask_yes, ask_no) such that ask_yes.price + ask_no.price < max_total_cost.
    Returns a list of opportunities with price, size, and profit per share.

    Args:
        yes_orderbook: OrderbookSnapshot for YES/UP outcome
        no_orderbook: OrderbookSnapshot for NO/DOWN outcome
        max_total_cost: Maximum total cost for 1 YES + 1 NO share (default: 1.0)
        max_levels: How many price levels to check on each side (default: 5)

    Returns:
        List of dicts with keys: 'yes_price', 'no_price', 'yes_size', 'no_size', 'total_cost', 'max_size', 'profit_per_share'
    """
    opportunities = []
    yes_asks = yes_orderbook.asks[:max_levels]
    no_asks = no_orderbook.asks[:max_levels]

    for yes_level in yes_asks:
        for no_level in no_asks:
            total_cost = yes_level.price + no_level.price
            if total_cost < max_total_cost:
                max_size = min(yes_level.size, no_level.size)
                profit_per_share = max_total_cost - total_cost
                opportunities.append(
                    {
                        "yes_price": yes_level.price,
                        "no_price": no_level.price,
                        "yes_size": yes_level.size,
                        "no_size": no_level.size,
                        "total_cost": total_cost,
                        "max_size": max_size,
                        "profit_per_share": profit_per_share,
                    }
                )
    # Sort by profit per share, descending
    opportunities.sort(key=lambda x: x["profit_per_share"], reverse=True)
    return opportunities
