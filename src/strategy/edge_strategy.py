from constants import INTERESTING_SERIES_SLUGS
from src.polymarket_client import PolymarketClient

def run_strategy(model, client: PolymarketClient, feature_df, feature_cols, market_name="ethusdt", edge_threshold=2.0):
    """
    Run the edge-based trading strategy using the real PolymarketClient:
    - Get the newest ETH hourly market (using get_newest_future_series_markets)
    - Get the order book (using get_order_book with the market's token)
    - Get the model probability for up (YES)
    - Generate orders if edge > threshold
    Returns: dict with market info, model_prob, order_book, orders
    """
    # 1. Get the newest ETH hourly market
    markets = client.get_newest_future_series_markets(INTERESTING_SERIES_SLUGS.get(market_name))
    market = markets[0]
    token = market["token"] if "token" in market else market["id"]  # fallback to id if token missing
    # 2. Get order book
    order_book_summary = client.get_order_book(token)
    # Convert order book summary to the expected format: {'yes': [(price, size)], 'no': [(price, size)]}
    yes_orders = [(float(order["price"]), float(order["size"])) for order in order_book_summary["buy"]]
    no_orders = [(float(order["price"]), float(order["size"])) for order in order_book_summary["sell"]]
    order_book = {"yes": yes_orders, "no": no_orders}
    # 3. Get model probability (use last row of feature_df)
    X_latest = feature_df[feature_cols].iloc[[-1]].values
    model_prob = model.predict_proba(X_latest)[0, 1]
    # 4. Generate orders
    from src.polymarket.trading import generate_orders
    orders = generate_orders(model_prob, order_book, threshold=edge_threshold)
    return {
        "market": market,
        "model_prob": model_prob,
        "order_book": order_book,
        "orders": orders
    } 