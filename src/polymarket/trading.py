from src.models import OrderArgsModel, BookSide

def calculate_edge(model_prob: float, market_price: float) -> float:
    """
    Calculate the edge (expected value) of buying YES at market_price given model_prob.
    Returns edge as a percentage.
    """
    return (model_prob - market_price) * 100

def generate_orders(
    model_prob: float,
    order_book: dict,
    token_id: str,
    threshold: float = 2.0,
    min_size: float = 1.0
) -> list[OrderArgsModel]:
    """
    Given model probability and order book, generate OrderArgsModel instances if edge > threshold (in %).
    Returns a list of OrderArgsModel objects.
    """
    orders = []
    # YES side (buy if model_prob > market_price + threshold)
    for price, size in order_book['yes']:
        edge = (model_prob - price) * 100
        if edge > threshold and size >= min_size:
            orders.append(OrderArgsModel(
                token_id=token_id,
                price=price,
                size=min(size, 100),  # Cap order size if desired
                side=BookSide.BUY
            ))
    # NO side (buy if (1-model_prob) > market_price + threshold)
    for price, size in order_book['no']:
        edge = ((1 - model_prob) - price) * 100
        if edge > threshold and size >= min_size:
            orders.append(OrderArgsModel(
                token_id=token_id,
                price=price,
                size=min(size, 100),
                side=BookSide.BUY
            ))
    return orders 