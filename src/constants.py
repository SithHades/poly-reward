from typing import Literal


DEFAULT_MIDPOINT = 0.5
DEFAULT_SPREAD = 0.0
DEFAULT_TICK_SIZE = 0.001

MARKETS = Literal["ethereum", "bitcoin", "solana", "xrp"]

TICKERS = {
    "ethereum": "ETH/USDT",
    "bitcoin": "BTC/USDT",
    "solana": "SOL/USDT",
    "xrp": "XRP/USDT",
}
