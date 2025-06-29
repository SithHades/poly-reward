from pathlib import Path
import pandas as pd


def load_eth_historical_data() -> pd.DataFrame:
    """
    Loads the historical ohlcv data from the data folder.
    File names are: f"{ticker}_{timeframe}-{year}-{month}.csv"
    """
    ticker = "ETHUSDT"
    base_path = Path("data")
    dfs = []

    for file in base_path.glob(f"{ticker.upper()}*.csv"):
        df = pd.read_csv(
            file,
            usecols=[0, 1, 2, 3, 4, 5],
            names=["timestamp", "open", "high", "low", "close", "volume"],
        )
        dfs.append(df)

    df = pd.concat(dfs)
    df["timestamp"] = df["timestamp"].divide(1000)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df.sort_index()

    return df
