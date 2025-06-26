import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features to OHLCV DataFrame:
    - log returns
    - moving averages (5, 15, 60 periods)
    - rolling volatility (15, 60 periods)
    - up/down target (1 if close > open, else 0)
    """
    df = df.copy()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_15'] = df['close'].rolling(window=15).mean()
    df['ma_60'] = df['close'].rolling(window=60).mean()
    df['vol_15'] = df['log_return'].rolling(window=15).std()
    df['vol_60'] = df['log_return'].rolling(window=60).std()
    df['target_up'] = (df['close'] > df['open']).astype(int)
    df = df.dropna()
    return df 