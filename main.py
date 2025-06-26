import time
import pandas as pd
from src.data.feature_engineering import add_features
from src.models.classifier import OnlineClassifier
from src.polymarket_client import PolymarketClient
from src.strategy.edge_strategy import run_strategy
from src.ui.terminal_ui import display_edge_dashboard

# --- Config ---
DATA_PATH = "data/ethusdt_ohlcv.csv"  # Path to your historical data
FEATURE_COLS = [
    'log_return', 'ma_5', 'ma_15', 'ma_60', 'vol_15', 'vol_60'
]
TARGET_COL = 'target_up'
EDGE_THRESHOLD = 2.0  # Minimum edge (%) to trigger an opportunity
REFRESH_INTERVAL = 30  # seconds

# --- Main Pipeline ---
def main():
    # 1. Load historical data (replace with live fetch if needed)
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'], index_col='timestamp')
    except Exception as e:
        print(f"Failed to load data from {DATA_PATH}: {e}")
        print("Using random data for demo.")
        import numpy as np
        idx = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='H')
        df = pd.DataFrame({
            'open': np.random.rand(200) * 2000 + 1000,
            'close': np.random.rand(200) * 2000 + 1000,
            'high': np.random.rand(200) * 2000 + 1000,
            'low': np.random.rand(200) * 2000 + 1000,
            'volume': np.random.rand(200) * 1000
        }, index=idx)
    # 2. Feature engineering
    feature_df = add_features(df)
    # 3. Train or load model
    model = OnlineClassifier()
    X, y = model.prepare_xy(feature_df, FEATURE_COLS, TARGET_COL)
    model.fit(X, y)
    # 4. Instantiate Polymarket client
    client = PolymarketClient()
    # 5. Main loop
    try:
        while True:
            # (Optional) Update model with new data here for online learning
            # 6. Run strategy
            strategy_result = run_strategy(
                model, client, feature_df, FEATURE_COLS, edge_threshold=EDGE_THRESHOLD
            )
            # 7. Display in terminal UI
            display_edge_dashboard(strategy_result)
            # 8. (Optional) Place orders using client.place_order() if desired
            # for order in strategy_result['orders']:
            #     ...
            time.sleep(REFRESH_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error in main loop: {e}")

if __name__ == "__main__":
    main()
