"""
ETH Candle Direction Prediction Demo

This demo shows how to use the EthCandlePredictor to:
1. Load historical ETH price data
2. Train multiple models
3. Make predictions
4. Evaluate performance
5. Use ensemble predictions

Based on your existing ethusdt.py script structure.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timezone, timedelta
import warnings

warnings.filterwarnings("ignore")

import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from src.eth_candle_predictor import (  # noqa: E402
    EthCandlePredictor,
    create_ensemble_predictor,
    ensemble_predict,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_eth_data(days: int = 30) -> pd.DataFrame:
    """
    Fetch ETH/USDT 1-minute data for the specified number of days.
    Based on your existing ethusdt.py structure.
    """
    logger.info(f"Fetching {days} days of ETH/USDT data...")

    exchange = ccxt.binance()
    symbol = "ETH/USDT"
    timeframe = "1m"

    # Calculate start date
    since = exchange.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    )

    all_ohlcv = []
    limit = 1000

    while True:
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if len(ohlcv) == 0:
                break

            all_ohlcv.extend(ohlcv)

            # Update 'since' to the timestamp of the last candle + 1 millisecond
            since = ohlcv[-1][0] + 1

            # Rate limiting
            time.sleep(exchange.rateLimit / 1000)

            logger.info(f"Fetched {len(all_ohlcv)} candles so far...")

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break

    # Convert to DataFrame
    df = pd.DataFrame(
        all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    # Remove duplicates and sort
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    logger.info(
        f"Fetched {len(df)} 1-minute candles from {df.index[0]} to {df.index[-1]}"
    )

    return df


def create_resampled_data(df_1min: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create resampled dataframes for different timeframes."""
    logger.info("Creating resampled data...")

    # Resample to different timeframes
    df_5min = (
        df_1min.resample("5min")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    df_15min = (
        df_1min.resample("15min")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    df_1h = (
        df_1min.resample("1H")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    logger.info("Created resampled data:")
    logger.info(f"  5-min: {len(df_5min)} candles")
    logger.info(f"  15-min: {len(df_15min)} candles")
    logger.info(f"  1-hour: {len(df_1h)} candles")

    return {"1min": df_1min, "5min": df_5min, "15min": df_15min, "1h": df_1h}


def analyze_historical_performance(df_15min: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """Analyze how often candles flip direction between 45min and close"""
    results = []

    print("🔍 Analyzing candle flipping behavior...")

    for i, ts in enumerate(df_1h.index):
        if (len(df_1h) <= 1000 and i % 100 == 0) or (
            len(df_1h) > 1000 and i % 1000 == 0
        ):
            print(f"   Processing candle {i + 1}/{len(df_1h)}")

        open_price = df_1h.loc[ts, "open"]
        close_price = df_1h.loc[ts, "close"]

        # Find 45-min mark price - this should be the CLOSE of the 3rd 15-min candle
        # 15-min candles in an hour: [0-15], [15-30], [30-45], [45-60]
        # We want the close price of the [30-45] candle, which ends at the 45-min mark
        ts_30 = ts + pd.Timedelta(minutes=30)  # Start of 3rd 15-min candle
        ts_45 = ts + pd.Timedelta(minutes=45)  # End of 3rd 15-min candle (45-min mark)

        # Find the 15-min candle that starts at the 30-minute mark
        # This candle closes at the 45-minute mark, giving us the price at 45 minutes
        matching_candles = df_15min.index[
            (df_15min.index >= ts_30) & (df_15min.index < ts_45)
        ]

        if len(matching_candles) > 0:
            # Use the close price of this candle (which represents the price at 45-min mark)
            price_45 = df_15min.loc[matching_candles[0], "close"]
        else:
            # Fallback: find all 15-min candles in the first 45 minutes and take the last one
            first_45_candles = df_15min.index[
                (df_15min.index >= ts) & (df_15min.index < ts_45)
            ]
            if len(first_45_candles) >= 3:
                # Take the close of the 3rd candle (index 2) which should end at 45 minutes
                price_45 = df_15min.loc[first_45_candles[2], "close"]
            else:
                # Skip this hour if we don't have enough data
                continue

        # Calculate changes
        delta_45 = price_45 - open_price
        delta_close = close_price - open_price

        direction_45 = "up" if delta_45 > 0 else "down" if delta_45 < 0 else "flat"
        direction_close = (
            "up" if delta_close > 0 else "down" if delta_close < 0 else "flat"
        )
        flipped = direction_45 != direction_close

        results.append(
            {
                "timestamp": ts,
                "open": open_price,
                "price_at_45min": price_45,
                "close": close_price,
                "delta_45_pct": (delta_45 / open_price) * 100,
                "delta_close_pct": (delta_close / open_price) * 100,
                "direction_45": direction_45,
                "direction_close": direction_close,
                "flipped": flipped,
                "green_candle": close_price > open_price,
                "hour_of_day": ts.hour,
                "day_of_week": ts.day_name(),
            }
        )

    flip_df = pd.DataFrame(results)  
     
    total_candles = len(flip_df)
    green_candles = flip_df["green_candle"].sum()
    red_candles = total_candles - green_candles
    flip_rate = flip_df["flipped"].mean()

    logger.info("Historical Analysis Results:")
    logger.info(f"  Total candles analyzed: {total_candles}")
    logger.info(
        f"  Green candles: {green_candles} ({green_candles / total_candles:.1%})"
    )
    logger.info(f"  Red candles: {red_candles} ({red_candles / total_candles:.1%})")
    logger.info(f"  Overall flip rate: {flip_rate:.1%}")

    # Analyze flip rates by different thresholds
    for threshold in [0.1, 0.25, 0.5, 1.0]:
        subset_up = flip_df[flip_df["delta_45_pct"] > threshold]
        subset_down = flip_df[flip_df["delta_45_pct"] < -threshold]

        if len(subset_up) > 0:
            flip_rate_up = subset_up["flipped"].mean()
            logger.info(
                f"  Flip rate when up >{threshold}% at 45min: {flip_rate_up:.1%} ({len(subset_up)} samples)"
            )

        if len(subset_down) > 0:
            flip_rate_down = subset_down["flipped"].mean()
            logger.info(
                f"  Flip rate when down <-{threshold}% at 45min: {flip_rate_down:.1%} ({len(subset_down)} samples)"
            )
    
    return flip_df


def train_single_model(
    df_1min: pd.DataFrame, model_type: str = "random_forest"
) -> tuple:
    """Train a single model and return it with performance metrics."""
    logger.info(f"Training {model_type} model...")

    predictor = EthCandlePredictor(model_type=model_type)
    performance = predictor.train(df_1min, test_size=0.2)

    return predictor, performance


def train_ensemble_models(df_1min: pd.DataFrame) -> dict[str, EthCandlePredictor]:
    """Train ensemble of different models."""
    logger.info("Training ensemble of models...")

    models = create_ensemble_predictor(df_1min)

    return models


def make_predictions_demo(models: dict[str, EthCandlePredictor], df_1min: pd.DataFrame):
    """Demonstrate making predictions on recent data."""
    logger.info("Making predictions on recent data...")

    # Get the most recent complete hours for prediction
    df_1h = (
        df_1min.resample("1H")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    # Get last 10 complete hours
    recent_hours = df_1h.index[-10:]

    predictions = []

    for hour_start in recent_hours:
        try:
            # Make individual model predictions
            individual_preds = {}
            for model_name, model in models.items():
                pred = model.predict(df_1min, hour_start)
                individual_preds[model_name] = pred

            # Make ensemble prediction
            ensemble_pred = ensemble_predict(models, df_1min, hour_start)

            # Get actual result
            actual_close = df_1h.loc[hour_start, "close"]
            actual_open = df_1h.loc[hour_start, "open"]
            actual_direction = "green" if actual_close > actual_open else "red"

            prediction_result = {
                "timestamp": hour_start,
                "actual_direction": actual_direction,
                "actual_change_pct": (actual_close - actual_open) / actual_open * 100,
                "ensemble_prediction": ensemble_pred.predicted_direction,
                "ensemble_confidence": ensemble_pred.confidence,
                "ensemble_prob_green": ensemble_pred.probability_green,
                "individual_predictions": individual_preds,
            }

            predictions.append(prediction_result)

        except Exception as e:
            logger.warning(f"Failed to predict for {hour_start}: {e}")
            continue

    # Print results
    logger.info("\nRecent Predictions vs Actual Results:")
    logger.info("=" * 80)

    correct_predictions = 0
    total_predictions = len(predictions)

    for pred in predictions:
        ensemble_correct = (
            "✓" if pred["ensemble_prediction"] == pred["actual_direction"] else "✗"
        )
        if pred["ensemble_prediction"] == pred["actual_direction"]:
            correct_predictions += 1

        logger.info(
            f"{pred['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
            f"Actual: {pred['actual_direction']:>5} ({pred['actual_change_pct']:+.2f}%) | "
            f"Predicted: {pred['ensemble_prediction']:>5} "
            f"({pred['ensemble_confidence']:.1%}) {ensemble_correct}"
        )

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    logger.info(
        f"\nEnsemble Accuracy on Recent Data: {accuracy:.1%} ({correct_predictions}/{total_predictions})"
    )

    return predictions


def plot_prediction_analysis(
    df_1min: pd.DataFrame, flip_df: pd.DataFrame, predictions: list[dict] = None
):
    """Create comprehensive plots for prediction analysis."""

    # 1. Historical flip rate analysis
    fig1 = px.histogram(
        flip_df,
        x="delta_45_pct",
        color="flipped",
        nbins=40,
        title="Historical Flip Probabilities by 45-Minute Change (%)",
        labels={"delta_45_pct": "45-Minute Change (%)", "count": "Number of Candles"},
    )
    fig1.show()

    # 2. Green vs Red candle distribution by 45-min change
    fig2 = px.histogram(
        flip_df,
        x="delta_45_pct",
        color="green_candle",
        nbins=40,
        title="Green vs Red Candles by 45-Minute Change (%)",
        labels={"delta_45_pct": "45-Minute Change (%)", "count": "Number of Candles"},
    )
    fig2.show()

    # 3. Recent price action with predictions (if available)
    if predictions:
        # Get recent hourly data
        df_1h = (
            df_1min.resample("1H")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        # Create subplot
        fig3 = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("ETH Price (Last 48 Hours)", "Prediction Confidence"),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
        )

        # Plot recent price data
        recent_data = df_1h.tail(48)  # Last 48 hours
        fig3.add_trace(
            go.Candlestick(
                x=recent_data.index,
                open=recent_data["open"],
                high=recent_data["high"],
                low=recent_data["low"],
                close=recent_data["close"],
                name="ETH Price",
            ),
            row=1,
            col=1,
        )

        # Add prediction markers
        pred_timestamps = [pred["timestamp"] for pred in predictions]
        pred_confidences = [pred["ensemble_confidence"] for pred in predictions]
        pred_colors = [
            "green" if pred["ensemble_prediction"] == "green" else "red"
            for pred in predictions
        ]
        pred_correct = [
            "✓" if pred["ensemble_prediction"] == pred["actual_direction"] else "✗"
            for pred in predictions
        ]

        fig3.add_trace(
            go.Scatter(
                x=pred_timestamps,
                y=pred_confidences,
                mode="markers+text",
                marker=dict(size=10, color=pred_colors),
                text=pred_correct,
                textposition="top center",
                name="Predictions",
                hovertemplate="Time: %{x}<br>Confidence: %{y:.1%}<br>Correct: %{text}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig3.update_layout(title="Recent ETH Price Action with Predictions", height=800)
        fig3.show()


def main():
    """Main demo function."""
    logger.info("Starting ETH Candle Direction Prediction Demo")

    # 1. Fetch data
    df_1min = fetch_eth_data(days=50)  # 2 weeks of data

    # 2. Create resampled data
    resampled_data = create_resampled_data(df_1min)

    df_15min = resampled_data["15min"]
    df_1h = resampled_data["1h"]

    # 3. Analyze historical behavior
    flip_df = analyze_historical_performance(df_15min, df_1h)

    return

    # 4. Train models
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING MODELS")
    logger.info("=" * 50)

    # Train individual models
    rf_model, rf_performance = train_single_model(df_1min, "random_forest")
    gb_model, gb_performance = train_single_model(df_1min, "gradient_boost")
    lr_model, lr_performance = train_single_model(df_1min, "logistic")

    # Create ensemble
    models = {
        "random_forest": rf_model,
        "gradient_boost": gb_model,
        "logistic": lr_model,
    }

    # 5. Make predictions on recent data
    logger.info("\n" + "=" * 50)
    logger.info("MAKING PREDICTIONS")
    logger.info("=" * 50)

    predictions = make_predictions_demo(models, df_1min)

    # 6. Create plots
    logger.info("\n" + "=" * 50)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 50)

    plot_prediction_analysis(df_1min, flip_df, predictions)

    # 7. Save models for future use
    logger.info("\n" + "=" * 50)
    logger.info("SAVING MODELS")
    logger.info("=" * 50)

    for model_name, model in models.items():
        filepath = f"models/eth_candle_predictor_{model_name}.joblib"
        os.makedirs("models", exist_ok=True)
        model.save_model(filepath)
        logger.info(f"Saved {model_name} model to {filepath}")

    logger.info("\nDemo completed successfully!")

    return {
        "data": resampled_data,
        "models": models,
        "historical_analysis": flip_df,
        "predictions": predictions,
    }


if __name__ == "__main__":
    # Run the demo
    results = main()
