"""
ETH Candle Direction Predictor

This module provides statistical models to predict whether a 1-hour ETH candle
will finish green (close > open) or red (close < open) based on the first 45 minutes of data.

The model uses various technical indicators and statistical features extracted from
1-minute, 5-minute, and 15-minute data within the first 45 minutes of each hour.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timezone, timedelta
import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib


@dataclass
class PredictionResult:
    """Result of a candle direction prediction"""

    probability_green: float
    probability_red: float
    predicted_direction: str  # 'green' or 'red'
    confidence: float
    features_used: Dict[str, float]
    model_name: str


@dataclass
class ModelPerformance:
    """Model performance metrics"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    confusion_matrix: np.ndarray


class EthCandlePredictor:
    """
    Statistical model to predict ETH 1-hour candle direction based on first 45 minutes of data.

    Features extracted include:
    - Price action metrics (open, high, low, current price at 45min)
    - Technical indicators (RSI, MACD, moving averages)
    - Volume patterns
    - Volatility measures
    - Market microstructure features
    - Time-based features
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the predictor with specified model type.

        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boost', 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.logger = logging.getLogger(__name__)
        self.is_trained = False

        # Initialize model based on type
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
            )
        elif model_type == "gradient_boost":
            self.model = GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
        elif model_type == "logistic":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def extract_features(
        self, df_1min: pd.DataFrame, target_hour: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Extract comprehensive features from the first 45 minutes of an hourly candle.

        Args:
            df_1min: 1-minute OHLCV DataFrame with timestamp index
            target_hour: The start timestamp of the hour we're analyzing

        Returns:
            Dictionary of extracted features
        """
        # Get data for the first 45 minutes of the target hour
        start_time = target_hour
        end_time = target_hour + pd.Timedelta(minutes=45)

        # Filter data for the first 45 minutes
        mask = (df_1min.index >= start_time) & (df_1min.index < end_time)
        period_data = df_1min[mask].copy()

        if len(period_data) < 30:  # Need at least 30 minutes of data
            return {}

        # Find the price closest to the 45-minute mark
        ts_45 = target_hour + pd.Timedelta(minutes=45)
        time_diffs = np.abs(df_1min.index - ts_45)
        closest_idx = time_diffs.argmin()

        # Validate we found a reasonable match (within 2 minutes of target)
        if time_diffs[closest_idx] > pd.Timedelta(minutes=2):
            self.logger.warning(f"No data close to 45-minute mark for {target_hour}")
            return {}

        current_price = df_1min.iloc[closest_idx]["close"]

        features = {}

        # Basic price action features
        open_price = period_data.iloc[0]["open"]
        high_price = period_data["high"].max()
        low_price = period_data["low"].min()

        features["open_price"] = open_price
        features["high_price"] = high_price
        features["low_price"] = low_price
        features["current_price"] = current_price

        # Price movement features
        features["price_change_abs"] = current_price - open_price
        features["price_change_pct"] = (current_price - open_price) / open_price * 100
        features["range_pct"] = (high_price - low_price) / open_price * 100
        features["position_in_range"] = (
            (current_price - low_price) / (high_price - low_price)
            if high_price != low_price
            else 0.5
        )

        # Trend features
        features["upward_momentum"] = current_price > open_price
        features["upper_half"] = current_price > (high_price + low_price) / 2

        # Volume features
        total_volume = period_data["volume"].sum()
        avg_volume = period_data["volume"].mean()
        features["total_volume"] = total_volume
        features["avg_volume"] = avg_volume
        features["volume_trend"] = (
            period_data["volume"].iloc[-15:].mean()
            / period_data["volume"].iloc[:15].mean()
            if len(period_data) >= 30
            else 1.0
        )

        # Volatility features
        returns = period_data["close"].pct_change().dropna()
        features["volatility"] = returns.std() * 100
        features["realized_vol"] = np.sqrt((returns**2).sum()) * 100

        # Technical indicators
        if len(period_data) >= 14:
            # RSI calculation
            rsi = self._calculate_rsi(period_data["close"], period=14)
            features["rsi"] = rsi.iloc[-1] if not rsi.empty else 50

        if len(period_data) >= 20:
            # Moving averages
            ma_10 = period_data["close"].rolling(10).mean()
            ma_20 = period_data["close"].rolling(20).mean()
            features["ma_10"] = ma_10.iloc[-1] if not ma_10.empty else current_price
            features["ma_20"] = ma_20.iloc[-1] if not ma_20.empty else current_price
            features["price_vs_ma10"] = (
                (current_price - features["ma_10"]) / features["ma_10"] * 100
            )
            features["price_vs_ma20"] = (
                (current_price - features["ma_20"]) / features["ma_20"] * 100
            )
            features["ma_cross"] = features["ma_10"] > features["ma_20"]

        # MACD (if enough data)
        if len(period_data) >= 26:
            macd_line, macd_signal = self._calculate_macd(period_data["close"])
            if len(macd_line) > 0:
                features["macd"] = macd_line.iloc[-1]
                features["macd_signal"] = macd_signal.iloc[-1]
                features["macd_histogram"] = features["macd"] - features["macd_signal"]

        # Candle pattern features
        features["doji_like"] = (
            abs(current_price - open_price) / (high_price - low_price) < 0.1
            if high_price != low_price
            else True
        )
        features["hammer_like"] = (
            (current_price - low_price) / (high_price - low_price) > 0.7
            if high_price != low_price
            else False
        )
        features["shooting_star_like"] = (
            (high_price - current_price) / (high_price - low_price) > 0.7
            if high_price != low_price
            else False
        )

        # Time-based features
        hour = target_hour.hour
        day_of_week = target_hour.dayofweek  # 0=Monday, 6=Sunday
        features["hour"] = hour
        features["day_of_week"] = day_of_week
        features["is_weekend"] = day_of_week >= 5
        features["is_market_open"] = 9 <= hour <= 16  # US market hours (approximate)

        # Market microstructure (if we have tick-level features)
        features["num_price_changes"] = (period_data["close"].diff() != 0).sum()
        features["up_moves"] = (period_data["close"].diff() > 0).sum()
        features["down_moves"] = (period_data["close"].diff() < 0).sum()
        features["up_down_ratio"] = (
            features["up_moves"] / features["down_moves"]
            if features["down_moves"] > 0
            else 2.0
        )

        # Recent momentum features (last 15 minutes vs first 15 minutes)
        if len(period_data) >= 30:
            first_15_avg = period_data["close"].iloc[:15].mean()
            last_15_avg = period_data["close"].iloc[-15:].mean()
            features["momentum_15min"] = (
                (last_15_avg - first_15_avg) / first_15_avg * 100
            )

            first_15_vol = period_data["volume"].iloc[:15].mean()
            last_15_vol = period_data["volume"].iloc[-15:].mean()
            features["volume_momentum_15min"] = (
                (last_15_vol - first_15_vol) / first_15_vol * 100
                if first_15_vol > 0
                else 0
            )

        # Support/Resistance levels (simplified)
        features["near_period_high"] = (current_price / high_price) > 0.95
        features["near_period_low"] = (current_price / low_price) < 1.05

        return features

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self, prices: pd.Series, fast=12, slow=26, signal=9
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD line and signal line"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

    def prepare_training_data(
        self, df_1min: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data by extracting features and labels for all complete hourly periods.

        Args:
            df_1min: 1-minute OHLCV DataFrame with timestamp index

        Returns:
            Tuple of (features_df, labels_series)
        """
        # Resample to get hourly candles
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

        features_list = []
        labels_list = []

        self.logger.info(f"Processing {len(df_1h)} hourly candles for training data...")

        for i, (hour_start, hour_data) in enumerate(df_1h.iterrows()):
            # Extract features for this hour
            features = self.extract_features(df_1min, hour_start)

            if not features:  # Skip if not enough data
                continue

            # Create label: 1 if green candle (close > open), 0 if red candle
            label = 1 if hour_data["close"] > hour_data["open"] else 0

            features_list.append(features)
            labels_list.append(label)

            if i % 100 == 0:
                self.logger.info(f"Processed {i}/{len(df_1h)} candles...")

        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        labels_series = pd.Series(labels_list)

        # Store feature names
        self.feature_names = list(features_df.columns)

        self.logger.info(
            f"Prepared training data: {len(features_df)} samples with {len(self.feature_names)} features"
        )
        self.logger.info(
            f"Label distribution: {labels_series.value_counts().to_dict()}"
        )

        return features_df, labels_series

    def train(self, df_1min: pd.DataFrame, test_size: float = 0.2) -> ModelPerformance:
        """
        Train the model on historical data.

        Args:
            df_1min: 1-minute OHLCV DataFrame with timestamp index
            test_size: Fraction of data to use for testing

        Returns:
            Model performance metrics
        """
        # Prepare training data
        X, y = self.prepare_training_data(df_1min)

        if len(X) < 100:
            raise ValueError("Need at least 100 samples for training")

        # Split data (using temporal split to avoid lookahead bias)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        performance = ModelPerformance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            confusion_matrix=cm,
        )

        self.logger.info(f"Model Performance:")
        self.logger.info(f"  Accuracy: {accuracy:.3f}")
        self.logger.info(f"  Precision: {precision:.3f}")
        self.logger.info(f"  Recall: {recall:.3f}")
        self.logger.info(f"  F1 Score: {f1:.3f}")
        self.logger.info(f"  AUC Score: {auc:.3f}")

        # Feature importance (for tree-based models)
        if hasattr(self.model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            self.logger.info("Top 10 Most Important Features:")
            for _, row in feature_importance.head(10).iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.3f}")

        return performance

    def predict(
        self, df_1min: pd.DataFrame, target_hour: pd.Timestamp
    ) -> PredictionResult:
        """
        Predict the direction of a 1-hour candle based on first 45 minutes of data.

        Args:
            df_1min: 1-minute OHLCV DataFrame with timestamp index
            target_hour: The start timestamp of the hour to predict

        Returns:
            Prediction result with probabilities and confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Extract features
        features = self.extract_features(df_1min, target_hour)

        if not features:
            raise ValueError("Not enough data to extract features for prediction")

        # Prepare features DataFrame
        features_df = pd.DataFrame([features])

        # Ensure all training features are present
        for feature_name in self.feature_names:
            if feature_name not in features_df.columns:
                features_df[feature_name] = 0  # Default value for missing features

        # Reorder columns to match training data
        features_df = features_df[self.feature_names]

        # Scale features
        features_scaled = self.scaler.transform(features_df)

        # Make prediction
        probabilities = self.model.predict_proba(features_scaled)[0]
        prob_red = probabilities[0]
        prob_green = probabilities[1]

        predicted_direction = "green" if prob_green > prob_red else "red"
        confidence = max(prob_green, prob_red)

        return PredictionResult(
            probability_green=prob_green,
            probability_red=prob_red,
            predicted_direction=predicted_direction,
            confidence=confidence,
            features_used=features,
            model_name=self.model_type,
        )

    def cross_validate(
        self, df_1min: pd.DataFrame, cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation on the model.

        Args:
            df_1min: 1-minute OHLCV DataFrame with timestamp index
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary of cross-validation scores
        """
        # Prepare data
        X, y = self.prepare_training_data(df_1min)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=tscv, scoring="accuracy"
        )
        cv_auc_scores = cross_val_score(
            self.model, X_scaled, y, cv=tscv, scoring="roc_auc"
        )

        return {
            "accuracy_mean": cv_scores.mean(),
            "accuracy_std": cv_scores.std(),
            "auc_mean": cv_auc_scores.mean(),
            "auc_std": cv_auc_scores.std(),
            "individual_scores": cv_scores.tolist(),
        }

    def save_model(self, filepath: str):
        """Save the trained model and scaler to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model and scaler from disk"""
        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.model_type = model_data["model_type"]
        self.is_trained = True

        self.logger.info(f"Model loaded from {filepath}")


def create_ensemble_predictor(df_1min: pd.DataFrame) -> Dict[str, EthCandlePredictor]:
    """
    Create and train an ensemble of different models.

    Args:
        df_1min: 1-minute OHLCV DataFrame with timestamp index

    Returns:
        Dictionary of trained models
    """
    models = {}

    for model_type in ["random_forest", "gradient_boost", "logistic"]:
        logger = logging.getLogger(__name__)
        logger.info(f"Training {model_type} model...")

        predictor = EthCandlePredictor(model_type=model_type)
        performance = predictor.train(df_1min)

        models[model_type] = predictor

        logger.info(
            f"{model_type} - Accuracy: {performance.accuracy:.3f}, AUC: {performance.auc_score:.3f}"
        )

    return models


def ensemble_predict(
    models: Dict[str, EthCandlePredictor],
    df_1min: pd.DataFrame,
    target_hour: pd.Timestamp,
) -> PredictionResult:
    """
    Make ensemble prediction using multiple models.

    Args:
        models: Dictionary of trained models
        df_1min: 1-minute OHLCV DataFrame with timestamp index
        target_hour: The start timestamp of the hour to predict

    Returns:
        Ensemble prediction result
    """
    predictions = []
    features_dict = {}

    for model_name, model in models.items():
        try:
            pred = model.predict(df_1min, target_hour)
            predictions.append(pred)
            if not features_dict:  # Use features from first successful prediction
                features_dict = pred.features_used
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Model {model_name} failed to predict: {e}"
            )

    if not predictions:
        raise ValueError("All models failed to make predictions")

    # Average the probabilities
    avg_prob_green = np.mean([p.probability_green for p in predictions])
    avg_prob_red = np.mean([p.probability_red for p in predictions])

    # Determine final prediction
    predicted_direction = "green" if avg_prob_green > avg_prob_red else "red"
    confidence = max(avg_prob_green, avg_prob_red)

    return PredictionResult(
        probability_green=avg_prob_green,
        probability_red=avg_prob_red,
        predicted_direction=predicted_direction,
        confidence=confidence,
        features_used=features_dict,
        model_name="ensemble",
    )
