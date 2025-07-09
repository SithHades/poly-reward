"""
Feature engineering for crypto candle direction prediction.

Generates technical indicators, rolling statistics, and time-aware features
that are computed only using data available up to the current timepoint.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for crypto candle prediction.
    All features respect the temporal constraint - no future data leakage.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_columns = []
        
    def rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            series: Price series (typically close)
            period: RSI period
            
        Returns:
            RSI values
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def bollinger_bands(self, series: pd.Series, period: int = 20, 
                       std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            series: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def macd(self, series: pd.Series, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            series: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            high: High prices
            low: Low prices  
            close: Close prices
            period: ATR period
            
        Returns:
            ATR values
        """
        high_low = high - low
        high_close_prev = np.abs(high - close.shift(1))
        low_close_prev = np.abs(low - close.shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume indicators
        """
        result = df.copy()
        
        # Volume moving averages
        result['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        result['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        result['volume_ratio'] = df['volume'] / result['volume_sma_20']
        
        # Volume rate of change
        result['volume_roc'] = df['volume'].pct_change(periods=5)
        
        # On-Balance Volume (OBV)
        obv = (df['volume'] * np.where(df['close'] > df['close'].shift(1), 1, -1)).cumsum()
        result['obv'] = obv
        result['obv_sma'] = obv.rolling(window=20).mean()
        
        # Volume-Price Trend (VPT)
        vpt = (df['volume'] * df['close'].pct_change()).cumsum()
        result['vpt'] = vpt
        
        return result
    
    def price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price action and momentum features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price action features
        """
        result = df.copy()
        
        # Basic price features
        result['hl_ratio'] = (df['high'] - df['low']) / df['close']
        result['oc_ratio'] = (df['close'] - df['open']) / df['open']
        result['body_size'] = np.abs(df['close'] - df['open']) / df['close']
        result['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        result['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        
        # Price position within candle
        result['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Rolling price features
        for period in [5, 10, 20, 50]:
            result[f'price_change_{period}'] = df['close'].pct_change(periods=period)
            result[f'high_change_{period}'] = df['high'].pct_change(periods=period)
            result[f'low_change_{period}'] = df['low'].pct_change(periods=period)
            
            # Price position relative to moving averages
            sma = df['close'].rolling(window=period).mean()
            result[f'price_vs_sma_{period}'] = (df['close'] - sma) / sma
            
            # Volatility measures
            result[f'volatility_{period}'] = df['close'].rolling(window=period).std() / sma
        
        return result
    
    def momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum and trend features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum features
        """
        result = df.copy()
        
        # RSI for multiple periods
        for period in [7, 14, 21]:
            result[f'rsi_{period}'] = self.rsi(df['close'], period)
        
        # MACD
        macd_line, signal_line, histogram = self.macd(df['close'])
        result['macd'] = macd_line
        result['macd_signal'] = signal_line
        result['macd_histogram'] = histogram
        
        # Stochastic
        k_percent, d_percent = self.stochastic(df['high'], df['low'], df['close'])
        result['stoch_k'] = k_percent
        result['stoch_d'] = d_percent
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(df['close'])
        result['bb_upper'] = bb_upper
        result['bb_middle'] = bb_middle
        result['bb_lower'] = bb_lower
        result['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        result['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # ATR
        result['atr'] = self.atr(df['high'], df['low'], df['close'])
        result['atr_ratio'] = result['atr'] / df['close']
        
        # Rate of Change for multiple periods
        for period in [3, 5, 10, 20]:
            result[f'roc_{period}'] = df['close'].pct_change(periods=period)
        
        return result
    
    def time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time-based features.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with time features
        """
        result = df.copy()
        
        # Extract time components
        result['hour'] = df.index.hour
        result['day_of_week'] = df.index.dayofweek
        result['day_of_month'] = df.index.day
        result['month'] = df.index.month
        
        # Cyclical encoding for time features
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        # Time since start features
        if len(df) > 0:
            start_time = df.index[0]
            result['hours_since_start'] = (df.index - start_time).total_seconds() / 3600
            result['days_since_start'] = (df.index - start_time).days
        
        return result
    
    def multi_timeframe_features(self, data_dict: Dict[str, pd.DataFrame],
                                target_timeframe: str = '1m') -> pd.DataFrame:
        """
        Create features using multiple timeframes.
        
        Args:
            data_dict: Dictionary mapping timeframe to DataFrame
            target_timeframe: The timeframe to use as base
            
        Returns:
            DataFrame with multi-timeframe features
        """
        if target_timeframe not in data_dict:
            raise ValueError(f"Target timeframe {target_timeframe} not found in data")
        
        base_df = data_dict[target_timeframe].copy()
        
        # Add features from higher timeframes
        higher_timeframes = ['5m', '15m', '30m', '1h', '4h']
        
        for tf in higher_timeframes:
            if tf not in data_dict:
                continue
                
            tf_data = data_dict[tf]
            
            # Resample to match base timeframe
            resampled = tf_data.reindex(base_df.index, method='ffill')
            
            # Add prefixed features
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in resampled.columns:
                    base_df[f'{tf}_{col}'] = resampled[col]
            
            # Add some derived features
            if 'close' in resampled.columns:
                base_df[f'{tf}_rsi'] = self.rsi(resampled['close'])
                
                # Trend direction
                base_df[f'{tf}_trend'] = np.where(
                    resampled['close'] > resampled['close'].shift(1), 1, -1
                )
        
        return base_df
    
    def create_labels(self, df: pd.DataFrame, prediction_horizon: int = 60) -> pd.Series:
        """
        Create labels for 1-hour candle direction prediction.
        
        Args:
            df: DataFrame with minute-level OHLCV data
            prediction_horizon: Minutes ahead to predict (60 for 1-hour)
            
        Returns:
            Series with binary labels (1 for up, 0 for down)
        """
        # Get the close price at prediction horizon
        future_close = df['close'].shift(-prediction_horizon)
        current_close = df['close']
        
        # Label is 1 if future close > current close, 0 otherwise
        labels = (future_close > current_close).astype(int)
        
        return labels
    
    def create_candle_progress_features(self, df: pd.DataFrame, 
                                      current_time: datetime) -> pd.DataFrame:
        """
        Create features based on progress within current 1-hour candle.
        
        Args:
            df: DataFrame with minute-level data
            current_time: Current timestamp within the candle
            
        Returns:
            DataFrame with candle progress features
        """
        result = df.copy()
        
        # Find the start of current 1-hour candle
        candle_start = current_time.replace(minute=0, second=0, microsecond=0)
        minutes_elapsed = (current_time - candle_start).total_seconds() / 60
        
        # Progress through candle (0-59 minutes)
        result['candle_progress'] = minutes_elapsed / 60
        result['minutes_remaining'] = 60 - minutes_elapsed
        
        # Features based on candle progress
        result['is_early_candle'] = int(minutes_elapsed < 15)
        result['is_mid_candle'] = int((minutes_elapsed >= 15) and (minutes_elapsed < 45))
        result['is_late_candle'] = int(minutes_elapsed >= 45)
        
        # Current candle statistics (only using data up to current time)
        candle_data = df[df.index >= candle_start]
        if len(candle_data) > 0:
            result['candle_open'] = candle_data['open'].iloc[0]
            result['candle_high'] = candle_data['high'].max()
            result['candle_low'] = candle_data['low'].min()
            result['candle_volume'] = candle_data['volume'].sum()
            
            # Current position within candle range
            current_close = df['close'].iloc[-1]
            candle_range = result['candle_high'] - result['candle_low']
            if candle_range > 0:
                result['position_in_candle'] = (current_close - result['candle_low']) / candle_range
            else:
                result['position_in_candle'] = 0.5
        
        return result
    
    def engineer_features(self, df: pd.DataFrame, 
                         multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                         current_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Main feature engineering pipeline.
        
        Args:
            df: Base DataFrame with OHLCV data
            multi_tf_data: Optional multi-timeframe data
            current_time: Current time for candle progress features
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        # Start with base data
        features = df.copy()
        
        # Add price action features
        features = self.price_action_features(features)
        logger.info("Added price action features")
        
        # Add momentum features
        features = self.momentum_features(features)
        logger.info("Added momentum features")
        
        # Add volume features
        features = self.volume_indicators(features)
        logger.info("Added volume features")
        
        # Add time features
        features = self.time_features(features)
        logger.info("Added time features")
        
        # Add multi-timeframe features if available
        if multi_tf_data:
            features = self.multi_timeframe_features(multi_tf_data, '1m')
            logger.info("Added multi-timeframe features")
        
        # Add candle progress features if current time provided
        if current_time:
            features = self.create_candle_progress_features(features, current_time)
            logger.info("Added candle progress features")
        
        # Store feature column names (excluding original OHLCV)
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe']
        self.feature_columns = [col for col in features.columns if col not in original_cols]
        
        logger.info(f"Feature engineering complete. Generated {len(self.feature_columns)} features")
        
        return features
    
    def get_feature_columns(self) -> List[str]:
        """Get list of engineered feature column names."""
        return self.feature_columns.copy()
    
    def prepare_model_input(self, features: pd.DataFrame, 
                           sequence_length: int = 60) -> np.ndarray:
        """
        Prepare features for model input with sequence formatting.
        
        Args:
            features: DataFrame with engineered features
            sequence_length: Length of input sequences
            
        Returns:
            3D array suitable for sequence models (samples, timesteps, features)
        """
        # Get only feature columns (no OHLCV)
        feature_data = features[self.feature_columns]
        
        # Handle missing values
        feature_data = feature_data.fillna(method='ffill').fillna(0)
        
        # Create sequences
        sequences = []
        for i in range(sequence_length, len(feature_data)):
            sequence = feature_data.iloc[i-sequence_length:i].values
            sequences.append(sequence)
        
        return np.array(sequences)