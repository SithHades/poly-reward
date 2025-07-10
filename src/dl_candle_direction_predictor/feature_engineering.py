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
    
    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create labels for CURRENT proper 1-hour candle direction prediction.
        
        Labels predict whether the CURRENT proper 1-hour candle (the one in progress)
        will close higher than it opens. For example:
        - At 10:15, predict if the 10:00-11:00 candle will close green/red
        - At 10:45, predict if the 10:00-11:00 candle will close green/red
        
        This is used for Polymarket betting where you want to predict the final
        direction of the current candle while it's still in progress.
        
        Args:
            df: DataFrame with minute-level OHLCV data (datetime index)
            
        Returns:
            Series with binary labels (1 for green candle, 0 for red candle)
        """
        labels = []
        
        for timestamp in df.index:
            # Find the current proper 1-hour candle that contains this timestamp
            current_candle_start = timestamp.replace(minute=0, second=0, microsecond=0)
            current_candle_end = current_candle_start + timedelta(hours=1)
            
            # Get all data for the current candle (complete candle)
            current_candle_data = df[(df.index >= current_candle_start) & (df.index < current_candle_end)]
            
            if len(current_candle_data) > 0:
                # Get open (first minute) and close (last minute) of current candle
                candle_open = current_candle_data['open'].iloc[0]
                candle_close = current_candle_data['close'].iloc[-1]
                
                # Label is 1 if current candle closes higher than it opens (green)
                label = 1 if candle_close > candle_open else 0
                labels.append(label)
            else:
                # No complete data available for current candle
                labels.append(None)
        
        return pd.Series(labels, index=df.index)
    
    def create_candle_progress_features(self, df: pd.DataFrame, 
                                      current_time: datetime) -> pd.DataFrame:
        """
        Create features based on progress within current proper 1-hour candle.
        
        These features help predict whether the CURRENT candle will close green or red.
        The current candle is the proper 1-hour candle containing current_time.
        For example:
        - If current_time is 10:15, current candle is 10:00-11:00 (predicting its final direction)
        - If current_time is 10:45, current candle is 10:00-11:00 (predicting its final direction)
        
        Args:
            df: DataFrame with minute-level data
            current_time: Current timestamp within the candle
            
        Returns:
            DataFrame with candle progress features for prediction
        """
        result = df.copy()
        
        # Find the current proper 1-hour candle (always starts at top of hour)
        current_candle_start = current_time.replace(minute=0, second=0, microsecond=0)
        current_candle_end = current_candle_start + timedelta(hours=1)
        
        # Calculate progress within the current candle
        minutes_elapsed = (current_time - current_candle_start).total_seconds() / 60
        minutes_remaining = 60 - minutes_elapsed
        
        # Progress through current candle (0.0 to 1.0)
        result['candle_progress'] = minutes_elapsed / 60
        result['minutes_remaining'] = minutes_remaining
        result['minutes_elapsed'] = minutes_elapsed
        
        # Features based on candle progress (for betting timing)
        result['is_early_candle'] = int(minutes_elapsed < 15)
        result['is_mid_candle'] = int((minutes_elapsed >= 15) and (minutes_elapsed < 45))
        result['is_late_candle'] = int(minutes_elapsed >= 45)  # Prime betting time
        result['is_very_late_candle'] = int(minutes_elapsed >= 55)  # Last-minute bets
        
        # Current candle performance so far (only using data up to current time)
        candle_data = df[(df.index >= current_candle_start) & (df.index <= current_time)]
        
        if len(candle_data) > 0:
            result['current_candle_open'] = candle_data['open'].iloc[0]
            result['current_candle_high'] = candle_data['high'].max()
            result['current_candle_low'] = candle_data['low'].min()
            result['current_candle_volume'] = candle_data['volume'].sum()
            result['current_candle_close'] = candle_data['close'].iloc[-1]  # Current price
            
            # Key feature: How is the current candle performing so far?
            result['current_candle_change_pct'] = (
                (result['current_candle_close'] - result['current_candle_open']) / 
                result['current_candle_open']
            ) * 100
            
            # Current candle direction so far (what Polymarket might show)
            result['current_candle_direction_so_far'] = int(
                result['current_candle_close'] > result['current_candle_open']
            )
            
            # How strong is the current move?
            result['current_candle_body_size_pct'] = abs(
                result['current_candle_change_pct']
            )
            
            # Position within the candle's range so far
            candle_range = result['current_candle_high'] - result['current_candle_low']
            if candle_range > 0:
                result['current_price_position_in_range'] = (
                    result['current_candle_close'] - result['current_candle_low']
                ) / candle_range
            else:
                result['current_price_position_in_range'] = 0.5
            
            # Volatility within current candle
            if len(candle_data) > 1:
                result['current_candle_volatility'] = candle_data['close'].std()
                result['current_candle_volatility_pct'] = (
                    result['current_candle_volatility'] / result['current_candle_open'] * 100
                )
            else:
                result['current_candle_volatility'] = 0.0
                result['current_candle_volatility_pct'] = 0.0
            
            # Momentum within current candle (recent minutes vs early minutes)
            if len(candle_data) >= 10:  # At least 10 minutes of data
                early_avg = candle_data['close'].iloc[:5].mean() if len(candle_data) >= 5 else candle_data['close'].iloc[0]
                recent_avg = candle_data['close'].iloc[-5:].mean()
                result['current_candle_momentum'] = (recent_avg - early_avg) / early_avg * 100
            else:
                result['current_candle_momentum'] = 0.0
        
        # Trading session features (important for crypto markets)
        result['hour_of_day'] = current_time.hour
        result['is_asian_session'] = int(0 <= current_time.hour < 8)  # UTC
        result['is_european_session'] = int(8 <= current_time.hour < 16)
        result['is_american_session'] = int(16 <= current_time.hour < 24)
        
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