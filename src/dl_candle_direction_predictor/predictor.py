"""
Real-time prediction interface for crypto candle direction prediction.

Provides a high-level interface for loading trained models and making
real-time predictions with confidence scores.
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from .model import CandlePredictionModel
from .data_loader import BinanceDataLoader
from .feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class CandleDirectionPredictor:
    """
    Real-time candle direction predictor with confidence scoring.
    
    This class provides a complete interface for:
    1. Loading trained models
    2. Fetching real-time market data
    3. Engineering features
    4. Making predictions with confidence scores
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 symbols: Optional[List[str]] = None,
                 data_dir: str = "data"):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model (optional, can load later)
            symbols: List of symbols to support
            data_dir: Directory for data storage
        """
        self.model_path = model_path
        self.symbols = symbols or ['ETHUSDT', 'BTCUSDT', 'SOLUSDT', 'XRPUSDT']
        
        # Initialize components
        self.data_loader = BinanceDataLoader(data_dir)
        self.feature_engineer = FeatureEngineer()
        self.model = None
        
        # Configuration
        self.sequence_length = 60  # 60 minutes of lookback
        self.prediction_horizon = 60  # 60 minutes ahead (1 hour)
        
        # Cache for efficiency
        self._feature_cache = {}
        self._data_cache = {}
        
        logger.info(f"Initialized predictor for symbols: {self.symbols}")
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            self.model = CandlePredictionModel()
            self.model.load_model(model_path)
            self.model_path = model_path
            
            logger.info(f"Loaded model from {model_path}")
            logger.info(f"Model summary:\n{self.model.get_model_summary()}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def _prepare_data(self, symbol: str, current_time: Optional[datetime] = None,
                     use_cache: bool = True) -> pd.DataFrame:
        """
        Prepare data for prediction by fetching recent market data.
        
        Args:
            symbol: Trading symbol
            current_time: Current time (defaults to now)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with recent market data
        """
        if current_time is None:
            current_time = datetime.now()
        
        cache_key = f"{symbol}_{current_time.strftime('%Y%m%d_%H%M')}"
        
        # Check cache first
        if use_cache and cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        try:
            # Get the last few hours of 1-minute data
            lookback_hours = 6  # Get extra data for feature engineering
            
            # Fetch recent 1-minute data
            data_1m = self.data_loader.fetch_live_data(
                symbol=symbol,
                timeframe='1m',
                limit=lookback_hours * 60  # 6 hours * 60 minutes
            )
            
            # Also get higher timeframe data for multi-timeframe features
            data_5m = self.data_loader.fetch_live_data(symbol, '5m', limit=lookback_hours * 12)
            data_15m = self.data_loader.fetch_live_data(symbol, '15m', limit=lookback_hours * 4)
            data_1h = self.data_loader.fetch_live_data(symbol, '1h', limit=lookback_hours)
            
            # Combine into multi-timeframe dict
            multi_tf_data = {
                '1m': data_1m,
                '5m': data_5m,
                '15m': data_15m,
                '1h': data_1h
            }
            
            # Filter to current time (simulate real-time constraint)
            filtered_data = {}
            for tf, df in multi_tf_data.items():
                if len(df) > 0:
                    filtered_data[tf] = df[df.index <= current_time]
            
            # Cache the result
            if use_cache:
                self._data_cache[cache_key] = filtered_data
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Failed to prepare data for {symbol}: {e}")
            raise
    
    def _engineer_features(self, data: Dict[str, pd.DataFrame], 
                          symbol: str, current_time: datetime,
                          use_cache: bool = True) -> pd.DataFrame:
        """
        Engineer features for prediction.
        
        Args:
            data: Multi-timeframe data dictionary
            symbol: Trading symbol
            current_time: Current time
            use_cache: Whether to use cached features
            
        Returns:
            DataFrame with engineered features
        """
        cache_key = f"{symbol}_{current_time.strftime('%Y%m%d_%H%M')}_features"
        
        # Check cache first
        if use_cache and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        try:
            # Use 1-minute data as base
            base_data = data['1m']
            
            if len(base_data) == 0:
                raise ValueError(f"No base data available for {symbol}")
            
            # Engineer features
            features = self.feature_engineer.engineer_features(
                df=base_data,
                multi_tf_data=data,
                current_time=current_time
            )
            
            # Cache the result
            if use_cache:
                self._feature_cache[cache_key] = features
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to engineer features for {symbol}: {e}")
            raise
    
    def predict_candle_direction(self, symbol: str, 
                               current_time: Optional[datetime] = None,
                               return_features: bool = False) -> Dict[str, Union[str, float]]:
        """
        Predict the direction of the next proper 1-hour candle.
        
        This predicts whether the NEXT proper 1-hour candle (starting at top of hour) 
        will close higher than it opens. For example:
        - At 10:15, predicts the 11:00-12:00 candle direction
        - At 10:45, predicts the 11:00-12:00 candle direction
        - At 10:59, predicts the 11:00-12:00 candle direction
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            current_time: Current time within the candle (defaults to now)
            return_features: Whether to return engineered features
            
        Returns:
            Dictionary with prediction results:
            {
                'symbol': 'ETHUSDT',
                'current_time': datetime,
                'target_candle_start': datetime,  # Start of predicted candle
                'target_candle_end': datetime,    # End of predicted candle  
                'direction': 'up' or 'down',     # Predicted candle direction
                'confidence': float (0-1),       # Model confidence
                'current_candle_progress': float (0-1),  # Progress in current candle
                'minutes_into_current_candle': int,      # Minutes elapsed in current candle
                'model_type': str,
                'features': DataFrame (if return_features=True)
            }
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        if symbol not in self.symbols:
            logger.warning(f"Symbol {symbol} not in supported symbols: {self.symbols}")
        
        if current_time is None:
            current_time = datetime.now()
        
        try:
            # Prepare data
            data = self._prepare_data(symbol, current_time)
            
            # Engineer features
            features = self._engineer_features(data, symbol, current_time)
            
            # Check if we have enough data
            if len(features) < self.sequence_length:
                raise ValueError(f"Insufficient data: need {self.sequence_length} samples, got {len(features)}")
            
            # Prepare model input
            model_input = self.feature_engineer.prepare_model_input(
                features, self.sequence_length
            )
            
            if len(model_input) == 0:
                raise ValueError("No valid sequences could be created from features")
            
            # Get the latest sequence (most recent data)
            latest_sequence = model_input[-1]  # Shape: (sequence_length, n_features)
            
            # Convert to torch tensor and add batch dimension
            X = torch.tensor(latest_sequence, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, features)
            
            # Make prediction
            direction, confidence = self.model.predict_direction_and_confidence(X)
            
            # Calculate current candle progress (proper 1-hour candle)
            current_candle_start = current_time.replace(minute=0, second=0, microsecond=0)
            current_candle_end = current_candle_start + timedelta(hours=1)
            minutes_into_current_candle = (current_time - current_candle_start).total_seconds() / 60
            current_candle_progress = minutes_into_current_candle / 60
            
            # Calculate target candle (the one we're predicting)
            target_candle_start = current_candle_end  # Next proper hour
            target_candle_end = target_candle_start + timedelta(hours=1)
            
            result = {
                'symbol': symbol,
                'current_time': current_time,
                'target_candle_start': target_candle_start,
                'target_candle_end': target_candle_end,
                'direction': direction,
                'confidence': confidence,
                'current_candle_progress': current_candle_progress,
                'minutes_into_current_candle': int(minutes_into_current_candle),
                'model_type': self.model.model_type
            }
            
            if return_features:
                result['features'] = features
            
            logger.info(f"Prediction for {symbol}: {direction} (confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            raise
    
    def predict_multiple_symbols(self, symbols: Optional[List[str]] = None,
                                current_time: Optional[datetime] = None) -> Dict[str, Dict]:
        """
        Make predictions for multiple symbols.
        
        Args:
            symbols: List of symbols (defaults to self.symbols)
            current_time: Current time (defaults to now)
            
        Returns:
            Dictionary mapping symbol to prediction result
        """
        symbols = symbols or self.symbols
        current_time = current_time or datetime.now()
        
        results = {}
        
        for symbol in symbols:
            try:
                result = self.predict_candle_direction(symbol, current_time)
                results[symbol] = result
            except Exception as e:
                logger.error(f"Failed to predict for {symbol}: {e}")
                results[symbol] = {
                    'error': str(e),
                    'symbol': symbol,
                    'current_time': current_time
                }
        
        return results
    
    def simulate_real_time_predictions(self, symbol: str, 
                                     start_time: datetime,
                                     end_time: datetime,
                                     interval_minutes: int = 5) -> pd.DataFrame:
        """
        Simulate real-time predictions over a historical period.
        
        Args:
            symbol: Trading symbol
            start_time: Start of simulation
            end_time: End of simulation
            interval_minutes: Minutes between predictions
            
        Returns:
            DataFrame with simulation results
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        results = []
        current_time = start_time
        
        logger.info(f"Starting simulation for {symbol} from {start_time} to {end_time}")
        
        while current_time <= end_time:
            try:
                # Make prediction at current time
                prediction = self.predict_candle_direction(symbol, current_time)
                
                # Add timestamp info
                prediction['simulation_time'] = current_time.isoformat()
                results.append(prediction)
                
                # Move to next interval
                current_time += timedelta(minutes=interval_minutes)
                
            except Exception as e:
                logger.warning(f"Simulation failed at {current_time}: {e}")
                current_time += timedelta(minutes=interval_minutes)
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        logger.info(f"Simulation complete. Generated {len(df)} predictions.")
        
        return df
    
    def get_candle_progress_info(self, current_time: Optional[datetime] = None) -> Dict[str, Union[int, float, bool, datetime]]:
        """
        Get information about current position within the proper 1-hour candle.
        
        The current candle is always the proper 1-hour candle (starting at top of hour).
        Also provides information about the target candle being predicted.
        
        Args:
            current_time: Current time (defaults to now)
            
        Returns:
            Dictionary with candle progress information
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Find the current proper 1-hour candle (always starts at top of hour)
        current_candle_start = current_time.replace(minute=0, second=0, microsecond=0)
        current_candle_end = current_candle_start + timedelta(hours=1)
        
        # Calculate progress in current candle
        elapsed = current_time - current_candle_start
        remaining = current_candle_end - current_time
        
        minutes_elapsed = elapsed.total_seconds() / 60
        minutes_remaining = remaining.total_seconds() / 60
        progress = minutes_elapsed / 60
        
        # Target candle (the one we predict)
        target_candle_start = current_candle_end
        target_candle_end = target_candle_start + timedelta(hours=1)
        
        return {
            'current_time': current_time,
            'current_candle_start': current_candle_start,
            'current_candle_end': current_candle_end,
            'target_candle_start': target_candle_start,
            'target_candle_end': target_candle_end,
            'minutes_elapsed': int(minutes_elapsed),
            'minutes_remaining': int(minutes_remaining),
            'progress': progress,
            'is_early_candle': minutes_elapsed < 15,
            'is_mid_candle': 15 <= minutes_elapsed < 45,
            'is_late_candle': minutes_elapsed >= 45
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data and features."""
        self._data_cache.clear()
        self._feature_cache.clear()
        logger.info("Cleared prediction cache")
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is supported.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if symbol is valid
        """
        return self.data_loader.validate_symbol(symbol)
    
    def get_model_info(self) -> Dict[str, Union[str, int, float, List[str], None]]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {'status': 'No model loaded'}
        
        return {
            'model_type': self.model.model_type,
            'input_size': self.model.input_size,
            'device': str(self.model.device),
            'model_path': self.model_path,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'supported_symbols': self.symbols,
            'model_summary': self.model.get_model_summary()
        }
    
    def benchmark_prediction_speed(self, symbol: str = 'ETHUSDT', 
                                 num_predictions: int = 10) -> Dict[str, float]:
        """
        Benchmark prediction speed.
        
        Args:
            symbol: Symbol to use for benchmarking
            num_predictions: Number of predictions to make
            
        Returns:
            Dictionary with timing statistics
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        import time
        times = []
        
        logger.info(f"Benchmarking prediction speed with {num_predictions} predictions")
        
        for i in range(num_predictions):
            start_time = time.time()
            
            try:
                self.predict_candle_direction(symbol)
                elapsed = time.time() - start_time
                times.append(elapsed)
            except Exception as e:
                logger.warning(f"Benchmark prediction {i} failed: {e}")
                continue
        
        if not times:
            raise ValueError("All benchmark predictions failed")
        
        return {
            'num_predictions': len(times),
            'mean_time_seconds': np.mean(times),
            'median_time_seconds': np.median(times),
            'min_time_seconds': np.min(times),
            'max_time_seconds': np.max(times),
            'std_time_seconds': np.std(times),
            'predictions_per_second': 1.0 / np.mean(times)
        }