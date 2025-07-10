"""
Deep Learning Candle Direction Predictor

A comprehensive module for predicting 1-hour crypto candle directions using deep learning.
Supports real-time predictions at any point during a candle with confidence scores.
"""

from .predictor import CandleDirectionPredictor
from .model import CandlePredictionModel
from .data_loader import BinanceDataLoader
from .feature_engineering import FeatureEngineer

__version__ = "1.0.0"
__author__ = "Poly Reward Team"

__all__ = [
    "CandleDirectionPredictor",
    "CandlePredictionModel", 
    "BinanceDataLoader",
    "FeatureEngineer"
]