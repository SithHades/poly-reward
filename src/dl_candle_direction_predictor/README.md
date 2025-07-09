# Deep Learning Candle Direction Predictor 🕯️📈

A comprehensive deep learning module for predicting proper 1-hour crypto candle directions (green/red) with confidence scores. This module predicts whether the NEXT proper 1-hour candle (starting at top of hour, e.g., 10:00-11:00, 11:00-12:00) will close higher than it opens. Supports real-time predictions at any point during the current candle and is designed for integration with prediction market agents like Polymarket.

## 🎯 Overview

This module provides:
- **Proper 1-hour candle predictions** - Predicts the next candle starting at top of hour (e.g., 10:00-11:00)
- **Real-time predictions** at any minute/second within the current candle
- **Clear prediction target** - Whether the target candle closes higher than it opens (green vs red)
- **Multiple model architectures** (LSTM, Transformer, CNN-LSTM)
- **Comprehensive feature engineering** with technical indicators
- **Multi-timeframe analysis** (1m, 5m, 15m, 1h data)
- **Confidence scoring** for prediction market integration
- **Backtesting and evaluation** tools
- **Live simulation capabilities**

### 🎯 Prediction Target

**Important**: This model predicts proper 1-hour candles only:
- ✅ **Proper candles**: 01:00-02:00, 10:00-11:00, 15:00-16:00
- ❌ **Not predicted**: 01:12-02:12, 10:15-11:15, arbitrary 60-minute periods

**Examples**:
- At 10:15 → Predicts 11:00-12:00 candle direction
- At 10:45 → Predicts 11:00-12:00 candle direction  
- At 10:59 → Predicts 11:00-12:00 candle direction

## 📊 Supported Markets

- ETHUSDT, BTCUSDT, SOLUSDT, XRPUSDT
- Any Binance spot trading pairs (easily extensible)

## 🏗️ Architecture

### Module Structure
```
src/dl_candle_direction_predictor/
├── __init__.py              # Module initialization
├── data_loader.py           # Binance data fetching (ccxt + zip files)
├── feature_engineering.py   # Technical indicators & features
├── model.py                 # Deep learning model architectures
├── predictor.py             # Real-time prediction interface
├── train.py                 # Model training script
├── evaluate.py              # Backtesting & evaluation
├── requirements.txt         # Dependencies
└── README.md               # This file
```

### Model Architectures

1. **LSTM with Attention**
   - Bidirectional LSTM layers
   - Attention mechanism for sequence focus
   - Dropout for regularization

2. **Transformer**
   - Multi-head self-attention
   - Positional encoding
   - Layer normalization

3. **CNN-LSTM Hybrid**
   - 1D CNN for feature extraction
   - LSTM for sequence modeling
   - Combined approach for pattern recognition

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r src/dl_candle_direction_predictor/requirements.txt

# Or install specific requirements in your environment
pip install torch numpy pandas ccxt scikit-learn matplotlib seaborn
```

### Basic Usage

```python
from src.dl_candle_direction_predictor import CandleDirectionPredictor

# Initialize predictor with trained model
predictor = CandleDirectionPredictor(model_path="path/to/trained/model")

# Make a prediction
result = predictor.predict_candle_direction('ETHUSDT')
print(f"Target Candle: {result['target_candle_start']} - {result['target_candle_end']}")
print(f"Direction: {result['direction']}")  # 'up' or 'down'
print(f"Confidence: {result['confidence']:.3f}")  # 0.0 - 1.0
print(f"Current Candle Progress: {result['current_candle_progress']:.2f}")  # 0.0 - 1.0
```

### Training a Model

```python
from src.dl_candle_direction_predictor.train import CandleTrainer

# Configuration
config = {
    'model_type': 'lstm',  # 'lstm', 'transformer', 'cnn_lstm'
    'symbols': ['ETHUSDT', 'BTCUSDT'],
    'start_date': '2023-01-01',
    'end_date': '2024-01-01',
    'num_epochs': 100,
    'batch_size': 32,
    'sequence_length': 60,  # 60 minutes of lookback
    'prediction_horizon': 60,  # 60 minutes ahead
}

# Train model
trainer = CandleTrainer(config)
model = trainer.run_full_training_pipeline()
```

### Running Evaluation

```python
from src.dl_candle_direction_predictor.evaluate import CandleEvaluator

# Initialize evaluator
evaluator = CandleEvaluator(model_path="path/to/trained/model")

# Run backtest
results = evaluator.historical_backtest(
    symbol='ETHUSDT',
    start_date='2024-01-01',
    end_date='2024-02-01',
    prediction_interval=5  # Make prediction every 5 minutes
)

# Generate performance report
report = evaluator.generate_performance_report({'ETHUSDT': results})
print(f"Overall Accuracy: {report['overall_metrics']['overall_accuracy']:.4f}")
```

## 📈 Features

### Data Sources
- **Live Data**: ccxt library for real-time Binance data
- **Historical Data**: Binance Vision public datasets
- **Multi-Timeframe**: 1s, 1m, 5m, 15m, 30m, 1h, 4h, 1d

### Feature Engineering
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, ATR
- **Price Action**: Body size, shadows, position within candle
- **Volume Indicators**: OBV, VPT, volume ratios
- **Time Features**: Hour, day of week, cyclical encoding
- **Multi-Timeframe**: Features from higher timeframes
- **Candle Progress**: Position within current 1-hour candle

### Model Features
- **Sequence Modeling**: 60-minute lookback for temporal patterns
- **Mixed Precision Training**: For faster training on modern GPUs
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **Model Checkpointing**: Save best models automatically

### Prediction Interface
- **Real-time Ready**: Sub-second prediction times
- **Confidence Scores**: Calibrated probability outputs
- **Caching**: Efficient data and feature caching
- **Multi-Symbol**: Batch predictions across symbols
- **Simulation**: Historical simulation capabilities

## 🔧 Advanced Usage

### Custom Model Configuration

```python
# LSTM Configuration
lstm_config = {
    'model_type': 'lstm',
    'hidden_size': 256,
    'num_layers': 3,
    'dropout': 0.3,
    'learning_rate': 0.0005,
    'weight_decay': 0.01
}

# Transformer Configuration
transformer_config = {
    'model_type': 'transformer',
    'd_model': 256,
    'nhead': 8,
    'num_layers': 6,
    'dropout': 0.1,
    'learning_rate': 0.0001
}

# CNN-LSTM Configuration
cnn_lstm_config = {
    'model_type': 'cnn_lstm',
    'cnn_filters': [128, 64, 32],
    'lstm_hidden': 256,
    'lstm_layers': 2,
    'dropout': 0.25
}
```

### Custom Feature Engineering

```python
from src.dl_candle_direction_predictor.feature_engineering import FeatureEngineer

# Initialize feature engineer
fe = FeatureEngineer()

# Engineer features with custom parameters
features = fe.engineer_features(
    df=data_1m,
    multi_tf_data={'1m': data_1m, '5m': data_5m, '1h': data_1h},
    current_time=datetime.now()
)

# Get feature names
feature_names = fe.get_feature_columns()
print(f"Generated {len(feature_names)} features")
```

### Live Trading Integration

```python
import time
from datetime import datetime

# Initialize predictor
predictor = CandleDirectionPredictor(model_path="trained_model")

# Trading loop example
while True:
    for symbol in ['ETHUSDT', 'BTCUSDT']:
        try:
            # Get prediction
            result = predictor.predict_candle_direction(symbol)
            
            # Check confidence threshold
            if result['confidence'] > 0.75:
                direction = result['direction']
                confidence = result['confidence']
                progress = result['candle_progress']
                
                print(f"{symbol}: {direction} ({confidence:.3f}) - {progress:.2f} through candle")
                
                # Your trading logic here
                # e.g., place order on prediction market
                
        except Exception as e:
            print(f"Error predicting {symbol}: {e}")
    
    # Wait before next prediction
    time.sleep(60)  # Predict every minute
```

## 🧪 Command Line Interface

### Training
```bash
# Train LSTM model
python -m src.dl_candle_direction_predictor.train \
    --model-type lstm \
    --symbols ETHUSDT BTCUSDT \
    --epochs 100 \
    --batch-size 32 \
    --output-dir models/lstm_v1

# Train Transformer model
python -m src.dl_candle_direction_predictor.train \
    --model-type transformer \
    --symbols ETHUSDT BTCUSDT SOLUSDT \
    --epochs 150 \
    --output-dir models/transformer_v1
```

### Evaluation
```bash
# Evaluate model performance
python -m src.dl_candle_direction_predictor.evaluate \
    --model-path models/lstm_v1 \
    --symbols ETHUSDT BTCUSDT \
    --start-date 2024-01-01 \
    --end-date 2024-02-01 \
    --prediction-interval 5 \
    --output-dir evaluation_results
```

## 📊 Performance Metrics

The module provides comprehensive evaluation metrics:

- **Overall Accuracy**: Percentage of correct predictions
- **Confidence Calibration**: Accuracy by confidence level
- **Candle Progress Analysis**: Accuracy by position in candle
- **Symbol-specific Performance**: Per-symbol accuracy
- **Temporal Analysis**: Performance over time
- **Rolling Metrics**: Moving averages of performance

### Sample Output
```
EVALUATION SUMMARY
==================================================
Overall Accuracy: 0.6234
Mean Confidence: 0.7456
Total Predictions: 15,432

Accuracy by Symbol:
  ETHUSDT: 0.6345 (5,123 predictions)
  BTCUSDT: 0.6189 (4,987 predictions)
  SOLUSDT: 0.6178 (5,322 predictions)

Confidence Analysis:
  High Confidence (>0.8): 0.7123 accuracy (3,456 predictions)
  Medium Confidence (0.6-0.8): 0.6234 accuracy (8,765 predictions)
  Low Confidence (<0.6): 0.5432 accuracy (3,211 predictions)
```

## 🎛️ Configuration

### Data Configuration
```python
data_config = {
    'data_dir': 'data',           # Data storage directory
    'symbols': ['ETHUSDT'],       # Trading symbols
    'start_date': '2023-01-01',   # Training start date
    'end_date': '2024-01-01',     # Training end date
    'timeframes': ['1m', '5m', '15m', '1h']  # Data timeframes
}
```

### Training Configuration
```python
training_config = {
    'sequence_length': 60,        # Lookback window (minutes)
    'prediction_horizon': 60,     # Prediction horizon (minutes)
    'batch_size': 32,            # Training batch size
    'num_epochs': 100,           # Maximum training epochs
    'early_stopping_patience': 15, # Early stopping patience
    'val_split': 0.2,            # Validation split ratio
    'test_split': 0.1            # Test split ratio
}
```

### Model Configuration
```python
model_config = {
    'model_type': 'lstm',        # Model architecture
    'hidden_size': 128,          # Hidden layer size
    'num_layers': 2,             # Number of layers
    'dropout': 0.2,              # Dropout rate
    'learning_rate': 0.001,      # Learning rate
    'weight_decay': 0.01         # L2 regularization
}
```

## 🚨 Important Notes

### Temporal Constraints
- **No Future Data**: All features use only data available up to the current timepoint
- **Real-time Compatible**: Predictions can be made at any point during a candle
- **Progressive Accuracy**: Model accuracy may improve as more data becomes available within the candle

### Risk Considerations
- **Market Volatility**: Crypto markets are highly volatile and unpredictable
- **Model Limitations**: Past performance doesn't guarantee future results
- **Testing Required**: Always backtest thoroughly before live trading
- **Position Sizing**: Use confidence scores for appropriate position sizing

### Performance Optimization
- **GPU Usage**: Models automatically use CUDA if available
- **Mixed Precision**: Enabled for faster training on modern GPUs
- **Caching**: Data and features are cached for efficiency
- **Batch Processing**: Support for batch predictions across symbols

## 🔮 Future Enhancements

- **Additional Architectures**: Graph Neural Networks, Attention mechanisms
- **Alternative Data**: Order book data, social sentiment, news
- **Ensemble Methods**: Combining multiple models
- **Reinforcement Learning**: Direct trading strategy optimization
- **Real-time Streaming**: WebSocket integration for live data
- **Advanced Features**: Market microstructure indicators

## 📚 References

- Binance API Documentation
- PyTorch Deep Learning Framework
- Technical Analysis Indicators
- Time Series Forecasting Methods
- Attention Mechanisms in Deep Learning

## 🤝 Contributing

This module is designed to be modular and extensible. Key areas for contribution:

1. **New Model Architectures**: Add new deep learning models
2. **Feature Engineering**: Implement additional technical indicators
3. **Data Sources**: Integrate new data providers
4. **Evaluation Metrics**: Add new performance measures
5. **Optimization**: Improve training and inference speed

## ⚖️ License

This module is part of the larger poly-reward project. Refer to the main project license for terms and conditions.

---

**Disclaimer**: This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Always conduct thorough testing and risk management before using any trading strategies.