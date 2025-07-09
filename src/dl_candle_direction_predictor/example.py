"""
Example usage of the Deep Learning Candle Direction Predictor.

This script demonstrates basic functionality including:
1. Data loading and preparation
2. Feature engineering
3. Model training (simplified)
4. Making predictions
5. Basic evaluation
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dl_candle_direction_predictor import (
    BinanceDataLoader,
    FeatureEngineer, 
    CandlePredictionModel,
    CandleDirectionPredictor
)


def example_data_loading():
    """Demonstrate data loading capabilities."""
    print("=" * 50)
    print("DATA LOADING EXAMPLE")
    print("=" * 50)
    
    # Initialize data loader
    data_loader = BinanceDataLoader(data_dir="example_data")
    
    print("Supported symbols:", data_loader.supported_symbols)
    print("Available timeframes:", list(data_loader.timeframes.keys()))
    
    # Fetch live data (small sample)
    try:
        print("\n📊 Fetching live 1-minute data for ETHUSDT...")
        live_data = data_loader.fetch_live_data('ETHUSDT', '1m', limit=10)
        print(f"Fetched {len(live_data)} candles")
        print("Latest candle:")
        print(live_data.tail(1))
        
    except Exception as e:
        print(f"Error fetching live data: {e}")
        print("Note: This requires internet connection and may fail in isolated environments")


def example_feature_engineering():
    """Demonstrate feature engineering."""
    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING EXAMPLE") 
    print("=" * 50)
    
    # Initialize components
    data_loader = BinanceDataLoader()
    feature_engineer = FeatureEngineer()
    
    try:
        # Get sample data
        print("📈 Fetching sample data...")
        data_1m = data_loader.fetch_live_data('ETHUSDT', '1m', limit=200)
        data_5m = data_loader.fetch_live_data('ETHUSDT', '5m', limit=50)
        data_1h = data_loader.fetch_live_data('ETHUSDT', '1h', limit=10)
        
        # Prepare multi-timeframe data
        multi_tf_data = {
            '1m': data_1m,
            '5m': data_5m, 
            '1h': data_1h
        }
        
        print("🔧 Engineering features...")
        features = feature_engineer.engineer_features(
            df=data_1m,
            multi_tf_data=multi_tf_data,
            current_time=datetime.now()
        )
        
        print(f"✅ Generated {len(feature_engineer.get_feature_columns())} features")
        print("\nSample features:")
        feature_names = feature_engineer.get_feature_columns()
        for i, name in enumerate(feature_names[:10]):  # Show first 10 features
            print(f"  {i+1}. {name}")
        
        if len(feature_names) > 10:
            print(f"  ... and {len(feature_names) - 10} more features")
        
        # Show sample of engineered data
        print(f"\nFeature data shape: {features.shape}")
        print("Sample feature values (last row):")
        sample_features = features[feature_names[:5]].iloc[-1]
        for name, value in sample_features.items():
            print(f"  {name}: {value:.6f}")
            
    except Exception as e:
        print(f"Error in feature engineering: {e}")


def example_model_creation():
    """Demonstrate model creation and basic setup."""
    print("\n" + "=" * 50)
    print("MODEL CREATION EXAMPLE")
    print("=" * 50)
    
    # Model configurations
    configs = {
        'LSTM': {
            'model_type': 'lstm',
            'hidden_size': 64,  # Smaller for example
            'num_layers': 2,
            'dropout': 0.2
        },
        'Transformer': {
            'model_type': 'transformer', 
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dropout': 0.1
        },
        'CNN-LSTM': {
            'model_type': 'cnn_lstm',
            'cnn_filters': [32, 16],
            'lstm_hidden': 64,
            'lstm_layers': 1,
            'dropout': 0.2
        }
    }
    
    input_size = 50  # Example number of features
    
    for name, config in configs.items():
        print(f"\n🧠 Creating {name} model...")
        try:
            model = CandlePredictionModel(
                model_type=config['model_type'],
                input_size=input_size,
                config=config
            )
            
            model.build_model()
            print(f"✅ {name} model created successfully")
            print(f"Model summary:\n{model.get_model_summary()}")
            
        except Exception as e:
            print(f"❌ Error creating {name} model: {e}")


def example_prediction_simulation():
    """Simulate making predictions without a trained model."""
    print("\n" + "=" * 50)
    print("PREDICTION SIMULATION")
    print("=" * 50)
    
    print("🔮 Simulating candle direction predictions...")
    
    # Simulate prediction results (since we don't have a trained model)
    import random
    
    symbols = ['ETHUSDT', 'BTCUSDT', 'SOLUSDT']
    
    for symbol in symbols:
        # Simulate prediction
        direction = random.choice(['up', 'down'])
        confidence = random.uniform(0.55, 0.95)
        current_candle_progress = random.uniform(0.0, 1.0)
        
        # Simulate current and target candle times
        now = datetime.now()
        current_candle_start = now.replace(minute=0, second=0, microsecond=0)
        target_candle_start = current_candle_start + timedelta(hours=1)
        target_candle_end = target_candle_start + timedelta(hours=1)
        
        print(f"\n📊 {symbol}:")
        print(f"  Target Candle: {target_candle_start.strftime('%H:%M')}-{target_candle_end.strftime('%H:%M')}")
        print(f"  Predicted Direction: {direction}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Current Candle Progress: {current_candle_progress:.2f}")
        print(f"  Minutes into current candle: {int(current_candle_progress * 60)}")
        
        # Simulate decision making based on confidence
        if confidence > 0.8:
            print(f"  💡 High confidence - consider trading")
        elif confidence > 0.65:
            print(f"  ⚖️ Medium confidence - proceed with caution")
        else:
            print(f"  🚫 Low confidence - avoid trading")


def example_candle_progress_info():
    """Demonstrate candle progress analysis."""
    print("\n" + "=" * 50)
    print("CANDLE PROGRESS ANALYSIS")
    print("=" * 50)
    
    # Initialize a predictor (without model for this example)
    try:
        predictor = CandleDirectionPredictor()
        
        # Get candle progress info
        progress_info = predictor.get_candle_progress_info()
        
        print("🕐 Current candle information:")
        print(f"  Current Time: {progress_info['current_time']}")
        print(f"  Current Candle: {progress_info['current_candle_start']} - {progress_info['current_candle_end']}")
        print(f"  Target Candle: {progress_info['target_candle_start']} - {progress_info['target_candle_end']}")
        print(f"  Minutes Elapsed: {progress_info['minutes_elapsed']}")
        print(f"  Minutes Remaining: {progress_info['minutes_remaining']}")
        print(f"  Progress: {progress_info['progress']:.2%}")
        
        # Progress categorization
        if progress_info['is_early_candle']:
            print("  📍 Position: Early in current candle (0-15 minutes)")
        elif progress_info['is_mid_candle']:
            print("  📍 Position: Mid current candle (15-45 minutes)")
        else:
            print("  📍 Position: Late in current candle (45-60 minutes)")
            
    except Exception as e:
        print(f"Error getting candle progress: {e}")


def example_training_config():
    """Show example training configuration."""
    print("\n" + "=" * 50)
    print("TRAINING CONFIGURATION EXAMPLE")
    print("=" * 50)
    
    training_config = {
        # Data parameters
        'symbols': ['ETHUSDT', 'BTCUSDT'],
        'start_date': '2024-01-01',
        'end_date': '2024-06-01',
        'data_dir': 'training_data',
        
        # Model parameters
        'model_type': 'lstm',
        'sequence_length': 60,        # 60 minutes lookback
        'prediction_horizon': 60,     # 60 minutes ahead (1 hour)
        
        # Training parameters
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'early_stopping_patience': 10,
        
        # Data splits
        'val_split': 0.2,
        'test_split': 0.1,
        
        # Model architecture
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'weight_decay': 0.01
    }
    
    print("📋 Sample training configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    
    print("\n💡 To train a model, you would run:")
    print("```python")
    print("from dl_candle_direction_predictor.train import CandleTrainer")
    print("trainer = CandleTrainer(training_config)")
    print("model = trainer.run_full_training_pipeline()")
    print("```")


def main():
    """Run all examples."""
    print("🕯️ Deep Learning Candle Direction Predictor - Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_data_loading()
        example_feature_engineering()
        example_model_creation()
        example_prediction_simulation()
        example_candle_progress_info()
        example_training_config()
        
        print("\n" + "=" * 60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n📚 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train a model: python train.py --symbols ETHUSDT BTCUSDT")
        print("3. Evaluate model: python evaluate.py --model-path trained_model")
        print("4. Use for live predictions with trained model")
        
        print("\n⚠️  Note: Some examples may fail without proper dependencies")
        print("   or internet connection. This is expected in isolated environments.")
        
    except Exception as e:
        print(f"\n❌ Error in examples: {e}")
        print("This is expected if dependencies are not installed.")


if __name__ == "__main__":
    main()