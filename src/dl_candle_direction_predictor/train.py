"""
Training script for crypto candle direction prediction models.

Handles data preparation, feature engineering, model training, and validation
with comprehensive logging and model checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
import argparse
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .model import CandlePredictionModel
from .data_loader import BinanceDataLoader
from .feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandleTrainer:
    """
    Comprehensive trainer for candle direction prediction models.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.data_loader = BinanceDataLoader(config.get('data_dir', 'data'))
        self.feature_engineer = FeatureEngineer()
        
        # Model configuration
        self.model = None
        self.scaler = StandardScaler()
        
        # Training parameters
        self.sequence_length = config.get('sequence_length', 60)
        self.prediction_horizon = config.get('prediction_horizon', 60)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.num_epochs = config.get('num_epochs', 100)
        self.early_stopping_patience = config.get('early_stopping_patience', 15)
        
        # Data parameters
        self.symbols = config.get('symbols', ['ETHUSDT', 'BTCUSDT', 'SOLUSDT'])
        self.start_date = config.get('start_date', '2023-01-01')
        self.end_date = config.get('end_date', '2024-01-01')
        self.val_split = config.get('val_split', 0.2)
        self.test_split = config.get('test_split', 0.1)
        
        # Output paths
        self.output_dir = Path(config.get('output_dir', 'models'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized trainer with config: {config}")
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data by downloading, processing, and engineering features.
        
        Returns:
            Tuple of (features, labels, feature_names)
        """
        logger.info("Starting data preparation...")
        
        all_features = []
        all_labels = []
        
        for symbol in self.symbols:
            logger.info(f"Processing {symbol}...")
            
            try:
                # Get historical data
                data_1m = self.data_loader.get_historical_data(
                    symbol=symbol,
                    timeframe='1m',
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                
                # Get multi-timeframe data
                data_5m = self.data_loader.get_historical_data(symbol, '5m', self.start_date, self.end_date)
                data_15m = self.data_loader.get_historical_data(symbol, '15m', self.start_date, self.end_date)
                data_1h = self.data_loader.get_historical_data(symbol, '1h', self.start_date, self.end_date)
                
                multi_tf_data = {
                    '1m': data_1m,
                    '5m': data_5m,
                    '15m': data_15m,
                    '1h': data_1h
                }
                
                # Engineer features
                features = self.feature_engineer.engineer_features(
                    df=data_1m,
                    multi_tf_data=multi_tf_data
                )
                
                # Create labels (for proper 1-hour candles)
                labels = self.feature_engineer.create_labels(data_1m)
                
                # Align features and labels
                min_length = min(len(features), len(labels))
                features = features.iloc[:min_length]
                labels = labels.iloc[:min_length]
                
                # Remove rows with NaN values
                valid_indices = ~(features.isnull().any(axis=1) | labels.isnull())
                features = features[valid_indices]
                labels = labels[valid_indices]
                
                if len(features) > 0:
                    all_features.append(features)
                    all_labels.append(labels)
                    logger.info(f"Processed {len(features)} samples for {symbol}")
                else:
                    logger.warning(f"No valid data for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid data could be processed for any symbol")
        
        # Combine all data
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_labels = pd.concat(all_labels, ignore_index=True)
        
        # Get feature names
        feature_names = self.feature_engineer.get_feature_columns()
        
        logger.info(f"Data preparation complete. Total samples: {len(combined_features)}")
        logger.info(f"Features: {len(feature_names)}")
        logger.info(f"Class distribution: {combined_labels.value_counts().to_dict()}")
        
        return combined_features, combined_labels, feature_names
    
    def create_sequences(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling.
        
        Args:
            features: Feature DataFrame
            labels: Labels Series
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        logger.info(f"Creating sequences with length {self.sequence_length}...")
        
        # Get only feature columns
        feature_columns = self.feature_engineer.get_feature_columns()
        feature_data = features[feature_columns]
        
        # Normalize features
        feature_data_scaled = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(feature_data_scaled)):
            # Get sequence of features
            sequence = feature_data_scaled[i-self.sequence_length:i]
            X_sequences.append(sequence)
            
            # Get corresponding label
            y_sequences.append(labels.iloc[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        logger.info(f"Created {len(X_sequences)} sequences")
        logger.info(f"Sequence shape: {X_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting data...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_split, random_state=42, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = self.val_split / (1 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Validation: {len(X_val)} samples")
        logger.info(f"  Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_loaders(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch data loaders.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                   input_size: int) -> CandlePredictionModel:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            input_size: Number of input features
            
        Returns:
            Trained model
        """
        logger.info("Starting model training...")
        
        # Initialize model
        model_config = {
            'hidden_size': self.config.get('hidden_size', 128),
            'num_layers': self.config.get('num_layers', 2),
            'dropout': self.config.get('dropout', 0.2),
            'learning_rate': self.learning_rate,
            'weight_decay': self.config.get('weight_decay', 0.01)
        }
        
        self.model = CandlePredictionModel(
            model_type=self.config.get('model_type', 'lstm'),
            input_size=input_size,
            config=model_config
        )
        
        self.model.build_model()
        
        logger.info(f"Model summary:\n{self.model.get_model_summary()}")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Training phase
            train_losses = []
            train_accuracies = []
            
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch = X_batch.to(self.model.device)
                y_batch = y_batch.to(self.model.device)
                
                loss, accuracy = self.model.train_step(X_batch, y_batch)
                train_losses.append(loss)
                train_accuracies.append(accuracy)
            
            # Validation phase
            val_losses = []
            val_accuracies = []
            
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.model.device)
                y_batch = y_batch.to(self.model.device)
                
                loss, accuracy = self.model.validate_step(X_batch, y_batch)
                val_losses.append(loss)
                val_accuracies.append(accuracy)
            
            # Calculate averages
            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accuracies)
            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accuracies)
            
            # Update history
            self.model.history['train_loss'].append(avg_train_loss)
            self.model.history['train_acc'].append(avg_train_acc)
            self.model.history['val_loss'].append(avg_val_loss)
            self.model.history['val_acc'].append(avg_val_acc)
            
            # Learning rate scheduling
            self.model.scheduler.step(avg_val_loss)
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
            logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                self.model.save_model(self.output_dir / 'best_model')
                logger.info("Saved new best model")
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_model(self.output_dir / 'best_model')
        
        logger.info("Training completed!")
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model on test data...")
        
        # Convert to tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        # Make predictions
        predictions, confidences = self.model.predict(X_test_tensor)
        
        # Calculate metrics
        test_accuracy = (predictions == y_test).mean()
        
        # Classification report
        class_report = classification_report(
            y_test, predictions,
            target_names=['Down', 'Up'],
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, predictions)
        
        evaluation_results = {
            'test_accuracy': float(test_accuracy),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'mean_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences))
        }
        
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Mean Confidence: {np.mean(confidences):.4f}")
        
        return evaluation_results
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        if self.model is None or not self.model.history['train_loss']:
            logger.warning("No training history to plot")
            return
        
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.model.history['train_loss'], label='Train Loss')
        ax1.plot(self.model.history['val_loss'], label='Val Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(self.model.history['train_acc'], label='Train Acc')
        ax2.plot(self.model.history['val_acc'], label='Val Acc')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300)
        plt.show()
        
        logger.info(f"Training history plot saved to {self.output_dir / 'training_history.png'}")
    
    def save_training_results(self, evaluation_results: Dict) -> None:
        """
        Save training configuration and results.
        
        Args:
            evaluation_results: Results from model evaluation
        """
        results = {
            'config': self.config,
            'feature_names': self.feature_engineer.get_feature_columns(),
            'evaluation_results': evaluation_results,
            'training_completed': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {self.output_dir / 'training_results.json'}")
    
    def run_full_training_pipeline(self) -> CandlePredictionModel:
        """
        Run the complete training pipeline.
        
        Returns:
            Trained model
        """
        logger.info("Starting full training pipeline...")
        
        # 1. Prepare data
        features, labels, feature_names = self.prepare_data()
        
        # 2. Create sequences
        X_sequences, y_sequences = self.create_sequences(features, labels)
        
        # 3. Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_sequences, y_sequences)
        
        # 4. Create data loaders
        train_loader, val_loader = self.create_data_loaders(X_train, y_train, X_val, y_val)
        
        # 5. Train model
        input_size = X_sequences.shape[2]  # Number of features
        trained_model = self.train_model(train_loader, val_loader, input_size)
        
        # 6. Evaluate model
        evaluation_results = self.evaluate_model(X_test, y_test)
        
        # 7. Plot training history
        self.plot_training_history()
        
        # 8. Save results
        self.save_training_results(evaluation_results)
        
        logger.info("Full training pipeline completed successfully!")
        return trained_model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train crypto candle direction prediction model')
    
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--model-type', type=str, default='lstm',
                       choices=['lstm', 'transformer', 'cnn_lstm'],
                       help='Model architecture to use')
    parser.add_argument('--symbols', nargs='+', default=['ETHUSDT', 'BTCUSDT'],
                       help='Crypto symbols to train on')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for models and results')
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override config with command line arguments
    config.update({
        'model_type': args.model_type,
        'symbols': args.symbols,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'output_dir': args.output_dir
    })
    
    # Initialize trainer
    trainer = CandleTrainer(config)
    
    # Run training
    trained_model = trainer.run_full_training_pipeline()
    
    logger.info("Training script completed successfully!")


if __name__ == '__main__':
    main()