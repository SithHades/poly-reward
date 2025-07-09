"""
Deep learning model for crypto candle direction prediction.

Implements multiple model architectures including LSTM, Transformer, and hybrid models
for predicting 1-hour candle directions with confidence scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class LSTMCandlePredictor(nn.Module):
    """
    LSTM-based model for candle direction prediction.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMCandlePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # Binary classification: up/down
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Logits for binary classification
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(context_vector)
        
        return output


class TransformerCandlePredictor(nn.Module):
    """
    Transformer-based model for candle direction prediction.
    """
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dropout: float = 0.1):
        """
        Initialize Transformer model.
        
        Args:
            input_size: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super(TransformerCandlePredictor, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Logits for binary classification
        """
        # Project to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global pooling
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # Classification
        output = self.classifier(x)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class CNNLSTMCandlePredictor(nn.Module):
    """
    Hybrid CNN-LSTM model for candle direction prediction.
    """
    
    def __init__(self, input_size: int, cnn_filters: List[int] = [64, 32, 16],
                 lstm_hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.2):
        """
        Initialize CNN-LSTM model.
        
        Args:
            input_size: Number of input features
            cnn_filters: List of CNN filter sizes
            lstm_hidden: LSTM hidden size
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(CNNLSTMCandlePredictor, self).__init__()
        
        # CNN feature extraction
        self.conv1d_layers = nn.ModuleList()
        in_channels = input_size
        
        for out_channels in cnn_filters:
            self.conv1d_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Logits for binary classification
        """
        # Transpose for Conv1d: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        for conv_layer in self.conv1d_layers:
            x = conv_layer(x)
        
        # Transpose back for LSTM: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Use last hidden state
        output = self.classifier(h_n[-1])
        
        return output


class CandlePredictionModel:
    """
    Main model class that wraps the PyTorch models and provides training/inference interface.
    """
    
    def __init__(self, model_type: str = 'lstm', input_size: int = None, 
                 config: Optional[Dict] = None, device: Optional[str] = None):
        """
        Initialize the prediction model.
        
        Args:
            model_type: Type of model ('lstm', 'transformer', 'cnn_lstm')
            input_size: Number of input features
            config: Model configuration dictionary
            device: Device to run the model on
        """
        self.model_type = model_type
        self.input_size = input_size
        self.config = config or {}
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        logger.info(f"Initialized {model_type} model on {self.device}")
    
    def build_model(self) -> None:
        """Build the PyTorch model based on configuration."""
        if self.input_size is None:
            raise ValueError("input_size must be specified to build model")
        
        if self.model_type == 'lstm':
            self.model = LSTMCandlePredictor(
                input_size=self.input_size,
                hidden_size=self.config.get('hidden_size', 128),
                num_layers=self.config.get('num_layers', 2),
                dropout=self.config.get('dropout', 0.2)
            )
        elif self.model_type == 'transformer':
            self.model = TransformerCandlePredictor(
                input_size=self.input_size,
                d_model=self.config.get('d_model', 128),
                nhead=self.config.get('nhead', 8),
                num_layers=self.config.get('num_layers', 4),
                dropout=self.config.get('dropout', 0.1)
            )
        elif self.model_type == 'cnn_lstm':
            self.model = CNNLSTMCandlePredictor(
                input_size=self.input_size,
                cnn_filters=self.config.get('cnn_filters', [64, 32, 16]),
                lstm_hidden=self.config.get('lstm_hidden', 128),
                lstm_layers=self.config.get('lstm_layers', 2),
                dropout=self.config.get('dropout', 0.2)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"Built {self.model_type} model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> Tuple[float, float]:
        """
        Single training step.
        
        Args:
            X_batch: Input batch
            y_batch: Target batch
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            logits = self.model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
        
        # Backward pass
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y_batch).float().mean()
        
        return loss.item(), accuracy.item()
    
    def validate_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> Tuple[float, float]:
        """
        Single validation step.
        
        Args:
            X_batch: Input batch
            y_batch: Target batch
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y_batch).float().mean()
        
        return loss.item(), accuracy.item()
    
    def predict(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Input tensor
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(X)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
        
        return predictions.cpu().numpy(), confidence_scores.cpu().numpy()
    
    def predict_direction_and_confidence(self, X: torch.Tensor) -> Tuple[str, float]:
        """
        Predict direction and confidence for a single sample.
        
        Args:
            X: Input tensor (1, sequence_length, features)
            
        Returns:
            Tuple of (direction, confidence)
        """
        predictions, confidences = self.predict(X)
        
        direction = 'up' if predictions[0] == 1 else 'down'
        confidence = float(confidences[0])
        
        return direction, confidence
    
    def save_model(self, path: str) -> None:
        """
        Save the model and configuration.
        
        Args:
            path: Path to save the model
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'history': self.history
        }, save_path / 'model.pth')
        
        # Save configuration
        config_to_save = {
            'model_type': self.model_type,
            'input_size': self.input_size,
            'config': self.config,
            'device': str(self.device)
        }
        
        with open(save_path / 'config.json', 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str) -> None:
        """
        Load the model and configuration.
        
        Args:
            path: Path to load the model from
        """
        load_path = Path(path)
        
        # Load configuration
        with open(load_path / 'config.json', 'r') as f:
            saved_config = json.load(f)
        
        self.model_type = saved_config['model_type']
        self.input_size = saved_config['input_size']
        self.config = saved_config['config']
        
        # Build model
        self.build_model()
        
        # Load model state
        checkpoint = torch.load(load_path / 'model.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.history = checkpoint['history']
        
        logger.info(f"Model loaded from {load_path}")
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        if self.model is None:
            return "Model not built yet"
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = f"""
Model Type: {self.model_type}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Input Size: {self.input_size}
Device: {self.device}
Configuration: {self.config}
        """
        
        return summary.strip()