"""
Evaluation and backtesting script for crypto candle direction prediction models.

Provides comprehensive evaluation including historical backtesting,
live simulation, and performance analysis.
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from .predictor import CandleDirectionPredictor
from .data_loader import BinanceDataLoader
from .feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandleEvaluator:
    """
    Comprehensive evaluator for candle direction prediction models.
    """
    
    def __init__(self, model_path: str, config: Optional[Dict] = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model
            config: Evaluation configuration
        """
        self.model_path = model_path
        self.config = config or {}
        
        # Initialize predictor
        self.predictor = CandleDirectionPredictor(model_path)
        
        # Evaluation parameters
        self.symbols = self.config.get('symbols', ['ETHUSDT', 'BTCUSDT', 'SOLUSDT'])
        self.prediction_intervals = self.config.get('prediction_intervals', [5, 10, 15, 30, 45])
        
        logger.info(f"Initialized evaluator with model: {model_path}")
    
    def historical_backtest(self, symbol: str, start_date: str, end_date: str,
                           prediction_interval: int = 5) -> pd.DataFrame:
        """
        Run historical backtest for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for backtest
            end_date: End date for backtest
            prediction_interval: Minutes between predictions
            
        Returns:
            DataFrame with backtest results
        """
        logger.info(f"Running historical backtest for {symbol} from {start_date} to {end_date}")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Run simulation
        results_df = self.predictor.simulate_real_time_predictions(
            symbol=symbol,
            start_time=start_dt,
            end_time=end_dt,
            interval_minutes=prediction_interval
        )
        
        if results_df.empty:
            logger.warning(f"No results for {symbol} backtest")
            return results_df
        
        # Add actual outcomes for evaluation
        results_df = self._add_actual_outcomes(results_df, symbol)
        
        # Calculate performance metrics
        results_df = self._calculate_performance_metrics(results_df)
        
        logger.info(f"Backtest completed for {symbol}. {len(results_df)} predictions made.")
        
        return results_df
    
    def _add_actual_outcomes(self, results_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add actual candle outcomes to backtest results.
        
        Args:
            results_df: Results DataFrame from simulation
            symbol: Trading symbol
            
        Returns:
            DataFrame with actual outcomes added
        """
        if results_df.empty:
            return results_df
        
        # Get start and end times
        start_time = pd.to_datetime(results_df['current_time'].min())
        end_time = pd.to_datetime(results_df['current_time'].max()) + timedelta(hours=2)
        
        try:
            # Get historical 1-hour data to verify actual outcomes
            hourly_data = self.predictor.data_loader.get_historical_data(
                symbol=symbol,
                timeframe='1h',
                start_date=start_time,
                end_date=end_time
            )
            
            actual_outcomes = []
            
            for _, row in results_df.iterrows():
                current_time = pd.to_datetime(row['current_time'])
                
                # Find the target proper 1-hour candle that this prediction is for
                # The target is always the next proper hour after the current time
                current_candle_start = current_time.replace(minute=0, second=0, microsecond=0)
                target_candle_start = current_candle_start + timedelta(hours=1)
                
                # Get the actual 1-hour candle data for the target candle
                candle_data = hourly_data[hourly_data.index == target_candle_start]
                
                if len(candle_data) > 0:
                    candle = candle_data.iloc[0]
                    # Actual direction: up if close > open, down otherwise
                    actual_direction = 'up' if candle['close'] > candle['open'] else 'down'
                    actual_outcomes.append(actual_direction)
                else:
                    actual_outcomes.append(None)
            
            results_df['actual_direction'] = actual_outcomes
            
            # Calculate prediction accuracy
            valid_predictions = results_df['actual_direction'].notna()
            results_df['prediction_correct'] = (
                results_df['direction'] == results_df['actual_direction']
            ).astype(float)
            
            # Only set correct for valid predictions
            results_df.loc[~valid_predictions, 'prediction_correct'] = None
            
        except Exception as e:
            logger.error(f"Failed to add actual outcomes: {e}")
            results_df['actual_direction'] = None
            results_df['prediction_correct'] = None
        
        return results_df
    
    def _calculate_performance_metrics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance metrics for backtest results.
        
        Args:
            results_df: Results DataFrame with actual outcomes
            
        Returns:
            DataFrame with performance metrics added
        """
        if 'prediction_correct' not in results_df.columns:
            return results_df
        
        # Rolling accuracy (window of 20 predictions)
        results_df['rolling_accuracy'] = (
            results_df['prediction_correct']
            .rolling(window=20, min_periods=5)
            .mean()
        )
        
        # Confidence-weighted accuracy
        results_df['confidence_weighted_correct'] = (
            results_df['prediction_correct'] * results_df['confidence']
        )
        
        # Cumulative accuracy
        results_df['cumulative_accuracy'] = (
            results_df['prediction_correct']
            .expanding()
            .mean()
        )
        
        return results_df
    
    def evaluate_multiple_symbols(self, symbols: Optional[List[str]] = None,
                                 start_date: str = '2024-01-01',
                                 end_date: str = '2024-02-01',
                                 prediction_interval: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Evaluate multiple symbols.
        
        Args:
            symbols: List of symbols to evaluate
            start_date: Start date for evaluation
            end_date: End date for evaluation
            prediction_interval: Minutes between predictions
            
        Returns:
            Dictionary mapping symbol to results DataFrame
        """
        symbols = symbols or self.symbols
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.historical_backtest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    prediction_interval=prediction_interval
                )
            except Exception as e:
                logger.error(f"Failed to evaluate {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        
        return results
    
    def analyze_prediction_accuracy_by_confidence(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze prediction accuracy by confidence levels.
        
        Args:
            results_df: Results DataFrame with predictions and outcomes
            
        Returns:
            Dictionary with confidence analysis
        """
        if results_df.empty or 'prediction_correct' not in results_df.columns:
            return {}
        
        # Remove rows with missing actual outcomes
        valid_results = results_df.dropna(subset=['prediction_correct'])
        
        if valid_results.empty:
            return {}
        
        # Define confidence bins
        confidence_bins = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_labels = ['0.0-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        
        valid_results['confidence_bin'] = pd.cut(
            valid_results['confidence'],
            bins=confidence_bins,
            labels=bin_labels,
            include_lowest=True
        )
        
        # Calculate metrics by confidence bin
        confidence_analysis = {}
        
        for bin_label in bin_labels:
            bin_data = valid_results[valid_results['confidence_bin'] == bin_label]
            
            if len(bin_data) > 0:
                confidence_analysis[bin_label] = {
                    'count': len(bin_data),
                    'accuracy': bin_data['prediction_correct'].mean(),
                    'mean_confidence': bin_data['confidence'].mean(),
                    'std_confidence': bin_data['confidence'].std()
                }
        
        return confidence_analysis
    
    def analyze_prediction_accuracy_by_candle_progress(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze prediction accuracy by position within the 1-hour candle.
        
        Args:
            results_df: Results DataFrame with predictions and outcomes
            
        Returns:
            Dictionary with candle progress analysis
        """
        if results_df.empty or 'prediction_correct' not in results_df.columns:
            return {}
        
        # Remove rows with missing actual outcomes
        valid_results = results_df.dropna(subset=['prediction_correct'])
        
        if valid_results.empty:
            return {}
        
        # Define candle progress bins
        progress_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
        bin_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
        
        # Use the correct field name for current candle progress
        progress_field = 'current_candle_progress' if 'current_candle_progress' in valid_results.columns else 'candle_progress'
        
        valid_results['progress_bin'] = pd.cut(
            valid_results[progress_field],
            bins=progress_bins,
            labels=bin_labels,
            include_lowest=True
        )
        
        # Calculate metrics by progress bin
        progress_analysis = {}
        
        for bin_label in bin_labels:
            bin_data = valid_results[valid_results['progress_bin'] == bin_label]
            
            if len(bin_data) > 0:
                progress_analysis[bin_label] = {
                    'count': len(bin_data),
                    'accuracy': bin_data['prediction_correct'].mean(),
                    'mean_confidence': bin_data['confidence'].mean(),
                    'std_confidence': bin_data['confidence'].std()
                }
        
        return progress_analysis
    
    def generate_performance_report(self, results: Dict[str, pd.DataFrame]) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            results: Dictionary mapping symbol to results DataFrame
            
        Returns:
            Dictionary with performance report
        """
        report = {
            'overall_metrics': {},
            'by_symbol': {},
            'confidence_analysis': {},
            'candle_progress_analysis': {}
        }
        
        # Combine all results
        all_results = []
        for symbol, df in results.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy['symbol'] = symbol
                all_results.append(df_copy)
        
        if not all_results:
            logger.warning("No valid results to analyze")
            return report
        
        combined_results = pd.concat(all_results, ignore_index=True)
        valid_combined = combined_results.dropna(subset=['prediction_correct'])
        
        if valid_combined.empty:
            logger.warning("No valid predictions with actual outcomes")
            return report
        
        # Overall metrics
        report['overall_metrics'] = {
            'total_predictions': len(valid_combined),
            'overall_accuracy': valid_combined['prediction_correct'].mean(),
            'mean_confidence': valid_combined['confidence'].mean(),
            'confidence_std': valid_combined['confidence'].std(),
            'up_predictions': (valid_combined['direction'] == 'up').sum(),
            'down_predictions': (valid_combined['direction'] == 'down').sum()
        }
        
        # By symbol analysis
        for symbol, df in results.items():
            if not df.empty:
                valid_df = df.dropna(subset=['prediction_correct'])
                if not valid_df.empty:
                    report['by_symbol'][symbol] = {
                        'predictions': len(valid_df),
                        'accuracy': valid_df['prediction_correct'].mean(),
                        'mean_confidence': valid_df['confidence'].mean(),
                        'up_accuracy': valid_df[valid_df['direction'] == 'up']['prediction_correct'].mean() if (valid_df['direction'] == 'up').any() else None,
                        'down_accuracy': valid_df[valid_df['direction'] == 'down']['prediction_correct'].mean() if (valid_df['direction'] == 'down').any() else None
                    }
        
        # Confidence analysis
        report['confidence_analysis'] = self.analyze_prediction_accuracy_by_confidence(combined_results)
        
        # Candle progress analysis
        report['candle_progress_analysis'] = self.analyze_prediction_accuracy_by_candle_progress(combined_results)
        
        return report
    
    def plot_performance_analysis(self, results: Dict[str, pd.DataFrame], 
                                 output_dir: Optional[str] = None) -> None:
        """
        Create performance analysis plots.
        
        Args:
            results: Dictionary mapping symbol to results DataFrame
            output_dir: Directory to save plots
        """
        output_path = None
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Combine all results
        all_results = []
        for symbol, df in results.items():
            if not df.empty and 'prediction_correct' in df.columns:
                df_copy = df.copy()
                df_copy['symbol'] = symbol
                all_results.append(df_copy)
        
        if not all_results:
            logger.warning("No valid results to plot")
            return
        
        combined_results = pd.concat(all_results, ignore_index=True)
        valid_results = combined_results.dropna(subset=['prediction_correct'])
        
        if valid_results.empty:
            logger.warning("No valid predictions to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Accuracy by confidence
        if 'confidence' in valid_results.columns:
            confidence_bins = pd.cut(valid_results['confidence'], bins=10)
            accuracy_by_conf = valid_results.groupby(confidence_bins)['prediction_correct'].mean()
            
            axes[0, 0].bar(range(len(accuracy_by_conf)), accuracy_by_conf.values)
            axes[0, 0].set_title('Accuracy by Confidence Level')
            axes[0, 0].set_xlabel('Confidence Bins')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_xticks(range(len(accuracy_by_conf)))
            axes[0, 0].set_xticklabels([f'{interval.left:.2f}-{interval.right:.2f}' 
                                       for interval in accuracy_by_conf.index], rotation=45)
        
        # 2. Accuracy by candle progress
        progress_field = 'current_candle_progress' if 'current_candle_progress' in valid_results.columns else 'candle_progress'
        if progress_field in valid_results.columns:
            progress_bins = pd.cut(valid_results[progress_field], bins=10)
            accuracy_by_progress = valid_results.groupby(progress_bins)['prediction_correct'].mean()
            
            axes[0, 1].plot(range(len(accuracy_by_progress)), accuracy_by_progress.values, marker='o')
            axes[0, 1].set_title('Accuracy by Current Candle Progress')
            axes[0, 1].set_xlabel('Current Candle Progress')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].grid(True)
        
        # 3. Rolling accuracy over time
        if 'current_time' in valid_results.columns:
            valid_results['current_time'] = pd.to_datetime(valid_results['current_time'])
            time_sorted = valid_results.sort_values('current_time')
            time_sorted['rolling_acc'] = time_sorted['prediction_correct'].rolling(window=50, min_periods=10).mean()
            
            axes[1, 0].plot(time_sorted['current_time'], time_sorted['rolling_acc'])
            axes[1, 0].set_title('Rolling Accuracy Over Time (50-prediction window)')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Rolling Accuracy')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Accuracy by symbol
        symbol_accuracy = valid_results.groupby('symbol')['prediction_correct'].mean()
        axes[1, 1].bar(symbol_accuracy.index, symbol_accuracy.values)
        axes[1, 1].set_title('Accuracy by Symbol')
        axes[1, 1].set_xlabel('Symbol')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_dir and output_path is not None:
            plt.savefig(str(output_path / 'performance_analysis.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
        
        logger.info("Performance analysis plots created")
    
    def save_evaluation_results(self, results: Dict[str, pd.DataFrame],
                               report: Dict, output_dir: str) -> None:
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results
            report: Performance report
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results for each symbol
        for symbol, df in results.items():
            if not df.empty:
                df.to_csv(output_path / f'{symbol}_evaluation_results.csv', index=False)
        
        # Save performance report
        with open(output_path / 'performance_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate crypto candle direction prediction model')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--symbols', nargs='+', default=['ETHUSDT', 'BTCUSDT'],
                       help='Crypto symbols to evaluate')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='Start date for evaluation (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-02-01',
                       help='End date for evaluation (YYYY-MM-DD)')
    parser.add_argument('--prediction-interval', type=int, default=5,
                       help='Minutes between predictions')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'symbols': args.symbols,
        'prediction_intervals': [args.prediction_interval]
    }
    
    # Initialize evaluator
    evaluator = CandleEvaluator(args.model_path, config)
    
    # Run evaluation
    logger.info("Starting model evaluation...")
    
    results = evaluator.evaluate_multiple_symbols(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        prediction_interval=args.prediction_interval
    )
    
    # Generate performance report
    report = evaluator.generate_performance_report(results)
    
    # Create plots
    evaluator.plot_performance_analysis(results, args.output_dir)
    
    # Save results
    evaluator.save_evaluation_results(results, report, args.output_dir)
    
    # Print summary
    logger.info("Evaluation completed!")
    logger.info(f"Overall accuracy: {report['overall_metrics'].get('overall_accuracy', 'N/A'):.4f}")
    logger.info(f"Total predictions: {report['overall_metrics'].get('total_predictions', 'N/A')}")
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall Accuracy: {report['overall_metrics'].get('overall_accuracy', 'N/A'):.4f}")
    print(f"Mean Confidence: {report['overall_metrics'].get('mean_confidence', 'N/A'):.4f}")
    print(f"Total Predictions: {report['overall_metrics'].get('total_predictions', 'N/A')}")
    
    print("\nAccuracy by Symbol:")
    for symbol, metrics in report['by_symbol'].items():
        print(f"  {symbol}: {metrics['accuracy']:.4f} ({metrics['predictions']} predictions)")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()