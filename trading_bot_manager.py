#!/usr/bin/env python3
"""
Trading Bot Manager

Utility script for managing the ETH Polymarket Trading Bot with additional features:
- Configuration management
- Bot status monitoring  
- Manual prediction testing
- Performance reporting
- Risk monitoring
"""

import argparse
import asyncio
import logging
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Literal

from constants import MARKETS
from polymarket_hourly_trading_bot import PolymarketHourlyTradingBot, TradingConfig
from parsing_utils import ET, extract_datetime_from_slug, get_current_market_slug, get_next_market_slug
from src.eth_candle_predictor import EthCandlePredictor
from src.polymarket_client import PolymarketClient


class TradingBotManager:
    """Manager for the ETH Polymarket Trading Bot"""
    
    def __init__(self, config_path: str = "config/trading_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.bot = None
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger("BotManager")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing config file: {e}")
            return {}
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        
        level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('file', 'trading_bot.log')
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def create_trading_config(self) -> TradingConfig:
        """Create TradingConfig from YAML configuration"""
        model_config = self.config.get('model', {})
        trading_config = self.config.get('trading', {})
        markets_config = self.config.get('markets', {})
        data_config = self.config.get('data', {})
        
        return TradingConfig(
            model_type=model_config.get('type', 'logistic'),
            confidence_threshold=model_config.get('confidence_threshold', 0.75),
            max_position_size=trading_config.get('max_position_size', 5.0),
            max_daily_trades=trading_config.get('max_daily_trades', 24),
            order_timeout_minutes=trading_config.get('order_timeout_minutes', 30),
            ethusdt_series_slug=markets_config.get('ethusdt_series_slug', 'ethusdt'),
            min_liquidity=markets_config.get('min_liquidity', 100.0),
            max_markets_to_check=markets_config.get('max_markets_to_check', 12),
            data_refresh_interval=data_config.get('refresh_interval_seconds', 10),
            prediction_window_start=data_config.get('prediction_window', {}).get('start_minute', 43),
            prediction_window_end=data_config.get('prediction_window', {}).get('end_minute', 47)
        )
    
    async def start_bot(self):
        """Start the trading bot"""
        self.logger.info("Starting ETH Polymarket Trading Bot...")
        
        trading_config = self.create_trading_config()
        self.bot = PolymarketHourlyTradingBot(trading_config)
        
        try:
            await self.bot.run()
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        except Exception as e:
            self.logger.error(f"Bot error: {e}")
    
    async def test_prediction(self):
        """Test the prediction model with current data"""
        self.logger.info("Testing prediction model...")
        
        try:
            # Create predictor
            model_config = self.config.get('model', {})
            predictor = EthCandlePredictor(model_type=model_config.get('type', 'logistic'))
            
            # Create minimal bot to get data
            trading_config = self.create_trading_config()
            bot = PolymarketHourlyTradingBot(trading_config)
            
            # Load data and train model
            await bot.load_historical_data()
            bot.train_model()
            
            # Get current hour
            current_time = datetime.now(timezone.utc)
            current_hour = current_time.replace(minute=0, second=0, microsecond=0)
            
            # Make prediction
            prediction = bot.predictor.predict(bot.ohlcv_data, current_hour)
            
            print(f"\n=== ETH Candle Prediction Test ===")
            print(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"Prediction Hour: {current_hour.strftime('%Y-%m-%d %H:00 UTC')}")
            print(f"Predicted Direction: {prediction.predicted_direction.upper()}")
            print(f"Confidence: {prediction.confidence:.1%}")
            print(f"Probability Green: {prediction.probability_green:.1%}")
            print(f"Probability Red: {prediction.probability_red:.1%}")
            print(f"Model: {prediction.model_name}")
            
            # Check if confidence meets threshold
            threshold = model_config.get('confidence_threshold', 0.75)
            print(f"\nTrading Decision:")
            if prediction.confidence >= threshold:
                print(f"✅ TRADE - Confidence {prediction.confidence:.1%} >= {threshold:.1%}")
            else:
                print(f"❌ NO TRADE - Confidence {prediction.confidence:.1%} < {threshold:.1%}")
            
        except Exception as e:
            self.logger.error(f"Error testing prediction: {e}")
    
    async def check_markets(self, market_slug: MARKETS):
        """Check available Polymarket markets"""
        self.logger.info(f"Checking {market_slug} Polymarket markets...")
        
        try:            
            client = PolymarketClient()

            current_market_slug = get_current_market_slug(market_slug)
            next_market_slug = get_next_market_slug(market_slug)
            markets = [client.get_market_by_slug(current_market_slug), client.get_market_by_slug(next_market_slug)]

            print(f"\n=== {market_slug} Hourly Markets ===")
            print(f"{market_slug} markets found: {len(markets)}")
            
            current_time = datetime.now(timezone.utc)
            
            print(f"\nActive {market_slug} Markets:")
            for market in markets:                
                print(f"    Ends: {market.end_date_iso.strftime('%Y-%m-%d %H:%M:%S') if market.end_date_iso else 'N/A'}")
                print(f"    Tokens: {market.tokens}")
                print(f"    Accepting Orders: {market.accepting_orders}")
                print(f"    Active: {market.active}")
                print(f"    Closed: {market.closed}")
                print(f"    Minimum Order Size: {market.minimum_order_size}")
                print(f"    Minimum Tick Size: {market.minimum_tick_size}")

                
        except Exception as e:
            self.logger.error(f"Error checking {market_slug} markets: {e}")
    
    async def check_positions(self):
        """Check current positions and orders"""
        self.logger.info("Checking current positions...")
        
        try:
            client = PolymarketClient()
            positions = client.get_positions()
            orders = client.get_orders()
            
            print(f"\n=== Current Positions ===")
            print(f"Open orders: {len(orders)}")
            
            total_reserved_exposure = 0
            for order in orders:
                exposure = float(order.price) * float(order.original_size)
                total_reserved_exposure += exposure
                
                print(f"\nOrder ID: {order.order_id}")
                print(f"Market: {order.market_id}")
                print(f"Side: {order.side}")
                print(f"Price: ${order.price:.3f}")
                print(f"Size: {order.original_size}")
                print(f"Exposure: ${exposure:.2f}")
                print(f"Status: {order.status}")
            
            print(f"\nTotal Reserved Exposure: ${total_reserved_exposure:.2f}")
            
            print(f"\n=== Current Positions ===")
            print(f"Positions: {len(positions)}")
            total_exposure = 0
            for position in positions:
                print(f"Token ID: {position.token_id}")
                print(f"Size: {position.size}")
                print(f"Avg Price: {position.avg_price}")
                print(f"Current Price: {position.current_price}")
                exposure = float(position.current_price) * float(position.size)
                total_exposure += exposure

            # Check against risk limits
            risk_config = self.config.get('risk', {})
            max_exposure = risk_config.get('max_total_exposure', 200.0)
            
            if (total_reserved_exposure + total_exposure) > max_exposure:
                print(f"⚠️  WARNING: Total reserved exposure ${total_reserved_exposure:.2f} + total exposure ${total_exposure:.2f} exceeds limit ${max_exposure:.2f}")
            else:
                print(f"✅ Risk OK: Total exposure within limits")
                
        except Exception as e:
            self.logger.error(f"Error checking positions: {e}")
    
    def validate_config(self):
        """Validate configuration"""
        print(f"\n=== Configuration Validation ===")
        
        required_sections = ['model', 'trading', 'markets', 'data', 'risk', 'logging']
        for section in required_sections:
            if section in self.config:
                print(f"✅ {section}: OK")
            else:
                print(f"❌ {section}: Missing")
        
        # Check model settings
        model_config = self.config.get('model', {})
        if model_config.get('type') in ['logistic', 'random_forest', 'gradient_boost']:
            print(f"✅ Model type: {model_config.get('type')}")
        else:
            print(f"❌ Invalid model type: {model_config.get('type')}")
        
        # Check confidence threshold
        threshold = model_config.get('confidence_threshold', 0)
        if 0.5 <= threshold <= 1.0:
            print(f"✅ Confidence threshold: {threshold:.1%}")
        else:
            print(f"❌ Invalid confidence threshold: {threshold}")
        
        print(f"\nConfiguration file: {self.config_path}")
        print(f"Config loaded successfully: {bool(self.config)}")


async def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="ETH Polymarket Trading Bot Manager")
    parser.add_argument('--config', '-c', default='config/trading_config.yaml',
                       help='Path to configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start bot command
    subparsers.add_parser('start', help='Start the trading bot')
    
    # Test commands
    subparsers.add_parser('test-prediction', help='Test prediction model with current data')
    
    # Add check-markets command with slug argument
    check_markets_parser = subparsers.add_parser('check-markets', help='Check available Polymarket markets')
    check_markets_parser.add_argument('--slug', type=str, help='Market slug to check')
    
    subparsers.add_parser('check-positions', help='Check current positions and orders')
    subparsers.add_parser('validate-config', help='Validate configuration file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create manager
    manager = TradingBotManager(args.config)
    
    # Execute command
    if args.command == 'start':
        await manager.start_bot()
    elif args.command == 'test-prediction':
        await manager.test_prediction()
    elif args.command == 'check-markets':
        await manager.check_markets()
    elif args.command == 'check-positions':
        await manager.check_positions()
    elif args.command == 'validate-config':
        manager.validate_config()


if __name__ == "__main__":
    asyncio.run(main())