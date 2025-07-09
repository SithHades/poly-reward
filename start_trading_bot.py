#!/usr/bin/env python3
"""
Quick start script for the ETH Polymarket Trading Bot

This script provides a simple interface to start the trading bot with
proper environment setup and error handling.
"""

import sys
import asyncio
import logging
from pathlib import Path

def setup_environment():
    """Setup the environment and check dependencies"""
    print("🔧 Setting up environment...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Creating template...")
        env_template = """# Polymarket Configuration
PK=your_polymarket_private_key_here
BROWSER_ADDRESS=your_wallet_address_here

# Optional: API Keys for enhanced features
COINBASE_API_KEY=your_coinbase_api_key
COINBASE_API_SECRET=your_coinbase_api_secret
"""
        with open(".env", "w") as f:
            f.write(env_template)
        
        print("📝 Please edit .env file with your credentials before running the bot")
        return False
    
    # Check configuration file
    config_file = Path("config/trading_config.yaml")
    if not config_file.exists():
        print("❌ Configuration file not found: config/trading_config.yaml")
        return False
    
    print("✅ Environment setup complete")
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        "ccxt", "pandas", "sklearn", "aiohttp", 
        "py_clob_client", "pyyaml", "python-dotenv"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please run: uv sync")
        return False
    
    print("✅ All dependencies found")
    return True

async def main():
    """Main function"""
    print("🚀 ETH Polymarket Trading Bot Launcher")
    print("=" * 50)
    
    # Check environment and dependencies
    if not setup_environment():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Import and start the bot
    try:
        from trading_bot_manager import TradingBotManager
        
        print("🤖 Starting trading bot...")
        manager = TradingBotManager()
        await manager.start_bot()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all files are in the correct location")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())