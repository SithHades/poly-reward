"""
Data loader for Binance market data.

Supports fetching data via:
1. ccxt for live/recent data
2. Binance Vision public zip files for historical data
"""

import os
import io
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import ccxt
import requests
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BinanceDataLoader:
    """
    Comprehensive data loader for Binance OHLCV data.
    Supports multiple timeframes and data sources.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ccxt exchange
        self.exchange = ccxt.binance({
            'apiKey': '',  # Not needed for public data
            'secret': '',
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        self.base_url = "https://data.binance.vision/data/spot"
        
        # Common crypto symbols
        self.supported_symbols = [
            'ETHUSDT', 'BTCUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'BNBUSDT'
        ]
        
        # Available timeframes
        self.timeframes = {
            '1s': '1s',
            '1m': '1m', 
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d'
        }
    
    def fetch_live_data(self, symbol: str, timeframe: str = '1m', 
                       limit: int = 1000) -> pd.DataFrame:
        """
        Fetch recent/live OHLCV data using ccxt.
        
        Args:
            symbol: Trading pair (e.g., 'ETHUSDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching live data for {symbol} {timeframe}")
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Add symbol and timeframe info
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching live data: {e}")
            raise
    
    def download_historical_zip(self, symbol: str, timeframe: str,
                               year: int, month: int, day: Optional[int] = None) -> str:
        """
        Download historical data zip file from Binance Vision.
        
        Args:
            symbol: Trading pair (e.g., 'ETHUSDT')
            timeframe: Timeframe (e.g., '1m', '1s')
            year: Year (e.g., 2024)
            month: Month (1-12)
            day: Day (1-31), required for daily data
            
        Returns:
            Path to downloaded zip file
        """
        # Determine URL structure based on timeframe and parameters
        if timeframe in ['1s', '1m'] and day is not None:
            # Daily data for high-frequency timeframes
            url_type = "daily"
            date_str = f"{year:04d}-{month:02d}-{day:02d}"
            filename = f"{symbol}-{timeframe}-{date_str}.zip"
        else:
            # Monthly data for lower frequency or when day not specified
            url_type = "monthly"
            date_str = f"{year:04d}-{month:02d}"
            filename = f"{symbol}-{timeframe}-{date_str}.zip"
        
        url = f"{self.base_url}/{url_type}/klines/{symbol}/{timeframe}/{filename}"
        
        # Local file path
        local_path = self.data_dir / filename
        
        # Skip download if file already exists
        if local_path.exists():
            logger.info(f"File already exists: {local_path}")
            return str(local_path)
        
        try:
            logger.info(f"Downloading {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded: {local_path}")
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            if local_path.exists():
                local_path.unlink()  # Remove partial file
            raise
    
    def load_from_zip(self, zip_path: str) -> pd.DataFrame:
        """
        Load OHLCV data from a Binance Vision zip file.
        
        Args:
            zip_path: Path to zip file
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                # Get the CSV filename (should be only one file)
                csv_filename = zip_file.namelist()[0]
                
                with zip_file.open(csv_filename) as csv_file:
                    # Binance CSV columns
                    columns = [
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                        'ignore'
                    ]
                    
                    df = pd.read_csv(csv_file, names=columns)
                    
                    # Keep only essential columns
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
                    # Convert timestamp
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')
                    
                    # Convert to float
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    
                    return df
                    
        except Exception as e:
            logger.error(f"Error loading zip file {zip_path}: {e}")
            raise
    
    def get_historical_data(self, symbol: str, timeframe: str,
                           start_date: Union[str, datetime],
                           end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get historical data for a date range.
        Automatically downloads and combines data from multiple files if needed.
        
        Args:
            symbol: Trading pair (e.g., 'ETHUSDT')
            timeframe: Timeframe (e.g., '1m', '1h')
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime)
            
        Returns:
            Combined DataFrame with OHLCV data
        """
        # Convert to datetime objects
        start_dt = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        end_dt = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date
        
        dfs = []
        current_date = start_dt
        
        while current_date <= end_dt:
            try:
                # For high-frequency data, download daily files
                if timeframe in ['1s', '1m']:
                    zip_path = self.download_historical_zip(
                        symbol, timeframe, 
                        current_date.year, current_date.month, current_date.day
                    )
                    current_date += timedelta(days=1)
                else:
                    # For lower frequency, download monthly files
                    zip_path = self.download_historical_zip(
                        symbol, timeframe,
                        current_date.year, current_date.month
                    )
                    # Move to next month
                    if current_date.month == 12:
                        current_date = current_date.replace(year=current_date.year + 1, month=1)
                    else:
                        current_date = current_date.replace(month=current_date.month + 1)
                
                df = self.load_from_zip(zip_path)
                
                # Filter to date range
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                
                if not df.empty:
                    df['symbol'] = symbol
                    df['timeframe'] = timeframe
                    dfs.append(df)
                    
            except Exception as e:
                logger.warning(f"Could not load data for {current_date}: {e}")
                continue
        
        if not dfs:
            raise ValueError(f"No data found for {symbol} {timeframe} between {start_dt} and {end_dt}")
        
        # Combine all dataframes
        combined_df = pd.concat(dfs).sort_index()
        
        # Remove duplicates
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        logger.info(f"Loaded {len(combined_df)} candles for {symbol} {timeframe}")
        return combined_df
    
    def get_multi_timeframe_data(self, symbol: str, timeframes: List[str],
                                start_date: Union[str, datetime],
                                end_date: Union[str, datetime]) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple timeframes.
        
        Args:
            symbol: Trading pair
            timeframes: List of timeframes
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        data = {}
        
        for tf in timeframes:
            try:
                data[tf] = self.get_historical_data(symbol, tf, start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to load {symbol} {tf}: {e}")
                
        return data
    
    def get_latest_complete_candle(self, symbol: str, timeframe: str = '1h') -> pd.Series:
        """
        Get the latest complete candle for a symbol.
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            
        Returns:
            Latest complete candle as Series
        """
        df = self.fetch_live_data(symbol, timeframe, limit=2)
        
        # Return the second-to-last candle (last complete one)
        if len(df) >= 2:
            return df.iloc[-2]
        else:
            return df.iloc[-1]
    
    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is supported by the exchange."""
        try:
            markets = self.exchange.load_markets()
            return symbol in markets
        except Exception:
            return symbol in self.supported_symbols