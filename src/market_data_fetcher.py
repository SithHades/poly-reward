"""
Market Data Fetcher

Alternative data sources for ETH price data and Polymarket integration
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import pandas as pd


class AlternativeDataFetcher:
    """Fetch market data from alternative sources"""
    
    def __init__(self):
        self.logger = logging.getLogger("DataFetcher")
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_eth_price_coingecko(self) -> Optional[Dict]:
        """Fetch current ETH price from CoinGecko API"""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": "ethereum",
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_last_updated_at": "true"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("ethereum", {})
                    
        except Exception as e:
            self.logger.error(f"Error fetching from CoinGecko: {e}")
        
        return None
    
    async def fetch_eth_ohlcv_coinbase(self, hours: int = 2) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Coinbase Pro API"""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)
            
            url = "https://api.exchange.coinbase.com/products/ETH-USD/candles"
            params = {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "granularity": 60  # 1-minute candles
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data, columns=["timestamp", "low", "high", "open", "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                    df = df.set_index("timestamp").sort_index()
                    
                    # Reorder columns to match CCXT format
                    df = df[["open", "high", "low", "close", "volume"]]
                    
                    return df
                    
        except Exception as e:
            self.logger.error(f"Error fetching from Coinbase: {e}")
        
        return None
    
    async def fetch_polymarket_markets_gamma(self, keyword: str = "ethereum") -> List[Dict]:
        """Fetch markets from Polymarket Gamma API"""
        try:
            url = "https://gamma-api.polymarket.com/markets"
            params = {
                "active": "true",
                "closed": "false",
                "limit": 100
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Filter for keyword
                    if keyword:
                        filtered_markets = []
                        for market in data:
                            market_text = f"{market.get('question', '')} {market.get('description', '')}".lower()
                            if keyword.lower() in market_text:
                                filtered_markets.append(market)
                        return filtered_markets
                    
                    return data
                    
        except Exception as e:
            self.logger.error(f"Error fetching from Gamma API: {e}")
        
        return []
    
    async def get_market_health_check(self) -> Dict[str, bool]:
        """Check if various market data sources are available"""
        health = {
            "coingecko": False,
            "coinbase": False,
            "polymarket_gamma": False
        }
        
        # Test CoinGecko
        try:
            eth_price = await self.fetch_eth_price_coingecko()
            health["coingecko"] = bool(eth_price and "usd" in eth_price)
        except:
            pass
        
        # Test Coinbase
        try:
            ohlcv_data = await self.fetch_eth_ohlcv_coinbase(hours=1)
            health["coinbase"] = bool(ohlcv_data is not None and len(ohlcv_data) > 0)
        except:
            pass
        
        # Test Polymarket Gamma
        try:
            markets = await self.fetch_polymarket_markets_gamma()
            health["polymarket_gamma"] = bool(markets and len(markets) > 0)
        except:
            pass
        
        return health


class RobustMarketDataProvider:
    """Robust market data provider with multiple fallbacks"""
    
    def __init__(self):
        self.logger = logging.getLogger("RobustDataProvider")
        self.ccxt_exchange = None
        
        # Try to import and initialize CCXT
        try:
            import ccxt
            self.ccxt_exchange = ccxt.binance()
        except ImportError:
            self.logger.warning("CCXT not available, using alternative sources")
    
    async def get_eth_current_price(self) -> Optional[float]:
        """Get current ETH price from best available source"""
        
        # Try CCXT first
        if self.ccxt_exchange:
            try:
                ticker = self.ccxt_exchange.fetch_ticker("ETH/USDT")
                return float(ticker["last"])
            except Exception as e:
                self.logger.warning(f"CCXT failed: {e}")
        
        # Fallback to CoinGecko
        async with AlternativeDataFetcher() as fetcher:
            eth_data = await fetcher.fetch_eth_price_coingecko()
            if eth_data and "usd" in eth_data:
                return float(eth_data["usd"])
        
        self.logger.error("All ETH price sources failed")
        return None
    
    async def get_eth_ohlcv_data(self, hours: int = 2) -> Optional[pd.DataFrame]:
        """Get ETH OHLCV data from best available source"""
        
        # Try CCXT first with robust pagination
        if self.ccxt_exchange:
            try:
                since = self.ccxt_exchange.parse8601(
                    (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
                )
                
                all_ohlcv = []
                limit = 1000  # Fetch in chunks to be more robust
                target_candles = hours * 60  # Approximate number of 1-minute candles needed
                
                while True:
                    try:
                        ohlcv = self.ccxt_exchange.fetch_ohlcv("ETH/USDT", "1m", since, limit)
                        
                        if len(ohlcv) == 0:
                            break
                        
                        all_ohlcv.extend(ohlcv)
                        since = ohlcv[-1][0] + 1  # Move to next timestamp
                        
                        # Rate limiting to avoid hitting API limits
                        if self.ccxt_exchange.rateLimit:
                            await asyncio.sleep(self.ccxt_exchange.rateLimit / 1000)
                        
                        # Progress tracking for larger requests
                        if len(all_ohlcv) % 2000 == 0:
                            self.logger.info(f"Fetched {len(all_ohlcv)} candles...")
                        
                        # Stop if we have enough data
                        if len(all_ohlcv) >= target_candles:
                            break
                            
                    except Exception as e:
                        self.logger.warning(f"CCXT chunk fetch failed: {e}")
                        break
                
                if all_ohlcv:
                    df = pd.DataFrame(
                        all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    df = df.set_index("timestamp").sort_index()
                    
                    # Remove duplicates (keep last occurrence)
                    df = df[~df.index.duplicated(keep="last")]
                    
                    # Reorder columns
                    df = df[["open", "high", "low", "close", "volume"]]
                    
                    # Filter to requested time range
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - timedelta(hours=hours)
                    df = df[df.index >= start_time]
                    
                    return df
                    
            except Exception as e:
                self.logger.warning(f"CCXT OHLCV failed: {e}")
        
        # Fallback to Coinbase
        async with AlternativeDataFetcher() as fetcher:
            return await fetcher.fetch_eth_ohlcv_coinbase(hours)
    
    async def check_data_sources(self) -> Dict[str, bool]:
        """Check health of all data sources"""
        health = {"ccxt_binance": False}
        
        # Test CCXT
        if self.ccxt_exchange:
            try:
                self.ccxt_exchange.fetch_ticker("ETH/USDT")
                health["ccxt_binance"] = True
            except:
                pass
        
        # Test alternative sources
        async with AlternativeDataFetcher() as fetcher:
            alt_health = await fetcher.get_market_health_check()
            health.update(alt_health)
        
        return health


async def test_data_sources():
    """Test all data sources and print results"""
    print("Testing market data sources...")
    
    provider = RobustMarketDataProvider()
    
    # Check all sources
    health = await provider.check_data_sources()
    
    print("\nData Source Health Check:")
    for source, is_healthy in health.items():
        status = "✅ OK" if is_healthy else "❌ FAILED"
        print(f"  {source}: {status}")
    
    # Test current price
    print(f"\nTesting current ETH price...")
    price = await provider.get_eth_current_price()
    if price:
        print(f"  Current ETH Price: ${price:,.2f}")
    else:
        print(f"  ❌ Failed to get ETH price")
    
    # Test OHLCV data
    print(f"\nTesting OHLCV data (last 1 hour)...")
    ohlcv = await provider.get_eth_ohlcv_data(hours=1)
    if ohlcv is not None and len(ohlcv) > 0:
        print(f"  ✅ Got {len(ohlcv)} candles")
        print(f"  Latest: {ohlcv.index[-1]} - Close: ${ohlcv.iloc[-1]['close']:.2f}")
    else:
        print(f"  ❌ Failed to get OHLCV data")


if __name__ == "__main__":
    asyncio.run(test_data_sources())