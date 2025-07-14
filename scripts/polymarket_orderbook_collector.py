import aiohttp
import asyncio
import json
import pandas as pd
import time
from datetime import datetime, UTC, timedelta
import os
import logging
from websockets import connect
import sys
import hashlib
from typing import Dict, Set, List, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.parsing_utils import (
    get_next_market_slug, 
    get_current_market_slug,
    slug_to_datetime,
    create_slug_from_datetime,
    get_next_market_hour_et,
    get_current_market_hour_et,
    ET
)
from src.polymarket_client import PolymarketClient
from src.constants import MARKETS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orderbook_collector.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PolymarketOrderbookCollector:
    def __init__(self, output_dir="orderbook_data"):
        self.ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self.output_dir = output_dir
        self.orderbook_data = []
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes
        self.client = PolymarketClient()
        
        # Market tracking for all crypto types
        self.current_markets = {}  # slug -> {market_id, asset_ids, start_time, end_time, crypto}
        self.monitored_asset_ids = set()
        
        # Storage optimization: mapping long identifiers to short ones
        self.asset_id_mapping = {}  # asset_id -> short_id
        self.slug_mapping = {}  # slug -> short_slug
        self.reverse_asset_mapping = {}  # short_id -> asset_id
        self.reverse_slug_mapping = {}  # short_slug -> slug
        self.next_asset_id = 1
        self.next_slug_id = 1
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for better organization
        os.makedirs(os.path.join(self.output_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "mappings"), exist_ok=True)
        
        # Load existing mappings if they exist
        self.load_mappings()

    def load_mappings(self):
        """Load existing mappings from files"""
        try:
            # Load asset ID mappings
            asset_mapping_file = os.path.join(self.output_dir, "mappings", "asset_id_mapping.json")
            if os.path.exists(asset_mapping_file):
                with open(asset_mapping_file, 'r') as f:
                    data = json.load(f)
                    self.asset_id_mapping = data.get('asset_id_mapping', {})
                    self.reverse_asset_mapping = data.get('reverse_asset_mapping', {})
                    self.next_asset_id = data.get('next_asset_id', 1)
                    logger.info(f"Loaded {len(self.asset_id_mapping)} asset ID mappings")

            # Load slug mappings
            slug_mapping_file = os.path.join(self.output_dir, "mappings", "slug_mapping.json")
            if os.path.exists(slug_mapping_file):
                with open(slug_mapping_file, 'r') as f:
                    data = json.load(f)
                    self.slug_mapping = data.get('slug_mapping', {})
                    self.reverse_slug_mapping = data.get('reverse_slug_mapping', {})
                    self.next_slug_id = data.get('next_slug_id', 1)
                    logger.info(f"Loaded {len(self.slug_mapping)} slug mappings")
        except Exception as e:
            logger.error(f"Error loading mappings: {e}")

    def save_mappings(self):
        """Save current mappings to files"""
        try:
            # Save asset ID mappings
            asset_mapping_file = os.path.join(self.output_dir, "mappings", "asset_id_mapping.json")
            with open(asset_mapping_file, 'w') as f:
                json.dump({
                    'asset_id_mapping': self.asset_id_mapping,
                    'reverse_asset_mapping': self.reverse_asset_mapping,
                    'next_asset_id': self.next_asset_id
                }, f, indent=2)

            # Save slug mappings
            slug_mapping_file = os.path.join(self.output_dir, "mappings", "slug_mapping.json")
            with open(slug_mapping_file, 'w') as f:
                json.dump({
                    'slug_mapping': self.slug_mapping,
                    'reverse_slug_mapping': self.reverse_slug_mapping,
                    'next_slug_id': self.next_slug_id
                }, f, indent=2)
            
            logger.info(f"Saved mappings: {len(self.asset_id_mapping)} assets, {len(self.slug_mapping)} slugs")
        except Exception as e:
            logger.error(f"Error saving mappings: {e}")

    def get_short_asset_id(self, asset_id: str) -> str:
        """Get or create a short identifier for an asset_id"""
        if asset_id not in self.asset_id_mapping:
            short_id = f"a{self.next_asset_id}"
            self.asset_id_mapping[asset_id] = short_id
            self.reverse_asset_mapping[short_id] = asset_id
            self.next_asset_id += 1
        return self.asset_id_mapping[asset_id]

    def get_short_slug(self, slug: str) -> str:
        """Get or create a short identifier for a slug"""
        if slug not in self.slug_mapping:
            short_slug = f"s{self.next_slug_id}"
            self.slug_mapping[slug] = short_slug
            self.reverse_slug_mapping[short_slug] = slug
            self.next_slug_id += 1
        return self.slug_mapping[slug]

    def get_markets_to_monitor(self) -> Dict[str, List[str]]:
        """Get market slugs that need monitoring for all crypto types"""
        current_time = datetime.now(ET)
        markets_to_monitor = {}
        
        # Check all market types
        for crypto in ["ethereum", "bitcoin", "solana", "xrp"]:
            crypto_markets = []
            
            # Get current market (if we're in monitoring window)
            current_market_slug = get_current_market_slug(crypto)
            current_market_time = slug_to_datetime(current_market_slug)
            
            if current_market_time:
                # Check if we're in the 1h before + active hour window
                monitor_start = current_market_time - timedelta(hours=1)
                monitor_end = current_market_time + timedelta(hours=1)
                
                if monitor_start <= current_time <= monitor_end:
                    crypto_markets.append(current_market_slug)
                    
            # Get next market (if we're in pre-monitoring window)
            next_market_slug = get_next_market_slug(crypto)
            next_market_time = slug_to_datetime(next_market_slug)
            
            if next_market_time:
                # Check if we're in the 1h before window
                monitor_start = next_market_time - timedelta(hours=1)
                monitor_end = next_market_time + timedelta(hours=1)
                
                if monitor_start <= current_time <= monitor_end:
                    crypto_markets.append(next_market_slug)
            
            if crypto_markets:
                markets_to_monitor[crypto] = list(set(crypto_markets))
        
        return markets_to_monitor

    async def setup_market_monitoring(self):
        """Setup monitoring for relevant markets across all crypto types"""
        markets_by_crypto = self.get_markets_to_monitor()
        
        for crypto, market_slugs in markets_by_crypto.items():
            logger.info(f"Setting up monitoring for {crypto} markets: {market_slugs}")
            
            for slug in market_slugs:
                if slug not in self.current_markets:
                    try:
                        market = self.client.get_market_by_slug(slug)
                        if market and market.tokens:
                            # Extract asset_ids from tokens
                            asset_ids = [token.token_id for token in market.tokens]
                            
                            market_time = slug_to_datetime(slug)
                            self.current_markets[slug] = {
                                'market_id': market.condition_id,
                                'asset_ids': asset_ids,
                                'start_time': market_time - timedelta(hours=1),
                                'end_time': market_time + timedelta(hours=1),
                                'market_time': market_time,
                                'crypto': crypto
                            }
                            self.monitored_asset_ids.update(asset_ids)
                            logger.info(f"Added {crypto} market {slug} with assets {asset_ids}")
                        else:
                            logger.warning(f"Could not find market data for {crypto} slug: {slug}")
                    except Exception as e:
                        logger.error(f"Error setting up {crypto} market {slug}: {str(e)}")

    def clean_expired_markets(self):
        """Remove markets that are no longer in monitoring window"""
        current_time = datetime.now(ET)
        expired_markets = []
        
        for slug, market_info in self.current_markets.items():
            if current_time > market_info['end_time']:
                expired_markets.append(slug)
                # Remove asset_ids from monitored set
                for asset_id in market_info['asset_ids']:
                    self.monitored_asset_ids.discard(asset_id)
        
        for slug in expired_markets:
            crypto = self.current_markets[slug]['crypto']
            del self.current_markets[slug]
            logger.info(f"Removed expired {crypto} market: {slug}")

    def process_websocket_message(self, message_data):
        """Process websocket message which contains an array of updates"""
        if not isinstance(message_data, list):
            logger.warning("Expected array in websocket message")
            return
            
        for data in message_data:
            asset_id = data.get('asset_id')
            if not asset_id or asset_id not in self.monitored_asset_ids:
                continue
                
            # Find which market this asset belongs to
            market_slug = None
            crypto = None
            for slug, market_info in self.current_markets.items():
                if asset_id in market_info['asset_ids']:
                    market_slug = slug
                    crypto = market_info['crypto']
                    break
            
            if not market_slug:
                continue
                
            event_type = data.get('event_type')
            if event_type == 'book':
                self.process_orderbook(data, market_slug, crypto)
            elif event_type == 'price_change':
                self.process_price_change(data, market_slug, crypto)

    def process_orderbook(self, data, market_slug, crypto):
        """Process full orderbook data with optimized storage"""
        timestamp_ms = int(data['timestamp'])
        timestamp_s = timestamp_ms / 1000  # Convert ms to seconds
        dt = datetime.fromtimestamp(timestamp_s, UTC).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Keep millisecond precision
        asset_id = data.get('asset_id')
        
        # Get optimized identifiers
        short_asset_id = self.get_short_asset_id(asset_id)
        short_slug = self.get_short_slug(market_slug)
        
        for bid in data.get('bids', []):
            self.orderbook_data.append({
                'timestamp': dt,
                'market_slug': short_slug,
                'asset_id': short_asset_id,
                'crypto': crypto,
                'side': 'bid',
                'price': float(bid['price']),
                'size': float(bid['size']),
                'event_type': 'book'
            })
        for ask in data.get('asks', []):
            self.orderbook_data.append({
                'timestamp': dt,
                'market_slug': short_slug,
                'asset_id': short_asset_id,
                'crypto': crypto,
                'side': 'ask',
                'price': float(ask['price']),
                'size': float(ask['size']),
                'event_type': 'book'
            })

    def process_price_change(self, data, market_slug, crypto):
        """Process price change updates with optimized storage"""
        timestamp_ms = int(data['timestamp'])
        timestamp_s = timestamp_ms / 1000  # Convert ms to seconds
        dt = datetime.fromtimestamp(timestamp_s, UTC).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Keep millisecond precision
        asset_id = data.get('asset_id')
        
        # Get optimized identifiers
        short_asset_id = self.get_short_asset_id(asset_id)
        short_slug = self.get_short_slug(market_slug)
        
        for change in data.get('changes', []):
            self.orderbook_data.append({
                'timestamp': dt,
                'market_slug': short_slug,
                'asset_id': short_asset_id,
                'crypto': crypto,
                'side': change['side'].lower(),
                'price': float(change['price']),
                'size': float(change['size']),
                'event_type': 'price_change'
            })

    def save_data(self):
        """Save collected data to CSV with optimized storage"""
        if not self.orderbook_data:
            logger.info("No data to save")
            return
            
        # Save mappings first
        self.save_mappings()
        
        df = pd.DataFrame(self.orderbook_data)
        
        # Group by crypto and market_slug for separate files
        for (crypto, market_slug), group in df.groupby(['crypto', 'market_slug']):
            # Create unique filename with timestamp to prevent overwrites
            now = datetime.now(UTC)
            date_str = now.strftime('%Y%m%d')
            time_str = now.strftime('%H%M%S')
            
            filename = f'orderbook_{crypto}_{market_slug}_{date_str}_{time_str}.csv'
            output_file = os.path.join(self.output_dir, "raw", filename)
            
            # Save with header
            group.to_csv(output_file, index=False)
            logger.info(f"Saved {len(group)} records to {output_file}")
        
        # Also save a consolidated file for the session
        consolidated_file = os.path.join(
            self.output_dir, 
            "processed", 
            f'consolidated_all_markets_{datetime.now(UTC).strftime("%Y%m%d_%H%M%S")}.csv'
        )
        df.to_csv(consolidated_file, index=False)
        logger.info(f"Saved consolidated data to {consolidated_file}")
        
        # Save a summary of current mappings
        summary_file = os.path.join(
            self.output_dir, 
            "mappings", 
            f'mapping_summary_{datetime.now(UTC).strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now(UTC).isoformat(),
                'total_asset_mappings': len(self.asset_id_mapping),
                'total_slug_mappings': len(self.slug_mapping),
                'active_markets': {slug: info['crypto'] for slug, info in self.current_markets.items()}
            }, f, indent=2)
        
        self.orderbook_data = []  # Clear data after saving

    async def websocket_listener(self):
        """Listen for WebSocket updates with reconnection logic"""
        while True:
            try:
                async with connect(self.ws_url) as websocket:
                    # Setup monitoring and subscribe to all relevant asset_ids
                    await self.setup_market_monitoring()
                    
                    if not self.monitored_asset_ids:
                        logger.info("No markets to monitor at this time, waiting...")
                        await asyncio.sleep(60)  # Wait 1 minute before checking again
                        continue
                        
                    # Subscribe to all monitored asset_ids
                    asset_ids_list = list(self.monitored_asset_ids)
                    subscription_msg = {
                        "type": "MARKET", 
                        "assets_ids": asset_ids_list
                    }
                    await websocket.send(json.dumps(subscription_msg))
                    
                    # Log subscription by crypto type
                    crypto_counts = {}
                    for slug, market_info in self.current_markets.items():
                        crypto = market_info['crypto']
                        crypto_counts[crypto] = crypto_counts.get(crypto, 0) + len(market_info['asset_ids'])
                    
                    logger.info(f"Subscribed to {len(asset_ids_list)} asset_ids across all markets:")
                    for crypto, count in crypto_counts.items():
                        logger.info(f"  {crypto}: {count} assets")
                    
                    message_count = 0
                    while True:
                        try:
                            message = await websocket.recv()
                            message_data = json.loads(message)
                            message_count += 1
                            
                            # Process the message (array of updates)
                            self.process_websocket_message(message_data)
                            
                            # Periodic maintenance
                            if message_count % 100 == 0:
                                self.clean_expired_markets()
                                # Re-setup monitoring in case new markets need to be added
                                await self.setup_market_monitoring()
                                
                                # Update subscription if monitored assets changed
                                current_asset_ids = list(self.monitored_asset_ids)
                                if set(current_asset_ids) != set(asset_ids_list):
                                    asset_ids_list = current_asset_ids
                                    if asset_ids_list:
                                        subscription_msg = {
                                            "type": "MARKET", 
                                            "assets_ids": asset_ids_list
                                        }
                                        await websocket.send(json.dumps(subscription_msg))
                                        logger.info(f"Updated subscription to {len(asset_ids_list)} asset_ids")
                            
                            # Save periodically
                            current_time = time.time()
                            if current_time - self.last_save_time >= self.save_interval:
                                self.save_data()
                                self.last_save_time = current_time
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {str(e)}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing message: {str(e)}")
                            continue
                        
            except Exception as e:
                logger.error(f"WebSocket connection error: {str(e)}")
                await asyncio.sleep(5)  # Wait before reconnecting
            except KeyboardInterrupt:
                logger.info("Stopping...")
                self.save_data()
                break

    async def run(self):
        """Main run method - monitors all market types simultaneously"""
        logger.info("Starting orderbook collector for all market types (ethereum, bitcoin, solana, xrp)")
        logger.info("Storage optimization: Using short identifiers for asset_ids and slugs")
        logger.info("Initial orderbook will be obtained from first websocket message")
        await self.websocket_listener()

async def main():
    collector = PolymarketOrderbookCollector()
    try:
        await collector.run()
    except (asyncio.exceptions.CancelledError, KeyboardInterrupt):
        logger.info("Shutting down...")
        collector.save_data()

if __name__ == "__main__":
    asyncio.run(main())
