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
    def __init__(self, crypto: MARKETS = "ethereum", output_dir="orderbook_data"):
        self.ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self.crypto = crypto
        self.output_dir = output_dir
        self.orderbook_data = []
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes
        self.client = PolymarketClient()
        
        # Market tracking
        self.current_markets = {}  # slug -> {market_id, asset_ids, start_time, end_time}
        self.monitored_asset_ids = set()
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for better organization
        os.makedirs(os.path.join(self.output_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "processed"), exist_ok=True)

    def get_markets_to_monitor(self) -> list[str]:
        """Get market slugs that need monitoring (1h before + during active hour)"""
        current_time = datetime.now(ET)
        markets_to_monitor = []
        
        # Get current market (if we're in monitoring window)
        current_market_slug = get_current_market_slug(self.crypto)
        current_market_time = slug_to_datetime(current_market_slug)
        
        if current_market_time:
            # Check if we're in the 1h before + active hour window
            monitor_start = current_market_time - timedelta(hours=1)
            monitor_end = current_market_time + timedelta(hours=1)
            
            if monitor_start <= current_time <= monitor_end:
                markets_to_monitor.append(current_market_slug)
                
        # Get next market (if we're in pre-monitoring window)
        next_market_slug = get_next_market_slug(self.crypto)
        next_market_time = slug_to_datetime(next_market_slug)
        
        if next_market_time:
            # Check if we're in the 1h before window
            monitor_start = next_market_time - timedelta(hours=1)
            monitor_end = next_market_time + timedelta(hours=1)
            
            if monitor_start <= current_time <= monitor_end:
                markets_to_monitor.append(next_market_slug)
        
        return list(set(markets_to_monitor))  # Remove duplicates

    async def setup_market_monitoring(self):
        """Setup monitoring for relevant markets"""
        market_slugs = self.get_markets_to_monitor()
        logger.info(f"Setting up monitoring for markets: {market_slugs}")
        
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
                            'market_time': market_time
                        }
                        self.monitored_asset_ids.update(asset_ids)
                        logger.info(f"Added market {slug} with assets {asset_ids}")
                    else:
                        logger.warning(f"Could not find market data for slug: {slug}")
                except Exception as e:
                    logger.error(f"Error setting up market {slug}: {str(e)}")

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
            del self.current_markets[slug]
            logger.info(f"Removed expired market: {slug}")

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
            for slug, market_info in self.current_markets.items():
                if asset_id in market_info['asset_ids']:
                    market_slug = slug
                    break
            
            if not market_slug:
                continue
                
            event_type = data.get('event_type')
            if event_type == 'book':
                self.process_orderbook(data, market_slug)
            elif event_type == 'price_change':
                self.process_price_change(data, market_slug)

    def process_orderbook(self, data, market_slug):
        """Process full orderbook data"""
        timestamp = int(data['timestamp']) / 1000  # Convert ms to seconds
        dt = datetime.fromtimestamp(timestamp, UTC).strftime('%Y-%m-%d %H:%M:%S')
        asset_id = data.get('asset_id')
        
        for bid in data.get('bids', []):
            self.orderbook_data.append({
                'timestamp': dt,
                'market_slug': market_slug,
                'asset_id': asset_id,
                'side': 'bid',
                'price': float(bid['price']),
                'size': float(bid['size']),
                'event_type': 'book'
            })
        for ask in data.get('asks', []):
            self.orderbook_data.append({
                'timestamp': dt,
                'market_slug': market_slug,
                'asset_id': asset_id,
                'side': 'ask',
                'price': float(ask['price']),
                'size': float(ask['size']),
                'event_type': 'book'
            })

    def process_price_change(self, data, market_slug):
        """Process price change updates"""
        timestamp = int(data['timestamp']) / 1000  # Convert ms to seconds
        dt = datetime.fromtimestamp(timestamp, UTC).strftime('%Y-%m-%d %H:%M:%S')
        asset_id = data.get('asset_id')
        
        for change in data.get('changes', []):
            self.orderbook_data.append({
                'timestamp': dt,
                'market_slug': market_slug,
                'asset_id': asset_id,
                'side': change['side'].lower(),
                'price': float(change['price']),
                'size': float(change['size']),
                'event_type': 'price_change'
            })

    def save_data(self):
        """Save collected data to CSV with unique timestamps to prevent overwrites"""
        if not self.orderbook_data:
            logger.info("No data to save")
            return
            
        df = pd.DataFrame(self.orderbook_data)
        
        # Group by market_slug and asset_id for separate files
        for (market_slug, asset_id), group in df.groupby(['market_slug', 'asset_id']):
            # Create unique filename with timestamp to prevent overwrites
            now = datetime.now(UTC)
            date_str = now.strftime('%Y%m%d')
            time_str = now.strftime('%H%M%S')
            
            filename = f'orderbook_{market_slug}_{asset_id}_{date_str}_{time_str}.csv'
            output_file = os.path.join(self.output_dir, "raw", filename)
            
            # Save with header
            group.to_csv(output_file, index=False)
            logger.info(f"Saved {len(group)} records to {output_file}")
        
        # Also save a consolidated file for the session
        consolidated_file = os.path.join(
            self.output_dir, 
            "processed", 
            f'consolidated_{self.crypto}_{datetime.now(UTC).strftime("%Y%m%d_%H%M%S")}.csv'
        )
        df.to_csv(consolidated_file, index=False)
        logger.info(f"Saved consolidated data to {consolidated_file}")
        
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
                    logger.info(f"Subscribed to {len(asset_ids_list)} asset_ids: {asset_ids_list}")
                    
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
        """Main run method - no longer fetches initial orderbook via REST"""
        logger.info(f"Starting orderbook collector for {self.crypto} markets")
        logger.info("Initial orderbook will be obtained from first websocket message")
        await self.websocket_listener()

async def main():
    collector = PolymarketOrderbookCollector(crypto="ethereum")
    try:
        await collector.run()
    except (asyncio.exceptions.CancelledError, KeyboardInterrupt):
        logger.info("Shutting down...")
        collector.save_data()

if __name__ == "__main__":
    asyncio.run(main())
