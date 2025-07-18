import aiohttp
import asyncio
import json
import pandas as pd
import time
from datetime import datetime, UTC, timedelta
import os
import logging
from websockets import connect, ConnectionClosed, WebSocketException
import sys
import hashlib
from typing import Dict, Set, List, Optional
from enum import Enum

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

class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"

class PolymarketOrderbookCollector:
    def __init__(self, output_dir="orderbook_data"):
        self.ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self.output_dir = output_dir
        self.orderbook_data = []
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes
        self.client = PolymarketClient()
        
        # Connection state management
        self.connection_state = ConnectionState.DISCONNECTED
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.base_reconnect_delay = 5  # seconds
        self.max_reconnect_delay = 300  # 5 minutes
        
        # Market monitoring timing
        self.market_check_interval = 30  # Check markets every 30 seconds
        self.last_market_check = 0
        self.market_transition_threshold = 300  # 5 minutes before market expires
        
        # WebSocket health monitoring
        self.last_message_time = 0
        self.websocket_timeout = 120  # 2 minutes without messages triggers reconnection
        self.ping_interval = 60  # Send ping every minute
        self.last_ping_time = 0
        
        # Market tracking for all crypto types
        self.current_markets = {}  # slug -> {market_id, asset_ids, start_time, end_time, crypto}
        self.monitored_asset_ids = set()
        self.previous_asset_ids = set()  # Track changes in monitored assets
        
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
        markets_changed = False
        
        for crypto, market_slugs in markets_by_crypto.items():
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
                            markets_changed = True
                            logger.info(f"Added {crypto} market {slug} with assets {asset_ids}")
                        else:
                            logger.warning(f"Could not find market data for {crypto} slug: {slug}")
                    except Exception as e:
                        logger.error(f"Error setting up {crypto} market {slug}: {str(e)}")
        
        return markets_changed

    def clean_expired_markets(self):
        """Remove markets that are no longer in monitoring window"""
        current_time = datetime.now(ET)
        expired_markets = []
        markets_changed = False
        
        for slug, market_info in self.current_markets.items():
            if current_time > market_info['end_time']:
                expired_markets.append(slug)
                # Remove asset_ids from monitored set
                for asset_id in market_info['asset_ids']:
                    self.monitored_asset_ids.discard(asset_id)
                markets_changed = True
        
        for slug in expired_markets:
            crypto = self.current_markets[slug]['crypto']
            del self.current_markets[slug]
            logger.info(f"Removed expired {crypto} market: {slug}")
        
        return markets_changed

    def check_market_transitions(self):
        """Check if any markets are about to expire and need proactive reconnection"""
        current_time = datetime.now(ET)
        transition_needed = False
        
        for slug, market_info in self.current_markets.items():
            time_to_expiry = (market_info['end_time'] - current_time).total_seconds()
            if 0 < time_to_expiry < self.market_transition_threshold:
                logger.info(f"Market {slug} ({market_info['crypto']}) expires in {time_to_expiry:.0f} seconds")
                transition_needed = True
        
        return transition_needed

    def get_reconnect_delay(self):
        """Calculate exponential backoff delay for reconnection"""
        if self.reconnect_attempts == 0:
            return 0
        
        delay = min(
            self.base_reconnect_delay * (2 ** (self.reconnect_attempts - 1)),
            self.max_reconnect_delay
        )
        return delay

    def reset_reconnect_attempts(self):
        """Reset reconnection attempts after successful connection"""
        self.reconnect_attempts = 0

    async def check_markets_and_reconnect_if_needed(self, websocket):
        """Check markets and determine if reconnection is needed"""
        current_time = time.time()
        
        # Check if it's time to check markets
        if current_time - self.last_market_check < self.market_check_interval:
            return False
            
        self.last_market_check = current_time
        
        # Clean expired markets and setup new ones
        expired_changed = self.clean_expired_markets()
        new_changed = await self.setup_market_monitoring()
        
        # Check if monitored assets have changed
        current_asset_ids = set(self.monitored_asset_ids)
        assets_changed = current_asset_ids != self.previous_asset_ids
        
        # Check if we're approaching market transitions
        transition_needed = self.check_market_transitions()
        
        if expired_changed or new_changed or assets_changed or transition_needed:
            logger.info("Market changes detected, reconnection needed")
            logger.info(f"  Expired markets: {expired_changed}")
            logger.info(f"  New markets: {new_changed}")
            logger.info(f"  Assets changed: {assets_changed}")
            logger.info(f"  Transition needed: {transition_needed}")
            return True
            
        return False

    async def send_ping(self, websocket):
        """Send ping to keep connection alive"""
        current_time = time.time()
        if current_time - self.last_ping_time > self.ping_interval:
            try:
                await websocket.ping()
                self.last_ping_time = current_time
                logger.debug("Sent WebSocket ping")
            except Exception as e:
                logger.error(f"Error sending ping: {e}")
                raise

    def check_connection_health(self):
        """Check if connection is healthy based on last message time"""
        current_time = time.time()
        if self.last_message_time > 0 and current_time - self.last_message_time > self.websocket_timeout:
            logger.warning(f"No messages received for {current_time - self.last_message_time:.0f} seconds")
            return False
        return True

    def process_websocket_message(self, message_data):
        """Process websocket message which contains an array of updates"""
        self.last_message_time = time.time()
        
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
        """Listen for WebSocket updates with robust reconnection logic"""
        while True:
            try:
                self.connection_state = ConnectionState.CONNECTING
                logger.info(f"Connecting to WebSocket (attempt {self.reconnect_attempts + 1})")
                
                async with connect(
                    self.ws_url,
                    ping_interval=None,  # We'll handle pings manually
                    ping_timeout=20,
                    close_timeout=10,
                ) as websocket:
                    self.connection_state = ConnectionState.CONNECTED
                    logger.info("WebSocket connected successfully")
                    self.reset_reconnect_attempts()
                    
                    # Setup monitoring and subscribe to all relevant asset_ids
                    await self.setup_market_monitoring()
                    
                    if not self.monitored_asset_ids:
                        logger.info("No markets to monitor at this time, waiting...")
                        await asyncio.sleep(60)  # Wait 1 minute before checking again
                        continue
                        
                    # Subscribe to all monitored asset_ids
                    asset_ids_list = list(self.monitored_asset_ids)
                    self.previous_asset_ids = set(asset_ids_list)
                    
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
                    
                    # Initialize timing
                    self.last_message_time = time.time()
                    self.last_ping_time = time.time()
                    
                    # Main message processing loop
                    while True:
                        try:
                            # Check if we need to reconnect due to market changes
                            if await self.check_markets_and_reconnect_if_needed(websocket):
                                logger.info("Reconnecting due to market changes...")
                                break
                            
                            # Check connection health
                            if not self.check_connection_health():
                                logger.warning("Connection appears unhealthy, reconnecting...")
                                break
                            
                            # Send ping if needed
                            await self.send_ping(websocket)
                            
                            # Try to receive a message with timeout
                            try:
                                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                                message_data = json.loads(message)
                                
                                # Process the message (array of updates)
                                self.process_websocket_message(message_data)
                                
                            except asyncio.TimeoutError:
                                # No message received, continue to next iteration
                                continue
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON decode error: {str(e)}")
                                continue
                            
                            # Save periodically
                            current_time = time.time()
                            if current_time - self.last_save_time >= self.save_interval:
                                self.save_data()
                                self.last_save_time = current_time
                                
                        except ConnectionClosed as e:
                            logger.warning(f"WebSocket connection closed: {e}")
                            break
                        except WebSocketException as e:
                            logger.error(f"WebSocket error: {e}")
                            break
                        except Exception as e:
                            logger.error(f"Error in message processing loop: {str(e)}")
                            # Continue processing unless it's a connection error
                            if "connection" in str(e).lower() or "close" in str(e).lower():
                                break
                            continue
                        
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed during connect: {e}")
            except WebSocketException as e:
                logger.error(f"WebSocket connection error: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error in WebSocket listener: {str(e)}")
            except KeyboardInterrupt:
                logger.info("Stopping...")
                self.save_data()
                break
            
            # Connection lost, prepare for reconnection
            self.connection_state = ConnectionState.RECONNECTING
            self.reconnect_attempts += 1
            
            if self.reconnect_attempts > self.max_reconnect_attempts:
                logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
                break
            
            delay = self.get_reconnect_delay()
            logger.info(f"Reconnecting in {delay} seconds (attempt {self.reconnect_attempts})")
            await asyncio.sleep(delay)

    async def run(self):
        """Main run method - monitors all market types simultaneously"""
        logger.info("Starting robust orderbook collector for all market types (ethereum, bitcoin, solana, xrp)")
        logger.info("Enhanced features:")
        logger.info("  - Time-based market monitoring (every 30 seconds)")
        logger.info("  - Proactive reconnection before market transitions")
        logger.info("  - Connection health monitoring with ping/pong")
        logger.info("  - Exponential backoff for reconnections")
        logger.info("  - Storage optimization with short identifiers")
        
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
