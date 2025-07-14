import aiohttp
import asyncio
import json
import pandas as pd
import time
from datetime import datetime, UTC
import os
import logging
from websockets import connect
import sys

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
    def __init__(self, asset_id, output_dir="orderbook_data"):
        self.rest_url = "https://clob.polymarket.com/book?token_id=19333565431154034959696570897114176039106580749219566804891884257974220925231"
        self.ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self.asset_id = asset_id
        self.output_dir = output_dir
        self.orderbook_data = []
        self.last_save_time = time.time()
        self.save_interval = 3600  # Save every hour
        os.makedirs(self.output_dir, exist_ok=True)

    async def fetch_initial_orderbook(self):
        """Fetch initial orderbook data via REST API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.rest_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('asset_id') == self.asset_id:
                            self.process_orderbook(data)
                            logger.info("Fetched initial orderbook")
                        else:
                            logger.warning(f"Market ID {self.asset_id} not found in REST response")
                    else:
                        logger.error(f"REST API error: {response.status}")
        except Exception as e:
            logger.error(f"Error fetching initial orderbook: {str(e)}")

    def process_orderbook(self, data):
        """Process orderbook data and store it"""
        timestamp = int(data['timestamp']) / 1000  # Convert ms to seconds
        dt = datetime.fromtimestamp(timestamp, UTC).strftime('%Y-%m-%d %H:%M:%S')
        for bid in data.get('bids', []):
            self.orderbook_data.append({
                'timestamp': dt,
                'side': 'bid',
                'price': float(bid['price']),
                'size': float(bid['size'])
            })
        for ask in data.get('asks', []):
            self.orderbook_data.append({
                'timestamp': dt,
                'side': 'ask',
                'price': float(ask['price']),
                'size': float(ask['size'])
            })

    def process_price_change(self, data):
        """Process price change updates"""
        timestamp = int(data['timestamp']) / 1000  # Convert ms to seconds
        dt = datetime.fromtimestamp(timestamp, UTC).strftime('%Y-%m-%d %H:%M:%S')
        for change in data.get('changes', []):
            self.orderbook_data.append({
                'timestamp': dt,
                'side': change['side'].lower(),
                'price': float(change['price']),
                'size': float(change['size'])
            })

    def save_data(self):
        """Save collected data to CSV"""
        if not self.orderbook_data:
            logger.info("No data to save")
            return
        df = pd.DataFrame(self.orderbook_data)
        date_str = datetime.now(UTC).strftime('%Y%m%d')
        output_file = os.path.join(self.output_dir, f'orderbook_{self.asset_id}_{date_str}.csv')
        df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))
        logger.info(f"Saved {len(self.orderbook_data)} records to {output_file}")
        self.orderbook_data = []  # Clear data after saving

    async def websocket_listener(self):
        """Listen for WebSocket updates with reconnection logic"""
        while True:
            try:
                async with connect(self.ws_url) as websocket:
                    # Subscribe to market channel
                    await websocket.send(json.dumps({"type": "MARKET", "assets_ids": [self.asset_id]}))
                    logger.info(f"Subscribed to market channel for {self.asset_id}")
                    
                    while True:
                        message = await websocket.recv()
                        data = json.loads(message)[0]
                        
                        if data.get('asset_id') == self.asset_id:
                            if data['event_type'] == 'book':
                                self.process_orderbook(data)
                            elif data['event_type'] == 'price_change':
                                self.process_price_change(data)
                            
                            # Save periodically
                            current_time = time.time()
                            if current_time - self.last_save_time >= self.save_interval:
                                self.save_data()
                                self.last_save_time = current_time
                        
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await asyncio.sleep(5)  # Wait before reconnecting
            except KeyboardInterrupt:
                logger.info("Stopping")
                self.save_data()
                break

    async def run(self):
        """Main run method"""
        await self.fetch_initial_orderbook()
        await self.websocket_listener()

async def main():
    # Replace with actual ETHUSDT market ID from Polymarket
    asset_id = "19333565431154034959696570897114176039106580749219566804891884257974220925231"
    collector = PolymarketOrderbookCollector(asset_id)
    try:
        await collector.run()
    except (asyncio.exceptions.CancelledError, KeyboardInterrupt):
        logger.info("Stopping")
        collector.save_data()

if __name__ == "__main__":
    asyncio.run(main())
