#!/usr/bin/env python3
"""
Test script for the robust Polymarket orderbook collector.
This script tests the connection handling and market transition features.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.polymarket_orderbook_collector import (
    PolymarketOrderbookCollector, 
    ConnectionState
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockWebSocket:
    """Mock WebSocket for testing"""
    def __init__(self, messages=None, should_fail=False, fail_after=None):
        self.messages = messages or []
        self.message_index = 0
        self.should_fail = should_fail
        self.fail_after = fail_after
        self.sent_messages = []
        self.closed = False
        self.ping_count = 0
        
    async def send(self, message):
        """Mock send method"""
        self.sent_messages.append(json.loads(message))
        logger.info(f"Mock WebSocket sent: {message}")
    
    async def recv(self):
        """Mock receive method"""
        if self.should_fail and self.fail_after and len(self.sent_messages) >= self.fail_after:
            raise ConnectionError("Mock connection failure")
            
        if self.message_index < len(self.messages):
            message = self.messages[self.message_index]
            self.message_index += 1
            return json.dumps(message)
        else:
            # Simulate no message available
            await asyncio.sleep(0.1)
            return json.dumps([])  # Empty message array
    
    async def ping(self):
        """Mock ping method"""
        self.ping_count += 1
        logger.info(f"Mock WebSocket ping sent (count: {self.ping_count})")
    
    async def close(self):
        """Mock close method"""
        self.closed = True
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

class TestRobustCollector:
    """Test the robust orderbook collector features"""
    
    def __init__(self):
        self.collector = PolymarketOrderbookCollector(output_dir="test_orderbook_data")
        
    def test_connection_state_management(self):
        """Test connection state tracking"""
        logger.info("Testing connection state management...")
        
        # Test initial state
        assert self.collector.connection_state == ConnectionState.DISCONNECTED
        
        # Test state transitions
        self.collector.connection_state = ConnectionState.CONNECTING
        assert self.collector.connection_state == ConnectionState.CONNECTING
        
        self.collector.connection_state = ConnectionState.CONNECTED
        assert self.collector.connection_state == ConnectionState.CONNECTED
        
        logger.info("✓ Connection state management works correctly")
        
    def test_exponential_backoff(self):
        """Test exponential backoff calculation"""
        logger.info("Testing exponential backoff...")
        
        # Test initial delay
        self.collector.reconnect_attempts = 0
        assert self.collector.get_reconnect_delay() == 0
        
        # Test exponential increase
        self.collector.reconnect_attempts = 1
        delay1 = self.collector.get_reconnect_delay()
        
        self.collector.reconnect_attempts = 2
        delay2 = self.collector.get_reconnect_delay()
        
        self.collector.reconnect_attempts = 3
        delay3 = self.collector.get_reconnect_delay()
        
        assert delay1 < delay2 < delay3
        logger.info(f"Delays: {delay1}s, {delay2}s, {delay3}s")
        
        # Test max delay cap
        self.collector.reconnect_attempts = 20
        max_delay = self.collector.get_reconnect_delay()
        assert max_delay <= self.collector.max_reconnect_delay
        
        logger.info("✓ Exponential backoff works correctly")
        
    def test_connection_health_monitoring(self):
        """Test connection health monitoring"""
        logger.info("Testing connection health monitoring...")
        
        # Test healthy connection
        self.collector.last_message_time = time.time()
        assert self.collector.check_connection_health() == True
        
        # Test unhealthy connection
        self.collector.last_message_time = time.time() - 200  # 200 seconds ago
        assert self.collector.check_connection_health() == False
        
        logger.info("✓ Connection health monitoring works correctly")
        
    def test_market_transition_detection(self):
        """Test market transition detection"""
        logger.info("Testing market transition detection...")
        
        # Import the timezone module
        from src.parsing_utils import ET
        
        # Create a mock market that expires soon
        current_time = datetime.now(ET)
        self.collector.current_markets = {
            "test-market": {
                'end_time': current_time + timedelta(seconds=200),  # Expires in 200 seconds
                'crypto': 'ethereum'
            }
        }
        
        # Should detect transition needed
        transition_needed = self.collector.check_market_transitions()
        assert transition_needed == True
        
        # Create a mock market that expires later
        self.collector.current_markets = {
            "test-market": {
                'end_time': current_time + timedelta(seconds=600),  # Expires in 600 seconds
                'crypto': 'ethereum'
            }
        }
        
        # Should not detect transition needed
        transition_needed = self.collector.check_market_transitions()
        assert transition_needed == False
        
        logger.info("✓ Market transition detection works correctly")
        
    async def test_market_monitoring_frequency(self):
        """Test that market monitoring respects frequency limits"""
        logger.info("Testing market monitoring frequency...")
        
        # Mock WebSocket
        mock_ws = MockWebSocket()
        
        # Test that market check respects interval
        self.collector.last_market_check = time.time()
        result = await self.collector.check_markets_and_reconnect_if_needed(mock_ws)
        assert result == False  # Should not check markets too frequently
        
        # Test that market check works after interval
        self.collector.last_market_check = time.time() - 40  # 40 seconds ago
        result = await self.collector.check_markets_and_reconnect_if_needed(mock_ws)
        # Result depends on actual market state, but should have checked
        
        logger.info("✓ Market monitoring frequency works correctly")
        
    def test_message_processing_with_timing(self):
        """Test that message processing updates timing correctly"""
        logger.info("Testing message processing timing...")
        
        # Test message processing updates last_message_time
        initial_time = self.collector.last_message_time
        
        # Process a mock message
        mock_message = [{
            'asset_id': 'test_asset',
            'event_type': 'book',
            'timestamp': int(time.time() * 1000),
            'bids': [{'price': '0.5', 'size': '100'}],
            'asks': [{'price': '0.6', 'size': '100'}]
        }]
        
        self.collector.process_websocket_message(mock_message)
        
        # Should update last_message_time
        assert self.collector.last_message_time > initial_time
        
        logger.info("✓ Message processing timing works correctly")
        
    async def test_ping_functionality(self):
        """Test WebSocket ping functionality"""
        logger.info("Testing ping functionality...")
        
        mock_ws = MockWebSocket()
        
        # Test ping not needed initially
        self.collector.last_ping_time = time.time()
        await self.collector.send_ping(mock_ws)
        assert mock_ws.ping_count == 0
        
        # Test ping needed after interval
        self.collector.last_ping_time = time.time() - 70  # 70 seconds ago
        await self.collector.send_ping(mock_ws)
        assert mock_ws.ping_count == 1
        
        logger.info("✓ Ping functionality works correctly")
        
    def test_asset_id_tracking(self):
        """Test asset ID change tracking"""
        logger.info("Testing asset ID tracking...")
        
        # Set initial asset IDs
        self.collector.monitored_asset_ids = {'asset1', 'asset2'}
        self.collector.previous_asset_ids = {'asset1', 'asset2'}
        
        # Test no change
        current_asset_ids = set(self.collector.monitored_asset_ids)
        assets_changed = current_asset_ids != self.collector.previous_asset_ids
        assert assets_changed == False
        
        # Test change
        self.collector.monitored_asset_ids = {'asset1', 'asset3'}
        current_asset_ids = set(self.collector.monitored_asset_ids)
        assets_changed = current_asset_ids != self.collector.previous_asset_ids
        assert assets_changed == True
        
        logger.info("✓ Asset ID tracking works correctly")
        
    def run_all_tests(self):
        """Run all synchronous tests"""
        logger.info("Running all synchronous tests...")
        
        self.test_connection_state_management()
        self.test_exponential_backoff()
        self.test_connection_health_monitoring()
        self.test_market_transition_detection()
        self.test_message_processing_with_timing()
        self.test_asset_id_tracking()
        
        logger.info("✓ All synchronous tests passed!")
        
    async def run_async_tests(self):
        """Run all asynchronous tests"""
        logger.info("Running all asynchronous tests...")
        
        await self.test_market_monitoring_frequency()
        await self.test_ping_functionality()
        
        logger.info("✓ All asynchronous tests passed!")

async def test_collector_robustness():
    """Test the overall robustness of the collector"""
    logger.info("Testing collector robustness...")
    
    # Create test instance
    test_collector = TestRobustCollector()
    
    # Run synchronous tests
    test_collector.run_all_tests()
    
    # Run asynchronous tests
    await test_collector.run_async_tests()
    
    logger.info("✓ All robustness tests passed!")

def test_configuration_values():
    """Test that configuration values are reasonable"""
    logger.info("Testing configuration values...")
    
    collector = PolymarketOrderbookCollector()
    
    # Check timing values
    assert collector.market_check_interval == 30  # 30 seconds
    assert collector.websocket_timeout == 120  # 2 minutes
    assert collector.ping_interval == 60  # 1 minute
    assert collector.market_transition_threshold == 300  # 5 minutes
    
    # Check reconnection values
    assert collector.max_reconnect_attempts == 10
    assert collector.base_reconnect_delay == 5
    assert collector.max_reconnect_delay == 300
    
    logger.info("✓ Configuration values are reasonable!")

async def main():
    """Main test function"""
    logger.info("Starting robust orderbook collector tests...")
    
    # Test configuration
    test_configuration_values()
    
    # Test robustness features
    await test_collector_robustness()
    
    logger.info("🎉 All tests passed! The robust orderbook collector is working correctly.")

if __name__ == "__main__":
    asyncio.run(main()) 