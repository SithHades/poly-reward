import unittest
from unittest.mock import patch, MagicMock
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from polymarket_client import PolymarketClient

class TestPolymarketClient(unittest.TestCase):
    def setUp(self):
        self.mock_clob = MagicMock()
        patcher = patch('polymarket_client.ClobClient', return_value=self.mock_clob)
        self.addCleanup(patcher.stop)
        self.mock_clob_class = patcher.start()
        self.client = PolymarketClient(key='test', paper_trading=False)
        self.client.client = self.mock_clob  # Ensure the mock is used
        self.client.logger.setLevel(logging.CRITICAL)  # Silence logs during tests

    def test_get_market_data(self):
        self.mock_clob.get_market.return_value = {'market': 'data'}
        result = self.client.get_market_data('market_id')
        self.assertEqual(result, {'market': 'data'})
        self.mock_clob.get_market.assert_called_once_with('market_id')

    def test_place_order(self):
        self.mock_clob.create_and_post_order.return_value = {'order': 'placed'}
        result = self.client.place_order(0.5, 10, 'buy', 'token_id')
        self.assertEqual(result, {'order': 'placed'})
        self.mock_clob.create_and_post_order.assert_called()

    def test_place_order_paper_trading(self):
        client = PolymarketClient(key='test', paper_trading=True)
        result = client.place_order(0.5, 10, 'buy', 'token_id')
        self.assertEqual(result['status'], 'paper')
        self.assertEqual(result['price'], 0.5)
        self.assertEqual(result['size'], 10)
        self.assertEqual(result['side'], 'buy')
        self.assertEqual(result['token_id'], 'token_id')

    def test_cancel_order(self):
        self.mock_clob.cancel_order.return_value = {'order': 'cancelled'}
        result = self.client.cancel_order('order_id')
        self.assertEqual(result, {'order': 'cancelled'})
        self.mock_clob.cancel_order.assert_called_once_with('order_id')

    def test_cancel_order_paper_trading(self):
        client = PolymarketClient(key='test', paper_trading=True)
        result = client.cancel_order('order_id')
        self.assertEqual(result['status'], 'paper')
        self.assertEqual(result['order_id'], 'order_id')

    def test_get_order_status(self):
        self.mock_clob.get_order.return_value = {'order': 'status'}
        result = self.client.get_order_status('order_id')
        self.assertEqual(result, {'order': 'status'})
        self.mock_clob.get_order.assert_called_once_with('order_id')

    def test_get_positions(self):
        self.mock_clob.get_positions.return_value = {'positions': []}
        # Patch hasattr to True for get_positions
        with patch.object(self.client.client, 'get_positions', create=True, return_value={'positions': []}):
            result = self.client.get_positions('user_address')
            self.assertEqual(result, {'positions': []})

    def test_rate_limiting(self):
        # Test that _rate_limit does not raise and updates last_request_time
        before = self.client.last_request_time
        self.client._rate_limit()
        self.assertTrue(self.client.last_request_time >= before)

    def test_exponential_backoff(self):
        # Simulate failure and then success
        call_count = {'count': 0}
        def flaky_func():
            if call_count['count'] < 2:
                call_count['count'] += 1
                raise Exception('fail')
            return 'success'
        result = self.client._with_backoff(flaky_func)
        self.assertEqual(result, 'success')

    def test_error_handling_final_failure(self):
        def always_fail():
            raise Exception('fail')
        with self.assertRaises(Exception):
            self.client._with_backoff(always_fail)

if __name__ == '__main__':
    unittest.main() 