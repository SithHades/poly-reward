#!/usr/bin/env python3
"""
Test script to validate the orderbook optimization system.
This script creates sample data and tests the encoding/decoding functionality.
"""

import os
import sys
import json
import pandas as pd
import tempfile
import shutil
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from polymarket_orderbook_collector import PolymarketOrderbookCollector
from decode_orderbook_data import OrderbookDecoder

def test_optimization_system():
    """Test the complete optimization system"""
    print("Testing Polymarket Orderbook Optimization System")
    print("=" * 50)
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Initialize the collector
        collector = PolymarketOrderbookCollector(output_dir=temp_dir)
        
        # Test 1: Create sample data with long identifiers
        print("\n1. Testing short identifier creation...")
        
        # Sample long identifiers (similar to real Polymarket data)
        long_asset_ids = [
            "0x1234567890abcdef1234567890abcdef12345678901234567890abcdef1234567890abcdef12345678",
            "0xabcdef1234567890abcdef1234567890abcdef12345678901234567890abcdef1234567890abcdef12",
            "0x567890abcdef1234567890abcdef1234567890abcdef12345678901234567890abcdef1234567890ab"
        ]
        
        long_slugs = [
            "ethereum-up-or-down-january-15-3pm-et",
            "bitcoin-up-or-down-january-15-3pm-et", 
            "solana-up-or-down-january-15-4pm-et"
        ]
        
        # Test mapping creation
        short_asset_ids = [collector.get_short_asset_id(aid) for aid in long_asset_ids]
        short_slugs = [collector.get_short_slug(slug) for slug in long_slugs]
        
        print(f"Long asset IDs (avg length: {sum(len(aid) for aid in long_asset_ids) / len(long_asset_ids):.1f}):")
        for i, aid in enumerate(long_asset_ids):
            print(f"  {aid[:50]}... -> {short_asset_ids[i]}")
        
        print(f"\nLong slugs (avg length: {sum(len(slug) for slug in long_slugs) / len(long_slugs):.1f}):")
        for i, slug in enumerate(long_slugs):
            print(f"  {slug} -> {short_slugs[i]}")
        
        # Test 2: Create sample orderbook data
        print("\n2. Testing sample data creation...")
        
        sample_data = []
        for i in range(100):  # Create 100 sample records
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # Use different combinations of assets and slugs
            asset_idx = i % len(short_asset_ids)
            slug_idx = i % len(short_slugs)
            crypto = ["ethereum", "bitcoin", "solana"][slug_idx]
            
            sample_data.append({
                'timestamp': timestamp,
                'market_slug': short_slugs[slug_idx],
                'asset_id': short_asset_ids[asset_idx],
                'crypto': crypto,
                'side': 'bid' if i % 2 == 0 else 'ask',
                'price': 0.5 + (i % 10) * 0.01,
                'size': 100.0 + (i % 50) * 10.0,
                'event_type': 'book'
            })
        
        # Save sample data to CSV
        df = pd.DataFrame(sample_data)
        test_file = os.path.join(temp_dir, "test_orderbook_data.csv")
        df.to_csv(test_file, index=False)
        print(f"Created test file with {len(sample_data)} records")
        
        # Test 3: Save mappings
        print("\n3. Testing mapping persistence...")
        collector.save_mappings()
        
        # Check if mapping files exist
        mappings_dir = os.path.join(temp_dir, "mappings")
        asset_mapping_file = os.path.join(mappings_dir, "asset_id_mapping.json")
        slug_mapping_file = os.path.join(mappings_dir, "slug_mapping.json")
        
        if os.path.exists(asset_mapping_file) and os.path.exists(slug_mapping_file):
            print("✓ Mapping files created successfully")
        else:
            print("✗ Mapping files not created")
            return False
        
        # Test 4: Load and decode data
        print("\n4. Testing data decoding...")
        
        decoder = OrderbookDecoder(temp_dir)
        
        # Show mappings summary
        print("\nMapping Summary:")
        decoder.show_mappings_summary()
        
        # Decode the test file
        decoded_file = decoder.decode_csv_file(test_file)
        
        # Compare original and decoded data
        original_df = pd.read_csv(test_file)
        decoded_df = pd.read_csv(decoded_file)
        
        print(f"\nOriginal data sample (first 3 rows):")
        print(original_df[['market_slug', 'asset_id', 'crypto']].head(3))
        
        print(f"\nDecoded data sample (first 3 rows):")
        print(decoded_df[['market_slug', 'asset_id', 'crypto']].head(3))
        
        # Test 5: Calculate storage savings
        print("\n5. Testing storage savings...")
        
        # Create a comparison CSV with original long identifiers
        comparison_data = []
        for i, row in original_df.iterrows():
            # Map back to original long identifiers
            original_asset_id = decoder.reverse_asset_mapping[row['asset_id']]
            original_slug = decoder.reverse_slug_mapping[row['market_slug']]
            
            comparison_data.append({
                'timestamp': row['timestamp'],
                'market_slug': original_slug,
                'asset_id': original_asset_id,
                'crypto': row['crypto'],
                'side': row['side'],
                'price': row['price'],
                'size': row['size'],
                'event_type': row['event_type']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = os.path.join(temp_dir, "comparison_full_data.csv")
        comparison_df.to_csv(comparison_file, index=False)
        
        # Calculate file sizes
        optimized_size = os.path.getsize(test_file)
        full_size = os.path.getsize(comparison_file)
        
        savings_percentage = ((full_size - optimized_size) / full_size) * 100
        
        print(f"Optimized file size: {optimized_size:,} bytes")
        print(f"Full identifier file size: {full_size:,} bytes")
        print(f"Storage savings: {savings_percentage:.1f}%")
        
        # Test 6: Verify data integrity
        print("\n6. Testing data integrity...")
        
        # Check if decoded data matches original long identifiers
        integrity_check = True
        for i, row in decoded_df.iterrows():
            expected_asset_id = long_asset_ids[i % len(long_asset_ids)]
            expected_slug = long_slugs[i % len(long_slugs)]
            
            if row['asset_id'] != expected_asset_id or row['market_slug'] != expected_slug:
                integrity_check = False
                break
        
        if integrity_check:
            print("✓ Data integrity verified - all identifiers correctly decoded")
        else:
            print("✗ Data integrity check failed")
            return False
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print(f"Storage optimization provides {savings_percentage:.1f}% file size reduction")
        
        return True

if __name__ == "__main__":
    success = test_optimization_system()
    sys.exit(0 if success else 1)