#!/usr/bin/env python3
"""
Simple test script to validate the orderbook optimization system without pandas.
This script creates sample data and tests the encoding/decoding functionality.
"""

import os
import sys
import json
import tempfile
import csv
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from polymarket_orderbook_collector import PolymarketOrderbookCollector
from decode_orderbook_data import OrderbookDecoder

def test_optimization_system():
    """Test the complete optimization system"""
    print("Testing Polymarket Orderbook Optimization System (Simple)")
    print("=" * 55)
    
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
        
        # Save sample data to CSV without pandas
        test_file = os.path.join(temp_dir, "test_orderbook_data.csv")
        with open(test_file, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'market_slug', 'asset_id', 'crypto', 'side', 'price', 'size', 'event_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_data)
        
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
        
        # Test 4: Load and decode data (without pandas)
        print("\n4. Testing data decoding...")
        
        # Create a simple decoder test without pandas
        # Load mapping files manually
        with open(asset_mapping_file, 'r') as f:
            asset_data = json.load(f)
            reverse_asset_mapping = asset_data['reverse_asset_mapping']
        
        with open(slug_mapping_file, 'r') as f:
            slug_data = json.load(f)
            reverse_slug_mapping = slug_data['reverse_slug_mapping']
        
        print(f"\nMapping Summary:")
        print(f"Asset ID mappings: {len(reverse_asset_mapping)}")
        print(f"Slug mappings: {len(reverse_slug_mapping)}")
        
        # Read and decode test file
        decoded_file = os.path.join(temp_dir, "test_orderbook_data_decoded.csv")
        
        with open(test_file, 'r', newline='') as infile, open(decoded_file, 'w', newline='') as outfile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                # Decode the short identifiers
                row['asset_id'] = reverse_asset_mapping.get(row['asset_id'], row['asset_id'])
                row['market_slug'] = reverse_slug_mapping.get(row['market_slug'], row['market_slug'])
                writer.writerow(row)
        
        print(f"Decoded data saved to: {decoded_file}")
        
        # Test 5: Calculate storage savings
        print("\n5. Testing storage savings...")
        
        # Create a comparison CSV with original long identifiers
        comparison_file = os.path.join(temp_dir, "comparison_full_data.csv")
        
        with open(comparison_file, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'market_slug', 'asset_id', 'crypto', 'side', 'price', 'size', 'event_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, row in enumerate(sample_data):
                # Map back to original long identifiers
                asset_idx = i % len(long_asset_ids)
                slug_idx = i % len(long_slugs)
                
                comparison_row = row.copy()
                comparison_row['asset_id'] = long_asset_ids[asset_idx]
                comparison_row['market_slug'] = long_slugs[slug_idx]
                writer.writerow(comparison_row)
        
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
        row_count = 0
        
        with open(decoded_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                expected_asset_id = long_asset_ids[i % len(long_asset_ids)]
                expected_slug = long_slugs[i % len(long_slugs)]
                
                if row['asset_id'] != expected_asset_id or row['market_slug'] != expected_slug:
                    integrity_check = False
                    break
                row_count += 1
        
        if integrity_check:
            print(f"✓ Data integrity verified - all {row_count} identifiers correctly decoded")
        else:
            print("✗ Data integrity check failed")
            return False
        
        # Test 7: Test multi-crypto functionality
        print("\n7. Testing multi-crypto functionality...")
        
        crypto_counts = {}
        for row in sample_data:
            crypto = row['crypto']
            crypto_counts[crypto] = crypto_counts.get(crypto, 0) + 1
        
        print("Data distribution by crypto:")
        for crypto, count in crypto_counts.items():
            print(f"  {crypto}: {count} records")
        
        print("\n" + "=" * 55)
        print("All tests passed! ✓")
        print(f"Storage optimization provides {savings_percentage:.1f}% file size reduction")
        print("System ready for multi-crypto orderbook collection")
        
        return True

if __name__ == "__main__":
    success = test_optimization_system()
    sys.exit(0 if success else 1)
"""
Simple test script to validate the orderbook optimization system without pandas.
This script creates sample data and tests the encoding/decoding functionality.
"""

import os
import sys
import json
import tempfile
import csv
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from polymarket_orderbook_collector import PolymarketOrderbookCollector
from decode_orderbook_data import OrderbookDecoder

def test_optimization_system():
    """Test the complete optimization system"""
    print("Testing Polymarket Orderbook Optimization System (Simple)")
    print("=" * 55)
    
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
        
        # Save sample data to CSV without pandas
        test_file = os.path.join(temp_dir, "test_orderbook_data.csv")
        with open(test_file, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'market_slug', 'asset_id', 'crypto', 'side', 'price', 'size', 'event_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_data)
        
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
        
        # Test 4: Load and decode data (without pandas)
        print("\n4. Testing data decoding...")
        
        # Create a simple decoder test without pandas
        # Load mapping files manually
        with open(asset_mapping_file, 'r') as f:
            asset_data = json.load(f)
            reverse_asset_mapping = asset_data['reverse_asset_mapping']
        
        with open(slug_mapping_file, 'r') as f:
            slug_data = json.load(f)
            reverse_slug_mapping = slug_data['reverse_slug_mapping']
        
        print(f"\nMapping Summary:")
        print(f"Asset ID mappings: {len(reverse_asset_mapping)}")
        print(f"Slug mappings: {len(reverse_slug_mapping)}")
        
        # Read and decode test file
        decoded_file = os.path.join(temp_dir, "test_orderbook_data_decoded.csv")
        
        with open(test_file, 'r', newline='') as infile, open(decoded_file, 'w', newline='') as outfile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                # Decode the short identifiers
                row['asset_id'] = reverse_asset_mapping.get(row['asset_id'], row['asset_id'])
                row['market_slug'] = reverse_slug_mapping.get(row['market_slug'], row['market_slug'])
                writer.writerow(row)
        
        print(f"Decoded data saved to: {decoded_file}")
        
        # Test 5: Calculate storage savings
        print("\n5. Testing storage savings...")
        
        # Create a comparison CSV with original long identifiers
        comparison_file = os.path.join(temp_dir, "comparison_full_data.csv")
        
        with open(comparison_file, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'market_slug', 'asset_id', 'crypto', 'side', 'price', 'size', 'event_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, row in enumerate(sample_data):
                # Map back to original long identifiers
                asset_idx = i % len(long_asset_ids)
                slug_idx = i % len(long_slugs)
                
                comparison_row = row.copy()
                comparison_row['asset_id'] = long_asset_ids[asset_idx]
                comparison_row['market_slug'] = long_slugs[slug_idx]
                writer.writerow(comparison_row)
        
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
        row_count = 0
        
        with open(decoded_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                expected_asset_id = long_asset_ids[i % len(long_asset_ids)]
                expected_slug = long_slugs[i % len(long_slugs)]
                
                if row['asset_id'] != expected_asset_id or row['market_slug'] != expected_slug:
                    integrity_check = False
                    break
                row_count += 1
        
        if integrity_check:
            print(f"✓ Data integrity verified - all {row_count} identifiers correctly decoded")
        else:
            print("✗ Data integrity check failed")
            return False
        
        # Test 7: Test multi-crypto functionality
        print("\n7. Testing multi-crypto functionality...")
        
        crypto_counts = {}
        for row in sample_data:
            crypto = row['crypto']
            crypto_counts[crypto] = crypto_counts.get(crypto, 0) + 1
        
        print("Data distribution by crypto:")
        for crypto, count in crypto_counts.items():
            print(f"  {crypto}: {count} records")
        
        print("\n" + "=" * 55)
        print("All tests passed! ✓")
        print(f"Storage optimization provides {savings_percentage:.1f}% file size reduction")
        print("System ready for multi-crypto orderbook collection")
        
        return True

if __name__ == "__main__":
    success = test_optimization_system()
    sys.exit(0 if success else 1)