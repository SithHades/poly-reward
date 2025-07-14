#!/usr/bin/env python3
"""
Utility script to decode optimized orderbook data back to original format.
This script reads the mapping files and converts short identifiers back to full asset_ids and slugs.
"""

import os
import json
import pandas as pd
import argparse
import sys
from datetime import datetime
from typing import Dict, Optional

class OrderbookDecoder:
    def __init__(self, data_dir: str = "orderbook_data"):
        self.data_dir = data_dir
        self.mappings_dir = os.path.join(data_dir, "mappings")
        
        # Load mappings
        self.asset_id_mapping = {}
        self.slug_mapping = {}
        self.reverse_asset_mapping = {}
        self.reverse_slug_mapping = {}
        
        self.load_mappings()
    
    def load_mappings(self):
        """Load the mapping files"""
        try:
            # Load asset ID mappings
            asset_mapping_file = os.path.join(self.mappings_dir, "asset_id_mapping.json")
            if os.path.exists(asset_mapping_file):
                with open(asset_mapping_file, 'r') as f:
                    data = json.load(f)
                    self.asset_id_mapping = data.get('asset_id_mapping', {})
                    self.reverse_asset_mapping = data.get('reverse_asset_mapping', {})
                    print(f"Loaded {len(self.asset_id_mapping)} asset ID mappings")
            else:
                print(f"Warning: Asset mapping file not found at {asset_mapping_file}")

            # Load slug mappings
            slug_mapping_file = os.path.join(self.mappings_dir, "slug_mapping.json")
            if os.path.exists(slug_mapping_file):
                with open(slug_mapping_file, 'r') as f:
                    data = json.load(f)
                    self.slug_mapping = data.get('slug_mapping', {})
                    self.reverse_slug_mapping = data.get('reverse_slug_mapping', {})
                    print(f"Loaded {len(self.slug_mapping)} slug mappings")
            else:
                print(f"Warning: Slug mapping file not found at {slug_mapping_file}")
        except Exception as e:
            print(f"Error loading mappings: {e}")
            sys.exit(1)
    
    def decode_asset_id(self, short_id: str) -> str:
        """Decode short asset ID back to original"""
        return self.reverse_asset_mapping.get(short_id, short_id)
    
    def decode_slug(self, short_slug: str) -> str:
        """Decode short slug back to original"""
        return self.reverse_slug_mapping.get(short_slug, short_slug)
    
    def decode_csv_file(self, input_file: str, output_file: Optional[str] = None) -> str:
        """Decode a CSV file with optimized data"""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Check if it's already decoded or needs decoding
        if 'asset_id' in df.columns and 'market_slug' in df.columns:
            # Check if the data looks like it needs decoding
            sample_asset_id = df['asset_id'].iloc[0] if len(df) > 0 else ""
            sample_slug = df['market_slug'].iloc[0] if len(df) > 0 else ""
            
            if sample_asset_id.startswith('a') and sample_slug.startswith('s'):
                # Decode the data
                df['asset_id'] = df['asset_id'].apply(self.decode_asset_id)
                df['market_slug'] = df['market_slug'].apply(self.decode_slug)
                print(f"Decoded {len(df)} records")
            else:
                print("Data appears to already be in decoded format")
        else:
            print("CSV file does not contain expected columns (asset_id, market_slug)")
            return input_file
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_decoded.csv"
        
        # Save decoded data
        df.to_csv(output_file, index=False)
        print(f"Decoded data saved to: {output_file}")
        
        return output_file
    
    def decode_directory(self, input_dir: str, output_dir: Optional[str] = None):
        """Decode all CSV files in a directory"""
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir is None:
            output_dir = os.path.join(input_dir, "decoded")
        
        os.makedirs(output_dir, exist_ok=True)
        
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"No CSV files found in {input_dir}")
            return
        
        print(f"Found {len(csv_files)} CSV files to decode")
        
        for csv_file in csv_files:
            input_path = os.path.join(input_dir, csv_file)
            output_path = os.path.join(output_dir, csv_file)
            
            try:
                self.decode_csv_file(input_path, output_path)
            except Exception as e:
                print(f"Error decoding {csv_file}: {e}")
    
    def show_mappings_summary(self):
        """Display summary of available mappings"""
        print("\n=== Mapping Summary ===")
        print(f"Asset ID mappings: {len(self.asset_id_mapping)}")
        print(f"Slug mappings: {len(self.slug_mapping)}")
        
        if self.asset_id_mapping:
            print("\nSample asset ID mappings:")
            for i, (original, short) in enumerate(list(self.asset_id_mapping.items())[:5]):
                print(f"  {short} -> {original[:50]}{'...' if len(original) > 50 else ''}")
            if len(self.asset_id_mapping) > 5:
                print(f"  ... and {len(self.asset_id_mapping) - 5} more")
        
        if self.slug_mapping:
            print("\nSample slug mappings:")
            for i, (original, short) in enumerate(list(self.slug_mapping.items())[:5]):
                print(f"  {short} -> {original}")
            if len(self.slug_mapping) > 5:
                print(f"  ... and {len(self.slug_mapping) - 5} more")

def main():
    parser = argparse.ArgumentParser(description="Decode optimized orderbook data")
    parser.add_argument("--data-dir", default="orderbook_data", 
                      help="Directory containing orderbook data and mappings")
    parser.add_argument("--input-file", help="Single CSV file to decode")
    parser.add_argument("--input-dir", help="Directory of CSV files to decode")
    parser.add_argument("--output-file", help="Output file for single file decoding")
    parser.add_argument("--output-dir", help="Output directory for batch decoding")
    parser.add_argument("--show-mappings", action="store_true", 
                      help="Show mapping summary")
    
    args = parser.parse_args()
    
    # Initialize decoder
    decoder = OrderbookDecoder(args.data_dir)
    
    # Show mappings summary if requested
    if args.show_mappings:
        decoder.show_mappings_summary()
    
    # Process input
    if args.input_file:
        try:
            decoder.decode_csv_file(args.input_file, args.output_file)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.input_dir:
        try:
            decoder.decode_directory(args.input_dir, args.output_dir)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif not args.show_mappings:
        # Default behavior: decode all files in raw directory
        raw_dir = os.path.join(args.data_dir, "raw")
        if os.path.exists(raw_dir):
            try:
                decoder.decode_directory(raw_dir)
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)
        else:
            print("No input specified and raw directory not found. Use --help for usage.")

if __name__ == "__main__":
    main()