"""
Polymarket Orderbook Data Processor

This module provides utilities for processing and analyzing Polymarket orderbook data
collected from hourly prediction markets for crypto pairs (BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT).
"""

import polars as pl
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta


class PolymarketDataProcessor:
    """Process and analyze Polymarket orderbook data with KLINES integration."""
    
    def __init__(self, data_root: str = "orderbook_data"):
        self.data_root = Path(data_root)
        self.mappings_dir = self.data_root / "mappings"
        self.processed_dir = self.data_root / "processed"
        self.klines_dir = Path("data")
        
        # Load mappings
        self.asset_mappings = self._load_asset_mappings()
        self.slug_mappings = self._load_slug_mappings()
    
    def _load_asset_mappings(self) -> Dict[str, str]:
        """Load asset ID mappings from the mapping file."""
        mapping_file = self.mappings_dir / "asset_id_mapping.json"
        if not mapping_file.exists():
            return {}
            
        with open(mapping_file, 'r') as f:
            data = json.load(f)
            return data.get("reverse_asset_mapping", {})
    
    def _load_slug_mappings(self) -> Dict[str, str]:
        """Load slug mappings if they exist."""
        # Based on README, slug mappings might not always exist
        # We'll work with short identifiers for now
        return {}
    
    def load_orderbook_data(self, file_path: Optional[str] = None) -> pl.DataFrame:
        """Load orderbook data from consolidated CSV files."""
        if file_path:
            files = [Path(file_path)]
        else:
            # Load latest consolidated file
            files = list(self.processed_dir.glob("consolidated_all_markets_*.csv"))
            files.sort(key=lambda x: x.name)
            files = files[-1:] if files else []
        
        if not files:
            raise FileNotFoundError("No orderbook data files found")
        
        # Load data using polars for performance
        df_list = []
        for file in files:
            df = pl.read_csv(file)
            df_list.append(df)
        
        # Combine all dataframes
        combined_df = pl.concat(df_list) if len(df_list) > 1 else df_list[0]
        
        # Convert timestamp to datetime
        combined_df = combined_df.with_columns([
            pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S%.f")
        ])
        
        return combined_df
    
    def load_klines_data(self, crypto_pair: str = "ETHUSDT") -> pl.DataFrame:
        """Load 1-minute KLINES data for the specified crypto pair."""
        pattern = f"{crypto_pair}-1m-*.csv"
        klines_files = list(self.klines_dir.glob(pattern))
        
        if not klines_files:
            raise FileNotFoundError(f"No KLINES data found for {crypto_pair}")
        
        klines_files.sort()
        
        df_list = []
        for file in klines_files:
            # Assuming standard KLINES format: timestamp,open,high,low,close,volume
            df = pl.read_csv(file)
            if df.height > 0:  # Only add non-empty dataframes
                df_list.append(df)
        
        if not df_list:
            raise ValueError(f"All KLINES files for {crypto_pair} are empty")
        
        combined_df = pl.concat(df_list)
        
        # Ensure timestamp column is datetime
        if "timestamp" in combined_df.columns:
            combined_df = combined_df.with_columns([
                pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S%.f")
            ])
        elif combined_df.columns[0] in ["open_time", "time"]:
            # If first column is timestamp with different name
            first_col = combined_df.columns[0]
            combined_df = combined_df.rename({first_col: "timestamp"})
            combined_df = combined_df.with_columns([
                pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S%.f")
            ])
        
        return combined_df.sort("timestamp")
    
    def resample_orderbook_to_intervals(self, df: pl.DataFrame, interval: str = "1m") -> pl.DataFrame:
        """
        Resample orderbook data to specified intervals with OHLC-style metrics.
        
        Args:
            df: Orderbook dataframe
            interval: Resampling interval (e.g., "1m", "5m", "1h")
        """
        # Group by crypto, market_slug, and time intervals
        resampled = (
            df.group_by_dynamic(
                "timestamp", 
                every=interval,
                by=["crypto", "market_slug", "asset_id"]
            ).agg([
                # Price metrics
                pl.col("price").first().alias("open_price"),
                pl.col("price").max().alias("high_price"),
                pl.col("price").min().alias("low_price"),
                pl.col("price").last().alias("close_price"),
                
                # Volume metrics
                pl.col("size").sum().alias("total_volume"),
                pl.col("size").count().alias("tick_count"),
                
                # Bid/Ask analysis
                pl.col("price").filter(pl.col("side") == "buy").mean().alias("avg_bid"),
                pl.col("price").filter(pl.col("side") == "sell").mean().alias("avg_ask"),
                pl.col("size").filter(pl.col("side") == "buy").sum().alias("bid_volume"),
                pl.col("size").filter(pl.col("side") == "sell").sum().alias("ask_volume"),
            ])
        )
        
        # Calculate spread and other derived metrics
        resampled = resampled.with_columns([
            (pl.col("avg_ask") - pl.col("avg_bid")).alias("bid_ask_spread"),
            (pl.col("bid_volume") / (pl.col("bid_volume") + pl.col("ask_volume"))).alias("bid_ratio"),
        ])
        
        return resampled.sort(["timestamp", "crypto", "market_slug"])
    
    def calculate_market_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate market microstructure features for analysis."""
        
        features = df.with_columns([
            # Price features
            (pl.col("close_price") / pl.col("open_price") - 1).alias("price_return"),
            (pl.col("high_price") / pl.col("low_price") - 1).alias("price_range"),
            
            # Volume features  
            pl.col("total_volume").log().alias("log_volume"),
            (pl.col("bid_volume") - pl.col("ask_volume")).alias("volume_imbalance"),
            
            # Market efficiency features
            pl.col("bid_ask_spread").alias("spread"),
            (pl.col("bid_ask_spread") / ((pl.col("avg_bid") + pl.col("avg_ask")) / 2)).alias("spread_pct"),
        ])
        
        # Calculate rolling features
        window_sizes = [5, 10, 30]  # 5min, 10min, 30min windows for 1min data
        
        for window in window_sizes:
            features = features.with_columns([
                pl.col("price_return").rolling_mean(window).over("crypto", "market_slug").alias(f"price_return_ma_{window}"),
                pl.col("total_volume").rolling_mean(window).over("crypto", "market_slug").alias(f"volume_ma_{window}"),
                pl.col("spread_pct").rolling_mean(window).over("crypto", "market_slug").alias(f"spread_ma_{window}"),
                pl.col("bid_ratio").rolling_mean(window).over("crypto", "market_slug").alias(f"bid_ratio_ma_{window}"),
            ])
        
        return features
    
    def merge_with_klines(self, orderbook_df: pl.DataFrame, klines_df: pl.DataFrame, 
                         crypto_pair: str) -> pl.DataFrame:
        """Merge orderbook data with KLINES data for correlation analysis."""
        
        # Filter orderbook for the specific crypto
        crypto_map = {
            "ETHUSDT": "ethereum",
            "BTCUSDT": "bitcoin", 
            "SOLUSDT": "solana",
            "XRPUSDT": "xrp"
        }
        
        crypto_name = crypto_map.get(crypto_pair, crypto_pair.replace("USDT", "").lower())
        ob_filtered = orderbook_df.filter(pl.col("crypto") == crypto_name)
        
        # Ensure both datasets have the same timestamp resolution
        klines_1min = klines_df.with_columns([
            pl.col("timestamp").dt.truncate("1m").alias("timestamp_1m")
        ])
        
        ob_1min = ob_filtered.with_columns([
            pl.col("timestamp").dt.truncate("1m").alias("timestamp_1m")
        ])
        
        # Merge on 1-minute timestamps
        merged = ob_1min.join(
            klines_1min,
            left_on="timestamp_1m",
            right_on="timestamp_1m",
            how="inner"
        )
        
        return merged
    
    def identify_market_opportunities(self, df: pl.DataFrame, 
                                    min_spread_threshold: float = 0.02) -> pl.DataFrame:
        """
        Identify potential market making opportunities based on:
        1. High spreads
        2. Volume imbalances  
        3. Price momentum
        4. Market inefficiencies
        """
        
        opportunities = df.filter(
            (pl.col("spread_pct") > min_spread_threshold) &  # Wide spreads
            (pl.col("total_volume") > pl.col("volume_ma_10")) &  # Above average volume
            (pl.col("tick_count") > 5)  # Sufficient market activity
        ).with_columns([
            # Opportunity scoring
            (pl.col("spread_pct") * pl.col("total_volume") * 
             (1 + pl.col("volume_imbalance").abs())).alias("opportunity_score"),
            
            # Market direction signals
            pl.when(pl.col("bid_ratio") > 0.6).then("bullish")
            .when(pl.col("bid_ratio") < 0.4).then("bearish")
            .otherwise("neutral").alias("market_sentiment"),
            
            # Risk indicators
            pl.col("price_range").alias("volatility_risk"),
        ])
        
        return opportunities.sort("opportunity_score", descending=True)


def create_sample_analysis():
    """Create a sample analysis script demonstrating the capabilities."""
    
    sample_code = '''
# Example usage of PolymarketDataProcessor

from analysis.data_processor import PolymarketDataProcessor
import polars as pl

# Initialize processor
processor = PolymarketDataProcessor()

# Load latest orderbook data
print("Loading orderbook data...")
orderbook_df = processor.load_orderbook_data()
print(f"Loaded {len(orderbook_df)} orderbook records")

# Load KLINES data for Ethereum
print("Loading ETHUSDT KLINES data...")
eth_klines = processor.load_klines_data("ETHUSDT")
print(f"Loaded {len(eth_klines)} KLINES records")

# Resample orderbook data to 1-minute intervals
print("Resampling orderbook data...")
resampled = processor.resample_orderbook_to_intervals(orderbook_df, "1m")

# Calculate market features
print("Calculating market features...")
features = processor.calculate_market_features(resampled)

# Merge with KLINES for correlation analysis
print("Merging with KLINES data...")
merged_data = processor.merge_with_klines(features, eth_klines, "ETHUSDT")

# Identify opportunities
print("Identifying market making opportunities...")
opportunities = processor.identify_market_opportunities(features)

print(f"Found {len(opportunities)} potential opportunities")
print("Top 5 opportunities:")
print(opportunities.head())
'''
    
    return sample_code


if __name__ == "__main__":
    # Quick test of the processor
    processor = PolymarketDataProcessor()
    print("PolymarketDataProcessor initialized successfully!")
    print(f"Found {len(processor.asset_mappings)} asset mappings")