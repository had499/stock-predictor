"""
Example usage of the Stock Data Loader

This script demonstrates how to use the data loader for various scenarios.
"""

from data_loader import StockDataLoader, load_stock_data, load_multiple_stocks
from datetime import datetime, timedelta
import pandas as pd

def main():
    print("=== Stock Data Loader Examples ===\n")
    
    # Example 1: Simple data loading
    print("1. Loading single stock data (AAPL):")
    print("-" * 40)
    try:
        aapl_data = load_stock_data("AAPL", "2023-01-01", "2023-12-31")
        print(f"✓ Successfully loaded {len(aapl_data)} records")
        print(f"✓ Columns: {list(aapl_data.columns)}")
        print(f"✓ Date range: {aapl_data['date'].min()} to {aapl_data['date'].max()}")
        print(f"✓ Sample data:")
        print(aapl_data.head(3))
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Multiple stocks
    print("2. Loading multiple stocks:")
    print("-" * 40)
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    try:
        multi_data = load_multiple_stocks(symbols, "2023-01-01", "2023-12-31")
        for symbol, data in multi_data.items():
            if data is not None:
                print(f"✓ {symbol}: {len(data)} records")
            else:
                print(f"✗ {symbol}: Failed to load")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Advanced usage with data loader
    print("3. Advanced usage with data loader:")
    print("-" * 40)
    try:
        loader = StockDataLoader()
        
        # Load data
        data = loader.load_stock_data("AAPL", "2023-01-01", "2023-12-31")
        print(f"✓ Loaded {len(data)} records for AAPL")
        
        # Get stock information
        info = loader.get_stock_info("AAPL")
        print(f"✓ Stock info: {info['name']} ({info['sector']})")
        
        # Show some statistics
        print(f"✓ Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print(f"✓ Average volume: {data['volume'].mean():,.0f}")
        print(f"✓ Available columns: {list(data.columns)}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 4: Different time intervals
    print("4. Loading data with different intervals:")
    print("-" * 40)
    try:
        loader = StockDataLoader()
        
        # Daily data (default)
        daily_data = loader.load_stock_data("AAPL", "2023-12-01", "2023-12-31", interval="1d")
        print(f"✓ Daily data: {len(daily_data)} records")
        
        # Weekly data
        weekly_data = loader.load_stock_data("AAPL", "2023-01-01", "2023-12-31", interval="1wk")
        print(f"✓ Weekly data: {len(weekly_data)} records")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 5: Error handling
    print("5. Error handling examples:")
    print("-" * 40)
    
    # Invalid symbol
    try:
        invalid_data = load_stock_data("INVALID_SYMBOL_123", "2023-01-01", "2023-12-31")
    except Exception as e:
        print(f"✓ Caught expected error for invalid symbol: {type(e).__name__}")
    
    # Invalid date range
    try:
        invalid_dates = load_stock_data("AAPL", "2023-12-31", "2023-01-01")
    except Exception as e:
        print(f"✓ Caught expected error for invalid date range: {type(e).__name__}")
    
    print("\n" + "="*60 + "\n")
    print("Examples completed! 🎉")

if __name__ == "__main__":
    main()
