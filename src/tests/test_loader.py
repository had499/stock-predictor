"""
Simple test script for the data loader
"""

from data_loader import load_stock_data

def test_basic_loading():
    """Test basic data loading functionality"""
    print("Testing basic data loading...")
    
    try:
        # Test loading AAPL data
        data = load_stock_data("AAPL", "2023-01-01", "2023-01-31")
        
        print(f"✓ Successfully loaded {len(data)} records")
        print(f"✓ Columns: {list(data.columns)}")
        print(f"✓ Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"✓ Sample data:")
        print(data.head(2))
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_loading()
    if success:
        print("\n🎉 Data loader test passed!")
    else:
        print("\n❌ Data loader test failed!")
