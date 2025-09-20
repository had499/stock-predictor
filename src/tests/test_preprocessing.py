"""
Test script for the preprocessing module.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.preprocessor import StockPreprocessor, preprocess_stock_data, clean_stock_data

def test_basic_preprocessing():
    """Test basic preprocessing functionality."""
    print("Testing basic preprocessing...")
    
    # Create test data
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-10', freq='D'),
        'open': [100, 101, 102, np.nan, 104, 105, 106, 107, 108, 109],
        'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
        'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
    })
    
    # Test preprocessing
    preprocessor = StockPreprocessor(
        handle_missing_values='forward_fill',
        remove_outliers=False,
        add_time_features=True,
        normalization_method='none'
    )
    
    processed = preprocessor.fit_transform(data)
    
    # Check results
    assert processed.shape[0] == data.shape[0], "Row count should be preserved"
    assert not processed['open'].isna().any(), "Missing values should be filled"
    assert 'year' in processed.columns, "Time features should be added"
    assert 'month' in processed.columns, "Time features should be added"
    
    print("✓ Basic preprocessing test passed")
    return True

def test_outlier_removal():
    """Test outlier removal functionality."""
    print("Testing outlier removal...")
    
    # Create data with outliers
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-20', freq='D'),
        'open': [100] * 20,
        'high': [105] * 20,
        'low': [95] * 20,
        'close': [100] * 20,
        'volume': [1000000] * 20
    })
    
    # Add outliers
    data.loc[10, 'close'] = 1000  # Extreme outlier
    data.loc[15, 'volume'] = 10000000  # Volume outlier
    
    # Test outlier removal
    preprocessor = StockPreprocessor(
        handle_missing_values='none',
        remove_outliers=True,
        outlier_method='iqr',
        add_time_features=False,
        normalization_method='none'
    )
    
    processed = preprocessor.fit_transform(data)
    
    # Check that outliers are removed
    assert processed.shape[0] < data.shape[0], "Outliers should be removed"
    assert processed['close'].max() < 1000, "Extreme outlier should be removed"
    
    print("✓ Outlier removal test passed")
    return True

def test_normalization():
    """Test normalization functionality."""
    print("Testing normalization...")
    
    # Create test data
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-10', freq='D'),
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
        'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
    })
    
    # Test different normalization methods
    methods = ['standard', 'robust', 'minmax']
    
    for method in methods:
        preprocessor = StockPreprocessor(
            handle_missing_values='none',
            remove_outliers=False,
            normalize_columns=['open', 'high', 'low', 'close'],
            normalization_method=method,
            add_time_features=False
        )
        
        processed = preprocessor.fit_transform(data)
        
        # Check that normalization was applied
        if method == 'standard':
            assert abs(processed['close'].mean()) < 0.1, "Standard normalization should center around 0"
        elif method == 'minmax':
            assert processed['close'].min() >= 0, "MinMax normalization should be >= 0"
            assert processed['close'].max() <= 1, "MinMax normalization should be <= 1"
    
    print("✓ Normalization test passed")
    return True

def test_convenience_functions():
    """Test convenience functions."""
    print("Testing convenience functions...")
    
    # Create test data
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-10', freq='D'),
        'open': [100, 101, 102, np.nan, 104, 105, 106, 107, 108, 109],
        'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
        'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
    })
    
    # Test preprocess_stock_data
    processed1 = preprocess_stock_data(data, handle_missing_values='forward_fill')
    assert processed1.shape[0] == data.shape[0], "Row count should be preserved"
    
    # Test clean_stock_data
    cleaned = clean_stock_data(data)
    assert cleaned.shape[0] == data.shape[0], "Row count should be preserved"
    assert not cleaned['open'].isna().any(), "Missing values should be filled"
    
    print("✓ Convenience functions test passed")
    return True

def test_sklearn_integration():
    """Test sklearn integration."""
    print("Testing sklearn integration...")
    
    # Create test data
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-10', freq='D'),
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
        'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
    })
    
    # Test sklearn interface
    preprocessor = StockPreprocessor()
    
    # Test fit
    preprocessor.fit(data)
    assert preprocessor._fitted, "Preprocessor should be fitted"
    
    # Test transform
    processed = preprocessor.transform(data)
    assert processed.shape[0] == data.shape[0], "Row count should be preserved"
    
    # Test fit_transform
    processed2 = preprocessor.fit_transform(data)
    assert processed2.shape[0] == data.shape[0], "Row count should be preserved"
    
    # Test get_params and set_params
    params = preprocessor.get_params()
    assert 'handle_missing_values' in params, "Should have handle_missing_values parameter"
    
    preprocessor.set_params(handle_missing_values='interpolate')
    assert preprocessor.handle_missing_values == 'interpolate', "Parameter should be updated"
    
    print("✓ Sklearn integration test passed")
    return True

def test_multiple_symbols():
    """Test preprocessing with multiple symbols."""
    print("Testing multiple symbols...")
    
    # Create data for multiple symbols
    symbols = ['AAPL', 'MSFT']
    all_data = []
    
    for i, symbol in enumerate(symbols):
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', '2023-01-10', freq='D'),
            'open': [100 + i * 10] * 10,
            'high': [105 + i * 10] * 10,
            'low': [95 + i * 10] * 10,
            'close': [102 + i * 10] * 10,
            'volume': [1000000 + i * 100000] * 10,
            'symbol': symbol
        })
        all_data.append(data)
    
    # Combine data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Test preprocessing per symbol
    preprocessor = StockPreprocessor(
        handle_missing_values='none',
        remove_outliers=False,
        add_time_features=True,
        normalization_method='none'
    )
    
    # Apply preprocessing per symbol
    processed_data = combined_data.groupby('symbol').apply(
        lambda x: preprocessor.fit_transform(x)
    ).reset_index(drop=True)
    
    # Check results
    assert processed_data.shape[0] == combined_data.shape[0], "Row count should be preserved"
    assert 'year' in processed_data.columns, "Time features should be added"
    assert processed_data['symbol'].nunique() == 2, "Both symbols should be preserved"
    
    print("✓ Multiple symbols test passed")
    return True

def main():
    """Run all tests."""
    print("=== TESTING PREPROCESSING MODULE ===\n")
    
    tests = [
        test_basic_preprocessing,
        test_outlier_removal,
        test_normalization,
        test_convenience_functions,
        test_sklearn_integration,
        test_multiple_symbols
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
    
    print(f"\n=== TEST RESULTS ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return passed == total

if __name__ == "__main__":
    main()
