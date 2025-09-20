"""
Example usage of the preprocessing module.

This script demonstrates how to use the StockPreprocessor for various
data cleaning and preprocessing tasks.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.preprocessor import StockPreprocessor, preprocess_stock_data, clean_stock_data

def create_sample_data():
    """Create sample stock data with various issues for testing."""
    print("Creating sample stock data with issues...")
    
    # Create date range
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Generate base price data
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Introduce some issues for testing
    # 1. Missing values
    data.loc[10:12, 'close'] = np.nan
    data.loc[50:52, 'volume'] = np.nan
    
    # 2. Outliers
    data.loc[100, 'close'] = data['close'].mean() * 3  # Extreme outlier
    data.loc[150, 'volume'] = data['volume'].mean() * 5  # Volume outlier
    
    # 3. Price anomalies
    data.loc[200, 'high'] = data.loc[200, 'low'] - 1  # high < low
    data.loc[250, 'open'] = data.loc[250, 'high'] + 1  # open > high
    
    # 4. Negative values
    data.loc[300, 'volume'] = -1000  # Negative volume
    
    print(f"✓ Sample data created: {data.shape}")
    print(f"✓ Missing values: {data.isnull().sum().sum()}")
    print(f"✓ Data range: {data['date'].min()} to {data['date'].max()}")
    
    return data

def test_basic_preprocessing():
    """Test basic preprocessing functionality."""
    print("\n" + "="*60)
    print("TESTING BASIC PREPROCESSING")
    print("="*60)
    
    # Create sample data
    data = create_sample_data()
    
    # Test basic preprocessing
    print("\n1. Testing basic preprocessing...")
    preprocessor = StockPreprocessor(
        handle_missing_values='forward_fill',
        remove_outliers=True,
        outlier_method='iqr',
        validate_ohlcv=True,
        fix_price_anomalies=True,
        add_time_features=True,
        normalization_method='none'
    )
    
    processed_data = preprocessor.fit_transform(data)
    
    print(f"✓ Original shape: {data.shape}")
    print(f"✓ Processed shape: {processed_data.shape}")
    print(f"✓ Missing values after: {processed_data.isnull().sum().sum()}")
    
    # Check time features
    time_features = [col for col in processed_data.columns if col in ['year', 'month', 'day', 'dayofweek', 'is_weekend']]
    print(f"✓ Time features added: {time_features}")
    
    return processed_data

def test_outlier_removal():
    """Test different outlier removal methods."""
    print("\n" + "="*60)
    print("TESTING OUTLIER REMOVAL METHODS")
    print("="*60)
    
    # Create sample data with outliers
    data = create_sample_data()
    
    methods = ['iqr', 'zscore', 'modified_zscore']
    
    for method in methods:
        print(f"\n2. Testing {method} outlier removal...")
        
        preprocessor = StockPreprocessor(
            handle_missing_values='forward_fill',
            remove_outliers=True,
            outlier_method=method,
            outlier_threshold=3.0,
            validate_ohlcv=True,
            add_time_features=False,
            normalization_method='none'
        )
        
        processed_data = preprocessor.fit_transform(data)
        
        print(f"✓ Original shape: {data.shape}")
        print(f"✓ After {method}: {processed_data.shape}")
        print(f"✓ Rows removed: {data.shape[0] - processed_data.shape[0]}")

def test_normalization():
    """Test different normalization methods."""
    print("\n" + "="*60)
    print("TESTING NORMALIZATION METHODS")
    print("="*60)
    
    # Create sample data
    data = create_sample_data()
    
    methods = ['standard', 'robust', 'minmax', 'none']
    
    for method in methods:
        print(f"\n3. Testing {method} normalization...")
        
        preprocessor = StockPreprocessor(
            handle_missing_values='forward_fill',
            remove_outliers=False,
            normalize_columns=['open', 'high', 'low', 'close'],
            normalization_method=method,
            add_time_features=False
        )
        
        processed_data = preprocessor.fit_transform(data)
        
        if method != 'none':
            print(f"✓ Close price stats after {method}:")
            print(f"  Mean: {processed_data['close'].mean():.4f}")
            print(f"  Std: {processed_data['close'].std():.4f}")
            print(f"  Min: {processed_data['close'].min():.4f}")
            print(f"  Max: {processed_data['close'].max():.4f}")

def test_convenience_functions():
    """Test convenience functions."""
    print("\n" + "="*60)
    print("TESTING CONVENIENCE FUNCTIONS")
    print("="*60)
    
    # Create sample data
    data = create_sample_data()
    
    print("\n4. Testing preprocess_stock_data()...")
    processed1 = preprocess_stock_data(
        data,
        handle_missing_values='interpolate',
        remove_outliers=True,
        add_time_features=True
    )
    print(f"✓ Processed shape: {processed1.shape}")
    
    print("\n5. Testing clean_stock_data()...")
    cleaned = clean_stock_data(data)
    print(f"✓ Cleaned shape: {cleaned.shape}")
    print(f"✓ Missing values: {cleaned.isnull().sum().sum()}")

def test_sklearn_integration():
    """Test sklearn pipeline integration."""
    print("\n" + "="*60)
    print("TESTING SKLEARN INTEGRATION")
    print("="*60)
    
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor
        
        # Create sample data
        data = create_sample_data()
        
        print("\n6. Testing sklearn pipeline...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', StockPreprocessor(
                handle_missing_values='forward_fill',
                remove_outliers=True,
                add_time_features=True,
                normalization_method='none'
            )),
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=10, random_state=42))
        ])
        
        # Prepare features and target
        X = data[['open', 'high', 'low', 'close', 'volume']]
        y = data['close'].shift(-1).dropna()  # Next day's close price
        X = X.iloc[:-1]  # Remove last row to match y
        
        # Fit pipeline
        pipeline.fit(X, y)
        
        # Make predictions
        predictions = pipeline.predict(X)
        
        print(f"✓ Pipeline created successfully")
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Predictions shape: {predictions.shape}")
        print(f"✓ Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")
        
    except ImportError as e:
        print(f"⚠️  Sklearn not available: {e}")

def test_multiple_symbols():
    """Test preprocessing with multiple symbols."""
    print("\n" + "="*60)
    print("TESTING MULTIPLE SYMBOLS")
    print("="*60)
    
    # Create data for multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    all_data = []
    
    for i, symbol in enumerate(symbols):
        # Create date range
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        
        # Generate different price patterns
        np.random.seed(42 + i)
        base_price = 100 + i * 50
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates)),
            'symbol': symbol
        })
        
        # Ensure price relationships
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        all_data.append(data)
    
    # Combine data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"\n7. Testing multiple symbols preprocessing...")
    print(f"✓ Combined data shape: {combined_data.shape}")
    print(f"✓ Symbols: {combined_data['symbol'].unique()}")
    
    # Preprocess with groupby
    preprocessor = StockPreprocessor(
        handle_missing_values='forward_fill',
        remove_outliers=True,
        add_time_features=True,
        normalization_method='none'
    )
    
    # Apply preprocessing per symbol
    processed_data = combined_data.groupby('symbol').apply(
        lambda x: preprocessor.fit_transform(x)
    ).reset_index(drop=True)
    
    print(f"✓ Processed shape: {processed_data.shape}")
    print(f"✓ Time features: {[col for col in processed_data.columns if col in ['year', 'month', 'dayofweek']]}")
    
    return processed_data

def main():
    """Run all preprocessing examples."""
    print("=== STOCK DATA PREPROCESSING EXAMPLES ===\n")
    
    try:
        # Test basic preprocessing
        test_basic_preprocessing()
        
        # Test outlier removal
        test_outlier_removal()
        
        # Test normalization
        test_normalization()
        
        # Test convenience functions
        test_convenience_functions()
        
        # Test sklearn integration
        test_sklearn_integration()
        
        # Test multiple symbols
        test_multiple_symbols()
        
        print("\n" + "="*60)
        print("🎉 ALL PREPROCESSING TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\n💡 Preprocessing Features:")
        print("1. Missing value handling (forward_fill, backward_fill, interpolate, drop)")
        print("2. Outlier detection and removal (IQR, Z-score, Modified Z-score)")
        print("3. Data normalization (Standard, Robust, MinMax)")
        print("4. Time feature extraction")
        print("5. OHLCV data validation and fixing")
        print("6. Volume processing and smoothing")
        print("7. Sklearn pipeline integration")
        print("8. Multiple symbol support")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
