"""
Complete pipeline example showing data loading, preprocessing, and feature engineering.

This script demonstrates how to use all three modules together in a complete
machine learning pipeline for stock prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_loader import StockDataLoader
from preprocessing.preprocessor import StockPreprocessor
from features.feature_engineering import FeatureEngineer

def create_mock_data():
    """Create mock stock data for demonstration."""
    print("Creating mock stock data...")
    
    # Create date range
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Generate realistic stock data
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
    
    # Add some issues for preprocessing demonstration
    data.loc[10:12, 'close'] = np.nan  # Missing values
    data.loc[100, 'close'] = data['close'].mean() * 3  # Outlier
    data.loc[200, 'volume'] = -1000  # Negative volume
    
    print(f"✓ Mock data created: {data.shape}")
    return data

def test_complete_pipeline():
    """Test the complete pipeline with mock data."""
    print("\n" + "="*60)
    print("TESTING COMPLETE PIPELINE")
    print("="*60)
    
    # Create mock data
    mock_data = create_mock_data()
    
    print("\n1. Data Loading Stage...")
    # Simulate data loading (normally would use StockDataLoader)
    print(f"✓ Data loaded: {mock_data.shape}")
    print(f"✓ Columns: {list(mock_data.columns)}")
    print(f"✓ Date range: {mock_data['date'].min()} to {mock_data['date'].max()}")
    print(f"✓ Missing values: {mock_data.isnull().sum().sum()}")
    
    print("\n2. Preprocessing Stage...")
    # Apply preprocessing
    preprocessor = StockPreprocessor(
        handle_missing_values='forward_fill',
        remove_outliers=True,
        outlier_method='iqr',
        validate_ohlcv=True,
        fix_price_anomalies=True,
        add_time_features=True,
        normalization_method='none'  # Don't normalize for feature engineering
    )
    
    preprocessed_data = preprocessor.fit_transform(mock_data)
    
    print(f"✓ Preprocessed shape: {preprocessed_data.shape}")
    print(f"✓ Missing values after: {preprocessed_data.isnull().sum().sum()}")
    print(f"✓ Time features added: {[col for col in preprocessed_data.columns if col in ['year', 'month', 'dayofweek']]}")
    
    print("\n3. Feature Engineering Stage...")
    # Apply feature engineering
    feature_engineer = FeatureEngineer(
        add_price_features=True,
        add_volume_features=True,
        add_technical_indicators=True,
        add_momentum_indicators=True,
        add_volatility_indicators=True,
        add_trend_indicators=True
    )
    
    features_data = feature_engineer.fit_transform(preprocessed_data)
    
    print(f"✓ Features shape: {features_data.shape}")
    print(f"✓ New features added: {features_data.shape[1] - preprocessed_data.shape[1]}")
    
    # Show some feature examples
    feature_cols = [col for col in features_data.columns if col not in preprocessed_data.columns]
    print(f"✓ Sample features: {feature_cols[:10]}")
    
    return features_data

def test_sklearn_pipeline():
    """Test sklearn pipeline integration."""
    print("\n" + "="*60)
    print("TESTING SKLEARN PIPELINE")
    print("="*60)
    
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Create mock data
        mock_data = create_mock_data()
        
        print("\n4. Creating sklearn pipeline...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', StockPreprocessor(
                handle_missing_values='forward_fill',
                remove_outliers=True,
                add_time_features=True,
                normalization_method='none'
            )),
            ('feature_engineer', FeatureEngineer(
                add_price_features=True,
                add_technical_indicators=True,
                add_momentum_indicators=True,
                add_volatility_indicators=True,
                add_trend_indicators=True
            )),
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
        ])
        
        print("✓ Pipeline created successfully")
        
        # Prepare data for training
        print("\n5. Preparing training data...")
        
        # Create target variable (next day's close price)
        mock_data['target'] = mock_data['close'].shift(-1)
        
        # Remove rows with missing target
        mock_data = mock_data.dropna(subset=['target'])
        
        # Separate features and target
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        X = mock_data[feature_cols]
        y = mock_data['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        
        # Train pipeline
        print("\n6. Training pipeline...")
        pipeline.fit(X_train, y_train)
        print("✓ Pipeline trained successfully")
        
        # Make predictions
        print("\n7. Making predictions...")
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✓ Mean Squared Error: {mse:.4f}")
        print(f"✓ R² Score: {r2:.4f}")
        print(f"✓ Prediction range: {y_pred.min():.2f} to {y_pred.max():.2f}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Sklearn not available: {e}")
        return False

def test_multiple_symbols_pipeline():
    """Test pipeline with multiple symbols."""
    print("\n" + "="*60)
    print("TESTING MULTIPLE SYMBOLS PIPELINE")
    print("="*60)
    
    # Create mock data for multiple symbols
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
    
    print(f"\n8. Multiple symbols data: {combined_data.shape}")
    print(f"✓ Symbols: {combined_data['symbol'].unique()}")
    
    # Apply preprocessing per symbol
    print("\n9. Applying preprocessing per symbol...")
    preprocessor = StockPreprocessor(
        handle_missing_values='forward_fill',
        remove_outliers=True,
        add_time_features=True,
        normalization_method='none'
    )
    
    preprocessed_data = combined_data.groupby('symbol').apply(
        lambda x: preprocessor.fit_transform(x)
    ).reset_index(drop=True)
    
    print(f"✓ Preprocessed shape: {preprocessed_data.shape}")
    
    # Apply feature engineering per symbol
    print("\n10. Applying feature engineering per symbol...")
    feature_engineer = FeatureEngineer(
        add_price_features=True,
        add_technical_indicators=True,
        add_momentum_indicators=True,
        add_volatility_indicators=True,
        add_trend_indicators=True
    )
    
    features_data = preprocessed_data.groupby('symbol').apply(
        lambda x: feature_engineer.fit_transform(x)
    ).reset_index(drop=True)
    
    print(f"✓ Features shape: {features_data.shape}")
    print(f"✓ Features per symbol: {features_data.groupby('symbol').size().to_dict()}")
    
    # Show feature examples
    feature_cols = [col for col in features_data.columns if col not in preprocessed_data.columns]
    print(f"✓ Sample features: {feature_cols[:10]}")
    
    return features_data

def main():
    """Run all pipeline examples."""
    print("=== COMPLETE STOCK PREDICTION PIPELINE ===\n")
    
    try:
        # Test complete pipeline
        features_data = test_complete_pipeline()
        
        # Test sklearn pipeline
        sklearn_success = test_sklearn_pipeline()
        
        # Test multiple symbols pipeline
        multi_symbol_data = test_multiple_symbols_pipeline()
        
        print("\n" + "="*60)
        print("🎉 ALL PIPELINE TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\n💡 Complete Pipeline Features:")
        print("1. Data Loading: StockDataLoader for downloading and combining stock data")
        print("2. Preprocessing: StockPreprocessor for data cleaning and preparation")
        print("3. Feature Engineering: FeatureEngineer for technical indicators and features")
        print("4. Sklearn Integration: Full pipeline compatibility")
        print("5. Multiple Symbols: Groupby operations for multi-stock analysis")
        print("6. Time Features: Automatic time-based feature extraction")
        print("7. Data Validation: OHLCV integrity checks and anomaly fixing")
        print("8. Outlier Handling: Multiple outlier detection and removal methods")
        print("9. Missing Value Handling: Various strategies for missing data")
        print("10. Normalization: Multiple normalization methods available")
        
        print(f"\n📊 Pipeline Results:")
        print(f"✓ Single symbol features: {features_data.shape}")
        print(f"✓ Multiple symbols features: {multi_symbol_data.shape}")
        print(f"✓ Sklearn integration: {'✓' if sklearn_success else '⚠️'}")
        
    except Exception as e:
        print(f"\n❌ Error during pipeline testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
