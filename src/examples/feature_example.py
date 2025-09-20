"""
Feature Engineering Example Usage

This script demonstrates how to use the feature engineering module
with data from the data loader.
"""

from data_loader import load_stock_data
from feature_engineering import FeatureEngineer, add_features, add_technical_indicators
import pandas as pd

def main():
    print("=== Feature Engineering Examples ===\n")
    
    # Load some sample data
    print("1. Loading sample data...")
    print("-" * 40)
    try:
        data = load_stock_data("AAPL", "2023-01-01", "2023-12-31")
        print(f"✓ Loaded {len(data)} records")
        print(f"✓ Original columns: {list(data.columns)}")
        print(f"✓ Sample data:")
        print(data.head(2))
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    print("\n" + "="*60 + "\n")
    
    # Example 1: Add all features
    print("2. Adding all features...")
    print("-" * 40)
    try:
        features = add_features(data)
        print(f"✓ Added {len(features.columns) - len(data.columns)} new features")
        print(f"✓ Total columns: {len(features.columns)}")
        
        # Show some feature categories
        price_features = [col for col in features.columns if any(x in col for x in ['price', 'return', 'change', 'ratio'])]
        technical_features = [col for col in features.columns if any(x in col for x in ['rsi', 'sma', 'ema', 'macd', 'bb_', 'stoch'])]
        volume_features = [col for col in features.columns if any(x in col for x in ['volume', 'vwap'])]
        
        print(f"✓ Price features: {len(price_features)}")
        print(f"✓ Technical indicators: {len(technical_features)}")
        print(f"✓ Volume features: {len(volume_features)}")
        
        # Show sample of new features
        print(f"✓ Sample new features: {list(features.columns[-10:])}")
        
    except Exception as e:
        print(f"✗ Error adding features: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Add only technical indicators
    print("3. Adding only technical indicators...")
    print("-" * 40)
    try:
        tech_data = add_technical_indicators(data)
        print(f"✓ Added {len(tech_data.columns) - len(data.columns)} technical indicators")
        
        # Show some technical indicators
        tech_cols = [col for col in tech_data.columns if any(x in col for x in ['rsi', 'sma', 'ema', 'macd', 'bb_', 'stoch'])]
        print(f"✓ Technical indicators: {tech_cols[:10]}...")  # Show first 10
        
    except Exception as e:
        print(f"✗ Error adding technical indicators: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Using FeatureEngineer class directly
    print("4. Using FeatureEngineer class directly...")
    print("-" * 40)
    try:
        engineer = FeatureEngineer()
        
        # Add specific feature types
        price_features = engineer.add_price_features(data)
        print(f"✓ Price features: {len(price_features.columns) - len(data.columns)} added")
        
        volume_features = engineer.add_volume_features(data)
        print(f"✓ Volume features: {len(volume_features.columns) - len(data.columns)} added")
        
        momentum_features = engineer.add_momentum_indicators(data)
        print(f"✓ Momentum features: {len(momentum_features.columns) - len(data.columns)} added")
        
        # Calculate individual indicators
        rsi = engineer.calculate_rsi(data['close'], period=14)
        sma_20 = engineer.calculate_sma(data['close'], period=20)
        macd, signal, hist = engineer.calculate_macd(data['close'])
        
        print(f"✓ RSI calculated: {len(rsi.dropna())} valid values")
        print(f"✓ SMA-20 calculated: {len(sma_20.dropna())} valid values")
        print(f"✓ MACD calculated: {len(macd.dropna())} valid values")
        
    except Exception as e:
        print(f"✗ Error using FeatureEngineer: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 4: Feature analysis
    print("5. Feature analysis...")
    print("-" * 40)
    try:
        features = add_features(data)
        
        # Check for missing values
        missing_data = features.isnull().sum()
        high_missing = missing_data[missing_data > len(features) * 0.5]
        
        if len(high_missing) > 0:
            print(f"⚠️  Features with >50% missing values: {len(high_missing)}")
            print(f"   Examples: {list(high_missing.head().index)}")
        else:
            print("✓ No features with excessive missing values")
        
        # Check feature ranges
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        print(f"✓ Numeric features: {len(numeric_cols)}")
        
        # Show some statistics
        print(f"✓ RSI range: {features['rsi_14'].min():.2f} - {features['rsi_14'].max():.2f}")
        print(f"✓ Price change range: {features['price_change'].min():.2f} - {features['price_change'].max():.2f}")
        print(f"✓ Volume ratio range: {features['volume_ratio_20'].min():.2f} - {features['volume_ratio_20'].max():.2f}")
        
    except Exception as e:
        print(f"✗ Error in feature analysis: {e}")
    
    print("\n" + "="*60 + "\n")
    print("Feature engineering examples completed! 🎉")
    print(f"Total features available: {len(features.columns) if 'features' in locals() else 'N/A'}")

if __name__ == "__main__":
    main()
