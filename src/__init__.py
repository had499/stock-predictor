"""
Stock Predictor Package

A comprehensive package for stock data loading, preprocessing, and feature engineering.
"""

# Data loading
from .data.data_loader import StockDataLoader

# Feature engineering
from .features.feature_engineering import FeatureEngineer, add_features, add_technical_indicators

# Preprocessing
from .preprocessing.preprocessor import StockPreprocessor, preprocess_stock_data, clean_stock_data

# Sklearn components for easy access
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

__version__ = "1.0.0"

__all__ = [
    # Data loading
    'StockDataLoader',
    
    # Feature engineering
    'FeatureEngineer',
    'add_features',
    'add_technical_indicators',
    
    # Preprocessing
    'StockPreprocessor',
    'preprocess_stock_data',
    'clean_stock_data',
    
    # Sklearn components
    'Pipeline',
    'StandardScaler',
    'RobustScaler',
    'MinMaxScaler',
    'RandomForestRegressor',
    'RandomForestClassifier',
    'cross_val_score',
    'GridSearchCV'
]
