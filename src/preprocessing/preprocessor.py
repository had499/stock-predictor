"""
Stock data preprocessing module.

This module provides comprehensive preprocessing capabilities for stock data,
including missing value handling, outlier detection, normalization, and time feature extraction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPreprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive stock data preprocessor.
    
    Handles missing values, outliers, normalization, and time features.
    Implements sklearn's BaseEstimator and TransformerMixin for
    easy integration with machine learning pipelines.
    """
    
    def __init__(
        self,
        # Missing value handling
        handle_missing_values: str = 'forward_fill',  # 'forward_fill', 'backward_fill', 'interpolate', 'drop', 'none'
        missing_value_columns: Optional[List[str]] = None,
        
        # Outlier detection and removal
        remove_outliers: bool = True,
        outlier_method: str = 'iqr',  # 'iqr', 'zscore', 'modified_zscore', 'isolation_forest'
        outlier_threshold: float = 3.0,
        outlier_columns: Optional[List[str]] = None,
        
        # Normalization
        normalize_columns: Optional[List[str]] = None,
        normalization_method: str = 'robust',  # 'standard', 'robust', 'minmax', 'none'
        
        # Time features
        add_time_features: bool = True,
        time_column: str = 'date',
        
        # Data validation
        validate_ohlcv: bool = True,
        min_volume_threshold: float = 0.0,
        
        # Volume processing
        normalize_volume: bool = True,
        volume_smoothing: bool = True,
        volume_smoothing_window: int = 5,
        
        # Price validation
        validate_price_relationships: bool = True,
        fix_price_anomalies: bool = True
    ):
        """
        Initialize the StockPreprocessor.
        
        Args:
            handle_missing_values (str): Method to handle missing values
            missing_value_columns (List[str]): Columns to apply missing value handling to
            remove_outliers (bool): Whether to remove outliers
            outlier_method (str): Method to detect outliers
            outlier_threshold (float): Threshold for outlier detection
            outlier_columns (List[str]): Columns to check for outliers
            normalize_columns (List[str]): Columns to normalize
            normalization_method (str): Normalization method to use
            add_time_features (bool): Whether to add time-based features
            time_column (str): Column name for time/date data
            validate_ohlcv (bool): Whether to validate OHLCV data integrity
            min_volume_threshold (float): Minimum volume threshold
            normalize_volume (bool): Whether to normalize volume data
            volume_smoothing (bool): Whether to apply smoothing to volume
            volume_smoothing_window (int): Window size for volume smoothing
            validate_price_relationships (bool): Whether to validate price relationships
            fix_price_anomalies (bool): Whether to fix price anomalies
        """
        self.handle_missing_values = handle_missing_values
        self.missing_value_columns = missing_value_columns
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.outlier_columns = outlier_columns
        self.normalize_columns = normalize_columns
        self.normalization_method = normalization_method
        self.add_time_features = add_time_features
        self.time_column = time_column
        self.validate_ohlcv = validate_ohlcv
        self.min_volume_threshold = min_volume_threshold
        self.normalize_volume = normalize_volume
        self.volume_smoothing = volume_smoothing
        self.volume_smoothing_window = volume_smoothing_window
        self.validate_price_relationships = validate_price_relationships
        self.fix_price_anomalies = fix_price_anomalies
        
        # Initialize scalers
        self._price_scaler = None
        self._volume_scaler = None
        self._fitted = False
        
        warnings.filterwarnings('ignore', category=FutureWarning)
    
    def fit(self, X, y=None):
        """Fit the preprocessor. Required by sklearn interface."""
        if isinstance(X, pd.DataFrame):
            self._fit_scalers(X)
        self._fitted = True
        return self
    
    def transform(self, X):
        """Transform the data. Required by sklearn interface."""
        if not self._fitted:
            self.fit(X)
        
        if isinstance(X, pd.DataFrame):
            return self._preprocess_dataframe(X)
        else:
            raise ValueError("Input must be a pandas DataFrame")
    
    def fit_transform(self, X, y=None):
        """Fit and transform the data. Required by sklearn interface."""
        return self.fit(X).transform(X)
    
    def _fit_scalers(self, data: pd.DataFrame) -> None:
        """Fit scalers for normalization."""
        if self.normalization_method == 'none':
            return
        
        # Determine columns to normalize
        if self.normalize_columns is None:
            price_cols = ['open', 'high', 'low', 'close']
            volume_cols = ['volume']
            self.normalize_columns = [col for col in price_cols + volume_cols if col in data.columns]
        
        if not self.normalize_columns:
            return
        
        # Fit scalers
        if self.normalization_method == 'standard':
            self._price_scaler = StandardScaler()
        elif self.normalization_method == 'robust':
            self._price_scaler = RobustScaler()
        elif self.normalization_method == 'minmax':
            self._price_scaler = MinMaxScaler()
        
        if self._price_scaler is not None:
            # Fit on non-missing values
            valid_data = data[self.normalize_columns].dropna()
            if len(valid_data) > 0:
                self._price_scaler.fit(valid_data)
    
    def _preprocess_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a single DataFrame."""
        logger.info("Starting data preprocessing...")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Sort by date if time column exists
        if self.time_column in df.columns:
            df = df.sort_values(self.time_column).reset_index(drop=True)
        elif df.index.name and 'date' in str(df.index.name).lower():
            df = df.sort_index()
        
        # Validate OHLCV data
        if self.validate_ohlcv:
            df = self._validate_ohlcv_data(df)
        
        # Handle missing values
        if self.handle_missing_values != 'none':
            df = self._handle_missing_values(df)
        
        # Fix price anomalies
        if self.fix_price_anomalies:
            df = self._fix_price_anomalies(df)
        
        # Validate price relationships
        if self.validate_price_relationships:
            df = self._validate_price_relationships(df)
        
        # Process volume
        if 'volume' in df.columns:
            df = self._process_volume(df)
        
        # Remove outliers
        if self.remove_outliers:
            df = self._remove_outliers(df)
        
        # Add time features
        if self.add_time_features:
            df = self._add_time_features(df)
        
        # Normalize data
        if self.normalization_method != 'none':
            df = self._normalize_data(df)
        
        logger.info(f"Preprocessing complete. Shape: {df.shape}")
        return df
    
    def _validate_ohlcv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV data integrity."""
        logger.info("Validating OHLCV data...")
        
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        existing_cols = [col for col in ohlcv_cols if col in data.columns]
        
        if not existing_cols:
            logger.warning("No OHLCV columns found for validation")
            return data
        
        # Check for negative values in price columns
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in data.columns]
        for col in price_cols:
            negative_count = (data[col] <= 0).sum()
            if negative_count > 0:
                logger.warning(f"Found {negative_count} non-positive values in {col}")
                data[col] = data[col].replace(0, np.nan)
        
        # Check for negative volume
        if 'volume' in data.columns:
            negative_volume = (data['volume'] < 0).sum()
            if negative_volume > 0:
                logger.warning(f"Found {negative_volume} negative volume values")
                data['volume'] = data['volume'].clip(lower=0)
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        logger.info(f"Handling missing values using {self.handle_missing_values} method...")
        
        df = data.copy()
        
        # Determine columns to process
        if self.missing_value_columns is None:
            # Default to OHLCV columns
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            self.missing_value_columns = [col for col in ohlcv_cols if col in df.columns]
        
        if not self.missing_value_columns:
            return df
        
        # Apply missing value handling
        for col in self.missing_value_columns:
            if col not in df.columns:
                continue
                
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue
                
            logger.info(f"Handling {missing_count} missing values in {col}")
            
            if self.handle_missing_values == 'forward_fill':
                df[col] = df[col].fillna(method='ffill')
            elif self.handle_missing_values == 'backward_fill':
                df[col] = df[col].fillna(method='bfill')
            elif self.handle_missing_values == 'interpolate':
                df[col] = df[col].interpolate(method='linear')
            elif self.handle_missing_values == 'drop':
                df = df.dropna(subset=[col])
        
        return df
    
    def _fix_price_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix common price anomalies."""
        logger.info("Fixing price anomalies...")
        
        df = data.copy()
        
        # Fix high < low anomalies
        if 'high' in df.columns and 'low' in df.columns:
            high_low_anomalies = (df['high'] < df['low']).sum()
            if high_low_anomalies > 0:
                logger.info(f"Fixing {high_low_anomalies} high < low anomalies")
                df['high'] = np.maximum(df['high'], df['low'])
        
        # Fix open/close outside high/low range
        if all(col in df.columns for col in ['open', 'high', 'low']):
            open_anomalies = ((df['open'] > df['high']) | (df['open'] < df['low'])).sum()
            if open_anomalies > 0:
                logger.info(f"Fixing {open_anomalies} open price anomalies")
                df['open'] = df['open'].clip(lower=df['low'], upper=df['high'])
        
        if all(col in df.columns for col in ['close', 'high', 'low']):
            close_anomalies = ((df['close'] > df['high']) | (df['close'] < df['low'])).sum()
            if close_anomalies > 0:
                logger.info(f"Fixing {close_anomalies} close price anomalies")
                df['close'] = df['close'].clip(lower=df['low'], upper=df['high'])
        
        return df
    
    def _validate_price_relationships(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix price relationships."""
        logger.info("Validating price relationships...")
        
        df = data.copy()
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
            df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        return df
    
    def _process_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process volume data."""
        logger.info("Processing volume data...")
        
        df = data.copy()
        
        if 'volume' not in df.columns:
            return df
        
        # Apply volume threshold
        if self.min_volume_threshold > 0:
            below_threshold = (df['volume'] < self.min_volume_threshold).sum()
            if below_threshold > 0:
                logger.info(f"Setting {below_threshold} low volume values to threshold")
                df['volume'] = df['volume'].clip(lower=self.min_volume_threshold)
        
        # Apply volume smoothing
        if self.volume_smoothing and self.volume_smoothing_window > 1:
            logger.info(f"Applying volume smoothing with window {self.volume_smoothing_window}")
            df['volume'] = df['volume'].rolling(
                window=self.volume_smoothing_window, 
                center=True
            ).mean().fillna(df['volume'])
        
        return df
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the data."""
        logger.info(f"Removing outliers using {self.outlier_method} method...")
        
        df = data.copy()
        
        # Determine columns to check for outliers
        if self.outlier_columns is None:
            # Default to price columns
            price_cols = ['open', 'high', 'low', 'close']
            self.outlier_columns = [col for col in price_cols if col in df.columns]
        
        if not self.outlier_columns:
            return df
        
        outlier_mask = pd.Series(False, index=df.index)
        
        for col in self.outlier_columns:
            if col not in df.columns:
                continue
            
            col_outliers = self._detect_outliers(df[col], method=self.outlier_method, threshold=self.outlier_threshold)
            outlier_mask |= col_outliers
        
        outliers_removed = outlier_mask.sum()
        if outliers_removed > 0:
            logger.info(f"Removing {outliers_removed} outlier rows")
            df = df[~outlier_mask].reset_index(drop=True)
        
        return df
    
    def _detect_outliers(self, series: pd.Series, method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
        """Detect outliers in a series."""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold
        
        elif method == 'modified_zscore':
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            return np.abs(modified_z_scores) > threshold
        
        else:
            logger.warning(f"Unknown outlier detection method: {method}")
            return pd.Series(False, index=series.index)
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        logger.info("Adding time features...")
        
        df = data.copy()
        
        # Determine time column
        time_col = None
        if self.time_column in df.columns:
            time_col = self.time_column
        elif df.index.name and 'date' in str(df.index.name).lower():
            time_col = df.index.name
            df = df.reset_index()
        
        if time_col is None:
            logger.warning("No time column found for time features")
            return df
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Add time features
        df['year'] = df[time_col].dt.year
        df['month'] = df[time_col].dt.month
        df['day'] = df[time_col].dt.day
        df['dayofweek'] = df[time_col].dt.dayofweek
        df['dayofyear'] = df[time_col].dt.dayofyear
        df['week'] = df[time_col].dt.isocalendar().week
        df['quarter'] = df[time_col].dt.quarter
        df['is_weekend'] = df[time_col].dt.dayofweek >= 5
        df['is_month_start'] = df[time_col].dt.is_month_start
        df['is_month_end'] = df[time_col].dt.is_month_end
        df['is_quarter_start'] = df[time_col].dt.is_quarter_start
        df['is_quarter_end'] = df[time_col].dt.is_quarter_end
        df['is_year_start'] = df[time_col].dt.is_year_start
        df['is_year_end'] = df[time_col].dt.is_year_end
        
        return df
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data using fitted scalers."""
        if self._price_scaler is None or self.normalization_method == 'none':
            return data
        
        logger.info(f"Normalizing data using {self.normalization_method} method...")
        
        df = data.copy()
        
        if self.normalize_columns:
            # Apply normalization to specified columns
            normalized_data = self._price_scaler.transform(df[self.normalize_columns])
            df[self.normalize_columns] = normalized_data
        
        return df
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'handle_missing_values': self.handle_missing_values,
            'missing_value_columns': self.missing_value_columns,
            'remove_outliers': self.remove_outliers,
            'outlier_method': self.outlier_method,
            'outlier_threshold': self.outlier_threshold,
            'outlier_columns': self.outlier_columns,
            'normalize_columns': self.normalize_columns,
            'normalization_method': self.normalization_method,
            'add_time_features': self.add_time_features,
            'time_column': self.time_column,
            'validate_ohlcv': self.validate_ohlcv,
            'min_volume_threshold': self.min_volume_threshold,
            'normalize_volume': self.normalize_volume,
            'volume_smoothing': self.volume_smoothing,
            'volume_smoothing_window': self.volume_smoothing_window,
            'validate_price_relationships': self.validate_price_relationships,
            'fix_price_anomalies': self.fix_price_anomalies
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self


# Convenience functions
def preprocess_stock_data(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Quick function to preprocess stock data.
    
    Args:
        data (pd.DataFrame): Stock data to preprocess
        **kwargs: Additional arguments passed to StockPreprocessor
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    preprocessor = StockPreprocessor(**kwargs)
    return preprocessor.fit_transform(data)


def clean_stock_data(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Quick function to clean stock data (basic preprocessing).
    
    Args:
        data (pd.DataFrame): Stock data to clean
        **kwargs: Additional arguments passed to StockPreprocessor
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    preprocessor = StockPreprocessor(
        handle_missing_values='forward_fill',
        remove_outliers=True,
        outlier_method='iqr',
        validate_ohlcv=True,
        fix_price_anomalies=True,
        add_time_features=True,
        normalization_method='none',
        **kwargs
    )
    return preprocessor.fit_transform(data)
