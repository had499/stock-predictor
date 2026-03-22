"""
Feature Engineering Module for Stock Predictor

This module provides comprehensive feature engineering capabilities for stock data,
including technical indicators, price-based features, and volume-based features.
Now implemented as sklearn transformers for easy integration with machine learning pipelines.

Author: Stock Predictor Project
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Any
import logging
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

from .calendar_features import add_calendar_features as _add_calendar_features
from .etf_features import add_etf_features as _add_etf_features, clean_etf as _clean_etf
from .leakage import remove_today_features as _remove_today_features
from .momentum_features import add_momentum_indicators as _add_momentum_indicators
from .price_features import add_price_features as _add_price_features
from .rank_features import add_rank_features as _add_rank_features
from .target_features import add_target_col as _add_target_col
from .technical_indicators import (
    add_technical_indicators as _add_technical_indicators,
    calculate_ema as _calculate_ema,
    calculate_macd as _calculate_macd,
    calculate_rsi as _calculate_rsi,
    calculate_sma as _calculate_sma,
)
from .trend_features import add_trend_indicators as _add_trend_indicators
from .volatility_features import add_volatility_indicators as _add_volatility_indicators
from .volume_features import add_volume_features as _add_volume_features

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A comprehensive feature engineering class for stock data.
    Provides various technical indicators and derived features.
    Implements sklearn's BaseEstimator and TransformerMixin for
    easy integration with machine learning pipelines.
    """
    
    def __init__(self, 
                 price_column: str = 'close',
                 volume_column: str = 'volume',
                 high_column: str = 'high',
                 low_column: str = 'low',
                 open_column: str = 'open',
                 add_price_features: bool = True,
                 add_volume_features: bool = True,
                 add_technical_indicators: bool = True,
                 add_momentum_indicators: bool = True,
                 add_volatility_indicators: bool = True,
                 add_trend_indicators: bool = True,
                 etf_columns: Optional[List[str]] = None,
                 sector_map: Optional[Dict[str, str]] = None,
                 target_horizon: int = 1):
        """
        Initialize the FeatureEngineer.
        
        Args:
            price_column (str): Column name for closing prices
            volume_column (str): Column name for volume
            high_column (str): Column name for high prices
            low_column (str): Column name for low prices
            open_column (str): Column name for open prices
            add_price_features (bool): Whether to add price-based features
            add_volume_features (bool): Whether to add volume features
            add_technical_indicators (bool): Whether to add technical indicators
            add_momentum_indicators (bool): Whether to add momentum indicators
            add_volatility_indicators (bool): Whether to add volatility indicators
            add_trend_indicators (bool): Whether to add trend indicators
            etf_columns (List[str], optional): List of ETF price columns (e.g., ['SPY','XLK','XLF',...])
            sector_map (Dict[str,str], optional): Mapping from ticker to sector ETF (e.g., {'AAPL':'XLK'})
            target_horizon (int): Number of days forward to predict (1=next day, 5=next week, etc.)
        """
        self.price_column = price_column
        self.volume_column = volume_column
        self.high_column = high_column
        self.low_column = low_column
        self.open_column = open_column
        self.add_price_features = add_price_features
        self.add_volume_features = add_volume_features
        self.add_technical_indicators = add_technical_indicators
        self.add_momentum_indicators = add_momentum_indicators
        self.add_volatility_indicators = add_volatility_indicators
        self.add_trend_indicators = add_trend_indicators
        self.etf_columns = etf_columns if etf_columns is not None else ['SPY']
        self.sector_map = sector_map if sector_map is not None else {}
        self.target_horizon = target_horizon
        
        warnings.filterwarnings('ignore', category=FutureWarning)
    
    def fit(self, X, y=None):
        """Fit the transformer. Required by sklearn interface."""
        return self
    
    def set_params(self, **params):
        """Set parameters for the transformer."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self
    
    def get_params(self, deep=True):
        """Get parameters for the transformer."""
        return {
            'price_column': self.price_column,
            'volume_column': self.volume_column,
            'high_column': self.high_column,
            'low_column': self.low_column,
            'open_column': self.open_column,
            'add_price_features': self.add_price_features,
            'add_volume_features': self.add_volume_features,
            'add_technical_indicators': self.add_technical_indicators,
            'add_momentum_indicators': self.add_momentum_indicators,
            'add_volatility_indicators': self.add_volatility_indicators,
            'add_trend_indicators': self.add_trend_indicators,
            'etf_columns': self.etf_columns,
            'sector_map': self.sector_map,
            'target_horizon': self.target_horizon,
        }
    
    def transform(self, X):
        """
        Transform method that adds features to the input data.
        
        Args:
            X (pd.DataFrame): Input data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with features added
        """
        return self._add_all_features(X)
    
    def _add_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all available features to the dataset."""
        # logger.info("Adding features to dataset...")
        # logger.info("Applying features per symbol...")
        df_sorted = data.copy()
        if 'date' in df_sorted.columns:
            df_sorted = df_sorted.sort_values(['symbol', 'date'])

        etf_columns = list(self.etf_columns)
        if self.sector_map:
            extra_etfs = set()
            for etfs in self.sector_map.values():
                if isinstance(etfs, str):
                    extra_etfs.add(etfs)
                elif isinstance(etfs, list):
                    extra_etfs.update(etfs)
            etf_columns = sorted(set(etf_columns) | extra_etfs)

        etf_returns = None
        if 'date' in df_sorted.columns and etf_columns:
            etf_prices = (
                df_sorted[df_sorted['symbol'].isin(etf_columns)]
                .pivot(index='date', columns='symbol', values=self.price_column)
            )
            if not etf_prices.empty:
                etf_returns = etf_prices.pct_change().shift(1).add_suffix("_return_1lag").reset_index()

        # Apply features per symbol using groupby
        df = df_sorted.groupby('symbol', group_keys=False, sort=False).apply(
            self._add_features_to_group,
            etf_returns=etf_returns,
            etf_columns=etf_columns
        ).reset_index(drop=True)

        # Add cross-sectional rank features (per date)
        df = _add_rank_features(df, etf_columns=etf_columns)

        # Add calendar/time features
        df = _add_calendar_features(df)

        # Remove features that may cause data leakage
        df = _remove_today_features(
            df,
            price_column=self.price_column,
            volume_column=self.volume_column,
            high_column=self.high_column,
            low_column=self.low_column,
            open_column=self.open_column,
        )
        df = _clean_etf(df, etf_columns=etf_columns)
        # logger.info(f"Added {len(df.columns) - len(data.columns)} new features")
        return df

    
    def _add_features_to_group(
        self,
        group_data: pd.DataFrame,
        etf_returns: Optional[pd.DataFrame] = None,
        etf_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Add features to a single symbol's data."""
        # Make a copy to avoid modifying original data
        df = group_data.copy()
        
        # Add different types of features based on configuration
        if self.add_price_features:
            df = _add_price_features(
                df,
                price_column=self.price_column,
                volume_column=self.volume_column,
                high_column=self.high_column,
                low_column=self.low_column,
                open_column=self.open_column,
            )
        
        if self.add_volume_features:
            df = _add_volume_features(
                df,
                price_column=self.price_column,
                volume_column=self.volume_column,
            )
        
        if self.add_technical_indicators:
            df = _add_technical_indicators(
                df,
                price_column=self.price_column,
                high_column=self.high_column,
                low_column=self.low_column,
            )
        
        if self.add_momentum_indicators:
            df = _add_momentum_indicators(
                df,
                price_column=self.price_column,
                high_column=self.high_column,
                low_column=self.low_column,
                volume_column=self.volume_column,
            )
        
        if self.add_volatility_indicators:
            df = _add_volatility_indicators(
                df,
                price_column=self.price_column,
                high_column=self.high_column,
                low_column=self.low_column,
            )
        
        if self.add_trend_indicators:
            df = _add_trend_indicators(
                df,
                price_column=self.price_column,
                high_column=self.high_column,
                low_column=self.low_column,
            )

        # Add ETF features
        df = _add_etf_features(
            df,
            price_column=self.price_column,
            etf_returns=etf_returns,
            etf_columns=etf_columns or [],
            sector_map=self.sector_map,
        )

        # Add target column
        df = _add_target_col(df, price_column=self.price_column, horizon=self.target_horizon)
        logger.info(f"Added {len(df.columns) - len(group_data.columns)} new features")
        return df

    def add_target_col(self, data: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        return _add_target_col(data, price_column=self.price_column, horizon=horizon)

    # Back-compat helpers used by old examples
    def add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return _add_price_features(
            data,
            price_column=self.price_column,
            volume_column=self.volume_column,
            high_column=self.high_column,
            low_column=self.low_column,
            open_column=self.open_column,
        )

    def add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return _add_volume_features(
            data,
            price_column=self.price_column,
            volume_column=self.volume_column,
        )

    def add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return _add_momentum_indicators(
            data,
            price_column=self.price_column,
            high_column=self.high_column,
            low_column=self.low_column,
            volume_column=self.volume_column,
        )

    def add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return _add_volatility_indicators(
            data,
            price_column=self.price_column,
            high_column=self.high_column,
            low_column=self.low_column,
        )

    def add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return _add_trend_indicators(
            data,
            price_column=self.price_column,
            high_column=self.high_column,
            low_column=self.low_column,
        )

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        return _calculate_rsi(prices, period=period)

    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        return _calculate_sma(prices, period)

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        return _calculate_ema(prices, period)

    def calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        return _calculate_macd(prices, fast=fast, slow=slow, signal=signal)


def add_features(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Convenience wrapper: apply full FeatureEngineer transform."""
    return FeatureEngineer(**kwargs).transform(data)


def add_technical_indicators(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Convenience wrapper: apply FeatureEngineer with technical indicators enabled."""
    fe = FeatureEngineer(
        add_price_features=False,
        add_volume_features=False,
        add_technical_indicators=True,
        add_momentum_indicators=False,
        add_volatility_indicators=False,
        add_trend_indicators=False,
        **kwargs,
    )
    return fe.transform(data)
