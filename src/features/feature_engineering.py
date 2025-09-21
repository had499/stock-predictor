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
                 add_trend_indicators: bool = True):
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
            'add_trend_indicators': self.add_trend_indicators
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
        logger.info("Adding features to dataset...")
        
        logger.info("Applying features per symbol...")
        # Apply features per symbol using groupby
        df = data.groupby('symbol').apply(self._add_features_to_group).reset_index(drop=True)

        # Remove features that may cause data leakage
        df = self._remove_today_features(df)

        
        logger.info(f"Added {len(df.columns) - len(data.columns)} new features")
        return df
    
    def _add_features_to_group(self, group_data: pd.DataFrame) -> pd.DataFrame:
        """Add features to a single symbol's data."""
        # Make a copy to avoid modifying original data
        df = group_data.copy()
        
        # Sort by date to ensure chronological order for technical indicators
        df = df.sort_values(['symbol', df.index.name or 'date']).reset_index(drop=True)
        
        # Add different types of features based on configuration
        if self.add_price_features:
            df = self._add_price_features(df)
        
        if self.add_volume_features:
            df = self._add_volume_features(df)
        
        if self.add_technical_indicators:
            df = self._add_technical_indicators(df)
        
        if self.add_momentum_indicators:
            df = self._add_momentum_indicators(df)
        
        if self.add_volatility_indicators:
            df = self._add_volatility_indicators(df)
        
        if self.add_trend_indicators:
            df = self._add_trend_indicators(df)

        # Add target column
        df = self.add_target_col(df)
        logger.info(f"Added {len(df.columns) - len(group_data.columns)} new features")
        return df
    
    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        df = data.copy()

        df['price_1lag'] = df[self.price_column].shift(1)  
        df['volume_1lag'] = df[self.volume_column].shift(1)  
        df['high_1lag'] = df[self.high_column].shift(1)  
        df['low_1lag'] = df[self.low_column].shift(1)  
        df['open_1lag'] = df[self.open_column].shift(1)
        
        # Basic price changes
        # df['price_change'] = df[self.price_column] - df[self.price_column].shift(1)
        # df['price_change_pct'] = df[self.price_column].pct_change()
        # df['log_return'] = np.log(df[self.price_column] / df[self.price_column].shift(1))
        
        # Open to close change
        df['open_close_change_1lag'] = df[self.price_column].shift(1) - df[self.open_column].shift(1)
        df['open_close_change_pct_1lag'] = (df[self.price_column].shift(1) - df[self.open_column].shift(1)) / df[self.open_column].shift(1)
        
        # High-low features
        df['hl_spread_1lag'] = df[self.high_column].shift(1) - df[self.low_column].shift(1)
        df['hl_spread_pct_1lag'] = (df[self.high_column].shift(1) - df[self.low_column]).shift(1) / df[self.low_column].shift(1)
        
        # Price position within daily range
        df['price_position_1lag'] = (df[self.price_column].shift(1) - df[self.low_column].shift(1)) / (df[self.high_column].shift(1) - df[self.low_column].shift(1))
        
        # Gap features
        df['gap_1lag'] = df[self.open_column].shift(1) - df[self.price_column].shift(2)
        df['gap_pct_1lag'] = df['gap'] / df[self.price_column].shift(2)
        
        # Price ratios
        df['close_open_ratio_1lag'] = df[self.price_column].shift(1) / df[self.open_column].shift(1)
        df['high_close_ratio_1lag'] = df[self.high_column].shift(1) / df[self.price_column].shift(1)
        df['low_close_ratio_1lag'] = df[self.low_column].shift(1) / df[self.price_column].shift(1)
        
        return df
    
    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df = data.copy()
        
        # Volume changes
        df['volume_change_1lag'] = df[self.volume_column].shift(1) - df[self.volume_column].shift(2)
        # df['volume_change_pct'] = df[self.volume_column].pct_change()
        
        # Volume moving averages (prevent leakage)
        df['volume_sma_5'] = df[self.volume_column].shift(1).rolling(window=5).mean()
        df['volume_sma_20'] = df[self.volume_column].shift(1).rolling(window=20).mean()
        df['volume_ema_5'] = df[self.volume_column].shift(1).ewm(span=5).mean()
        df['volume_ema_20'] = df[self.volume_column].shift(1).ewm(span=20).mean()
        
        # Volume ratios
        df['volume_ratio_5'] = df[self.volume_column].shift(1) / df['volume_sma_5']
        df['volume_ratio_20'] = df[self.volume_column].shift(1) / df['volume_sma_20']
        
        # Volume volatility
        df['volume_volatility'] = df[self.volume_column].shift(1).rolling(window=20).std()
        
        # Price-volume features
        df['price_volume_1lag'] = df[self.price_column] .shift(1)* df[self.volume_column].shift(1)
        df['vwap'] = df['price_volume_1lag'].rolling(window=20).sum() / df[self.volume_column].shift(1).rolling(window=20).sum()
        
        return df
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        df = data.copy()
        # RSI
        df['rsi_14'] = self._calculate_rsi(df[self.price_column], period=14).shift(1)
        df['rsi_21'] = self._calculate_rsi(df[self.price_column], period=21).shift(1)
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = self._calculate_sma(df[self.price_column], period).shift(1)
            df[f'ema_{period}'] = self._calculate_ema(df[self.price_column], period).shift(1)
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(df[self.price_column])
        df['macd'] = macd_line.shift(1)
        df['macd_signal'] = signal_line.shift(1)
        df['macd_histogram'] = histogram.shift(1)
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df[self.price_column])
        df['bb_upper'] = bb_upper.shift(1)
        df['bb_middle'] = bb_middle.shift(1)
        df['bb_lower'] = bb_lower.shift(1)
        df['bb_width'] = ((bb_upper - bb_lower) / bb_middle).shift(1)
        df['bb_position'] = ((df[self.price_column] - bb_lower) / (bb_upper - bb_lower)).shift(1)
        # Stochastic Oscillator
        stoch_k, stoch_d = self._calculate_stochastic(df[self.high_column], df[self.low_column], df[self.price_column])
        df['stoch_k'] = stoch_k.shift(1)
        df['stoch_d'] = stoch_d.shift(1)
        return df
    
    def _add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        df = data.copy()
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = self._calculate_roc(df[self.price_column], period).shift(1)
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df[self.high_column], df[self.low_column], df[self.price_column]).shift(1)
        
        # Commodity Channel Index
        df['cci'] = self._calculate_cci(df[self.high_column], df[self.low_column], df[self.price_column]).shift(1)
        
        # Money Flow Index
        df['mfi'] = self._calculate_mfi(df[self.high_column], df[self.low_column], df[self.price_column], df[self.volume_column]).shift(1)
        
        return df
    
    def _add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        df = data.copy()
        
        # Historical Volatility
        for period in [10, 20, 30]:
            df[f'volatility_{period}'] = self._calculate_volatility(df[self.price_column], period).shift(1)
        
        # Average True Range
        df['atr'] = self._calculate_atr(df[self.high_column], df[self.low_column], df[self.price_column]).shift(1)
        
        # True Range
        df['true_range'] = self._calculate_true_range(df[self.high_column], df[self.low_column], df[self.price_column]).shift(1)
        
        return df
    
    def _add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        df = data.copy()
        
        # ADX (Average Directional Index)
        df['adx'] = self._calculate_adx(df[self.high_column], df[self.low_column], df[self.price_column]).shift(1)
        
        # Parabolic SAR
        df['sar'] = self._calculate_sar(df[self.high_column], df[self.low_column], df[self.price_column]).shift(1)
        
        # Ichimoku Cloud
        ichimoku = self._calculate_ichimoku(df[self.high_column], df[self.low_column], df[self.price_column])
        df['ichimoku_conversion'] = ichimoku['conversion'].shift(1)
        df['ichimoku_base'] = ichimoku['base'].shift(1)
        df['ichimoku_span_a'] = ichimoku['span_a'].shift(1)
        df['ichimoku_span_b'] = ichimoku['span_b'].shift(1)
        
        return df
    
    # Technical Indicator Calculation Methods
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA)."""
        return prices.rolling(window=period).mean()
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average (EMA)."""
        return prices.ewm(span=period).mean()
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> tuple:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_roc(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change."""
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def _calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
        return mfi
    
    def _calculate_volatility(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Historical Volatility."""
        returns = prices.pct_change()
        return returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        true_range = self._calculate_true_range(high, low, close)
        return true_range.rolling(window=period).mean()
    
    def _calculate_true_range(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return true_range
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = self._calculate_atr(high, low, close, period)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx
    
    def _calculate_sar(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR (robust version)."""
        n = len(close)
        sar = pd.Series(np.nan, index=close.index)
        # Find first non-NaN low value for initialization
        first_valid = low.first_valid_index()
        if first_valid is not None:
            sar.iloc[first_valid] = low.iloc[first_valid]
        else:
            return sar  # all NaN
        # Use unshifted columns for SAR calculation
        for i in range(first_valid + 1, n):
            if pd.isna(sar.iloc[i-1]) or pd.isna(high.iloc[i]) or pd.isna(low.iloc[i]):
                sar.iloc[i] = np.nan
            elif close.iloc[i] > sar.iloc[i-1]:
                sar.iloc[i] = sar.iloc[i-1] + acceleration * (high.iloc[i] - sar.iloc[i-1])
            else:
                sar.iloc[i] = sar.iloc[i-1] - acceleration * (sar.iloc[i-1] - low.iloc[i])
        return sar
    
    def _calculate_ichimoku(self, high: pd.Series, low: pd.Series, close: pd.Series,
                          conversion_period: int = 9, base_period: int = 26, 
                          span_b_period: int = 52, displacement: int = 26) -> dict:
        """Calculate Ichimoku Cloud indicators."""
        conversion = (high.rolling(window=conversion_period).max() + 
                     low.rolling(window=conversion_period).min()) / 2
        
        base = (high.rolling(window=base_period).max() + 
               low.rolling(window=base_period).min()) / 2
        
        span_a = ((conversion + base) / 2).shift(displacement)
        
        span_b = ((high.rolling(window=span_b_period).max() + 
                  low.rolling(window=span_b_period).min()) / 2).shift(displacement)
        
        return {
            'conversion': conversion,
            'base': base,
            'span_a': span_a,
            'span_b': span_b
        }

    def _remove_today_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove features that may cause data leakage."""
        df = data.copy()

        df.drop(columns=[
            self.price_column,
            self.volume_column,
            self.high_column,   
            self.low_column,
            self.open_column,
            'dividends',
            'stock splits'

        ], inplace=True, errors='ignore')

        return df
    
    def add_target_col(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add target column: next day's percentage change in price."""
        df = data.copy()

        df['target'] = df[self.price_column].pct_change().shift(-1)  # Next day's closing price

        return df