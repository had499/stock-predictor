"""
Stock Data Loader Module

This module provides functionality to load stock data for given date ranges
using yfinance. It includes data validation, error handling, and various
data processing options. Now implemented as sklearn transformers for easy
integration with machine learning pipelines.

Author: Stock Predictor Project
"""

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
from typing import Union, List, Optional, Dict, Any
import logging
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataLoader(BaseEstimator, TransformerMixin):
    """
    A comprehensive stock data loader that handles data retrieval,
    validation, and preprocessing for stock prediction tasks.
    Implements sklearn's BaseEstimator and TransformerMixin for
    easy integration with machine learning pipelines.
    """
    
    def __init__(self, 
                 symbols: Optional[List[str]] = None,
                 start_date: Optional[Union[str, datetime]] = None,
                 end_date: Optional[Union[str, datetime]] = None,
                 interval: str = '1d',
                 auto_adjust: bool = True,
                 fill_missing: bool = True):
        """
        Initialize the StockDataLoader.
        
        Args:
            symbols (List[str], optional): Stock symbols to load (can be set later)
            start_date (str or datetime, optional): Start date for data loading
            end_date (str or datetime, optional): End date for data loading
            interval (str): Data interval ('1d', '1h', '5m', etc.)
            auto_adjust (bool): Whether to auto-adjust prices for splits/dividends
            fill_missing (bool): Whether to fill missing values
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.auto_adjust = auto_adjust
        self.fill_missing = fill_missing
        
        # Configure yfinance warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
    
    def fit(self, X=None, y=None):
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
            'symbols': self.symbols,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'interval': self.interval,
            'auto_adjust': self.auto_adjust,
            'fill_missing': self.fill_missing
        }
    
    def transform(self, X):
        """
        Transform method that loads stock data using the configured parameters.
        
        Args:
            X: Ignored (required by sklearn interface)
            
        Returns:
            pd.DataFrame: Combined stock data from all symbols
            
        Raises:
            ValueError: If symbols, start_date, or end_date are not set
        """
        if self.symbols is None:
            raise ValueError("Symbols must be set before calling transform()")
        if self.start_date is None:
            raise ValueError("Start date must be set before calling transform()")
        if self.end_date is None:
            raise ValueError("End date must be set before calling transform()")
        
        return self._load_and_combine_stocks()
    
    def _load_and_combine_stocks(self) -> pd.DataFrame:
        """Load data for multiple stocks and combine into a single DataFrame."""
        all_data = []
        
        for symbol in self.symbols:
            try:
                logger.info(f"Loading data for {symbol}...")
                data = self._load_single_stock(symbol)
                
                # Add symbol column if not already present
                if 'symbol' not in data.columns:
                    data['symbol'] = symbol
                
                all_data.append(data)
                logger.info(f"Successfully loaded {len(data)} records for {symbol}")
                
            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No data could be loaded for any of the provided symbols")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort by symbol and date
        combined_data = combined_data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        logger.info(f"Successfully combined data for {len(all_data)} symbols: {len(combined_data)} total records")
        return combined_data
    
    def _load_single_stock(self, symbol: str) -> pd.DataFrame:
        """Load data for a single stock symbol."""
        try:
            # Convert dates to string format if needed
            start_date = self.start_date
            end_date = self.end_date
            
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            
            # Validate date range
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
            
            logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
            
            # Download data using yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=self.interval,
                auto_adjust=self.auto_adjust
            )
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol} in the specified date range")
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            
            # Validate data
            self._validate_data(data, symbol)
            
            # Fill missing values if requested
            if self.fill_missing:
                data = self._fill_missing_values(data)
            
            # Add additional calculated columns
            data = self._add_calculated_columns(data)
            
            # Get stock info and add to data
            stock_info = self._get_stock_info(symbol)
            if not stock_info.get('error'):
                data['sector'] = stock_info['sector']
                data['industry'] = stock_info['industry']
                data['currency'] = stock_info['currency']
                data['exchange'] = stock_info['exchange']
                data['symbol'] = symbol
            
            logger.info(f"Successfully loaded {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            raise
    
    def _get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic information about a stock."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'symbol': symbol,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'N/A'),
                'exchange': info.get('exchange', 'N/A')
            }
        except Exception as e:
            logger.error(f"Error getting info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}
    
    def _validate_data(self, data: pd.DataFrame, symbol: str) -> None:
        """Validate the loaded stock data."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (data[col] <= 0).any():
                logger.warning(f"Found non-positive prices in {col} column for {symbol}")
        
        # Check for negative volume
        if (data['volume'] < 0).any():
            logger.warning(f"Found negative volume for {symbol}")
        
        # Check for missing values
        missing_data = data[required_columns].isnull().sum()
        if missing_data.any():
            logger.warning(f"Missing data found for {symbol}: {missing_data[missing_data > 0].to_dict()}")
    
    def _fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the stock data."""
        # Forward fill for price data
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill')
        
        # Fill volume with 0
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(0)
        
        return data
    
    def _add_calculated_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic calculated columns to the stock data."""
        # Only add basic price change - keep it minimal
        if 'close' in data.columns and 'open' in data.columns:
            data['price_change'] = data['close'] - data['open']
        
        return data
