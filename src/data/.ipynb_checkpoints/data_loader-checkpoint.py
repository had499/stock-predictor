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
from datetime import datetime, timedelta
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
                 symbol: Optional[str] = None,
                 start_date: Optional[Union[str, datetime]] = None,
                 end_date: Optional[Union[str, datetime]] = None,
                 interval: str = '1d',
                 auto_adjust: bool = True,
                 prepost: bool = False,
                 threads: bool = True,
                 proxy: Optional[str] = None,
                 validate_data: bool = True,
                 fill_missing: bool = True):
        """
        Initialize the StockDataLoader.
        
        Args:
            symbol (str, optional): Stock symbol to load (can be set later)
            start_date (str or datetime, optional): Start date for data loading
            end_date (str or datetime, optional): End date for data loading
            interval (str): Data interval ('1d', '1h', '5m', etc.)
            auto_adjust (bool): Whether to auto-adjust prices for splits/dividends
            prepost (bool): Whether to include pre/post market data
            threads (bool): Whether to use threads for downloading
            proxy (str, optional): Proxy URL for requests
            validate_data (bool): Whether to validate the loaded data
            fill_missing (bool): Whether to fill missing values
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.auto_adjust = auto_adjust
        self.prepost = prepost
        self.threads = threads
        self.proxy = proxy
        self.validate_data = validate_data
        self.fill_missing = fill_missing
        
        # Configure yfinance warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
    
    def fit(self, X=None, y=None):
        """
        Fit the transformer. For data loading, this is a no-op but required by sklearn.
        
        Args:
            X: Ignored (required by sklearn interface)
            y: Ignored (required by sklearn interface)
            
        Returns:
            self: Returns self for method chaining
        """
        return self
    
    def set_params(self, **params):
        """
        Set parameters for the transformer.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self: Returns self for method chaining
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self
    
    def get_params(self, deep=True):
        """
        Get parameters for the transformer.
        
        Args:
            deep (bool): If True, return parameters for sub-objects
            
        Returns:
            dict: Dictionary of parameters
        """
        return {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'interval': self.interval,
            'auto_adjust': self.auto_adjust,
            'prepost': self.prepost,
            'threads': self.threads,
            'proxy': self.proxy,
            'validate_data': self.validate_data,
            'fill_missing': self.fill_missing
        }
    
    def transform(self, X=None):
        """
        Transform method that loads stock data using the configured parameters.
        
        Args:
            X: Ignored (required by sklearn interface)
            
        Returns:
            pd.DataFrame: Loaded stock data
            
        Raises:
            ValueError: If symbol, start_date, or end_date are not set
        """
        if self.symbol is None:
            raise ValueError("Symbol must be set before calling transform()")
        if self.start_date is None:
            raise ValueError("Start date must be set before calling transform()")
        if self.end_date is None:
            raise ValueError("End date must be set before calling transform()")
        
        return self.load_stock_data(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            interval=self.interval,
            validate_data=self.validate_data,
            fill_missing=self.fill_missing
        )
    
    def load_stock_data(self, 
                       symbol: str,
                       start_date: Union[str, datetime],
                       end_date: Union[str, datetime],
                       interval: str = '1d',
                       validate_data: bool = True,
                       fill_missing: bool = True) -> pd.DataFrame:
        """
        Load stock data for a given symbol and date range.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
            start_date (str or datetime): Start date in 'YYYY-MM-DD' format or datetime object
            end_date (str or datetime): End date in 'YYYY-MM-DD' format or datetime object
            interval (str): Data interval ('1d', '1h', '5m', etc.)
            validate_data (bool): Whether to validate the loaded data
            fill_missing (bool): Whether to fill missing values
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
            
        Raises:
            ValueError: If symbol is invalid or date range is invalid
            ConnectionError: If unable to download data
        """
        try:
            # Convert dates to string format if needed
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
                interval=interval,
                auto_adjust=self.auto_adjust,
                prepost=self.prepost,
                threads=self.threads,
                proxy=self.proxy
            )
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol} in the specified date range")
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            
            # Validate data if requested
            if validate_data:
                self._validate_data(data, symbol)
            
            # Fill missing values if requested
            if fill_missing:
                data = self._fill_missing_values(data)
            
            # Add additional calculated columns
            data = self._add_calculated_columns(data)
            
            logger.info(f"Successfully loaded {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            raise
    
    def load_multiple_stocks(self, 
                           symbols: List[str],
                           start_date: Union[str, datetime],
                           end_date: Union[str, datetime],
                           interval: str = '1d',
                           validate_data: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple stocks simultaneously.
        
        Args:
            symbols (List[str]): List of stock symbols
            start_date (str or datetime): Start date
            end_date (str or datetime): End date
            interval (str): Data interval
            validate_data (bool): Whether to validate data
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.load_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    validate_data=validate_data
                )
                results[symbol] = data
            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {str(e)}")
                results[symbol] = None
        
        return results
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get basic information about a stock.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict[str, Any]: Stock information
        """
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
        """
        Validate the loaded stock data.
        
        Args:
            data (pd.DataFrame): Stock data to validate
            symbol (str): Stock symbol for error messages
        """
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
        """
        Fill missing values in the stock data.
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with missing values filled
        """
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
        """
        Add basic calculated columns to the stock data.
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with basic calculated columns
        """
        # Only add basic price change - keep it minimal
        if 'close' in data.columns and 'open' in data.columns:
            data['price_change'] = data['close'] - data['open']
        
        return data
    


# Convenience functions for quick data loading
def load_stock_data(symbol: str, 
                   start_date: Union[str, datetime], 
                   end_date: Union[str, datetime],
                   **kwargs) -> pd.DataFrame:
    """
    Quick function to load stock data for a single symbol.
    
    Args:
        symbol (str): Stock symbol
        start_date (str or datetime): Start date
        end_date (str or datetime): End date
        **kwargs: Additional arguments passed to StockDataLoader
        
    Returns:
        pd.DataFrame: Stock data
    """
    loader = StockDataLoader(symbol=symbol, start_date=start_date, end_date=end_date, **kwargs)
    return loader.transform()


def load_multiple_stocks(symbols: List[str],
                        start_date: Union[str, datetime],
                        end_date: Union[str, datetime],
                        **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Quick function to load data for multiple stocks.
    
    Args:
        symbols (List[str]): List of stock symbols
        start_date (str or datetime): Start date
        end_date (str or datetime): End date
        **kwargs: Additional arguments passed to StockDataLoader
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
    """
    loader = StockDataLoader(start_date=start_date, end_date=end_date, **kwargs)
    return loader.load_multiple_stocks(symbols, start_date, end_date, **kwargs)

