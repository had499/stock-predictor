"""
Stock Data Loader Module

This module provides functionality to load stock data for given date ranges
using yfinance. It includes data validation, error handling, and various
data processing options.

Author: Stock Predictor Project
"""

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Union, List, Optional, Dict, Any
import logging
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataLoader:
    """
    A comprehensive stock data loader that handles data retrieval,
    validation, and preprocessing for stock prediction tasks.
    """
    
    def __init__(self, 
                 default_period: int = 14,
                 auto_adjust: bool = True,
                 prepost: bool = False,
                 threads: bool = True,
                 proxy: Optional[str] = None):
        """
        Initialize the StockDataLoader.
        
        Args:
            default_period (int): Default period for technical indicators
            auto_adjust (bool): Whether to auto-adjust prices for splits/dividends
            prepost (bool): Whether to include pre/post market data
            threads (bool): Whether to use threads for downloading
            proxy (str, optional): Proxy URL for requests
        """
        self.default_period = default_period
        self.auto_adjust = auto_adjust
        self.prepost = prepost
        self.threads = threads
        self.proxy = proxy
        
        # Configure yfinance warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
    
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
        Add calculated columns to the stock data.
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with additional calculated columns
        """
        # Add daily returns
        if 'close' in data.columns:
            data['daily_return'] = data['close'].pct_change()
            data['log_return'] = np.log(data['close'] / data['close'].shift(1))
        
        # Add price change
        if 'close' in data.columns and 'open' in data.columns:
            data['price_change'] = data['close'] - data['open']
            data['price_change_pct'] = (data['close'] - data['open']) / data['open'] * 100
        
        # Add high-low spread
        if 'high' in data.columns and 'low' in data.columns:
            data['hl_spread'] = data['high'] - data['low']
            data['hl_spread_pct'] = (data['high'] - data['low']) / data['low'] * 100
        
        # Add volatility (rolling standard deviation of returns)
        if 'daily_return' in data.columns:
            data['volatility'] = data['daily_return'].rolling(window=20).std()
        
        return data
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = None, price_column: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) for the given data.
        
        Args:
            data (pd.DataFrame): Stock data
            period (int, optional): RSI period (defaults to self.default_period)
            price_column (str): Column to use for RSI calculation
            
        Returns:
            pd.Series: RSI values
        """
        if period is None:
            period = self.default_period
        
        if price_column not in data.columns:
            raise ValueError(f"Column '{price_column}' not found in data")
        
        prices = data[price_column]
        
        # Calculate daily differences
        diff = prices.diff()
        
        # Separate gains and losses
        gains = diff.clip(lower=0)
        losses = -diff.clip(upper=0)
        
        # Calculate EMA for gains and losses
        avg_gains = gains.ewm(span=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_sma(self, data: pd.DataFrame, period: int, price_column: str = 'close') -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            data (pd.DataFrame): Stock data
            period (int): SMA period
            price_column (str): Column to use for SMA calculation
            
        Returns:
            pd.Series: SMA values
        """
        if price_column not in data.columns:
            raise ValueError(f"Column '{price_column}' not found in data")
        
        return data[price_column].rolling(window=period).mean()
    
    def calculate_ema(self, data: pd.DataFrame, period: int, price_column: str = 'close') -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            data (pd.DataFrame): Stock data
            period (int): EMA period
            price_column (str): Column to use for EMA calculation
            
        Returns:
            pd.Series: EMA values
        """
        if price_column not in data.columns:
            raise ValueError(f"Column '{price_column}' not found in data")
        
        return data[price_column].ewm(span=period, adjust=False).mean()


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
        **kwargs: Additional arguments passed to StockDataLoader.load_stock_data
        
    Returns:
        pd.DataFrame: Stock data
    """
    loader = StockDataLoader()
    return loader.load_stock_data(symbol, start_date, end_date, **kwargs)


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
        **kwargs: Additional arguments passed to StockDataLoader.load_multiple_stocks
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
    """
    loader = StockDataLoader()
    return loader.load_multiple_stocks(symbols, start_date, end_date, **kwargs)

