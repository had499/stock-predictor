# Stock Data Loader

A comprehensive Python module for loading and processing stock market data using the `yfinance` library. This data loader is designed specifically for the stock predictor project and provides robust data handling with validation, error handling, and built-in technical indicators.

## Features

- **Easy Data Loading**: Load single or multiple stock symbols with simple function calls
- **Date Range Support**: Flexible date range input (string or datetime objects)
- **Data Validation**: Automatic validation of loaded data with warnings for issues
- **Missing Data Handling**: Automatic filling of missing values
- **Clean Data**: Returns raw OHLCV data without unnecessary feature engineering
- **Error Handling**: Comprehensive error handling with informative messages
- **Multiple Time Intervals**: Support for daily, weekly, hourly, and minute data
- **Stock Information**: Get basic company information for any stock symbol

## Installation

Make sure you have the required dependencies installed:

```bash
pip install -r ../requirements.txt
```

## Quick Start

### Basic Usage

```python
from data_loader import load_stock_data

# Load Apple stock data for 2023
data = load_stock_data("AAPL", "2023-01-01", "2023-12-31")
print(data.head())
```

### Multiple Stocks

```python
from data_loader import load_multiple_stocks

# Load multiple stocks
symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
multi_data = load_multiple_stocks(symbols, "2023-01-01", "2023-12-31")

for symbol, data in multi_data.items():
    if data is not None:
        print(f"{symbol}: {len(data)} records")
```

### Advanced Usage

```python
from data_loader import StockDataLoader

# Create loader instance
loader = StockDataLoader()

# Load data with custom settings
data = loader.load_stock_data(
    symbol="AAPL",
    start_date="2023-01-01",
    end_date="2023-12-31",
    interval="1d",
    validate_data=True,
    fill_missing=True
)

# Get stock information
info = loader.get_stock_info("AAPL")
print(f"Company: {info['name']}")
print(f"Sector: {info['sector']}")
```

## API Reference

### StockDataLoader Class

#### Constructor

```python
StockDataLoader(
    default_period=14,
    auto_adjust=True,
    prepost=False,
    threads=True,
    proxy=None
)
```

**Parameters:**
- `default_period` (int): Default period for technical indicators
- `auto_adjust` (bool): Whether to auto-adjust prices for splits/dividends
- `prepost` (bool): Whether to include pre/post market data
- `threads` (bool): Whether to use threads for downloading
- `proxy` (str, optional): Proxy URL for requests

#### Methods

##### load_stock_data()

```python
load_stock_data(
    symbol: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    interval: str = '1d',
    validate_data: bool = True,
    fill_missing: bool = True
) -> pd.DataFrame
```

Load stock data for a given symbol and date range.

**Parameters:**
- `symbol` (str): Stock symbol (e.g., 'AAPL', 'MSFT')
- `start_date` (str or datetime): Start date in 'YYYY-MM-DD' format or datetime object
- `end_date` (str or datetime): End date in 'YYYY-MM-DD' format or datetime object
- `interval` (str): Data interval ('1d', '1h', '5m', etc.)
- `validate_data` (bool): Whether to validate the loaded data
- `fill_missing` (bool): Whether to fill missing values

**Returns:**
- `pd.DataFrame`: Stock data with OHLCV columns and additional calculated columns

##### load_multiple_stocks()

```python
load_multiple_stocks(
    symbols: List[str],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    interval: str = '1d',
    validate_data: bool = True
) -> Dict[str, pd.DataFrame]
```

Load data for multiple stocks simultaneously.


##### get_stock_info()

```python
get_stock_info(symbol: str) -> Dict[str, Any]
```

Get basic information about a stock.

### Convenience Functions

#### load_stock_data()

Quick function to load stock data for a single symbol.

```python
load_stock_data(symbol, start_date, end_date, **kwargs) -> pd.DataFrame
```

#### load_multiple_stocks()

Quick function to load data for multiple stocks.

```python
load_multiple_stocks(symbols, start_date, end_date, **kwargs) -> Dict[str, pd.DataFrame]
```

## Data Format

The loaded data includes the following columns:

### Standard OHLCV Columns
- `date`: Date index
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

### Additional Calculated Columns
- `price_change`: Absolute price change (close - open)

## Error Handling

The data loader includes comprehensive error handling:

- **Invalid symbols**: Clear error messages for non-existent stock symbols
- **Invalid date ranges**: Validation that start date is before end date
- **Network issues**: Graceful handling of connection problems
- **Missing data**: Warnings and automatic filling of missing values
- **Data validation**: Checks for negative prices, missing columns, etc.

## Examples

See `example_usage.py` for comprehensive examples of how to use the data loader in various scenarios.

## Dependencies

- pandas
- numpy
- yfinance
- datetime (built-in)
- logging (built-in)
- warnings (built-in)
- typing (built-in)

## License

This module is part of the stock predictor project.
