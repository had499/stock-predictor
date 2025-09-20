# Feature Engineering Module

A comprehensive feature engineering module for stock prediction that provides technical indicators, price-based features, volume features, and momentum indicators.

## Features

### 🎯 **Technical Indicators**
- **RSI** (Relative Strength Index) - 14 and 21 period
- **Moving Averages** - SMA and EMA for multiple periods (5, 10, 20, 50, 100, 200)
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands** with width and position
- **Stochastic Oscillator** (%K and %D)

### 📈 **Price-Based Features**
- Price changes (absolute and percentage)
- Log returns
- Open-close changes
- High-low spreads
- Price position within daily range
- Gap analysis
- Price ratios

### 📊 **Volume Features**
- Volume changes and ratios
- Volume moving averages (SMA and EMA)
- Volume volatility
- VWAP (Volume Weighted Average Price)
- Price-volume relationships

### ⚡ **Momentum Indicators**
- Rate of Change (ROC) for multiple periods
- Williams %R
- Commodity Channel Index (CCI)
- Money Flow Index (MFI)

### 📉 **Volatility Indicators**
- Historical volatility (10, 20, 30 day periods)
- Average True Range (ATR)
- True Range

### 📈 **Trend Indicators**
- ADX (Average Directional Index)
- Parabolic SAR
- Ichimoku Cloud indicators

## Quick Start

### Basic Usage

```python
from data_loader import load_stock_data
from feature_engineering import add_features

# Load data
data = load_stock_data("AAPL", "2023-01-01", "2023-12-31")

# Add all features
features = add_features(data)
print(f"Added {len(features.columns) - len(data.columns)} features")
```

### Specific Feature Types

```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

# Add only technical indicators
tech_data = engineer.add_technical_indicators(data)

# Add only price features
price_data = engineer.add_price_features(data)

# Add only volume features
volume_data = engineer.add_volume_features(data)
```

## API Reference

### FeatureEngineer Class

#### Constructor
```python
FeatureEngineer()
```

#### Main Methods

##### add_all_features()
```python
add_all_features(data, price_column='close', volume_column='volume', 
                high_column='high', low_column='low', open_column='open')
```
Adds all available features to the dataset.

##### add_price_features()
```python
add_price_features(data, price_column='close', open_column='open', 
                  high_column='high', low_column='low')
```
Adds price-based features including returns, ratios, and spreads.

##### add_volume_features()
```python
add_volume_features(data, volume_column='volume', price_column='close')
```
Adds volume-based features including ratios and VWAP.

##### add_technical_indicators()
```python
add_technical_indicators(data, price_column='close', high_column='high', low_column='low')
```
Adds technical indicators (RSI, MACD, Bollinger Bands, etc.).

##### add_momentum_indicators()
```python
add_momentum_indicators(data, price_column='close', high_column='high', low_column='low')
```
Adds momentum indicators (ROC, Williams %R, CCI, MFI).

##### add_volatility_indicators()
```python
add_volatility_indicators(data, price_column='close', high_column='high', low_column='low')
```
Adds volatility indicators (ATR, historical volatility).

##### add_trend_indicators()
```python
add_trend_indicators(data, price_column='close', high_column='high', low_column='low')
```
Adds trend indicators (ADX, Parabolic SAR, Ichimoku).

#### Individual Indicator Methods

##### calculate_rsi()
```python
calculate_rsi(prices, period=14)
```
Calculate Relative Strength Index.

##### calculate_sma()
```python
calculate_sma(prices, period)
```
Calculate Simple Moving Average.

##### calculate_ema()
```python
calculate_ema(prices, period)
```
Calculate Exponential Moving Average.

##### calculate_macd()
```python
calculate_macd(prices, fast=12, slow=26, signal=9)
```
Calculate MACD and return (macd_line, signal_line, histogram).

##### calculate_bollinger_bands()
```python
calculate_bollinger_bands(prices, period=20, std_dev=2)
```
Calculate Bollinger Bands and return (upper, middle, lower).

### Convenience Functions

#### add_features()
```python
add_features(data, **kwargs)
```
Quick function to add all features.

#### add_technical_indicators()
```python
add_technical_indicators(data, **kwargs)
```
Quick function to add only technical indicators.

## Feature Categories

### Price Features
- `price_change`: Absolute price change
- `price_change_pct`: Percentage price change
- `log_return`: Logarithmic return
- `open_close_change`: Open to close change
- `open_close_change_pct`: Open to close change percentage
- `hl_spread`: High-low spread
- `hl_spread_pct`: High-low spread percentage
- `price_position`: Price position within daily range
- `gap`: Gap from previous close
- `gap_pct`: Gap percentage
- `close_open_ratio`: Close to open ratio
- `high_close_ratio`: High to close ratio
- `low_close_ratio`: Low to close ratio

### Volume Features
- `volume_change`: Volume change
- `volume_change_pct`: Volume change percentage
- `volume_sma_5`, `volume_sma_20`: Volume moving averages
- `volume_ema_5`, `volume_ema_20`: Volume exponential moving averages
- `volume_ratio_5`, `volume_ratio_20`: Volume ratios
- `volume_volatility`: Volume volatility
- `price_volume`: Price × volume
- `vwap`: Volume Weighted Average Price

### Technical Indicators
- `rsi_14`, `rsi_21`: RSI for different periods
- `sma_5`, `sma_10`, `sma_20`, `sma_50`, `sma_100`, `sma_200`: Simple moving averages
- `ema_5`, `ema_10`, `ema_20`, `ema_50`, `ema_100`, `ema_200`: Exponential moving averages
- `macd`, `macd_signal`, `macd_histogram`: MACD components
- `bb_upper`, `bb_middle`, `bb_lower`: Bollinger Bands
- `bb_width`, `bb_position`: Bollinger Band derived features
- `stoch_k`, `stoch_d`: Stochastic Oscillator

### Momentum Indicators
- `roc_5`, `roc_10`, `roc_20`: Rate of Change
- `williams_r`: Williams %R
- `cci`: Commodity Channel Index
- `mfi`: Money Flow Index

### Volatility Indicators
- `volatility_10`, `volatility_20`, `volatility_30`: Historical volatility
- `atr`: Average True Range
- `true_range`: True Range

### Trend Indicators
- `adx`: Average Directional Index
- `sar`: Parabolic SAR
- `ichimoku_conversion`, `ichimoku_base`: Ichimoku Cloud
- `ichimoku_span_a`, `ichimoku_span_b`: Ichimoku spans

## Usage Examples

### Complete Feature Engineering Pipeline

```python
from data_loader import load_stock_data
from feature_engineering import add_features

# Load data
data = load_stock_data("AAPL", "2023-01-01", "2023-12-31")

# Add all features
features = add_features(data)

# Check feature count
print(f"Original columns: {len(data.columns)}")
print(f"With features: {len(features.columns)}")
print(f"New features added: {len(features.columns) - len(data.columns)}")
```

### Custom Feature Selection

```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

# Start with base data
df = data.copy()

# Add specific feature types
df = engineer.add_price_features(df)
df = engineer.add_technical_indicators(df)
df = engineer.add_volume_features(df)

# Calculate individual indicators
rsi = engineer.calculate_rsi(df['close'], period=14)
macd, signal, hist = engineer.calculate_macd(df['close'])
```

### Feature Analysis

```python
# Check for missing values
missing_data = features.isnull().sum()
print("Missing values per feature:")
print(missing_data[missing_data > 0].sort_values(ascending=False))

# Check feature ranges
print(f"RSI range: {features['rsi_14'].min():.2f} - {features['rsi_14'].max():.2f}")
print(f"Price change range: {features['price_change'].min():.2f} - {features['price_change'].max():.2f}")

# Check correlation with target (if available)
correlations = features.corr()['target_column'].abs().sort_values(ascending=False)
print("Top correlated features:")
print(correlations.head(10))
```

## Performance Notes

- **Memory Usage**: Adding all features significantly increases memory usage
- **Missing Values**: Some indicators require warm-up periods (e.g., 200-day SMA)
- **Computational Cost**: Technical indicators are computationally efficient
- **Data Quality**: Features are calculated from raw OHLCV data

## Dependencies

- pandas
- numpy
- logging (built-in)
- warnings (built-in)
- typing (built-in)

## Error Handling

The module includes comprehensive error handling:
- **Missing Columns**: Validates required columns before processing
- **Invalid Parameters**: Checks parameter ranges and types
- **Data Validation**: Ensures data quality before feature calculation
- **Logging**: Provides detailed logging for debugging

## License

This module is part of the stock predictor project.
