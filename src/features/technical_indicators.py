import numpy as np
import pandas as pd


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    return prices.rolling(window=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    return prices.ewm(span=period).mean()


def calculate_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: pd.Series, period: int = 20, std_dev: float = 2
) -> tuple[pd.Series, pd.Series, pd.Series]:
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent


def add_technical_indicators(
    data: pd.DataFrame,
    *,
    price_column: str = "close",
    high_column: str = "high",
    low_column: str = "low",
) -> pd.DataFrame:
    df = data.copy()

    df["rsi_14"] = calculate_rsi(df[price_column], period=14).shift(1)
    df["rsi_21"] = calculate_rsi(df[price_column], period=21).shift(1)

    # SMA/EMA as price ratios (close / MA - 1) instead of raw dollar levels
    close = df[price_column]
    for period in [5, 10, 20, 50, 100, 200]:
        sma = calculate_sma(close, period)
        ema = calculate_ema(close, period)
        df[f"sma_ratio_{period}"] = (close / sma.replace(0, np.nan) - 1).shift(1)
        df[f"ema_ratio_{period}"] = (close / ema.replace(0, np.nan) - 1).shift(1)

    # MACD normalised by price (percentage terms)
    macd_line, signal_line, histogram = calculate_macd(close)
    df["macd_pct"] = (macd_line / close.replace(0, np.nan)).shift(1)
    df["macd_signal_pct"] = (signal_line / close.replace(0, np.nan)).shift(1)
    df["macd_histogram_pct"] = (histogram / close.replace(0, np.nan)).shift(1)

    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
    df["bb_width"] = ((bb_upper - bb_lower) / bb_middle.replace(0, np.nan)).shift(1)
    df["bb_position"] = (
        (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    ).shift(1)

    stoch_k, stoch_d = calculate_stochastic(df[high_column], df[low_column], close)
    df["stoch_k"] = stoch_k.shift(1)
    df["stoch_d"] = stoch_d.shift(1)

    return df
