import numpy as np
import pandas as pd


def calculate_volatility(prices: pd.Series, period: int) -> pd.Series:
    returns = prices.pct_change()
    return returns.rolling(window=period).std() * np.sqrt(252)


def calculate_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    return np.maximum(tr1, np.maximum(tr2, tr3))


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    true_range = calculate_true_range(high, low, close)
    return true_range.rolling(window=period).mean()


def add_volatility_indicators(
    data: pd.DataFrame,
    *,
    price_column: str = "close",
    high_column: str = "high",
    low_column: str = "low",
) -> pd.DataFrame:
    df = data.copy()
    returns = df[price_column].pct_change()

    for period in [10, 20, 30]:
        df[f"volatility_{period}"] = calculate_volatility(df[price_column], period).shift(1)

    df["atr"] = calculate_atr(df[high_column], df[low_column], df[price_column]).shift(1)
    df["true_range"] = calculate_true_range(df[high_column], df[low_column], df[price_column]).shift(1)

    # rv_1d removed — rolling(1).std() is always NaN
    df["rv_5d_lag"] = returns.rolling(window=5).std().shift(1)
    df["rv_22d_lag"] = returns.rolling(window=22).std().shift(1)

    return df
