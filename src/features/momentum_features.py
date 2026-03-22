import numpy as np
import pandas as pd


def calculate_roc(prices: pd.Series, period: int) -> pd.Series:
    return ((prices - prices.shift(period)) / prices.shift(period)) * 100


def calculate_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * ((highest_high - close) / (highest_high - lowest_low))


def calculate_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (typical_price - sma_tp) / (0.015 * mad)


def calculate_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()

    return 100 - (100 / (1 + (positive_flow / negative_flow)))


def add_momentum_indicators(
    data: pd.DataFrame,
    *,
    price_column: str = "close",
    high_column: str = "high",
    low_column: str = "low",
    volume_column: str = "volume",
) -> pd.DataFrame:
    df = data.copy()

    # ROC removed — redundant with return_Xlag in price_features.py

    df["williams_r"] = calculate_williams_r(df[high_column], df[low_column], df[price_column]).shift(1)
    df["cci"] = calculate_cci(df[high_column], df[low_column], df[price_column]).shift(1)
    df["mfi"] = calculate_mfi(
        df[high_column], df[low_column], df[price_column], df[volume_column]
    ).shift(1)

    return df
