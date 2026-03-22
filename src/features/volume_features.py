import numpy as np
import pandas as pd


def add_volume_features(
    data: pd.DataFrame,
    *,
    price_column: str = "close",
    volume_column: str = "volume",
) -> pd.DataFrame:
    df = data.copy()

    df["volume_change_1lag"] = df[volume_column].shift(1) - df[volume_column].shift(2)

    # Volume moving averages (prevent leakage)
    df["volume_sma_5"] = df[volume_column].shift(1).rolling(window=5, min_periods=5).mean()
    df["volume_sma_20"] = df[volume_column].shift(1).rolling(window=20, min_periods=20).mean()
    df["volume_ema_5"] = df[volume_column].shift(1).ewm(span=5, min_periods=5).mean()
    df["volume_ema_20"] = df[volume_column].shift(1).ewm(span=20, min_periods=20).mean()

    # Volume ratios
    df["volume_ratio_5"] = df[volume_column].shift(1) / df["volume_sma_5"].replace(0, np.nan)
    df["volume_ratio_20"] = df[volume_column].shift(1) / df["volume_sma_20"].replace(0, np.nan)

    # Volume volatility
    df["volume_volatility"] = df[volume_column].shift(1).rolling(window=20, min_periods=20).std()

    # VWAP ratio (close / vwap - 1) — normalised, scale-free
    price_vol_lag = df[price_column].shift(1) * df[volume_column].shift(1)
    vwap_denom = df[volume_column].shift(1).rolling(window=20, min_periods=20).sum()
    vwap = price_vol_lag.rolling(window=20, min_periods=20).sum() / vwap_denom.replace(0, np.nan)
    df["vwap_ratio"] = (df[price_column].shift(1) / vwap.replace(0, np.nan) - 1)

    return df
