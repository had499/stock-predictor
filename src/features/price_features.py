import numpy as np
import pandas as pd


def add_price_features(
    data: pd.DataFrame,
    *,
    price_column: str = "close",
    volume_column: str = "volume",
    high_column: str = "high",
    low_column: str = "low",
    open_column: str = "open",
) -> pd.DataFrame:
    df = data.copy()

    # Percentage-based features only (no raw dollar levels)
    df["open_close_change_pct_1lag"] = (
        (df[price_column].shift(1) - df[open_column].shift(1)) / df[open_column].shift(1)
    )

    # High-low features (percentage only)
    df["hl_spread_pct_1lag"] = (
        (df[high_column].shift(1) - df[low_column].shift(1)) / df[low_column].shift(1)
    )

    # Price position within daily range
    price_position_denom = df[high_column].shift(1) - df[low_column].shift(1)
    df["price_position_1lag"] = (
        (df[price_column].shift(1) - df[low_column].shift(1))
        / price_position_denom.replace(0, np.nan)
    )

    # Gap features
    df["gap_1lag"] = df[open_column].shift(1) - df[price_column].shift(2)
    df["gap_pct_1lag"] = df["gap_1lag"] / df[price_column].shift(2)

    # Price ratios
    df["close_open_ratio_1lag"] = df[price_column].shift(1) / df[open_column].shift(1)
    df["high_close_ratio_1lag"] = df[high_column].shift(1) / df[price_column].shift(1)
    df["low_close_ratio_1lag"] = df[low_column].shift(1) / df[price_column].shift(1)

    # Lagged returns (momentum/reversal signals)
    returns = df[price_column].pct_change()
    df["return_1lag"] = returns.shift(1)
    df["return_2lag"] = returns.shift(2)
    df["return_3lag"] = returns.shift(3)
    df["return_5lag"] = returns.shift(5)
    df["return_10lag"] = returns.shift(10)

    # Multi-day lagged returns (for longer horizon prediction)
    df["return_7d_1lag"] = df[price_column].shift(1) / df[price_column].shift(8) - 1
    df["return_7d_2lag"] = df[price_column].shift(8) / df[price_column].shift(15) - 1

    return df
