import numpy as np
import pandas as pd

from .volatility_features import calculate_atr


def calculate_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    plus_dm = high.diff()                        # high[t] - high[t-1]
    minus_dm = low.shift(1) - low                 # low[t-1] - low[t]  (fixed sign)

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr = calculate_atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(window=period).mean()


def calculate_sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    acceleration: float = 0.02,
    maximum: float = 0.2,
) -> pd.Series:
    """Parabolic SAR using only previous-bar data (no look-ahead)."""
    n = len(close)
    sar = pd.Series(np.nan, index=close.index)
    af = acceleration
    uptrend = True

    first_valid = low.first_valid_index()
    if first_valid is None:
        return sar
    start = low.index.get_loc(first_valid)
    if start + 1 >= n:
        return sar

    # Initialise: SAR = first low, EP = first high
    sar.iloc[start] = low.iloc[start]
    ep = high.iloc[start]

    for i in range(start + 1, n):
        prev_sar = sar.iloc[i - 1]
        if pd.isna(prev_sar) or pd.isna(high.iloc[i - 1]) or pd.isna(low.iloc[i - 1]):
            sar.iloc[i] = np.nan
            continue

        # Compute new SAR from previous bar's values only
        new_sar = prev_sar + af * (ep - prev_sar)

        if uptrend:
            # Clamp SAR to not exceed previous two lows
            new_sar = min(new_sar, low.iloc[i - 1])
            if i - 2 >= 0 and not pd.isna(low.iloc[i - 2]):
                new_sar = min(new_sar, low.iloc[i - 2])
            # Check for reversal using previous bar
            if low.iloc[i - 1] < new_sar:
                uptrend = False
                new_sar = ep
                ep = low.iloc[i - 1]
                af = acceleration
            else:
                if high.iloc[i - 1] > ep:
                    ep = high.iloc[i - 1]
                    af = min(af + acceleration, maximum)
        else:
            # Clamp SAR to not go below previous two highs
            new_sar = max(new_sar, high.iloc[i - 1])
            if i - 2 >= 0 and not pd.isna(high.iloc[i - 2]):
                new_sar = max(new_sar, high.iloc[i - 2])
            # Check for reversal
            if high.iloc[i - 1] > new_sar:
                uptrend = True
                new_sar = ep
                ep = high.iloc[i - 1]
                af = acceleration
            else:
                if low.iloc[i - 1] < ep:
                    ep = low.iloc[i - 1]
                    af = min(af + acceleration, maximum)

        sar.iloc[i] = new_sar

    return sar


def calculate_ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    conversion_period: int = 9,
    base_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> dict[str, pd.Series]:
    conversion = (high.rolling(window=conversion_period).max() + low.rolling(window=conversion_period).min()) / 2
    base = (high.rolling(window=base_period).max() + low.rolling(window=base_period).min()) / 2
    span_a = (conversion + base) / 2
    span_b = (high.rolling(window=span_b_period).max() + low.rolling(window=span_b_period).min()) / 2

    return {
        "conversion": conversion,
        "base": base,
        "span_a": span_a,
        "span_b": span_b,
    }


def add_trend_indicators(
    data: pd.DataFrame,
    *,
    price_column: str = "close",
    high_column: str = "high",
    low_column: str = "low",
) -> pd.DataFrame:
    df = data.copy()

    close = df[price_column]
    df["adx"] = calculate_adx(df[high_column], df[low_column], close).shift(1)

    # SAR as ratio to close (normalised, scale-free)
    sar_raw = calculate_sar(df[high_column], df[low_column], close)
    df["sar_ratio"] = (close / sar_raw.replace(0, np.nan) - 1).shift(1)

    # Ichimoku as ratios to close (normalised, scale-free)
    ichimoku = calculate_ichimoku(df[high_column], df[low_column], close)
    df["ichimoku_conversion_ratio"] = (close / ichimoku["conversion"].replace(0, np.nan) - 1).shift(1)
    df["ichimoku_base_ratio"] = (close / ichimoku["base"].replace(0, np.nan) - 1).shift(1)
    df["ichimoku_span_a_ratio"] = (close / ichimoku["span_a"].replace(0, np.nan) - 1).shift(1)
    df["ichimoku_span_b_ratio"] = (close / ichimoku["span_b"].replace(0, np.nan) - 1).shift(1)

    return df
