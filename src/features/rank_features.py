import numpy as np
import pandas as pd


# Only rank the most informative base features (not all 160+).
# Features are already lagged — DO NOT add an extra .shift(1).
_RANK_FEATURES = [
    "return_1lag", "return_5lag", "return_10lag",      # momentum
    "return_7d_1lag",                                    # weekly momentum
    "rv_5d_lag", "rv_22d_lag",                           # volatility
    "volume_ratio_5", "volume_ratio_20",                 # volume surprise
    "rsi_14",                                            # mean-reversion signal
    "bb_position",                                       # Bollinger position
    "mfi",                                               # money flow
    "atr",                                               # absolute risk
]


def add_rank_features(data: pd.DataFrame, *, etf_columns: list[str]) -> pd.DataFrame:
    df = data.copy()

    # Only rank a curated subset of features
    feature_cols = [c for c in _RANK_FEATURES if c in df.columns]

    mask = ~df["symbol"].isin(etf_columns)

    # Cross-sectional percentile per date (no extra lag — features are already shifted)
    pctile_features = df.groupby("date")[feature_cols].rank(pct=True).add_suffix("_pctile")

    df = pd.concat([df, pctile_features], axis=1)

    df.loc[~mask, pctile_features.columns] = np.nan

    return df
