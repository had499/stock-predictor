import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def add_etf_features(
    data: pd.DataFrame,
    *,
    price_column: str = "close",
    etf_returns: Optional[pd.DataFrame] = None,
    etf_columns: Optional[list[str]] = None,
    sector_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    df = data.copy()
    sector_map = sector_map or {}
    etf_columns = etf_columns or []

    if "date" not in df.columns:
        if df.index.name == "date":
            df = df.reset_index()
        else:
            logger.warning("No 'date' column found in input data. ETF features will not be added.")
            return df

    if etf_returns is None or etf_returns.empty:
        return df

    df = df.merge(etf_returns, on="date", how="left")

    df["stock_return_1lag"] = df[price_column].pct_change().shift(1)

    if sector_map and "symbol" in df.columns:
        symbol = df["symbol"].iloc[0]
        sector_etfs = sector_map.get(symbol, [])
        if isinstance(sector_etfs, str):
            sector_etfs = [sector_etfs]

        if sector_etfs:
            for sector_etf in sector_etfs:
                df[f"ETF_map_{sector_etf}"] = 1

            # Keep behavior consistent with existing code: compute relative returns/correlations
            # for all ETFs in etf_columns (not just sector_etfs).
            for sector_etf in etf_columns:
                col_name = f"{sector_etf}_return_1lag"
                if col_name in df.columns:
                    df[f"relative_return_1lag_{sector_etf}"] = df["stock_return_1lag"] - df[col_name]

                    for window in [20, 60, 120]:
                        df[f"corr_{sector_etf}_{window}d"] = (
                            df["stock_return_1lag"].rolling(window).corr(df[col_name])
                        )

    return df


def clean_etf(data: pd.DataFrame, *, etf_columns: list[str]) -> pd.DataFrame:
    df = data.copy()
    df = df[~df["symbol"].isin(etf_columns)]

    etf_flags = [col for col in df.columns if "ETF_map_" in col]
    if etf_flags:
        df[etf_flags] = df[etf_flags].fillna(0)

    return df
