import pandas as pd


def remove_today_features(
    data: pd.DataFrame,
    *,
    price_column: str = "close",
    volume_column: str = "volume",
    high_column: str = "high",
    low_column: str = "low",
    open_column: str = "open",
) -> pd.DataFrame:
    df = data.copy()

    df.drop(
        columns=[
            price_column,
            volume_column,
            high_column,
            low_column,
            open_column,
            "dividends",
            "stock splits",
            "price_change",
        ],
        inplace=True,
        errors="ignore",
    )

    return df
