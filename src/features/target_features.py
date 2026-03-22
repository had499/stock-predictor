import pandas as pd


def add_target_col(
    data: pd.DataFrame,
    *,
    price_column: str = "close",
    horizon: int = 1,
) -> pd.DataFrame:
    df = data.copy()

    if horizon == 1:
        df["target"] = df[price_column].pct_change().shift(-1)
    else:
        df["target"] = (df[price_column].shift(-horizon) / df[price_column]) - 1

    return df
