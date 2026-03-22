import pandas as pd


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    if "date" in df.columns:
        date_col = pd.to_datetime(df["date"])
        df["day_of_week"] = date_col.dt.dayofweek
        df["month"] = date_col.dt.month
        df["quarter"] = date_col.dt.quarter
        # year removed — ever-increasing integer creates spurious time trend
        df["is_month_end"] = date_col.dt.is_month_end.astype(int)
        df["is_quarter_end"] = date_col.dt.is_quarter_end.astype(int)
        df["is_year_end"] = date_col.dt.is_year_end.astype(int)
    return df
