"""
Financial Modeling Prep (FMP) analyst price target scraper.

Fetches the full historical time series of analyst price target events
per ticker via the FMP legacy API (free tier: 250 calls/day).

Requires a free FMP API key set as the environment variable:
    FMP_API_KEY=<your_key>

Register at: https://financialmodelingprep.com/register

Each event captures: published date, analyst firm, price target, stock
price at time of publication.

Exports: scrape_price_targets
"""

import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from .._cache import _load_incremental, _merge_and_save

logger = logging.getLogger(__name__)

_BASE_URL = "https://financialmodelingprep.com/api/v4/price-target"


def scrape_price_targets(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical analyst price targets from FMP.

    Returns a daily-aggregated DataFrame with mean/high/low consensus
    targets and a revision direction signal.

    Requires env var ``FMP_API_KEY``. Get a free key (250 calls/day) at
    https://financialmodelingprep.com/register

    Parameters
    ----------
    ticker : str
    start_date, end_date : str (ISO format)
    use_cache : bool

    Returns
    -------
    DataFrame with columns:
        date, ticker,
        pt_mean, pt_high, pt_low, pt_count,
        pt_revision_direction  (+1 raised / -1 lowered / 0 maintained)
    """
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FMP_API_KEY environment variable not set. "
            "Get a free key at https://financialmodelingprep.com/register"
        )

    start = start_date or "2021-01-01"
    end = end_date or datetime.today().strftime("%Y-%m-%d")

    existing_df, fetch_from = (
        _load_incremental("price_targets", ticker, start, end, tolerance_days=7)
        if use_cache else (None, start)
    )
    if fetch_from > end:
        return existing_df

    try:
        resp = requests.get(
            _BASE_URL,
            params={"symbol": ticker.upper(), "apikey": api_key},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("FMP price targets failed for %s: %s", ticker, e)
        return existing_df if existing_df is not None else _empty_price_target_df(ticker)

    if not data or not isinstance(data, list):
        logger.info("FMP: no price target data for %s", ticker)
        return existing_df if existing_df is not None else _empty_price_target_df(ticker)

    rows = []
    for item in data:
        try:
            pub_date = str(item.get("publishedDate", "") or "")[:10]
            if not pub_date or pub_date < fetch_from or pub_date > end:
                continue
            pt = item.get("priceTarget")
            if pt is None:
                continue
            rows.append({
                "date": pub_date,
                "ticker": ticker.upper(),
                "price_target": float(pt),
                "price_when_posted": float(item.get("priceWhenPosted") or 0) or None,
                "analyst_company": str(item.get("analystCompany") or ""),
            })
        except Exception:
            continue

    if not rows:
        logger.info("FMP: no price targets in range [%s, %s] for %s", fetch_from, end, ticker)
        return existing_df if existing_df is not None else _empty_price_target_df(ticker)

    raw = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Compute revision direction per event vs. the previous target from the same analyst
    raw["_prev_pt"] = raw.groupby("analyst_company")["price_target"].shift(1)
    raw["_direction"] = 0
    raw.loc[raw["price_target"] > raw["_prev_pt"], "_direction"] = 1
    raw.loc[raw["price_target"] < raw["_prev_pt"], "_direction"] = -1

    # Daily aggregation
    daily = raw.groupby("date").agg(
        pt_mean=("price_target", "mean"),
        pt_high=("price_target", "max"),
        pt_low=("price_target", "min"),
        pt_count=("price_target", "count"),
        pt_revision_direction=("_direction", "mean"),  # avg direction on that day
    ).reset_index()

    daily["ticker"] = ticker.upper()
    # Round direction to nearest int: overall bias on that day
    daily["pt_revision_direction"] = daily["pt_revision_direction"].round().astype(int)

    df = daily[["date", "ticker", "pt_mean", "pt_high", "pt_low",
                "pt_count", "pt_revision_direction"]]

    logger.info("FMP price targets: %d events → %d days for %s", len(raw), len(df), ticker)

    if use_cache:
        return _merge_and_save(df, existing_df, "price_targets", ticker, start, end)
    return df


def _empty_price_target_df(ticker: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="str"),
        "ticker": pd.Series(dtype="str"),
        "pt_mean": pd.Series(dtype="float"),
        "pt_high": pd.Series(dtype="float"),
        "pt_low": pd.Series(dtype="float"),
        "pt_count": pd.Series(dtype="int"),
        "pt_revision_direction": pd.Series(dtype="int"),
    })
