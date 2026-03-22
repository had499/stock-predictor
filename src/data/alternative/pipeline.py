"""
Alternative Data Pipeline

End-to-end pipeline that scrapes alternative data for one or more
tickers, engineers features, and returns a model-ready DataFrame.

Supports the same ``get_…(str | list[str])`` pattern as the news module.

Author: Stock Predictor Project
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Literal, Union, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .scrapers import (
    scrape_alternative_data,
    scrape_fred_macro,
    scrape_macro_data,
    scrape_cross_asset_data,
    scrape_etf_fund_flows,
)
from .features import build_alternative_features, build_macro_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================

def get_alternative_data(
    ticker: Union[str, List[str]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sources: Optional[List[str]] = None,
    use_cache: bool = True,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Scrape alternative data, engineer features, and return a daily
    DataFrame ready to merge with your stock price data.

    Accepts a **single ticker** (str) or a **list of tickers**.
    When given a list, tickers are scraped concurrently and macro
    data is fetched only once.

    Usage
    -----
    >>> from data.alternative import get_alternative_data
    >>> # Single ticker
    >>> alt = get_alternative_data("AAPL", "2021-01-01", "2025-03-01")
    >>> # Multiple tickers – concurrent, macro shared
    >>> alt = get_alternative_data(
    ...     ["AAPL", "MSFT", "GOOGL"],
    ...     start_date="2021-01-01",
    ...     end_date="2025-03-01",
    ... )

    Parameters
    ----------
    ticker : str or list[str]
        One or more stock ticker symbols.
    start_date, end_date : str or None
        ISO date strings.  Defaults to 2021-01-01 → today.
    sources : list[str] or None
        Which alternative data sources to use.  Options:
        'insider', 'earnings', 'short_interest', 'congress',
        'wiki', 'fred', 'earnings_revisions'.
        Defaults to all.
    use_cache : bool
        Cache raw scrapes to disk (parquet).
    max_workers : int
        Threads for concurrent ticker scraping.

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, <alternative features …>
        Sorted by (ticker, date).
    """
    from datetime import datetime

    start = start_date or "2021-01-01"
    end = end_date or datetime.today().strftime("%Y-%m-%d")

    if isinstance(ticker, str):
        return _process_single(ticker, start, end, sources, use_cache)

    tickers = list(ticker)
    if not tickers:
        return pd.DataFrame()

    return _process_batch(tickers, start, end, sources, use_cache, max_workers)


# ============================================================================
# Internal
# ============================================================================

def _process_single(
    ticker: str,
    start: str,
    end: str,
    sources: Optional[List[str]],
    use_cache: bool,
) -> pd.DataFrame:
    """Scrape + feature-engineer for a single ticker."""
    logger.info("Fetching alternative data for %s (%s → %s)", ticker, start, end)

    raw = scrape_alternative_data(
        ticker=ticker,
        start_date=start,
        end_date=end,
        sources=sources,
        use_cache=use_cache,
    )

    df = build_alternative_features(
        raw_data=raw,
        ticker=ticker,
        start_date=start,
        end_date=end,
    )

    logger.info("%s: %d rows × %d features", ticker, len(df), df.shape[1] - 2)
    return df


def _scrape_ticker(
    ticker: str,
    start: str,
    end: str,
    sources: Optional[List[str]],
    use_cache: bool,
    shared_macro: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Thread-safe: scrape one ticker, inject shared macro data."""
    try:
        # Remove 'fred' from per-ticker scrape (it's global)
        ticker_sources = None
        if sources:
            ticker_sources = [s for s in sources if s != "fred"]
        else:
            ticker_sources = [
                "insider", "earnings",
                "congress", "wiki",
            ]

        raw = scrape_alternative_data(
            ticker=ticker,
            start_date=start,
            end_date=end,
            sources=ticker_sources,
            use_cache=use_cache,
        )

        # Inject shared macro data
        if shared_macro is not None:
            raw["fred"] = shared_macro

        df = build_alternative_features(
            raw_data=raw,
            ticker=ticker,
            start_date=start,
            end_date=end,
        )
        return df

    except Exception as e:
        logger.warning("  ✗ %s failed: %s", ticker, e)
        return pd.DataFrame()


def _process_batch(
    tickers: List[str],
    start: str,
    end: str,
    sources: Optional[List[str]],
    use_cache: bool,
    max_workers: int,
) -> pd.DataFrame:
    """
    Efficient multi-ticker pipeline:
      1. Scrape ticker-specific alternative sources concurrently
      2. Concatenate all

    NOTE: FRED macro / cross-asset / ETF flows are NOT included here.
    Use ``get_macro_data()`` separately to avoid duplicate columns.
    """
    total = len(tickers)
    logger.info("Batch alternative data for %d tickers …", total)

    # ── Concurrent ticker scraping (no macro — that's handled by get_macro_data) ──
    frames: List[pd.DataFrame] = []

    # Exclude fred from per-ticker sources to avoid overlap with get_macro_data()
    ticker_sources = sources
    if ticker_sources is None:
        ticker_sources = [
            "insider", "earnings",
            "congress", "wiki", "earnings_revisions",
        ]
    else:
        ticker_sources = [s for s in ticker_sources if s != "fred"]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(
                _scrape_ticker, t, start, end, ticker_sources, use_cache, None,
            ): t
            for t in tickers
        }

        for i, future in enumerate(as_completed(future_map), 1):
            t = future_map[future]
            try:
                df = future.result()
                if not df.empty:
                    frames.append(df)
                    logger.info("[%d/%d] %s → %d rows", i, total, t, len(df))
                else:
                    logger.info("[%d/%d] %s → empty", i, total, t)
            except Exception as e:
                logger.warning("[%d/%d] %s failed: %s", i, total, t, e)

    if not frames:
        logger.warning("No alternative data collected for any ticker")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    logger.info(
        "Batch complete: %d tickers, %d rows × %d cols",
        total, len(combined), combined.shape[1],
    )
    return combined


# ============================================================================
# Standalone Macro Data Pipeline (ticker-agnostic)
# ============================================================================

def get_macro_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch and engineer a comprehensive set of macro-economic features
    that are **external to any individual ticker**.

    This downloads ~55 FRED series covering rates, yield curve, volatility,
    labor, inflation, currency, sentiment, housing, and leading indicators,
    then derives ~100+ features including momentum, regimes, z-scores,
    and cross-indicator composites.

    The result has **one row per business day** – merge into your stock
    DataFrame on ``date`` alone (no ticker column needed).

    Usage
    -----
    >>> from data.alternative import get_macro_data
    >>> macro = get_macro_data("2021-01-01", "2025-03-01")
    >>> macro.shape
    (1044, 120)

    >>> # Merge with your stock data
    >>> df = df.merge(macro, on="date", how="left")

    Parameters
    ----------
    start_date, end_date : str or None
        ISO date strings.  Defaults to 2021-01-01 → today.
    use_cache : bool
        Cache raw FRED download to ``datasets/alternative/``.

    Returns
    -------
    pd.DataFrame
        Columns: date, <raw indicators>, <derived features>.
        All numeric except ``date`` (str) and regime columns (str).
        Sorted by date ascending.
    """
    from datetime import datetime as _dt

    start = start_date or "2021-01-01"
    end = end_date or _dt.today().strftime("%Y-%m-%d")

    logger.info("Fetching expanded macro data (%s → %s) …", start, end)

    # Step 1: Download raw FRED series
    raw = scrape_macro_data(start_date=start, end_date=end, use_cache=use_cache)

    if raw.empty:
        logger.warning("No macro data fetched – returning empty DataFrame")
        return raw

    # Step 2: Download cross-asset data (BTC, copper) – new regime indicators
    try:
        cross_asset = scrape_cross_asset_data(
            start_date=start, end_date=end, use_cache=use_cache,
        )
        if not cross_asset.empty and "date" in cross_asset.columns:
            raw = raw.merge(cross_asset, on="date", how="left")
            logger.info("Cross-asset data merged: +%d columns", cross_asset.shape[1] - 1)
    except Exception as e:
        logger.warning("Cross-asset data fetch failed (non-fatal): %s", e)

    # Step 3: Download ETF fund flow proxies (SPY, QQQ, IWM, HYG, TLT)
    try:
        etf_flows = scrape_etf_fund_flows(
            start_date=start, end_date=end, use_cache=use_cache,
        )
        if not etf_flows.empty and "date" in etf_flows.columns:
            raw = raw.merge(etf_flows, on="date", how="left")
            logger.info("ETF fund flows merged: +%d columns", etf_flows.shape[1] - 1)
    except Exception as e:
        logger.warning("ETF fund flows fetch failed (non-fatal): %s", e)

    # Step 4: Engineer features
    featured = build_macro_features(raw)

    logger.info(
        "Macro pipeline complete: %d rows × %d columns",
        len(featured), featured.shape[1],
    )
    return featured
