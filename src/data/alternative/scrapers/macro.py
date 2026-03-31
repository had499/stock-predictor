"""
FRED macro-economic scrapers and cross-asset / ETF fund-flow scrapers.

All functions are ticker-agnostic (global / market-level data).

Exports: scrape_fred_macro, scrape_macro_data,
         scrape_cross_asset_data, scrape_etf_fund_flows
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

from .._cache import _load_cache_global, _save_cache_global

logger = logging.getLogger(__name__)


# ============================================================================
# FRED series definitions
# ============================================================================

# Key FRED series that are predictive for equity markets
_FRED_SERIES = {
    "DFF": "fed_funds_rate",              # Federal Funds Effective Rate
    "DGS10": "treasury_10y",              # 10-Year Treasury Yield
    "DGS2": "treasury_2y",               # 2-Year Treasury Yield
    "T10Y2Y": "yield_curve_10y2y",        # 10Y-2Y Spread (inversion signal)
    "T10Y3M": "yield_curve_10y3m",        # 10Y-3M Spread
    "VIXCLS": "vix",                      # CBOE Volatility Index
    "UNRATE": "unemployment_rate",        # Unemployment Rate (monthly)
    "CPIAUCSL": "cpi",                    # Consumer Price Index (monthly)
    "UMCSENT": "consumer_sentiment",      # U of Michigan Consumer Sentiment
    "DCOILWTICO": "oil_wti",             # WTI Crude Oil Price
    "DEXUSEU": "usd_eur",               # USD/EUR Exchange Rate
    "BAMLH0A0HYM2": "high_yield_spread", # ICE BofA High Yield Spread
    "ICSA": "initial_claims",            # Initial Jobless Claims (weekly)
}

# Expanded macro series – broader coverage for standalone macro features
_FRED_MACRO_EXPANDED = {
    # ── Interest rates & yield curve (full term structure) ──
    "DFF": "fed_funds_rate",
    "DGS1MO": "treasury_1m",
    "DGS3MO": "treasury_3m",
    "DGS6MO": "treasury_6m",
    "DGS1": "treasury_1y",
    "DGS2": "treasury_2y",
    "DGS5": "treasury_5y",
    "DGS7": "treasury_7y",
    "DGS10": "treasury_10y",
    "DGS20": "treasury_20y",
    "DGS30": "treasury_30y",
    "T10Y2Y": "yield_curve_10y2y",
    "T10Y3M": "yield_curve_10y3m",
    "T10YIE": "breakeven_inflation_10y",     # 10Y breakeven inflation
    "DFII10": "real_yield_10y",              # 10Y TIPS yield (real rate)

    # ── Volatility & risk ──
    "VIXCLS": "vix",
    "BAMLH0A0HYM2": "hy_spread",            # High-yield OAS spread
    "BAMLC0A4CBBB": "bbb_spread",           # BBB corporate spread
    "TEDRATE": "ted_spread",                 # TED spread (bank risk)

    # ── Labor market ──
    "UNRATE": "unemployment_rate",
    "ICSA": "initial_claims",
    "CCSA": "continued_claims",
    "PAYEMS": "nonfarm_payrolls",            # Total nonfarm payrolls
    "AWHAETP": "avg_weekly_hours",           # Avg weekly hours (leading)

    # ── Inflation & prices ──
    "CPIAUCSL": "cpi",
    "CPILFESL": "core_cpi",                 # Core CPI (ex food & energy)
    "PCEPI": "pce",                          # PCE price index (Fed's preferred)
    "PCEPILFE": "core_pce",                 # Core PCE
    "DCOILWTICO": "oil_wti",
    "DCOILBRENTEU": "oil_brent",            # Brent crude
    "GOLDAMGBD228NLBM": "gold_price",       # Gold London fixing
    "GASREGW": "gas_price",                  # Regular gas price (weekly)

    # ── Consumer & business sentiment ──
    "UMCSENT": "consumer_sentiment",
    "MICH": "inflation_expectations",        # U of Mich 1Y inflation expect.

    # ── Currency ──
    "DEXUSEU": "usd_eur",
    "DEXJPUS": "jpy_usd",                   # JPY/USD
    "DEXUSUK": "usd_gbp",                   # USD/GBP
    "DTWEXBGS": "usd_index_broad",          # Trade-weighted USD index (broad)

    # ── Money supply & financial conditions ──
    "M2SL": "m2_money_supply",              # M2 money supply
    "WALCL": "fed_balance_sheet",           # Fed total assets
    "NFCI": "nfci",                          # Chicago Fed National Financial Conditions Index
    "STLFSI4": "stress_index",               # St Louis Fed Financial Stress Index

    # ── Housing ──
    "MORTGAGE30US": "mortgage_30y",         # 30Y fixed mortgage rate
    "HOUST": "housing_starts",               # Housing starts (monthly)
    "PERMIT": "building_permits",            # Building permits (monthly)

    # ── Leading indicators ──
    "T10Y2Y": "yield_curve_10y2y",           # (already above, will dedup)
    "USSLIND": "leading_index",              # CB Leading Economic Index
    "RSXFS": "retail_sales_ex_food",         # Retail sales ex food services
    "INDPRO": "industrial_production",       # Industrial production index
    "DGORDER": "durable_goods_orders",       # Durable goods orders
    "ISRATIO": "inventory_sales_ratio",      # Total business inv/sales ratio
}


# ============================================================================
# FRED scrapers
# ============================================================================

def scrape_fred_macro(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    series: Optional[Dict[str, str]] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download macro-economic time series from FRED (CSV endpoint, no API key).

    Returns a date-indexed DataFrame with one column per indicator,
    forward-filled to daily frequency.

    Parameters
    ----------
    start_date, end_date : str (ISO format)
    series : dict mapping FRED series ID → column name.
             Defaults to a curated set of equity-relevant indicators.
    use_cache : bool
    """
    start = start_date or "2021-01-01"
    end = end_date or datetime.today().strftime("%Y-%m-%d")
    series_map = series or _FRED_SERIES

    if use_cache:
        cached = _load_cache_global("fred", start, end)
        if cached is not None:
            return cached

    frames = {}
    for fred_id, col_name in series_map.items():
        try:
            url = (
                f"https://fred.stlouisfed.org/graph/fredgraph.csv"
                f"?id={fred_id}&cosd={start}&coed={end}"
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()

            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), na_values=".")
            df.columns = ["date", col_name]
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            df = df.dropna(subset=[col_name])
            frames[col_name] = df.set_index("date")[col_name]

            logger.info("  FRED %s → %d observations", fred_id, len(df))
            time.sleep(0.3)

        except Exception as e:
            logger.warning("FRED %s failed: %s", fred_id, e)

    if not frames:
        return pd.DataFrame(columns=["date"])

    # Combine all series, forward-fill to daily
    combined = pd.DataFrame(frames)
    combined.index = pd.to_datetime(combined.index)

    # Create a full business-day index and forward-fill
    full_idx = pd.bdate_range(start=start, end=end)
    combined = combined.reindex(full_idx).ffill()
    combined.index.name = "date"
    combined = combined.reset_index()
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    # Derived features
    if "treasury_10y" in combined.columns and "treasury_2y" in combined.columns:
        combined["yield_curve_slope"] = combined["treasury_10y"] - combined["treasury_2y"]
        combined["yield_curve_inverted"] = (combined["yield_curve_slope"] < 0).astype(int)

    if "vix" in combined.columns:
        combined["vix_regime"] = pd.cut(
            combined["vix"], bins=[0, 15, 20, 30, 100],
            labels=["low_vol", "normal", "elevated", "crisis"],
        ).astype(str)

    if use_cache:
        _save_cache_global(combined, "fred", start, end)

    logger.info("FRED macro: %d rows × %d columns", *combined.shape)
    return combined


def scrape_macro_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download the **expanded** set of macro-economic time series from FRED.

    This is the standalone macro function – ticker-agnostic, covering
    ~55 series across rates, yield curve, volatility, labor, inflation,
    currency, sentiment, housing, and leading indicators.

    The result is a daily-frequency DataFrame (business days) with
    forward-filled values.

    Parameters
    ----------
    start_date, end_date : str (ISO format)
        Defaults to 2021-01-01 → today.
    use_cache : bool
        Cache the combined result to ``datasets/alternative/``.

    Returns
    -------
    pd.DataFrame
        Columns: date, <macro indicator columns …>
    """
    start = start_date or "2021-01-01"
    end = end_date or datetime.today().strftime("%Y-%m-%d")

    if use_cache:
        cached = _load_cache_global("macro_expanded", start, end)
        if cached is not None:
            return cached

    # De-duplicate series (some IDs appear twice in the dict)
    seen_ids: set = set()
    deduped: Dict[str, str] = {}
    for fred_id, col_name in _FRED_MACRO_EXPANDED.items():
        if fred_id not in seen_ids:
            deduped[fred_id] = col_name
            seen_ids.add(fred_id)

    total = len(deduped)
    frames: Dict[str, pd.Series] = {}
    failed: List[str] = []

    for i, (fred_id, col_name) in enumerate(deduped.items(), 1):
        try:
            url = (
                f"https://fred.stlouisfed.org/graph/fredgraph.csv"
                f"?id={fred_id}&cosd={start}&coed={end}"
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()

            from io import StringIO
            df_tmp = pd.read_csv(StringIO(resp.text), na_values=".")
            df_tmp.columns = ["date", col_name]
            df_tmp["date"] = pd.to_datetime(df_tmp["date"]).dt.strftime("%Y-%m-%d")
            df_tmp[col_name] = pd.to_numeric(df_tmp[col_name], errors="coerce")
            df_tmp = df_tmp.dropna(subset=[col_name])

            if not df_tmp.empty:
                frames[col_name] = df_tmp.set_index("date")[col_name]

            if i % 10 == 0 or i == total:
                logger.info("  FRED progress: %d/%d series downloaded", i, total)
            time.sleep(0.3)  # be polite to FRED

        except Exception as e:
            failed.append(fred_id)
            logger.warning("  FRED %s (%s) failed: %s", fred_id, col_name, e)

    if not frames:
        logger.warning("No FRED macro series downloaded")
        return pd.DataFrame(columns=["date"])

    # Combine all series, forward-fill to daily business days
    combined = pd.DataFrame(frames)
    combined.index = pd.to_datetime(combined.index)
    full_idx = pd.bdate_range(start=start, end=end)
    combined = combined.reindex(full_idx).ffill()
    combined.index.name = "date"
    combined = combined.reset_index()
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    if use_cache:
        _save_cache_global(combined, "macro_expanded", start, end)

    logger.info(
        "Expanded macro: %d rows × %d cols (%d series failed: %s)",
        *combined.shape, len(failed), ", ".join(failed) if failed else "none",
    )
    return combined


# ============================================================================
# Cross-asset data
# ============================================================================

def scrape_cross_asset_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download cross-asset prices (BTC, copper) via yfinance.

    These are ticker-agnostic regime indicators that complement FRED macro
    data with assets FRED doesn't cover:
      - **BTC** – risk appetite / speculative sentiment proxy
      - **Copper** – global industrial demand proxy ("Dr. Copper")

    Returns:
        date, btc_price, copper_price
    """
    start = start_date or "2021-01-01"
    end = end_date or datetime.today().strftime("%Y-%m-%d")

    if use_cache:
        cached = _load_cache_global("cross_asset", start, end)
        if cached is not None:
            return cached

    try:
        import yfinance as yf

        symbols = {
            "BTC-USD": "btc_price",
            "HG=F": "copper_price",
        }

        data = yf.download(
            list(symbols.keys()),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            return pd.DataFrame(columns=["date"])

        # Extract close prices
        frames = {}
        for sym, col_name in symbols.items():
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data["Close"][sym]
                else:
                    prices = data["Close"]
                frames[col_name] = prices
            except Exception:
                logger.warning("Cross-asset %s extraction failed", sym)

        if not frames:
            return pd.DataFrame(columns=["date"])

        combined = pd.DataFrame(frames)
        combined.index = pd.to_datetime(combined.index)

        # Fill to business days
        full_idx = pd.bdate_range(start=start, end=end)
        combined = combined.reindex(full_idx).ffill()
        combined.index.name = "date"
        combined = combined.reset_index()
        combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

        if use_cache:
            _save_cache_global(combined, "cross_asset", start, end)

        logger.info("Cross-asset data: %d rows × %d columns", *combined.shape)
        return combined

    except Exception as e:
        logger.warning("Cross-asset data fetch failed: %s", e)
        return pd.DataFrame(columns=["date"])


# ============================================================================
# ETF fund flows
# ============================================================================

def scrape_etf_fund_flows(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    etf_tickers: Optional[List[str]] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Estimate ETF fund flows from volume and price data.

    Uses **dollar volume** (volume × close) as a flow proxy and computes
    **abnormal volume** (z-score vs 20-day average) as the flow signal.

    Covers:
      - SPY, QQQ – broad equity appetite
      - IWM       – small-cap risk appetite
      - HYG       – credit / high-yield appetite
      - TLT       – flight-to-safety (long treasuries)

    Returns:
        date, <etf>_dollar_volume, <etf>_abnormal_volume,
        <etf>_flow_momentum, ...
    """
    start = start_date or "2021-01-01"
    end = end_date or datetime.today().strftime("%Y-%m-%d")

    tickers = etf_tickers or ["SPY", "QQQ", "IWM", "HYG", "TLT"]

    if use_cache:
        cached = _load_cache_global("etf_flows", start, end)
        if cached is not None:
            return cached

    try:
        import yfinance as yf

        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            return pd.DataFrame(columns=["date"])

        frames = {}
        for etf in tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    close = data["Close"][etf]
                    volume = data["Volume"][etf]
                else:
                    close = data["Close"]
                    volume = data["Volume"]

                etf_lower = etf.lower()

                # Dollar volume as flow proxy
                dollar_vol = close * volume
                frames[f"{etf_lower}_dollar_volume"] = dollar_vol

                # Abnormal volume (z-score vs 20-day avg)
                vol_ma20 = volume.rolling(20, min_periods=5).mean()
                vol_std20 = volume.rolling(20, min_periods=5).std().clip(lower=1)
                frames[f"{etf_lower}_abnormal_volume"] = (volume - vol_ma20) / vol_std20

                # Dollar volume momentum (5-day change)
                frames[f"{etf_lower}_flow_momentum"] = dollar_vol.pct_change(5)

            except Exception:
                logger.warning("ETF flow computation failed for %s", etf)

        if not frames:
            return pd.DataFrame(columns=["date"])

        combined = pd.DataFrame(frames)
        combined.index = pd.to_datetime(combined.index)

        full_idx = pd.bdate_range(start=start, end=end)
        combined = combined.reindex(full_idx).ffill()
        combined.index.name = "date"
        combined = combined.reset_index()
        combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

        if use_cache:
            _save_cache_global(combined, "etf_flows", start, end)

        logger.info("ETF fund flows: %d rows × %d columns", *combined.shape)
        return combined

    except Exception as e:
        logger.warning("ETF fund flows fetch failed: %s", e)
        return pd.DataFrame(columns=["date"])
