"""
Alternative Data Scrapers

Free, no-API-key scrapers for alternative datasets that provide
predictive signals beyond price/volume:

1. **SEC EDGAR** – Insider transactions (Form 4) & institutional
   holdings (13F filings).
2. **FRED** – Federal Reserve macro indicators (rates, inflation,
   unemployment, yield curve, VIX).
3. **Yahoo Finance** – Analyst ratings, short interest, earnings
   calendar/surprises, options put-call ratio.
4. **Capitol Trades** – US Congress member stock trades.
5. **Wikipedia Pageviews** – Retail attention proxy.

All scrapers return a DataFrame with at least (date, ticker) columns
and are cached to disk (parquet) for efficiency.

Author: Stock Predictor Project
"""

import pandas as pd
import numpy as np
import requests
import hashlib
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# SEC requires a contact email in the User-Agent
_SEC_HEADERS = {
    "User-Agent": "StockPredictor research@stockpredictor.dev",
    "Accept-Encoding": "gzip, deflate",
}

# Cache directory
_CACHE_DIR = Path(__file__).resolve().parents[3] / "datasets" / "alternative"


# ============================================================================
# Cache helpers (same pattern as news module)
# ============================================================================

def _cache_key(source: str, ticker: str, start: str, end: str) -> str:
    raw = f"{source}_{ticker}_{start}_{end}"
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{source}_{ticker}_{start}_{end}_{h}.parquet"


def _load_cache(source: str, ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    # Prefer an exact match for reproducibility
    fp = _CACHE_DIR / _cache_key(source, ticker, start, end)
    if fp.exists():
        logger.info("Cache hit (exact): %s", fp.name)
        return pd.read_parquet(fp)

    # If no exact match, attempt to find any cached files for this
    # (source, ticker) pair and combine them. This avoids redundant
    # re-downloads when overlapping date-range cache files already exist.
    pattern = f"{source}_{ticker}_*.parquet"
    matches = sorted(_CACHE_DIR.glob(pattern))
    if not matches:
        return None

    dfs = []
    for m in matches:
        try:
            dfs.append(pd.read_parquet(m))
        except Exception:
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    # If there's a 'date' column, filter to requested range and drop duplicates
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
        combined = combined[(combined["date"] >= start) & (combined["date"] <= end)]

    if combined.empty:
        return None

    # Deduplicate conservative by keeping last occurrence
    combined = combined.drop_duplicates()
    logger.info("Cache hit (merged %d files) for %s_%s", len(matches), source, ticker)
    return combined


def _save_cache(df: pd.DataFrame, source: str, ticker: str, start: str, end: str) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = _CACHE_DIR / _cache_key(source, ticker, start, end)
    df.to_parquet(fp, index=False)
    logger.info("Cached %d rows → %s", len(df), fp.name)


def _load_cache_global(source: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Cache for non-ticker-specific data (macro)."""
    fp = _CACHE_DIR / _cache_key(source, "GLOBAL", start, end)
    if fp.exists():
        logger.info("Cache hit (exact): %s", fp.name)
        return pd.read_parquet(fp)

    pattern = f"{source}_GLOBAL_*.parquet"
    matches = sorted(_CACHE_DIR.glob(pattern))
    if not matches:
        return None

    dfs = []
    for m in matches:
        try:
            dfs.append(pd.read_parquet(m))
        except Exception:
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
        combined = combined[(combined["date"] >= start) & (combined["date"] <= end)]

    if combined.empty:
        return None

    combined = combined.drop_duplicates()
    logger.info("Cache hit (merged %d files) for %s (GLOBAL)", len(matches), source)
    return combined


def _save_cache_global(df: pd.DataFrame, source: str, start: str, end: str) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = _CACHE_DIR / _cache_key(source, "GLOBAL", start, end)
    df.to_parquet(fp, index=False)
    logger.info("Cached %d rows → %s", len(df), fp.name)


# ============================================================================
# 1. SEC EDGAR – Insider Transactions (Form 4) via edgartools
# ============================================================================


def scrape_insider_trades(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
    max_filings: int = 500,
) -> pd.DataFrame:
    """
    Fetch insider transactions from SEC EDGAR via the edgartools library.

    Parses Form 4 filings and returns daily aggregated insider activity:
        date, ticker, insider_buys, insider_sells, insider_net_transactions,
        insider_buy_ratio, insider_transaction_count, insider_buy_value,
        insider_sell_value, insider_net_value, officer_buys, officer_sells

    Parameters
    ----------
    ticker : str
    start_date, end_date : str (ISO format)
    use_cache : bool
    max_filings : int
        Cap on filings to parse (avoids very long fetches for active tickers).
    """
    try:
        from edgar import get_filings as _edgar_get_filings
    except ImportError as exc:
        raise ImportError("edgartools is required: pip install edgartools") from exc

    start = start_date or "2021-01-01"
    end = end_date or datetime.today().strftime("%Y-%m-%d")

    if use_cache:
        cached = _load_cache("insider_edgar", ticker, start, end)
        if cached is not None:
            return cached

    all_transactions: List[Dict] = []

    try:
        filings = _edgar_get_filings(form="4", ticker=ticker, start_date=start, end_date=end)
    except Exception as e:
        logger.warning("edgartools Form 4 fetch failed for %s: %s", ticker, e)
        return _empty_insider_df(ticker)

    count = 0
    for filing in filings:
        if count >= max_filings:
            break
        try:
            ownership = filing.obj()
            filing_date = str(filing.filing_date)[:10]
            rel = str(getattr(ownership, "reporting_owner_relationship", "") or "")
            rel_lower = rel.lower()
            is_officer = int("officer" in rel_lower)
            is_director = int("director" in rel_lower)
            reporter_name = str(getattr(ownership, "reporting_owner_name", "") or "")

            for txn in (getattr(ownership, "transactions", None) or []):
                try:
                    tx_code = str(getattr(txn, "transaction_code", "") or "").strip()
                    if not tx_code:
                        continue
                    tx_date = getattr(txn, "transaction_date", None)
                    tx_date = str(tx_date)[:10] if tx_date else filing_date
                    if tx_date < start or tx_date > end:
                        continue
                    shares = float(getattr(txn, "shares", 0) or 0)
                    price = float(getattr(txn, "price_per_share", 0) or 0)
                    all_transactions.append({
                        "date": tx_date,
                        "ticker": ticker.upper(),
                        "insider_name": reporter_name,
                        "is_officer": is_officer,
                        "is_director": is_director,
                        "transaction_code": tx_code,
                        "shares": shares,
                        "price_per_share": price,
                        "total_value": shares * price,
                    })
                except Exception:
                    continue

            count += 1
        except Exception as e:
            logger.debug("Form 4 parse error for %s: %s", ticker, e)
            continue

    logger.info("%s: parsed %d filings → %d transactions", ticker, count, len(all_transactions))

    if not all_transactions:
        df = _empty_insider_df(ticker)
    else:
        raw = pd.DataFrame(all_transactions)
        raw["is_buy"] = (raw["transaction_code"] == "P").astype(int)
        raw["is_sell"] = (raw["transaction_code"] == "S").astype(int)
        raw["is_officer_buy"] = (
            (raw["transaction_code"] == "P") & (raw["is_officer"] == 1)
        ).astype(int)
        raw["is_officer_sell"] = (
            (raw["transaction_code"] == "S") & (raw["is_officer"] == 1)
        ).astype(int)
        raw["buy_value"] = raw["total_value"] * raw["is_buy"]
        raw["sell_value"] = raw["total_value"] * raw["is_sell"]

        daily = raw.groupby("date").agg(
            insider_buys=("is_buy", "sum"),
            insider_sells=("is_sell", "sum"),
            insider_transaction_count=("transaction_code", "count"),
            insider_buy_value=("buy_value", "sum"),
            insider_sell_value=("sell_value", "sum"),
            officer_buys=("is_officer_buy", "sum"),
            officer_sells=("is_officer_sell", "sum"),
        ).reset_index()

        daily["ticker"] = ticker.upper()
        daily["insider_net_transactions"] = daily["insider_buys"] - daily["insider_sells"]
        daily["insider_buy_ratio"] = (
            daily["insider_buys"] / daily["insider_transaction_count"].clip(lower=1)
        )
        daily["insider_net_value"] = daily["insider_buy_value"] - daily["insider_sell_value"]

        df = daily[["date", "ticker", "insider_buys", "insider_sells",
                     "insider_net_transactions", "insider_buy_ratio",
                     "insider_transaction_count", "insider_buy_value",
                     "insider_sell_value", "insider_net_value",
                     "officer_buys", "officer_sells"]]

    if use_cache:
        _save_cache(df, "insider_edgar", ticker, start, end)
    return df




def _empty_insider_df(ticker: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="str"),
        "ticker": pd.Series(dtype="str"),
        "insider_buys": pd.Series(dtype="int"),
        "insider_sells": pd.Series(dtype="int"),
        "insider_net_transactions": pd.Series(dtype="int"),
        "insider_buy_ratio": pd.Series(dtype="float"),
        "insider_transaction_count": pd.Series(dtype="int"),
        "insider_buy_value": pd.Series(dtype="float"),
        "insider_sell_value": pd.Series(dtype="float"),
        "insider_net_value": pd.Series(dtype="float"),
        "officer_buys": pd.Series(dtype="int"),
        "officer_sells": pd.Series(dtype="int"),
    })




# ============================================================================
# 3. FRED – Macro Economic Indicators (no API key needed for CSV)
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
# 4. Yahoo Finance – Short Interest
# ============================================================================


def scrape_yahoo_earnings(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Scrape historical earnings data from Yahoo Finance.

    Returns:
        date, ticker, earnings_estimate, earnings_actual,
        earnings_surprise, earnings_surprise_pct
    """
    start = start_date or "2021-01-01"
    end = end_date or datetime.today().strftime("%Y-%m-%d")

    if use_cache:
        cached = _load_cache("earnings", ticker, start, end)
        if cached is not None:
            return cached

    rows = []
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        # Earnings dates with actual vs estimate
        earnings = t.earnings_dates
        if earnings is not None and not earnings.empty:
            for idx, row_data in earnings.iterrows():
                try:
                    edate = pd.to_datetime(idx).strftime("%Y-%m-%d")
                except Exception:
                    continue

                if edate < start or edate > end:
                    continue

                estimate = row_data.get("EPS Estimate", np.nan)
                actual = row_data.get("Reported EPS", np.nan)

                estimate = pd.to_numeric(estimate, errors="coerce")
                actual = pd.to_numeric(actual, errors="coerce")

                surprise = actual - estimate if pd.notna(actual) and pd.notna(estimate) else np.nan
                surprise_pct = (
                    (surprise / abs(estimate) * 100) if pd.notna(surprise) and estimate != 0 else np.nan
                )

                rows.append({
                    "date": edate,
                    "ticker": ticker.upper(),
                    "earnings_estimate": estimate,
                    "earnings_actual": actual,
                    "earnings_surprise": surprise,
                    "earnings_surprise_pct": surprise_pct,
                    "scheduled": False,
                })

    except Exception as e:
        logger.warning("Earnings scrape failed for %s: %s", ticker, e)

    df = pd.DataFrame(rows) if rows else _empty_earnings_df(ticker)

    # Also attempt to include the next scheduled earnings date (if any)
    try:
        # Get the full earnings table and find the next date after `end`
        full_earnings = t.earnings_dates
        if full_earnings is not None and not full_earnings.empty:
            # Parse all dates
            all_dates = []
            for idx, _ in full_earnings.iterrows():
                try:
                    d = pd.to_datetime(idx).strftime("%Y-%m-%d")
                    all_dates.append(d)
                except Exception:
                    continue

            # Find the nearest future date after `end`
            future_dates = [d for d in all_dates if d > end]
            if future_dates:
                next_date = sorted(future_dates)[0]
                # Only add if not already present in df
                if next_date not in df["date"].values:
                    # Try to extract estimate if available in the table
                    try:
                        est_row = full_earnings.loc[pd.to_datetime(next_date)]
                        estimate = pd.to_numeric(est_row.get("EPS Estimate", np.nan), errors="coerce")
                    except Exception:
                        estimate = np.nan

                    new_row = {
                        "date": next_date,
                        "ticker": ticker.upper(),
                        "earnings_estimate": estimate,
                        "earnings_actual": np.nan,
                        "earnings_surprise": np.nan,
                        "earnings_surprise_pct": np.nan,
                        "scheduled": True,
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    except Exception:
        pass

    if use_cache and not df.empty:
        _save_cache(df, "earnings", ticker, start, end)
    return df


def _empty_earnings_df(ticker: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="str"),
        "ticker": pd.Series(dtype="str"),
        "earnings_estimate": pd.Series(dtype="float"),
        "earnings_actual": pd.Series(dtype="float"),
        "earnings_surprise": pd.Series(dtype="float"),
        "earnings_surprise_pct": pd.Series(dtype="float"),
        "scheduled": pd.Series(dtype="bool"),
    })


# ============================================================================
# 5. Capitol Trades – US Congress Stock Trades
# ============================================================================

def scrape_congress_trades(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Scrape US Congress stock trades from capitoltrades.com.

    Congress members are required to disclose stock transactions,
    and studies show their trades outperform the market.

    Returns:
        date, ticker, congress_buys, congress_sells,
        congress_net, congress_trade_count
    """
    start = start_date or "2021-01-01"
    end = end_date or datetime.today().strftime("%Y-%m-%d")

    if use_cache:
        cached = _load_cache("congress", ticker, start, end)
        if cached is not None:
            return cached

    rows = []
    base_url = "https://www.capitoltrades.com/trades"
    page = 1
    max_pages = 5

    while page <= max_pages:
        try:
            params = {
                "assetType": "stock",
                "ticker": ticker.upper(),
                "page": page,
            }
            resp = requests.get(base_url, params=params, headers=_HEADERS, timeout=15)
            if resp.status_code != 200:
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if not table:
                break

            tbody = table.find("tbody")
            if not tbody:
                break

            row_els = tbody.find_all("tr")
            if not row_els:
                break

            for tr in row_els:
                cells = tr.find_all("td")
                if len(cells) < 7:
                    continue

                trade_date = cells[3].get_text(strip=True)
                trade_type = cells[6].get_text(strip=True).lower()

                # Parse date
                try:
                    parsed_date = pd.to_datetime(trade_date).strftime("%Y-%m-%d")
                except Exception:
                    continue

                if parsed_date < start or parsed_date > end:
                    continue

                is_buy = 1 if "purchase" in trade_type or "buy" in trade_type else 0
                is_sell = 1 if "sale" in trade_type or "sell" in trade_type else 0

                rows.append({
                    "date": parsed_date,
                    "ticker": ticker.upper(),
                    "congress_buy": is_buy,
                    "congress_sell": is_sell,
                })

            # Check for next page
            next_link = soup.find("a", {"aria-label": "Next"})
            if not next_link:
                break
            page += 1
            time.sleep(1.0)

        except Exception as e:
            logger.warning("Congress trades page %d failed for %s: %s", page, ticker, e)
            break

    if not rows:
        df = _empty_congress_df(ticker)
    else:
        raw = pd.DataFrame(rows)
        daily = raw.groupby("date").agg(
            congress_buys=("congress_buy", "sum"),
            congress_sells=("congress_sell", "sum"),
            congress_trade_count=("congress_buy", "count"),
        ).reset_index()
        daily["ticker"] = ticker.upper()
        daily["congress_net"] = daily["congress_buys"] - daily["congress_sells"]
        df = daily[["date", "ticker", "congress_buys", "congress_sells",
                     "congress_net", "congress_trade_count"]]

    if use_cache:
        _save_cache(df, "congress", ticker, start, end)
    return df


def _empty_congress_df(ticker: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="str"),
        "ticker": pd.Series(dtype="str"),
        "congress_buys": pd.Series(dtype="int"),
        "congress_sells": pd.Series(dtype="int"),
        "congress_net": pd.Series(dtype="int"),
        "congress_trade_count": pd.Series(dtype="int"),
    })


# ============================================================================
# 6. Wikipedia Pageviews – Retail Attention Proxy
# ============================================================================

# Map tickers to Wikipedia article titles
_TICKER_TO_WIKI = {
    # Tech
    "AAPL": "Apple_Inc.", "MSFT": "Microsoft", "NVDA": "Nvidia",
    "GOOGL": "Alphabet_Inc.", "GOOG": "Alphabet_Inc.",
    "AMZN": "Amazon_(company)", "META": "Meta_Platforms",
    "TSLA": "Tesla,_Inc.", "ORCL": "Oracle_Corporation",
    "CRM": "Salesforce", "ADBE": "Adobe_Inc.",
    "INTC": "Intel", "AMD": "Advanced_Micro_Devices",
    "QCOM": "Qualcomm", "AVGO": "Broadcom_Inc.",
    "IBM": "IBM", "TXN": "Texas_Instruments",
    "NOW": "ServiceNow",
    # Finance
    "JPM": "JPMorgan_Chase", "BAC": "Bank_of_America",
    "WFC": "Wells_Fargo", "GS": "Goldman_Sachs",
    "C": "Citigroup", "MS": "Morgan_Stanley",
    "AXP": "American_Express", "BLK": "BlackRock",
    "SCHW": "Charles_Schwab_Corporation",
    "V": "Visa_Inc.", "MA": "Mastercard",
    "PYPL": "PayPal",
    # Healthcare
    "JNJ": "Johnson_%26_Johnson", "PFE": "Pfizer",
    "MRK": "Merck_%26_Co.", "UNH": "UnitedHealth_Group",
    "ABBV": "AbbVie", "LLY": "Eli_Lilly_and_Company",
    "BMY": "Bristol_Myers_Squibb", "AMGN": "Amgen",
    "GILD": "Gilead_Sciences",
    # Energy
    "XOM": "ExxonMobil", "CVX": "Chevron_Corporation",
    "COP": "ConocoPhillips", "SLB": "SLB_(company)",
    "SHEL": "Shell_plc", "EOG": "EOG_Resources",
    "PSX": "Phillips_66", "VLO": "Valero_Energy",
    # Consumer
    "WMT": "Walmart", "TGT": "Target_Corporation",
    "HD": "The_Home_Depot", "LOW": "Lowe%27s",
    "NKE": "Nike,_Inc.", "COST": "Costco",
    "SBUX": "Starbucks", "KO": "The_Coca-Cola_Company",
    "PEP": "PepsiCo", "MCD": "McDonald%27s",
    "PG": "Procter_%26_Gamble",
    # Media
    "DIS": "The_Walt_Disney_Company", "NFLX": "Netflix",
    "CMCSA": "Comcast",
    # Industrials
    "CAT": "Caterpillar_Inc.", "GE": "GE_Aerospace",
    "BA": "Boeing", "HON": "Honeywell",
    "LMT": "Lockheed_Martin", "RTX": "RTX_Corporation",
    "DE": "John_Deere", "UNP": "Union_Pacific_Corporation",
    "FDX": "FedEx", "UPS": "United_Parcel_Service",
    # Telecom
    "VZ": "Verizon_Communications", "T": "AT%26T",
    "TMUS": "T-Mobile_US",
    # Utilities
    "NEE": "NextEra_Energy", "DUK": "Duke_Energy",
    "SO": "Southern_Company",
    # Materials
    "LIN": "Linde_plc", "FCX": "Freeport-McMoRan",
    "NEM": "Newmont",
    # Real Estate
    "PLD": "Prologis", "AMT": "American_Tower",
    "SPG": "Simon_Property_Group", "O": "Realty_Income",
    # International
    "BABA": "Alibaba_Group", "TSM": "TSMC",
    "SHOP": "Shopify", "MELI": "MercadoLibre",
}


def scrape_wikipedia_pageviews(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily Wikipedia pageview counts as a retail attention proxy.

    Uses the Wikimedia REST API (free, no auth):
    https://wikimedia.org/api/rest_v1/

    Returns:
        date, ticker, wiki_pageviews, wiki_pageviews_ma7,
        wiki_attention_spike  (>2 std devs above 30-day mean)
    """
    start = start_date or "2021-01-01"
    end = end_date or datetime.today().strftime("%Y-%m-%d")

    if use_cache:
        cached = _load_cache("wiki", ticker, start, end)
        if cached is not None:
            return cached

    # Resolve article title
    article = _TICKER_TO_WIKI.get(ticker.upper())
    if not article:
        # Fallback: try the ticker as-is or common patterns
        article = f"{ticker.upper()}_(company)"

    start_ymd = start.replace("-", "")
    end_ymd = end.replace("-", "")

    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/all-agents/{article}/daily/{start_ymd}/{end_ymd}"
    )

    rows = []
    try:
        resp = requests.get(url, headers={
            "User-Agent": "StockPredictor/1.0 (research@stockpredictor.dev)",
        }, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("items", []):
                ts = item.get("timestamp", "")
                views = item.get("views", 0)
                try:
                    d = datetime.strptime(ts[:8], "%Y%m%d").strftime("%Y-%m-%d")
                except Exception:
                    continue
                rows.append({"date": d, "ticker": ticker.upper(), "wiki_pageviews": views})
        else:
            logger.warning("Wikipedia API returned %d for %s (%s)", resp.status_code, ticker, article)

    except Exception as e:
        logger.warning("Wikipedia pageviews failed for %s: %s", ticker, e)

    if not rows:
        df = _empty_wiki_df(ticker)
    else:
        df = pd.DataFrame(rows)
        df = df.sort_values("date").reset_index(drop=True)

        # Rolling features
        df["wiki_pageviews_ma7"] = df["wiki_pageviews"].rolling(7, min_periods=1).mean()
        df["wiki_pageviews_ma30"] = df["wiki_pageviews"].rolling(30, min_periods=1).mean()
        df["wiki_pageviews_std30"] = df["wiki_pageviews"].rolling(30, min_periods=1).std().fillna(0)
        df["wiki_attention_spike"] = (
            (df["wiki_pageviews"] > df["wiki_pageviews_ma30"] + 2 * df["wiki_pageviews_std30"])
        ).astype(int)
        # Drop intermediate columns
        df = df.drop(columns=["wiki_pageviews_ma30", "wiki_pageviews_std30"])

    if use_cache and not df.empty:
        _save_cache(df, "wiki", ticker, start, end)
    return df


def _empty_wiki_df(ticker: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="str"),
        "ticker": pd.Series(dtype="str"),
        "wiki_pageviews": pd.Series(dtype="int"),
        "wiki_pageviews_ma7": pd.Series(dtype="float"),
        "wiki_attention_spike": pd.Series(dtype="int"),
    })




# ============================================================================
# 8. Earnings Revisions – Analyst Upgrade/Downgrade History
# ============================================================================

def scrape_earnings_revisions(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Scrape analyst upgrade/downgrade history from Yahoo Finance.

    Returns daily-aggregated analyst rating changes:
        date, ticker, upgrades, downgrades, initiations,
        reiterations, total_revisions, net_revisions

    Uses a three-tier grade classification (bullish/neutral/bearish) to
    properly score direction from FromGrade → ToGrade, rather than relying
    solely on Yahoo's Action field.

    This is a **historical** time series (not a snapshot), making it one of
    the most valuable free alternative data sources for medium-term prediction.
    """
    start = start_date or "2021-01-01"
    end = end_date or datetime.today().strftime("%Y-%m-%d")

    if use_cache:
        cached = _load_cache("earnings_revisions", ticker, start, end)
        if cached is not None:
            return cached

    # ── Grade classification buckets ──
    # Different firms use different names for the same thing.
    # Map everything to a numeric score: +1 (bullish), 0 (neutral), -1 (bearish)
    _BULLISH = {
        "buy", "overweight", "outperform", "strong buy", "positive",
        "sector outperform", "top pick", "conviction buy", "accumulate",
        "above average", "add", "long-term buy", "market outperform",
        "strong outperform", "sector perform/outperform",
    }
    _BEARISH = {
        "sell", "underweight", "underperform", "reduce", "negative",
        "sector underperform", "strong sell", "avoid", "below average",
        "market underperform",
    }
    _NEUTRAL = {
        "hold", "neutral", "equal-weight", "market perform",
        "sector perform", "peer perform", "in-line", "perform",
        "sector weight", "market weight", "fair value",
    }

    def _grade_score(grade_str: str) -> Optional[int]:
        """Map a grade string to +1 (bullish), 0 (neutral), -1 (bearish)."""
        g = grade_str.lower().strip()
        if not g:
            return None
        if g in _BULLISH:
            return 1
        if g in _BEARISH:
            return -1
        if g in _NEUTRAL:
            return 0
        # Fuzzy fallback: check if any keyword is contained in the string
        for term in _BULLISH:
            if term in g:
                return 1
        for term in _BEARISH:
            if term in g:
                return -1
        for term in _NEUTRAL:
            if term in g:
                return 0
        return None  # Truly unknown grade

    def _classify_revision(row) -> int:
        """
        Classify a single analyst action as +1 (upgrade), -1 (downgrade), 0 (neutral).

        Priority:
        1. Yahoo's Action field ("up"/"down") — most reliable when present
        2. FromGrade → ToGrade comparison — catches "Buy→Hold" as downgrade
        3. ToGrade level alone — for initiations with no FromGrade
        """
        action = str(row.get("Action", row.get("action", ""))).lower().strip()

        # Priority 1: Yahoo's explicit action field
        if "up" == action:
            return 1
        if "down" == action:
            return -1

        # Priority 2: Compare FromGrade → ToGrade
        to_grade = str(row.get("ToGrade", row.get("tograde", ""))).strip()
        from_grade = str(row.get("FromGrade", row.get("fromgrade", ""))).strip()

        to_score = _grade_score(to_grade)
        from_score = _grade_score(from_grade)

        if to_score is not None and from_score is not None and to_score != from_score:
            return 1 if to_score > from_score else -1

        # Priority 3: For initiations, use the ToGrade level directly
        if action in ("init", "initiated", "initiate"):
            if to_score is not None:
                return to_score  # +1, 0, or -1
            return 0

        # Priority 4: If only ToGrade is known (no FromGrade)
        if to_score is not None and from_score is None:
            return to_score

        return 0  # Can't classify — treat as neutral

    rows = []
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        # Get upgrade/downgrade history
        ud = t.upgrades_downgrades
        if ud is not None and not ud.empty:
            ud = ud.reset_index()

            # Handle the date column (may be 'Date', 'GradeDate', or index)
            date_col = None
            for col in ["Date", "GradeDate", "date"]:
                if col in ud.columns:
                    date_col = col
                    break
            if date_col is None:
                date_col = ud.columns[0]

            ud["_date"] = pd.to_datetime(ud[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
            ud = ud.dropna(subset=["_date"])
            ud = ud[(ud["_date"] >= start) & (ud["_date"] <= end)]

            if not ud.empty:
                # Classify each row using grade-aware logic
                ud["_direction"] = ud.apply(_classify_revision, axis=1)

                # Detect initiations from Action column
                action_col = None
                for col in ["Action", "action"]:
                    if col in ud.columns:
                        action_col = col
                        break

                for date_val, group in ud.groupby("_date"):
                    directions = group["_direction"]
                    row = {
                        "date": date_val,
                        "ticker": ticker.upper(),
                        "upgrades": int((directions == 1).sum()),
                        "downgrades": int((directions == -1).sum()),
                        "initiations": 0,
                        "reiterations": 0,
                        "total_revisions": len(group),
                    }
                    if action_col:
                        actions = group[action_col].str.lower().fillna("")
                        row["initiations"] = int(actions.str.contains("init", na=False).sum())
                        row["reiterations"] = int(actions.str.contains("reit|main", na=False).sum())
                    row["net_revisions"] = row["upgrades"] - row["downgrades"]
                    rows.append(row)

    except Exception as e:
        logger.warning("Earnings revisions scrape failed for %s: %s", ticker, e)

    if rows:
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    else:
        df = pd.DataFrame({
            "date": pd.Series(dtype="str"),
            "ticker": pd.Series(dtype="str"),
            "upgrades": pd.Series(dtype="int"),
            "downgrades": pd.Series(dtype="int"),
            "initiations": pd.Series(dtype="int"),
            "reiterations": pd.Series(dtype="int"),
            "total_revisions": pd.Series(dtype="int"),
            "net_revisions": pd.Series(dtype="int"),
        })

    if use_cache and not df.empty:
        _save_cache(df, "earnings_revisions", ticker, start, end)
    return df


# ============================================================================
# 9. Cross-Asset Data – BTC, Copper (Ticker-Agnostic)
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
# 10. ETF Fund Flows – Dollar Volume Proxy (Ticker-Agnostic)
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


# ============================================================================
# Combined scraper – all sources for one ticker
# ============================================================================

def scrape_alternative_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sources: Optional[List[str]] = None,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Scrape all alternative data sources for a single ticker.

    Parameters
    ----------
    ticker : str
    start_date, end_date : str
    sources : list of str or None
        Sources to scrape.  Options: 'insider', 'institutional',
        'earnings', 'analyst', 'short_interest', 'options',
        'congress', 'wiki', 'fred'.
        Defaults to all.
    use_cache : bool

    Returns
    -------
    dict[str, DataFrame]
        Mapping from source name to its DataFrame.
    """
    all_sources = [
        "insider", "earnings", "congress",
        "wiki", "fred", "earnings_revisions",
    ]
    active = sources or all_sources

    results: Dict[str, pd.DataFrame] = {}

    scraper_map = {
        "insider": lambda: scrape_insider_trades(ticker, start_date, end_date, use_cache),
        "earnings": lambda: scrape_yahoo_earnings(ticker, start_date, end_date, use_cache),
        "congress": lambda: scrape_congress_trades(ticker, start_date, end_date, use_cache),
        "wiki": lambda: scrape_wikipedia_pageviews(ticker, start_date, end_date, use_cache),
        "fred": lambda: scrape_fred_macro(start_date, end_date, use_cache=use_cache),
        "earnings_revisions": lambda: scrape_earnings_revisions(ticker, start_date, end_date, use_cache),
    }

    for src in active:
        fn = scraper_map.get(src)
        if fn is None:
            logger.warning("Unknown alternative source: %s", src)
            continue
        try:
            df = fn()
            if df is not None and not df.empty:
                results[src] = df
                logger.info("  ✓ %s → %d rows for %s", src, len(df), ticker)
            else:
                logger.info("  ○ %s → empty for %s", src, ticker)
        except Exception as e:
            logger.warning("  ✗ %s failed for %s: %s", src, ticker, e)

    return results
