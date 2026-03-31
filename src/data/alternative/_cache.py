"""
Cache utilities for the alternative data module.

Provides disk-based parquet caching for per-ticker and global (ticker-agnostic)
scraped datasets.  All scraper sub-modules import from here.

Exports: CACHE_DIR, _cache_key, _load_cache, _save_cache,
         _load_cache_global, _save_cache_global, _empty_insider_df,
         _load_incremental, _merge_and_save,
         _load_incremental_global, _merge_and_save_global
"""

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Root cache directory (relative to this file: src/data/alternative/ → datasets/alternative/)
CACHE_DIR = Path(__file__).resolve().parents[3] / "datasets" / "alternative"


# ============================================================================
# Key helpers
# ============================================================================

def _cache_key(source: str, ticker: str, start: str, end: str) -> str:
    raw = f"{source}_{ticker}_{start}_{end}"
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{source}_{ticker}_{start}_{end}_{h}.parquet"


# ============================================================================
# Per-ticker cache
# ============================================================================

def _load_cache(source: str, ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    # Prefer an exact match for reproducibility
    fp = CACHE_DIR / _cache_key(source, ticker, start, end)
    if fp.exists():
        logger.info("Cache hit (exact): %s", fp.name)
        return pd.read_parquet(fp)

    # If no exact match, attempt to find any cached files for this
    # (source, ticker) pair and combine them. This avoids redundant
    # re-downloads when overlapping date-range cache files already exist.
    pattern = f"{source}_{ticker}_*.parquet"
    matches = sorted(CACHE_DIR.glob(pattern))
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = CACHE_DIR / _cache_key(source, ticker, start, end)
    df.to_parquet(fp, index=False)
    logger.info("Cached %d rows → %s", len(df), fp.name)


# ============================================================================
# Global (ticker-agnostic) cache — used for FRED macro and cross-asset data
# ============================================================================

def _load_cache_global(source: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Cache for non-ticker-specific data (macro)."""
    fp = CACHE_DIR / _cache_key(source, "GLOBAL", start, end)
    if fp.exists():
        logger.info("Cache hit (exact): %s", fp.name)
        return pd.read_parquet(fp)

    pattern = f"{source}_GLOBAL_*.parquet"
    matches = sorted(CACHE_DIR.glob(pattern))
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = CACHE_DIR / _cache_key(source, "GLOBAL", start, end)
    df.to_parquet(fp, index=False)
    logger.info("Cached %d rows → %s", len(df), fp.name)


# ============================================================================
# Incremental cache helpers
# ============================================================================

def _load_incremental(
    source: str, ticker: str, start: str, end: str,
    tolerance_days: int = 7,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load any existing cached data for (source, ticker) and return the date
    from which new data still needs to be fetched.

    Returns
    -------
    (existing_df, fetch_from)
        existing_df : all cached rows in [start, end], or None
        fetch_from  : ISO date string — scrape this date through `end`
                      equals `start` when there is no usable cache,
                      equals day-after max_cached_date otherwise
    """
    pattern = f"{source}_{ticker}_*.parquet"
    matches = sorted(CACHE_DIR.glob(pattern))
    if not matches:
        return None, start

    dfs = []
    for m in matches:
        try:
            dfs.append(pd.read_parquet(m))
        except Exception:
            continue

    if not dfs:
        return None, start

    combined = pd.concat(dfs, ignore_index=True)
    if "date" not in combined.columns or combined.empty:
        return None, start

    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    in_range = combined[(combined["date"] >= start) & (combined["date"] <= end)].drop_duplicates()

    if in_range.empty:
        return None, start

    max_cached = in_range["date"].max()
    if max_cached >= end:
        logger.info("Incremental cache: %s_%s fully covered up to %s", source, ticker, end)
        return in_range, end  # nothing new to fetch

    end_dt = datetime.strptime(end, "%Y-%m-%d")
    max_dt = datetime.strptime(max_cached, "%Y-%m-%d")
    if (end_dt - max_dt).days <= tolerance_days:
        logger.info("Incremental cache: %s_%s within tolerance (%d days gap), skipping fetch",
                    source, ticker, (end_dt - max_dt).days)
        return in_range, end

    fetch_from = (max_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info("Incremental cache: %s_%s cached to %s → fetching %s:%s",
                source, ticker, max_cached, fetch_from, end)
    return in_range, fetch_from


def _merge_and_save(
    new_df: pd.DataFrame,
    existing_df: Optional[pd.DataFrame],
    source: str, ticker: str, start: str, end: str,
) -> pd.DataFrame:
    """
    Merge new_df with existing cached data, delete stale cache files,
    write a single combined file, and return the combined DataFrame.
    """
    parts = [df for df in (existing_df, new_df) if df is not None and not df.empty]
    if not parts:
        return new_df if new_df is not None else pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
        combined = combined.sort_values("date")
    combined = combined.drop_duplicates().reset_index(drop=True)

    # Remove stale cache files before writing the merged one
    for old in CACHE_DIR.glob(f"{source}_{ticker}_*.parquet"):
        try:
            old.unlink()
        except Exception:
            pass

    _save_cache(combined, source, ticker, start, end)
    return combined


def _load_incremental_global(
    source: str, start: str, end: str
) -> Tuple[Optional[pd.DataFrame], str]:
    """Incremental load for ticker-agnostic (GLOBAL) sources."""
    pattern = f"{source}_GLOBAL_*.parquet"
    matches = sorted(CACHE_DIR.glob(pattern))
    if not matches:
        return None, start

    dfs = []
    for m in matches:
        try:
            dfs.append(pd.read_parquet(m))
        except Exception:
            continue

    if not dfs:
        return None, start

    combined = pd.concat(dfs, ignore_index=True)
    if "date" not in combined.columns or combined.empty:
        return None, start

    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    in_range = combined[(combined["date"] >= start) & (combined["date"] <= end)].drop_duplicates()

    if in_range.empty:
        return None, start

    max_cached = in_range["date"].max()
    if max_cached >= end:
        return in_range, end

    fetch_from = (datetime.strptime(max_cached, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info("Incremental cache: %s_GLOBAL cached to %s → fetching %s:%s",
                source, max_cached, fetch_from, end)
    return in_range, fetch_from


def _merge_and_save_global(
    new_df: pd.DataFrame,
    existing_df: Optional[pd.DataFrame],
    source: str, start: str, end: str,
) -> pd.DataFrame:
    """Merge + save for ticker-agnostic sources."""
    parts = [df for df in (existing_df, new_df) if df is not None and not df.empty]
    if not parts:
        return new_df if new_df is not None else pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
        combined = combined.sort_values("date")
    combined = combined.drop_duplicates().reset_index(drop=True)

    for old in CACHE_DIR.glob(f"{source}_GLOBAL_*.parquet"):
        try:
            old.unlink()
        except Exception:
            pass

    _save_cache_global(combined, source, start, end)
    return combined


# ============================================================================
# Shared empty-DataFrame factories
# ============================================================================

def _empty_insider_df(ticker: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="str"),
        "ticker": pd.Series(dtype="str"),
        "insider_buys": pd.Series(dtype="int"),
        "insider_sells": pd.Series(dtype="int"),
        "insider_net_transactions": pd.Series(dtype="int"),
        "insider_buy_value": pd.Series(dtype="float"),
        "insider_sell_value": pd.Series(dtype="float"),
        "insider_net_value": pd.Series(dtype="float"),
        "insider_transaction_count": pd.Series(dtype="int"),
        "insider_buy_ratio": pd.Series(dtype="float"),
        "officer_buys": pd.Series(dtype="int"),
        "officer_sells": pd.Series(dtype="int"),
        "grant_count": pd.Series(dtype="int"),
        "grant_shares": pd.Series(dtype="float"),
        "tax_withheld_count": pd.Series(dtype="int"),
        "tax_withheld_shares": pd.Series(dtype="float"),
        "option_exercises": pd.Series(dtype="int"),
        "gift_count": pd.Series(dtype="int"),
    })
