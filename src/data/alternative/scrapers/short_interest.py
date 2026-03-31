"""
FINRA consolidated short interest scraper.

Downloads bi-monthly short interest archive files from FINRA's public
catalog (no API key required) and extracts per-ticker data.

Coverage:
  - Exchange-listed stocks (NYSE/NASDAQ): June 2021 – present
  - Bi-monthly settlement dates (~24 snapshots/year)

Columns returned:
  date, ticker, shares_short, shares_short_prior,
  short_change_pct, avg_daily_volume, days_to_cover

Exports: scrape_short_interest
"""

import io
import logging
import re
import time
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .._cache import _load_incremental, _merge_and_save

logger = logging.getLogger(__name__)

_ARCHIVE_URL = "https://www.finra.org/finra-data/browse-catalog/equity-short-interest/files"
_HEADERS = {
    "User-Agent": "StockPredictor/1.0 (research@stockpredictor.dev)",
    "Accept": "text/html,application/xhtml+xml",
}

# Regex to extract a date from a FINRA file name, e.g. "2023-06-15" or "20230615"
_DATE_RE = re.compile(r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})")


def _parse_file_date(url_or_name: str) -> Optional[str]:
    """Extract YYYY-MM-DD from a FINRA archive file URL or name."""
    m = _DATE_RE.search(url_or_name)
    if not m:
        return None
    try:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    except Exception:
        return None


def _discover_archive_files(start: str, end: str) -> List[Tuple[str, str]]:
    """
    Scrape the FINRA archive page and return (settlement_date, file_url) pairs
    that fall within [start, end].
    """
    try:
        resp = requests.get(_ARCHIVE_URL, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("FINRA archive page fetch failed: %s", e)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # FINRA archive links are CSVs or pipe-delimited text files
        if not any(ext in href.lower() for ext in (".csv", ".txt", ".zip")):
            continue
        # Make absolute
        if href.startswith("/"):
            href = "https://www.finra.org" + href
        elif not href.startswith("http"):
            continue

        date_str = _parse_file_date(href)
        if date_str is None:
            # Try the link text
            date_str = _parse_file_date(a.get_text())
        if date_str is None:
            continue
        if date_str < start or date_str > end:
            continue

        results.append((date_str, href))

    logger.info("FINRA archive: found %d files in range [%s, %s]", len(results), start, end)
    return sorted(results)


def _parse_finra_file(content: bytes, ticker: str) -> Optional[pd.Series]:
    """
    Parse a FINRA short interest file and return the row for `ticker`, or None.

    FINRA files are pipe-delimited. Column names vary slightly across vintages
    so we normalise by position/keyword matching.
    """
    try:
        text = content.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text), sep="|", dtype=str, low_memory=False)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Find the symbol column
        sym_col = next(
            (c for c in df.columns if "symbol" in c or c in ("ticker", "issuesymbol")),
            None,
        )
        if sym_col is None:
            return None

        df[sym_col] = df[sym_col].str.strip().str.upper()
        row = df[df[sym_col] == ticker.upper()]
        if row.empty:
            return None

        row = row.iloc[0]

        def _num(col_keywords):
            for kw in col_keywords:
                for col in df.columns:
                    if kw in col:
                        try:
                            return float(str(row[col]).replace(",", ""))
                        except Exception:
                            pass
            return None

        return {
            "shares_short": _num(["currentshort", "current_short", "short_interest"]),
            "shares_short_prior": _num(["previousshort", "previous_short", "prior_short"]),
            "short_change_pct": _num(["percentchange", "percent_change", "%_change", "pct_change"]),
            "avg_daily_volume": _num(["averagedaily", "average_daily", "avg_daily"]),
            "days_to_cover": _num(["daystocover", "days_to_cover", "daystocov"]),
        }
    except Exception as e:
        logger.debug("FINRA file parse error: %s", e)
        return None


def scrape_short_interest(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch FINRA consolidated short interest for a ticker.

    Downloads bi-monthly archive files from FINRA's public catalog — no API
    key required. Exchange-listed data is available from June 2021 onward.

    Parameters
    ----------
    ticker : str
    start_date, end_date : str (ISO format)
    use_cache : bool

    Returns
    -------
    DataFrame with columns:
        date, ticker, shares_short, shares_short_prior,
        short_change_pct, avg_daily_volume, days_to_cover
    """
    start = start_date or "2021-06-01"  # exchange data starts June 2021
    end = end_date or datetime.today().strftime("%Y-%m-%d")

    existing_df, fetch_from = (
        _load_incremental("short_interest", ticker, start, end, tolerance_days=10)
        if use_cache else (None, start)
    )
    if fetch_from > end:
        return existing_df

    files = _discover_archive_files(fetch_from, end)
    if not files:
        logger.warning("FINRA: no archive files found for [%s, %s]", fetch_from, end)
        return existing_df if existing_df is not None else _empty_short_interest_df(ticker)

    rows = []
    for settlement_date, url in files:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            resp.raise_for_status()
            parsed = _parse_finra_file(resp.content, ticker)
            if parsed:
                parsed["date"] = settlement_date
                parsed["ticker"] = ticker.upper()
                rows.append(parsed)
            time.sleep(0.3)
        except Exception as e:
            logger.warning("FINRA file download failed (%s): %s", settlement_date, e)

    if not rows:
        logger.info("FINRA: no short interest data found for %s", ticker)
        return existing_df if existing_df is not None else _empty_short_interest_df(ticker)

    df = pd.DataFrame(rows)
    df = df[["date", "ticker", "shares_short", "shares_short_prior",
              "short_change_pct", "avg_daily_volume", "days_to_cover"]]
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("FINRA short interest: %d snapshots for %s", len(df), ticker)

    if use_cache:
        return _merge_and_save(df, existing_df, "short_interest", ticker, start, end)
    return df


def _empty_short_interest_df(ticker: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="str"),
        "ticker": pd.Series(dtype="str"),
        "shares_short": pd.Series(dtype="float"),
        "shares_short_prior": pd.Series(dtype="float"),
        "short_change_pct": pd.Series(dtype="float"),
        "avg_daily_volume": pd.Series(dtype="float"),
        "days_to_cover": pd.Series(dtype="float"),
    })
