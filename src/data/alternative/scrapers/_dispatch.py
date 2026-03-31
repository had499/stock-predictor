"""
Unified alternative-data dispatcher.

scrape_alternative_data() calls all individual per-ticker scrapers and
returns a dict mapping source name → DataFrame.  It is the main entry
point used by the pipeline.

Exports: scrape_alternative_data
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd

from .insider import scrape_insider_trades
from .macro import scrape_fred_macro
from .ticker import (
    scrape_yahoo_earnings,
    scrape_congress_trades,
    scrape_wikipedia_pageviews,
    scrape_earnings_revisions,
)
from .short_interest import scrape_short_interest
from .price_targets import scrape_price_targets

logger = logging.getLogger(__name__)


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
        "short_interest", "price_targets",
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
        "short_interest": lambda: scrape_short_interest(ticker, start_date, end_date, use_cache),
        "price_targets": lambda: scrape_price_targets(ticker, start_date, end_date, use_cache),
    }

    def _run(src):
        fn = scraper_map.get(src)
        if fn is None:
            logger.warning("Unknown alternative source: %s", src)
            return src, None
        try:
            df = fn()
            if df is not None and not df.empty:
                logger.info("  ✓ %s → %d rows for %s", src, len(df), ticker)
                return src, df
            else:
                logger.info("  ○ %s → empty for %s", src, ticker)
                return src, None
        except Exception as e:
            logger.warning("  ✗ %s failed for %s: %s", src, ticker, e)
            return src, None

    with ThreadPoolExecutor(max_workers=len(active)) as pool:
        for src, df in pool.map(_run, active):
            if df is not None:
                results[src] = df

    return results
