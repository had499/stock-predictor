"""
News Scraper Module

This module provides functionality to scrape daily financial news
for a given stock ticker over a specified time period by directly
scraping free sources (Google News RSS, Bing News, Finviz, Yahoo
Finance). No API keys required.

Author: Stock Predictor Project
"""

import pandas as pd
import numpy as np
import requests
import feedparser
import re
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common headers to avoid bot detection
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Default cache directory (relative to project root)
_CACHE_DIR = Path(__file__).resolve().parents[3] / "datasets" / "news"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(ticker: str, start_date: str, end_date: str, sources: str) -> str:
    """Deterministic filename for a scrape result."""
    raw = f"{ticker}_{start_date}_{end_date}_{sources}"
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{ticker}_{start_date}_{end_date}_{h}.parquet"


def _load_cache(
    ticker: str, start_date: str, end_date: str, sources: str,
    cache_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Return cached DataFrame if it exists, else None."""
    d = cache_dir or _CACHE_DIR
    fp = d / _cache_key(ticker, start_date, end_date, sources)
    if fp.exists():
        logger.info("Cache hit: %s", fp.name)
        return pd.read_parquet(fp)
    return None


def _save_cache(
    df: pd.DataFrame, ticker: str, start_date: str, end_date: str,
    sources: str, cache_dir: Optional[Path] = None,
) -> None:
    """Persist a scraped DataFrame to parquet."""
    d = cache_dir or _CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    fp = d / _cache_key(ticker, start_date, end_date, sources)
    df.to_parquet(fp, index=False)
    logger.info("Cached %d articles → %s", len(df), fp.name)

# ---------------------------------------------------------------------------
# Individual scrapers
# ---------------------------------------------------------------------------

def scrape_google_news_rss(
    ticker: str,
    company_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_results: int = 100,
    window_days: Optional[int] = None,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Fetch news articles from Google News RSS feed for a stock ticker.

    When *start_date* and *end_date* are provided the date range is split
    into sliding windows and fetched concurrently.  The window size is
    chosen automatically based on the span:

    - ≤ 3 months  →  7-day windows  (high granularity)
    - ≤ 1 year    → 14-day windows
    - > 1 year    → 30-day windows  (keeps request count manageable)

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        company_name: Optional full company name to broaden the query.
        start_date: ISO-format start date (inclusive).
        end_date: ISO-format end date (inclusive).
        max_results: Max articles **per window** (Google caps at ~100).
        window_days: Override automatic window size (days).
        max_workers: Number of concurrent fetch threads.

    Returns:
        DataFrame with columns: date, ticker, headline, description, source, url
    """
    logger.info("Fetching Google News RSS for %s …", ticker)

    # ---- Build sliding windows ----
    windows: List[tuple] = []
    if start_date and end_date:
        ws = datetime.strptime(start_date, "%Y-%m-%d")
        we = datetime.strptime(end_date, "%Y-%m-%d")
        span_days = (we - ws).days

        if window_days is None:
            if span_days <= 90:
                window_days = 7
            elif span_days <= 365:
                window_days = 14
            else:
                window_days = 30

        delta = timedelta(days=window_days)
        cur = ws
        while cur < we:
            win_end = min(cur + delta, we)
            windows.append((cur.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")))
            cur = win_end
        logger.info(
            "  %d windows (%d-day each) for %s → %s",
            len(windows), window_days, start_date, end_date,
        )
    else:
        windows.append((start_date, end_date))

    # ---- Fetch a single window (used by threads) ----
    def _fetch_window(w_start: Optional[str], w_end: Optional[str]) -> List[Dict[str, Any]]:
        query_parts = [f"{ticker} stock"]
        if company_name:
            query_parts.append(company_name)
        if w_start:
            query_parts.append(f"after:{w_start}")
        if w_end:
            query_parts.append(f"before:{w_end}")
        query = quote_plus(" ".join(query_parts))
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logger.warning("Google News RSS window %s–%s failed: %s", w_start, w_end, e)
            return []

        rows: List[Dict[str, Any]] = []
        for entry in feed.entries[:max_results]:
            rows.append({
                "date": _parse_rss_date(entry.get("published", "")),
                "ticker": ticker.upper(),
                "headline": _clean_html(entry.get("title", "")),
                "description": _clean_html(
                    entry.get("summary", entry.get("description", ""))
                ),
                "source": entry.get("source", {}).get("title", "Google News"),
                "url": entry.get("link", ""),
            })
        return rows

    # ---- Concurrent fetch ----
    all_articles: List[Dict[str, Any]] = []
    effective_workers = min(max_workers, len(windows))

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {
            pool.submit(_fetch_window, ws, we): (ws, we)
            for ws, we in windows
        }
        for future in as_completed(futures):
            all_articles.extend(future.result())

    df = pd.DataFrame(all_articles)
    if df.empty:
        return _empty_news_df()
    df = df.drop_duplicates(subset=["headline"], keep="first")
    return _filter_by_date(df, start_date, end_date)


def scrape_finviz_news(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Scrape news headlines from Finviz for a given ticker.

    Args:
        ticker: Stock ticker symbol.
        start_date: ISO-format start date (inclusive).
        end_date: ISO-format end date (inclusive).

    Returns:
        DataFrame with columns: date, ticker, headline, description, source, url
    """
    url = f"https://finviz.com/quote.ashx?t={ticker.upper()}&p=d"
    logger.info("Fetching Finviz news for %s …", ticker)

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Finviz request failed: %s", e)
        return _empty_news_df()

    soup = BeautifulSoup(resp.text, "html.parser")
    news_table = soup.find(id="news-table")
    if news_table is None:
        logger.warning("No news table found on Finviz for %s", ticker)
        return _empty_news_df()

    articles: List[Dict[str, Any]] = []
    current_date = datetime.today().strftime("%Y-%m-%d")

    for row in news_table.find_all("tr"):
        td_cells = row.find_all("td")
        if len(td_cells) < 2:
            continue

        date_cell = td_cells[0].text.strip()
        # Finviz date cell: either "Mar-07-26 09:30AM" or just "09:30AM"
        parts = date_cell.split()
        if len(parts) == 2:
            current_date = _parse_finviz_date(parts[0])

        link_tag = td_cells[1].find("a")
        if link_tag is None:
            continue

        headline = link_tag.text.strip()
        article_url = link_tag.get("href", "")

        articles.append({
            "date": current_date,
            "ticker": ticker.upper(),
            "headline": headline,
            "description": headline,  # Finviz only provides headlines
            "source": "Finviz",
            "url": article_url,
        })

    df = pd.DataFrame(articles)
    return _filter_by_date(df, start_date, end_date)


def scrape_yahoo_news_rss(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_results: int = 100,
) -> pd.DataFrame:
    """
    Fetch news articles from Yahoo Finance RSS feed for a given ticker.

    Args:
        ticker: Stock ticker symbol.
        start_date: ISO-format start date (inclusive).
        end_date: ISO-format end date (inclusive).
        max_results: Maximum number of articles to return.

    Returns:
        DataFrame with columns: date, ticker, headline, description, source, url
    """
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker.upper()}&region=US&lang=en-US"
    logger.info("Fetching Yahoo Finance RSS for %s …", ticker)

    try:
        feed = feedparser.parse(url)
    except Exception as e:
        logger.warning("Yahoo Finance RSS fetch failed: %s", e)
        return _empty_news_df()

    articles: List[Dict[str, Any]] = []
    for entry in feed.entries[:max_results]:
        pub_date = _parse_rss_date(entry.get("published", ""))
        headline = _clean_html(entry.get("title", ""))
        description = _clean_html(entry.get("summary", entry.get("description", "")))
        link = entry.get("link", "")

        articles.append({
            "date": pub_date,
            "ticker": ticker.upper(),
            "headline": headline,
            "description": description,
            "source": "Yahoo Finance",
            "url": link,
        })

    df = pd.DataFrame(articles)
    return _filter_by_date(df, start_date, end_date)


def scrape_bing_news(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_pages: int = 10,
) -> pd.DataFrame:
    """
    Scrape news articles from Bing News search results via HTML parsing.

    Bing News is a free, no-API-key source that returns headlines, links,
    source names, and relative/absolute publication dates.  This scraper
    paginates through multiple result pages to maximise coverage.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        start_date: ISO-format start date (inclusive).
        end_date: ISO-format end date (inclusive).
        max_pages: Maximum result pages to fetch (≈10 articles each).

    Returns:
        DataFrame with columns: date, ticker, headline, description, source, url
    """
    logger.info("Fetching Bing News for %s …", ticker)

    articles: List[Dict[str, Any]] = []
    seen: set = set()

    for page in range(max_pages):
        offset = page * 10 + 1
        url = (
            f"https://www.bing.com/news/search"
            f"?q={quote_plus(ticker + ' stock')}"
            f"&first={offset}"
            f'&qft=sortbydate%3d"1"'
        )
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Bing News page %d failed: %s", page, e)
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        cards = soup.select(".news-card")
        if not cards:
            break  # no more results

        for card in cards:
            title_tag = card.select_one("a.title")
            if title_tag is None:
                continue
            headline = title_tag.text.strip()
            if headline in seen:
                continue
            seen.add(headline)

            link = title_tag.get("href", "")
            source = card.get("data-author", "Bing News")

            # Resolve publication date from aria-label
            date_span = card.select_one("span[aria-label]")
            raw_date = date_span.get("aria-label", "") if date_span else ""
            pub_date = _parse_bing_date(raw_date)

            articles.append({
                "date": pub_date,
                "ticker": ticker.upper(),
                "headline": headline,
                "description": headline,
                "source": source,
                "url": link,
            })

        # Polite pause between pages
        time.sleep(1.0)

    df = pd.DataFrame(articles)
    if df.empty:
        return _empty_news_df()
    return _filter_by_date(df, start_date, end_date)


# ---------------------------------------------------------------------------
# Aggregated scraper
# ---------------------------------------------------------------------------

def scrape_news(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sources: Optional[List[str]] = None,
    company_name: Optional[str] = None,
    delay: float = 1.0,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Scrape news articles for a ticker from multiple sources and return a
    deduplicated, date-sorted DataFrame.

    Results are cached to disk (parquet) so that repeated calls with the
    same parameters are instant.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        start_date: ISO-format start date (inclusive), e.g. '2025-01-01'.
        end_date: ISO-format end date (inclusive), e.g. '2025-03-01'.
        sources: List of sources to use.
                 Options: 'google', 'bing', 'finviz', 'yahoo'.
                 Defaults to ['google', 'bing'] (best coverage).
        company_name: Optional company name used to widen Google News query.
        delay: Seconds to wait between source requests (polite crawling).
        use_cache: If True (default), read/write cached results.
        cache_dir: Override default cache directory.

    Returns:
        DataFrame with columns:
            date, ticker, headline, description, source, url
    """
    if sources is None:
        sources = ["google", "bing"]

    src_key = ",".join(sorted(sources))

    # ---- Check cache ----
    if use_cache and start_date and end_date:
        cached = _load_cache(ticker, start_date, end_date, src_key, cache_dir)
        if cached is not None:
            return cached

    scrapers = {
        "google": lambda: scrape_google_news_rss(
            ticker, company_name=company_name,
            start_date=start_date, end_date=end_date,
        ),
        "bing": lambda: scrape_bing_news(
            ticker, start_date=start_date, end_date=end_date,
        ),
        "finviz": lambda: scrape_finviz_news(
            ticker, start_date=start_date, end_date=end_date,
        ),
        "yahoo": lambda: scrape_yahoo_news_rss(
            ticker, start_date=start_date, end_date=end_date,
        ),
    }

    frames: List[pd.DataFrame] = []
    for src in sources:
        scraper_fn = scrapers.get(src.lower())
        if scraper_fn is None:
            logger.warning("Unknown news source: %s – skipping", src)
            continue
        try:
            df = scraper_fn()
            if not df.empty:
                frames.append(df)
                logger.info("  → %s returned %d articles", src, len(df))
        except Exception as e:
            logger.warning("Scraper '%s' failed: %s", src, e)

        time.sleep(delay)

    if not frames:
        logger.warning("No articles collected for %s", ticker)
        return _empty_news_df()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["headline"], keep="first")
    combined = combined.sort_values("date").reset_index(drop=True)

    logger.info("Total unique articles for %s: %d", ticker, len(combined))

    # ---- Save to cache ----
    if use_cache and start_date and end_date:
        _save_cache(combined, ticker, start_date, end_date, src_key, cache_dir)

    return combined


# ---------------------------------------------------------------------------
# Batch scraper – multiple tickers
# ---------------------------------------------------------------------------

def batch_scrape_news(
    tickers: List[str],
    start_date: str,
    end_date: str,
    sources: Optional[List[str]] = None,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Scrape news for **multiple tickers** and return a single DataFrame.

    Uses disk caching so only uncached tickers are actually scraped.
    A progress counter is printed to stdout.

    Args:
        tickers: List of stock ticker symbols.
        start_date: ISO start date.
        end_date: ISO end date.
        sources: Source list (default ['google', 'bing']).
        use_cache: Read/write cache (default True).
        cache_dir: Override cache directory.

    Returns:
        Concatenated DataFrame of all tickers' news.
    """
    frames: List[pd.DataFrame] = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] Scraping news for %s …", i, total, ticker)
        try:
            df = scrape_news(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                sources=sources,
                use_cache=use_cache,
                cache_dir=cache_dir,
            )
            if not df.empty:
                frames.append(df)
        except Exception as e:
            logger.warning("  ✗ %s failed: %s", ticker, e)

    if not frames:
        return _empty_news_df()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Batch complete: %d tickers, %d total articles",
        total, len(combined),
    )
    return combined


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _empty_news_df() -> pd.DataFrame:
    """Return an empty DataFrame with the standard news columns."""
    return pd.DataFrame(columns=["date", "ticker", "headline", "description", "source", "url"])


def _parse_rss_date(date_str: str) -> str:
    """Parse various RSS date formats into YYYY-MM-DD."""
    if not date_str:
        return datetime.today().strftime("%Y-%m-%d")
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Last resort: try pandas
    try:
        return pd.to_datetime(date_str).strftime("%Y-%m-%d")
    except Exception:
        return datetime.today().strftime("%Y-%m-%d")


def _parse_finviz_date(date_str: str) -> str:
    """Parse Finviz-style date like 'Mar-07-26' into 'YYYY-MM-DD'."""
    try:
        return datetime.strptime(date_str, "%b-%d-%y").strftime("%Y-%m-%d")
    except ValueError:
        return datetime.today().strftime("%Y-%m-%d")


def _parse_bing_date(raw: str) -> str:
    """Resolve a Bing News date string into YYYY-MM-DD.

    Bing uses either relative labels ("4 hours ago", "3 days ago") or
    absolute dates in DD/MM/YYYY format.
    """
    if not raw:
        return datetime.today().strftime("%Y-%m-%d")

    low = raw.lower().strip()
    now = datetime.today()

    # Relative: "X minutes/hours ago" → today
    if "minute" in low or "hour" in low:
        return now.strftime("%Y-%m-%d")

    # Relative: "X days ago"
    m = re.match(r"(\d+)\s*day", low)
    if m:
        return (now - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")

    # Absolute: DD/MM/YYYY (Bing international format)
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Fallback: let pandas try
    try:
        return pd.to_datetime(raw).strftime("%Y-%m-%d")
    except Exception:
        return now.strftime("%Y-%m-%d")


def _clean_html(text: str) -> str:
    """Strip HTML tags and extra whitespace from a string."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _filter_by_date(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Filter a news DataFrame to the given date window."""
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]
    return df.reset_index(drop=True)
