"""
Per-ticker alternative data scrapers.

Scrapes data that is specific to an individual stock ticker:
  - Yahoo Finance earnings history & revisions
  - Capitol Trades (US Congress stock trades)
  - Wikipedia pageview statistics

Exports: scrape_yahoo_earnings, scrape_congress_trades,
         scrape_wikipedia_pageviews, scrape_earnings_revisions
"""

import logging
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .._cache import (_load_cache, _save_cache,
                      _load_incremental, _merge_and_save)

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

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


# ============================================================================
# Yahoo Finance – earnings
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

    existing_df, fetch_from = _load_incremental("earnings", ticker, start, end) if use_cache else (None, start)
    if fetch_from > end:
        return existing_df

    rows = []
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        # Earnings dates with actual vs estimate — limit=40 fetches full history
        earnings = t.get_earnings_dates(limit=40)
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
        full_earnings = t.get_earnings_dates(limit=40)
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

    if use_cache:
        return _merge_and_save(df, existing_df, "earnings", ticker, start, end)
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
# Capitol Trades – US Congress stock trades
# ============================================================================

def _get_capitol_trades_issuer_id(ticker: str) -> Optional[str]:
    """Look up the capitoltrades.com numeric issuer ID for a ticker."""
    try:
        resp = requests.get(
            "https://www.capitoltrades.com/issuers",
            params={"search": ticker.upper()},
            headers=_HEADERS,
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        links = [l["href"] for l in soup.find_all("a", href=True) if "/issuers/" in l["href"]]
        return links[0].split("/")[-1] if links else None
    except Exception as e:
        logger.warning("Capitol Trades issuer lookup failed for %s: %s", ticker, e)
        return None


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

    existing_df, fetch_from = _load_incremental("congress", ticker, start, end) if use_cache else (None, start)
    if fetch_from > end:
        return existing_df

    issuer_id = _get_capitol_trades_issuer_id(ticker)
    if not issuer_id:
        logger.warning("Capitol Trades: no issuer ID found for %s", ticker)
        return existing_df if existing_df is not None else _empty_congress_df(ticker)

    rows = []
    page = 1
    max_pages = 50

    while page <= max_pages:
        try:
            resp = requests.get(
                "https://www.capitoltrades.com/trades",
                params={"issuer": issuer_id, "page": page},
                headers=_HEADERS,
                timeout=15,
            )
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

            earliest_on_page = None
            for tr in row_els:
                cells = tr.find_all("td")
                if len(cells) < 7:
                    continue

                trade_date = cells[3].get_text(strip=True)
                trade_type = cells[6].get_text(strip=True).lower()

                try:
                    parsed_date = pd.to_datetime(trade_date).strftime("%Y-%m-%d")
                except Exception:
                    continue

                if earliest_on_page is None or parsed_date < earliest_on_page:
                    earliest_on_page = parsed_date

                if parsed_date < fetch_from or parsed_date > end:
                    continue

                is_buy = 1 if "purchase" in trade_type or "buy" in trade_type else 0
                is_sell = 1 if "sale" in trade_type or "sell" in trade_type else 0

                rows.append({
                    "date": parsed_date,
                    "ticker": ticker.upper(),
                    "congress_buy": is_buy,
                    "congress_sell": is_sell,
                })

            # Stop early if all trades on this page predate our fetch window
            if earliest_on_page and earliest_on_page < fetch_from:
                break

            next_link = soup.find("a", {"aria-label": "Go to next page"})
            if not next_link:
                break
            page += 1
            time.sleep(0.1)

        except Exception as e:
            logger.warning("Congress trades page %d failed for %s: %s", page, ticker, e)
            break

    if not rows:
        new_df = _empty_congress_df(ticker)
    else:
        raw = pd.DataFrame(rows)
        daily = raw.groupby("date").agg(
            congress_buys=("congress_buy", "sum"),
            congress_sells=("congress_sell", "sum"),
            congress_trade_count=("congress_buy", "count"),
        ).reset_index()
        daily["ticker"] = ticker.upper()
        daily["congress_net"] = daily["congress_buys"] - daily["congress_sells"]
        new_df = daily[["date", "ticker", "congress_buys", "congress_sells",
                         "congress_net", "congress_trade_count"]]

    if use_cache:
        return _merge_and_save(new_df, existing_df, "congress", ticker, start, end)
    return new_df if existing_df is None else pd.concat([existing_df, new_df]).drop_duplicates()


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
# Wikipedia pageviews
# ============================================================================

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

    existing_df, fetch_from = _load_incremental("wiki", ticker, start, end) if use_cache else (None, start)
    if fetch_from > end:
        return existing_df

    # Resolve article title
    article = _TICKER_TO_WIKI.get(ticker.upper())
    if not article:
        # Fallback: try the ticker as-is or common patterns
        article = f"{ticker.upper()}_(company)"

    start_ymd = fetch_from.replace("-", "")
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

    if use_cache:
        return _merge_and_save(df, existing_df, "wiki", ticker, start, end)
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
# Yahoo Finance – earnings revisions (analyst upgrade/downgrade history)
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

    existing_df, fetch_from = _load_incremental("earnings_revisions", ticker, start, end) if use_cache else (None, start)
    if fetch_from > end:
        return existing_df

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

    if use_cache:
        return _merge_and_save(df, existing_df, "earnings_revisions", ticker, start, end)
    return df
