"""
SEC EDGAR insider-trade scraper.

Fetches Form 4 filings for a ticker via the edgartools library and
returns a daily-aggregated DataFrame of insider buy/sell activity.

Transaction codes captured:
  P  — open-market purchase  → insider_buys / insider_buy_value
  S  — open-market sale      → insider_sells / insider_sell_value
  A  — award / grant         → grant_count / grant_shares
  F  — tax withholding       → tax_withheld_count / tax_withheld_shares
  M  — option exercise       → option_exercises
  G  — gift                  → gift_count

Exports: scrape_insider_trades
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import pandas as pd

from .._cache import (_load_cache, _save_cache, _empty_insider_df,
                      _load_incremental, _merge_and_save)

logger = logging.getLogger(__name__)

# Silence noisy edgartools HTTP/cache loggers
for _noisy in ("httpx", "httpxthrottlecache.filecache.transport",
               "httpxthrottlecache.ratelimiter", "httpxthrottlecache.controller"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

import re as _re

# Whole-word patterns that indicate an executive officer (not a pure board director)
_OFFICER_PATTERN = _re.compile(
    r"\b(ceo|cfo|coo|cto|president|officer|evp|svp|vp|counsel|secretary"
    r"|treasurer|principal|controller|managing)\b"
)


def _is_officer(position: str) -> bool:
    return bool(_OFFICER_PATTERN.search(position.lower()))


def scrape_insider_trades(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
    max_filings: int = 500,
) -> pd.DataFrame:
    """
    Fetch insider transactions from SEC EDGAR via the edgartools library.

    Uses Form4.to_dataframe() to capture all transaction codes, then
    aggregates by day into the following columns:

        date, ticker,
        insider_buys, insider_sells, insider_net_transactions,
        insider_buy_value, insider_sell_value, insider_net_value,
        insider_transaction_count, insider_buy_ratio,
        officer_buys, officer_sells,
        grant_count, grant_shares,
        tax_withheld_count, tax_withheld_shares,
        option_exercises, gift_count

    Parameters
    ----------
    ticker : str
    start_date, end_date : str (ISO format)
    use_cache : bool
    max_filings : int
        Cap on filings to parse (avoids very long fetches for active tickers).
    """
    try:
        from edgar import Company as _EdgarCompany, set_identity as _edgar_set_identity
    except ImportError as exc:
        raise ImportError("edgartools is required: pip install edgartools") from exc

    _edgar_set_identity("stock-predictor research@example.com")

    start = start_date or "2021-01-01"
    end = end_date or datetime.today().strftime("%Y-%m-%d")

    existing_df, fetch_from = _load_incremental("insider_edgar", ticker, start, end) if use_cache else (None, start)
    if fetch_from > end:
        return existing_df

    all_transactions = []

    try:
        filings = _EdgarCompany(ticker).get_filings(form="4", filing_date=f"{fetch_from}:{end}", trigger_full_load=False)
    except Exception as e:
        logger.warning("edgartools Form 4 fetch failed for %s: %s", ticker, e)
        return existing_df if existing_df is not None else _empty_insider_df(ticker)

    # Cap the filing list before fetching — avoids building a huge iterator
    filings_list = []
    for filing in filings:
        filings_list.append(filing)
        if len(filings_list) >= max_filings:
            break

    def _fetch_filing(filing):
        """Download and parse one Form 4 filing; returns list of raw transaction dicts."""
        try:
            ownership = filing.obj()
            filing_date = str(filing.filing_date)[:10]
            txn_df: pd.DataFrame = ownership.to_dataframe()
            if txn_df is None or txn_df.empty:
                return []
            rows = []
            for _, row in txn_df.iterrows():
                try:
                    tx_code = str(row.get("Code", "") or "").strip()
                    if not tx_code:
                        continue
                    tx_date = row.get("Date", None)
                    tx_date = str(tx_date)[:10] if tx_date is not None else filing_date
                    if tx_date < start or tx_date > end:
                        continue
                    rows.append({
                        "date": tx_date,
                        "ticker": ticker.upper(),
                        "transaction_code": tx_code,
                        "shares": float(row.get("Shares", 0) or 0),
                        "value": float(row.get("Value", 0) or 0),
                        "is_officer": int(_is_officer(str(row.get("Position", "") or ""))),
                    })
                except Exception:
                    continue
            return rows
        except Exception as e:
            logger.debug("Form 4 parse error for %s: %s", ticker, e)
            return []

    # Fetch filings in parallel — EDGAR allows ~10 req/s; 8 workers stays safely under
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_filing, f): f for f in filings_list}
        for future in as_completed(futures):
            all_transactions.extend(future.result())

    logger.info("%s: parsed %d filings → %d transactions", ticker, len(filings_list), len(all_transactions))

    if not all_transactions:
        if existing_df is not None:
            return existing_df
        return _empty_insider_df(ticker)

    raw = pd.DataFrame(all_transactions)

    # Boolean masks per code
    is_buy = raw["transaction_code"] == "P"
    is_sell = raw["transaction_code"] == "S"
    is_grant = raw["transaction_code"] == "A"
    is_tax = raw["transaction_code"] == "F"
    is_exercise = raw["transaction_code"] == "M"
    is_gift = raw["transaction_code"] == "G"

    raw["is_buy"] = is_buy.astype(int)
    raw["is_sell"] = is_sell.astype(int)
    raw["buy_value"] = raw["value"] * is_buy
    raw["sell_value"] = raw["value"] * is_sell
    raw["is_officer_buy"] = (is_buy & (raw["is_officer"] == 1)).astype(int)
    raw["is_officer_sell"] = (is_sell & (raw["is_officer"] == 1)).astype(int)
    raw["is_grant"] = is_grant.astype(int)
    raw["grant_shares"] = raw["shares"] * is_grant
    raw["is_tax"] = is_tax.astype(int)
    raw["tax_shares"] = raw["shares"] * is_tax
    raw["is_exercise"] = is_exercise.astype(int)
    raw["is_gift"] = is_gift.astype(int)

    daily = raw.groupby("date").agg(
        insider_buys=("is_buy", "sum"),
        insider_sells=("is_sell", "sum"),
        insider_transaction_count=("transaction_code", "count"),
        insider_buy_value=("buy_value", "sum"),
        insider_sell_value=("sell_value", "sum"),
        officer_buys=("is_officer_buy", "sum"),
        officer_sells=("is_officer_sell", "sum"),
        grant_count=("is_grant", "sum"),
        grant_shares=("grant_shares", "sum"),
        tax_withheld_count=("is_tax", "sum"),
        tax_withheld_shares=("tax_shares", "sum"),
        option_exercises=("is_exercise", "sum"),
        gift_count=("is_gift", "sum"),
    ).reset_index()

    daily["ticker"] = ticker.upper()
    daily["insider_net_transactions"] = daily["insider_buys"] - daily["insider_sells"]
    daily["insider_buy_ratio"] = (
        daily["insider_buys"] / daily["insider_transaction_count"].clip(lower=1)
    )
    daily["insider_net_value"] = daily["insider_buy_value"] - daily["insider_sell_value"]

    df = daily[[
        "date", "ticker",
        "insider_buys", "insider_sells", "insider_net_transactions",
        "insider_buy_value", "insider_sell_value", "insider_net_value",
        "insider_transaction_count", "insider_buy_ratio",
        "officer_buys", "officer_sells",
        "grant_count", "grant_shares",
        "tax_withheld_count", "tax_withheld_shares",
        "option_exercises", "gift_count",
    ]]

    if use_cache:
        df = _merge_and_save(df, existing_df, "insider_edgar", ticker, start, end)
    return df
