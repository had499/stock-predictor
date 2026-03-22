"""
Alternative Data Module

Scrape, engineer, and serve alternative (non-price) features for
stock tickers – insider trades, macro indicators, earnings surprises,
analyst ratings, short interest, options flow, Congress trades,
and Wikipedia attention.

Quick start
-----------
>>> from data.alternative import get_alternative_data
>>> alt = get_alternative_data("AAPL", "2021-01-01", "2025-03-01")
>>> alt.head()

Multiple tickers (concurrent):
>>> alt = get_alternative_data(
...     ["AAPL", "MSFT", "GOOGL"],
...     start_date="2021-01-01",
...     end_date="2025-03-01",
... )

Standalone macro features (ticker-agnostic):
>>> from data.alternative import get_macro_data
>>> macro = get_macro_data("2021-01-01", "2025-03-01")
>>> df = df.merge(macro, on="date", how="left")
"""

from .scrapers import (
    scrape_alternative_data,
    scrape_insider_trades,
    scrape_fred_macro,
    scrape_macro_data,
    scrape_yahoo_earnings,
    scrape_congress_trades,
    scrape_wikipedia_pageviews,
    scrape_earnings_revisions,
    scrape_cross_asset_data,
    scrape_etf_fund_flows,
)
from .features import build_alternative_features, build_macro_features
from .pipeline import get_alternative_data, get_macro_data

__all__ = [
    # Pipeline (main entry points)
    "get_alternative_data",
    "get_macro_data",
    # Feature engineering
    "build_alternative_features",
    "build_macro_features",
    # Individual scrapers
    "scrape_alternative_data",
    "scrape_insider_trades",
    "scrape_fred_macro",
    "scrape_macro_data",
    "scrape_yahoo_earnings",
    "scrape_congress_trades",
    "scrape_wikipedia_pageviews",
    "scrape_earnings_revisions",
    "scrape_cross_asset_data",
    "scrape_etf_fund_flows",
]
