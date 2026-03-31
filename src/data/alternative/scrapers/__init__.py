"""
Alternative data scrapers sub-package.

Re-exports all public scraper functions so that the parent package's
``__init__.py`` and ``pipeline.py`` can import from a single location.
"""

from .insider import scrape_insider_trades
from .macro import (
    scrape_fred_macro,
    scrape_macro_data,
    scrape_cross_asset_data,
    scrape_etf_fund_flows,
)
from .ticker import (
    scrape_yahoo_earnings,
    scrape_congress_trades,
    scrape_wikipedia_pageviews,
    scrape_earnings_revisions,
)
from .short_interest import scrape_short_interest
from .price_targets import scrape_price_targets
from ._dispatch import scrape_alternative_data

__all__ = [
    "scrape_alternative_data",
    "scrape_insider_trades",
    "scrape_fred_macro",
    "scrape_macro_data",
    "scrape_cross_asset_data",
    "scrape_etf_fund_flows",
    "scrape_yahoo_earnings",
    "scrape_congress_trades",
    "scrape_wikipedia_pageviews",
    "scrape_earnings_revisions",
    "scrape_short_interest",
    "scrape_price_targets",
]
