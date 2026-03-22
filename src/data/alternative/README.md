# Alternative Data Module

Scrape and engineer **non-price alternative features** for stock tickers to give your model an informational edge beyond standard technical indicators.

## Quick Start

```python
from data.alternative import get_alternative_data

# Single ticker
alt = get_alternative_data("AAPL", "2021-01-01", "2025-03-01")

# Multiple tickers – concurrent scraping, FRED macro shared
alt = get_alternative_data(
    ["AAPL", "MSFT", "GOOGL"],
    start_date="2021-01-01",
    end_date="2025-03-01",
)

# Merge with your stock price DataFrame
merged = prices.merge(alt, on=["date", "ticker"], how="left")
```

## Data Sources & Features

| Source | Scraper | Features | Edge |
|--------|---------|----------|------|
| **SEC EDGAR (Form 4)** | `scrape_insider_trades()` | insider_buy_ratio, insider_net_transactions, insider_buys_30d, insider_activity_5d | Insiders have information advantage; cluster buys predict +3-5% over 6mo |
| **SEC EDGAR / Yahoo** | `scrape_institutional_holdings()` | institutional_holders_count, institutional_total_pct | High institutional ownership = more analyst coverage, lower volatility |
| **FRED** | `scrape_fred_macro()` | fed_funds_rate, treasury_10y/2y, yield_curve_slope, yield_curve_inverted, vix, vix_regime, unemployment, CPI, consumer_sentiment, oil_wti, high_yield_spread, initial_claims | Macro regime drives sector rotation and risk appetite |
| **Yahoo Finance** | `scrape_yahoo_earnings()` | earnings_surprise_pct, earnings_beat, days_since_earnings, days_to_next_earnings, earnings_surprise_ma | Post-earnings drift = strongest short-term anomaly |
| **Yahoo Finance** | `scrape_yahoo_analyst_ratings()` | analyst_consensus_score, analyst_buy_pct, analyst_total | Consensus shifts predict medium-term moves |
| **Yahoo Finance** | `scrape_yahoo_short_interest()` | short_pct_float, short_ratio, short_squeeze_risk | High short interest → squeeze potential or informed bearishness |
| **Yahoo Finance** | `scrape_options_flow()` | put_call_ratio, oi_put_call_ratio, options_sentiment | Options market leads equities; smart money trades options first |
| **Capitol Trades** | `scrape_congress_trades()` | congress_net, congress_trade_count, congress_activity_30d, congress_buy_signal | Congress trades outperform market (documented in academic lit) |
| **Wikipedia** | `scrape_wikipedia_pageviews()` | wiki_pageviews, wiki_pageviews_ma7, wiki_attention_spike, wiki_attention_trend | Retail attention spikes precede volume/volatility jumps |

## Derived Features

The `features.py` module adds computed signals on top of raw data:

- **Yield curve inversion** flag (recession predictor)
- **VIX regime** categorical (low_vol / normal / elevated / crisis)
- **Rate momentum** (20-day change in fed funds rate)
- **Credit stress** (5-day change in high-yield spread)
- **Oil return** (5-day % change)
- **Earnings proximity** (days since / until next report)
- **Congress buy signal** (net buys in trailing 30 days)
- **Attention trend** (7d MA / 30d MA of Wikipedia views)

## Architecture

```
src/data/alternative/
├── __init__.py      # Public API
├── scrapers.py      # Individual data source scrapers + caching
├── features.py      # Feature engineering per source
├── pipeline.py      # get_alternative_data() – main entry point
└── README.md        # This file
```

## Caching

All raw scrapes are cached to `datasets/alternative/` as parquet files.
Subsequent calls with the same parameters return instantly from cache.

## Parameters

```python
get_alternative_data(
    ticker="AAPL",              # str or list[str]
    start_date="2021-01-01",    # ISO date
    end_date="2025-03-01",      # ISO date
    sources=None,               # list or None (all sources)
    use_cache=True,             # disk caching
    max_workers=4,              # concurrent threads (multi-ticker)
)
```

Available sources: `'insider'`, `'institutional'`, `'earnings'`, `'analyst'`, `'short_interest'`, `'options'`, `'congress'`, `'wiki'`, `'fred'`.

## Dependencies

All dependencies are already in `requirements.txt`:
- `requests`, `beautifulsoup4`, `pandas`, `numpy`, `yfinance`, `pyarrow`

No API keys required – all sources are free and public.
