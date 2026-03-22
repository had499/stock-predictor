"""
News Data Module

Scrape, analyse, and vectorise financial news for stock tickers.

Quick start
-----------
>>> from src.data.news import get_news
>>> df = get_news("AAPL", start_date="2025-01-01", end_date="2025-03-01")
>>> df[["date", "headline", "headline_sentiment"]].head()

Components
----------
- **scraper**   – ``scrape_news()``, ``scrape_google_news_rss()``, etc.
- **sentiment** – ``VaderSentimentAnalyzer``, ``FinBERTSentimentAnalyzer``,
                  ``NewsTfidfVectorizer``, ``NewsEmbeddingVectorizer``,
                  ``aggregate_daily_sentiment()``
- **pipeline**  – ``NewsDataPipeline`` (sklearn transformer), ``get_news()``
"""

from .scraper import (
    scrape_news,
    batch_scrape_news,
    scrape_google_news_rss,
    scrape_bing_news,
    scrape_finviz_news,
    scrape_yahoo_news_rss,
)
from .sentiment import (
    VaderSentimentAnalyzer,
    FinBERTSentimentAnalyzer,
    NewsTfidfVectorizer,
    NewsEmbeddingVectorizer,
    FinBERTEmbeddingVectorizer,
    aggregate_daily_sentiment,
)
from .pipeline import (
    NewsDataPipeline,
    get_news,
    batch_get_news,
)

__all__ = [
    # Scraper
    "scrape_news",
    "batch_scrape_news",
    "scrape_google_news_rss",
    "scrape_bing_news",
    "scrape_finviz_news",
    "scrape_yahoo_news_rss",
    # Sentiment & vectorisation
    "VaderSentimentAnalyzer",
    "FinBERTSentimentAnalyzer",
    "NewsTfidfVectorizer",
    "NewsEmbeddingVectorizer",
    "FinBERTEmbeddingVectorizer",
    "aggregate_daily_sentiment",
    # Pipeline
    "NewsDataPipeline",
    "get_news",
    "batch_get_news",
]
