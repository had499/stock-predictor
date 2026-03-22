# News Data Module

Scrape, analyse, and vectorise financial news for stock tickers.

## Quick Start

```python
from src.data.news import get_news

# Single ticker – VADER sentiment (fast, no GPU needed)
df = get_news("AAPL", start_date="2025-01-01", end_date="2025-03-01")
df[["date", "headline", "headline_sentiment"]].head()

# Multiple tickers – models loaded once, scraped concurrently
df = get_news(
    ["AAPL", "MSFT", "GOOGL"],
    start_date="2021-01-01",
    end_date="2025-03-01",
    vectorize_method="embedding",
    aggregate=True,
)
```

## Features

| Component | Description |
|---|---|
| `scrape_news()` | Aggregates articles from Google News RSS, Bing News, Finviz, and Yahoo Finance |
| `batch_scrape_news()` | Scrape multiple tickers with disk caching |
| `VaderSentimentAnalyzer` | Fast rule-based sentiment via NLTK VADER |
| `FinBERTSentimentAnalyzer` | Transformer-based financial sentiment (ProsusAI/finbert) |
| `NewsTfidfVectorizer` | TF-IDF text features from headlines + descriptions |
| `NewsEmbeddingVectorizer` | Dense embeddings via Sentence-Transformers |
| `NewsDataPipeline` | End-to-end sklearn transformer: scrape → sentiment → vectors |
| `get_news()` | One-liner accepting `str` or `list[str]` – multi-ticker is concurrent & model-efficient |
| `aggregate_daily_sentiment()` | Collapse per-article sentiment to daily aggregates |

## Usage Examples

### 1. Scrape Only

```python
from src.data.news.scraper import scrape_news

articles = scrape_news("TSLA", start_date="2025-02-01", end_date="2025-02-28")
print(articles.shape)
```

### 2. VADER Sentiment (lightweight)

```python
from src.data.news import get_news

df = get_news("MSFT", sentiment_method="vader", aggregate=True)
# Returns daily: sentiment_mean, sentiment_median, sentiment_std, news_count
```

### 3. FinBERT Sentiment (more accurate for financial text)

```python
df = get_news("MSFT", sentiment_method="finbert", device="cpu")
```

### 4. Text Vectorisation

```python
# TF-IDF vectors
df = get_news("GOOGL", vectorize_method="tfidf", tfidf_features=50)

# Dense embeddings (Sentence-Transformers)
df = get_news("GOOGL", vectorize_method="embedding")
```

### 5. Multiple Tickers (efficient)

```python
from src.data.news import get_news

# Pass a list – scraping is concurrent, models loaded once
df = get_news(
    ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
    start_date="2021-01-01",
    end_date="2025-03-01",
    sentiment_method="vader",
    vectorize_method="embedding",
    aggregate=True,            # daily-level rows
    use_cache=True,            # cache raw articles to disk
    max_workers=4,             # concurrent scraping threads
)
# Returns DataFrame with columns: date, ticker, sentiment_*, emb_0…emb_383
```

### 6. Full sklearn Pipeline

```python
from src.data.news import NewsDataPipeline

pipe = NewsDataPipeline(
    ticker="AAPL",
    start_date="2025-01-01",
    end_date="2025-03-01",
    sentiment_method="vader",
    vectorize_method="tfidf",
    aggregate=False,
)
df = pipe.fit_transform()
```

### 7. Merge with Stock Data

```python
from src.data.news import get_news
from src.data.data_loader import StockDataLoader

# Daily sentiment
sentiment = get_news("AAPL", start_date="2025-01-01", aggregate=True)

# Stock prices
loader = StockDataLoader(symbols=["AAPL"], start_date="2025-01-01")
prices = loader.fit_transform()

# Merge
merged = prices.merge(sentiment, on=["date"], how="left")
```

## Dependencies

Core (always required):
- `requests`, `beautifulsoup4`, `feedparser`, `pandas`, `numpy`, `scikit-learn`

VADER sentiment:
- `nltk`

FinBERT sentiment:
- `transformers`, `torch`

Dense embeddings:
- `sentence-transformers`

All are listed in `requirements.txt`.
