"""
News Data Pipeline

End-to-end pipeline that scrapes news for one or more tickers,
computes sentiment, and optionally vectorises the text.  Follows the
sklearn transformer pattern used throughout the stock-predictor project.

Author: Stock Predictor Project
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Literal, Union, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import logging
from sklearn.base import BaseEstimator, TransformerMixin

from .scraper import scrape_news, batch_scrape_news
from .sentiment import (
    VaderSentimentAnalyzer,
    FinBERTSentimentAnalyzer,
    aggregate_daily_sentiment,
)
from .vectorizer import (
    NewsTfidfVectorizer,
    NewsEmbeddingVectorizer,
    FinBERTEmbeddingVectorizer,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared model cache  –  load expensive models once per session
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, object] = {}


def _get_sentiment_analyzer(
    method: str, device: str = "cpu",
) -> Union[VaderSentimentAnalyzer, FinBERTSentimentAnalyzer]:
    """Return a *fitted* sentiment analyzer, reusing a cached instance."""
    key = f"sent_{method}_{device}"
    if key not in _MODEL_CACHE:
        if method == "vader":
            analyzer = VaderSentimentAnalyzer()
        else:
            analyzer = FinBERTSentimentAnalyzer(device=device)
        # Fit with a dummy df so the model/lexicon is loaded once
        analyzer.fit(pd.DataFrame({"headline": ["init"], "description": ["init"]}))
        _MODEL_CACHE[key] = analyzer
        logger.info("Sentiment model '%s' loaded and cached", method)
    return _MODEL_CACHE[key]


def _get_vectorizer(
    method: Optional[str],
    embedding_model: str = "all-MiniLM-L6-v2",
    tfidf_features: int = 100,
    device: str = "cpu",
) -> Optional[Union[NewsTfidfVectorizer, NewsEmbeddingVectorizer]]:
    """Return a *fitted* vectorizer, reusing a cached instance."""
    if method is None:
        return None

    key = f"vec_{method}_{embedding_model}_{tfidf_features}_{device}"
    if key not in _MODEL_CACHE:
        if method == "tfidf":
            vec = NewsTfidfVectorizer(max_features=tfidf_features)
            vec.fit(pd.DataFrame({"headline": ["init"], "description": ["init"]}))
        elif method == "embedding":
            vec = NewsEmbeddingVectorizer(
                model_name=embedding_model, device=device,
            )
            vec.fit(pd.DataFrame({"headline": ["init"], "description": ["init"]}))
        elif method == "finbert_embedding":
            vec = FinBERTEmbeddingVectorizer(device=device, batch_size=16)
            vec.fit(pd.DataFrame({"headline": ["init"], "description": ["init"]}))
        else:
            return None
        _MODEL_CACHE[key] = vec
        logger.info("Vectorizer '%s' loaded and cached", method)
    return _MODEL_CACHE[key]


class NewsDataPipeline:
    """
    Scrape financial news for a ticker, compute sentiment, and
    optionally produce vector embeddings – all in a single
    `.fit_transform()` call.

    This transformer is designed to be used **standalone** (it
    scrapes its own data) rather than receiving data from a
    previous pipeline step.  Call ``get_news()`` for the simplest
    one-liner interface.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL').
    start_date : str or None
        ISO start date, e.g. '2025-01-01'.
    end_date : str or None
        ISO end date, e.g. '2025-03-01'.
    sources : list of str or None
        News sources to scrape.  Options: 'google', 'finviz', 'yahoo'.
        Defaults to all three.
    company_name : str or None
        Full company name used to broaden the Google News query.
    sentiment_method : {'vader', 'finbert'}
        Which sentiment model to use.
    vectorize_method : {None, 'tfidf', 'embedding'}
        Optionally add text vectors.
    tfidf_features : int
        Number of TF-IDF features (if vectorize_method='tfidf').
    embedding_model : str
        Sentence-Transformers model name (if vectorize_method='embedding').
    aggregate : bool
        If True, return daily-aggregated sentiment instead of per-article rows.
    device : str
        Device for FinBERT / Sentence-Transformers ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        ticker: str = "AAPL",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sources: Optional[List[str]] = None,
        company_name: Optional[str] = None,
        sentiment_method: Literal["vader", "finbert"] = "vader",
        vectorize_method: Optional[Literal["tfidf", "embedding", "finbert_embedding"]] = None,
        tfidf_features: int = 100,
        embedding_model: str = "all-MiniLM-L6-v2",
        aggregate: bool = False,
        device: str = "cpu",
        use_cache: bool = True,
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sources = sources
        self.company_name = company_name
        self.sentiment_method = sentiment_method
        self.vectorize_method = vectorize_method
        self.tfidf_features = tfidf_features
        self.embedding_model = embedding_model
        self.aggregate = aggregate
        self.device = device
        self.use_cache = use_cache

        # Internal transformer instances (built during fit)
        self._sentiment_analyzer = None
        self._vectorizer = None
        self._raw_news: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit_transform(self, X=None, y=None) -> pd.DataFrame:
        """Scrape, fit, and transform in one call."""
        return self.fit(X, y).transform(X)

    def fit(self, X=None, y=None):
        """
        Scrape news, then fit the sentiment + vectoriser models on it.
        ``X`` is ignored – the pipeline scrapes its own data.
        """
        self._raw_news = scrape_news(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            sources=self.sources,
            company_name=self.company_name,
            use_cache=self.use_cache,
        )

        if self._raw_news.empty:
            logger.warning("No news articles scraped – pipeline has nothing to fit.")
            return self

        # Use shared model cache for efficiency
        self._sentiment_analyzer = _get_sentiment_analyzer(
            self.sentiment_method, self.device,
        )
        self._vectorizer = _get_vectorizer(
            self.vectorize_method,
            embedding_model=self.embedding_model,
            tfidf_features=self.tfidf_features,
            device=self.device,
        )

        return self

    def transform(self, X=None) -> pd.DataFrame:
        """
        Apply sentiment & vectorisation to the scraped news.
        ``X`` is ignored – operates on internally scraped data.
        """
        if self._raw_news is None or self._raw_news.empty:
            logger.warning("No scraped news available – returning empty DataFrame.")
            return pd.DataFrame()

        df = self._raw_news.copy()

        # Sentiment
        if self._sentiment_analyzer is not None:
            df = self._sentiment_analyzer.transform(df)

        # Vectorisation
        if self._vectorizer is not None:
            df = self._vectorizer.transform(df)

        # Optionally aggregate to daily level
        if self.aggregate and "headline_sentiment" in df.columns:
            df = aggregate_daily_sentiment(df)

        return df

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def raw_news(self) -> Optional[pd.DataFrame]:
        """Access the raw scraped articles (available after fit)."""
        return self._raw_news


# ---------------------------------------------------------------------------
# Top-level convenience functions
# ---------------------------------------------------------------------------

def get_news(
    ticker: Union[str, List[str]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sources: Optional[List[str]] = None,
    company_name: Optional[str] = None,
    sentiment_method: Literal["vader", "finbert"] = "vader",
    vectorize_method: Optional[Literal["tfidf", "embedding", "finbert_embedding"]] = None,
    aggregate: bool = False,
    device: str = "cpu",
    use_cache: bool = True,
    max_workers: int = 4,
    result_cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Scrape news, compute sentiment, and return a DataFrame.

    Accepts a **single ticker** (str) or a **list of tickers** – when
    given a list the function scrapes all tickers concurrently and loads
    the sentiment/embedding model only once.

    Usage
    -----
    >>> from src.data.news import get_news
    >>> # Single ticker
    >>> df = get_news("AAPL", start_date="2025-01-01", end_date="2025-03-01")
    >>> # Multiple tickers – efficient, models loaded once
    >>> df = get_news(
    ...     ["AAPL", "MSFT", "GOOGL"],
    ...     start_date="2021-01-01", end_date="2025-03-01",
    ...     vectorize_method="embedding", aggregate=True,
    ... )

    Parameters
    ----------
    ticker : str or list[str]
        One or more stock ticker symbols.
    start_date, end_date : str or None
    sources : list[str] or None
    company_name : str or None
        Ignored when *ticker* is a list.
    sentiment_method : 'vader' | 'finbert'
    vectorize_method : None | 'tfidf' | 'embedding'
    aggregate : bool
    device : str
    use_cache : bool
        If True, read/write scraped articles from disk cache.
    max_workers : int
        Number of threads for concurrent scraping (only used when
        *ticker* is a list).  Default 4.

    Returns
    -------
    pd.DataFrame
        If a list of tickers is given, all tickers are concatenated
        and sorted by (ticker, date).
    """
    # ── single ticker shortcut ──
    if isinstance(ticker, str):
        return _get_news_single(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            sources=sources,
            company_name=company_name,
            sentiment_method=sentiment_method,
            vectorize_method=vectorize_method,
            aggregate=aggregate,
            device=device,
            use_cache=use_cache,
        )

    # ── multiple tickers ──
    tickers = list(ticker)
    if not tickers:
        return pd.DataFrame()

    return _get_news_batch(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        sources=sources,
        sentiment_method=sentiment_method,
        vectorize_method=vectorize_method,
        aggregate=aggregate,
        device=device,
        use_cache=use_cache,
        max_workers=max_workers,
        result_cache_dir=result_cache_dir,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_news_single(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sources: Optional[List[str]] = None,
    company_name: Optional[str] = None,
    sentiment_method: str = "vader",
    vectorize_method: Optional[str] = None,
    aggregate: bool = False,
    device: str = "cpu",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Process a single ticker (used by both get_news and the batch path)."""
    pipe = NewsDataPipeline(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        sources=sources,
        company_name=company_name,
        sentiment_method=sentiment_method,
        vectorize_method=vectorize_method,
        aggregate=aggregate,
        device=device,
        use_cache=use_cache,
    )
    pipe.fit()
    return pipe.transform()


def _scrape_single_ticker(
    ticker: str,
    start_date: Optional[str],
    end_date: Optional[str],
    sources: Optional[List[str]],
    use_cache: bool,
) -> pd.DataFrame:
    """Scrape news for one ticker (thread-safe, used by batch scraper)."""
    try:
        return scrape_news(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            sources=sources,
            use_cache=use_cache,
        )
    except Exception as e:
        logger.warning("  ✗ %s scraping failed: %s", ticker, e)
        return pd.DataFrame()


def _get_news_batch(
    tickers: List[str],
    start_date: Optional[str],
    end_date: Optional[str],
    sources: Optional[List[str]],
    sentiment_method: str,
    vectorize_method: Optional[str],
    aggregate: bool,
    device: str,
    use_cache: bool,
    max_workers: int,
    result_cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Efficient multi-ticker pipeline:
      1. Check result cache (parquet) — if hit, return immediately
      2. Scrape all tickers concurrently (I/O-bound)
      3. Load sentiment + vectorizer models ONCE
      4. Process all articles in a single pass
      5. Aggregate per ticker if requested
      6. Save result to cache for next time
    """
    # ── Result cache: skip all inference if we have a saved result ──
    cache_path = None
    if result_cache_dir is not None:
        cache_dir = Path(result_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Build a deterministic cache key from the call parameters
        key_str = f"{sorted(tickers)}_{start_date}_{end_date}_{sentiment_method}_{vectorize_method}_{aggregate}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]
        cache_path = cache_dir / f"news_result_{key_hash}.parquet"
        if cache_path.exists():
            logger.info("Result cache HIT: %s", cache_path)
            return pd.read_parquet(cache_path)
    total = len(tickers)
    logger.info("Batch processing %d tickers …", total)

    # ── Step 1: concurrent scraping ──
    raw_frames: List[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_ticker = {
            pool.submit(
                _scrape_single_ticker, t, start_date, end_date, sources, use_cache,
            ): t
            for t in tickers
        }
        for i, future in enumerate(as_completed(future_to_ticker), 1):
            t = future_to_ticker[future]
            try:
                df = future.result()
                if not df.empty:
                    raw_frames.append(df)
                    logger.info("[%d/%d] %s → %d articles", i, total, t, len(df))
                else:
                    logger.info("[%d/%d] %s → 0 articles", i, total, t)
            except Exception as e:
                logger.warning("[%d/%d] %s failed: %s", i, total, t, e)

    if not raw_frames:
        logger.warning("No articles scraped for any ticker")
        return pd.DataFrame()

    all_news = pd.concat(raw_frames, ignore_index=True)
    logger.info("Total raw articles: %d across %d tickers", len(all_news), total)

    # ── Step 2: load models once ──
    sent_analyzer = _get_sentiment_analyzer(sentiment_method, device)
    vectorizer = _get_vectorizer(
        vectorize_method, device=device,
    )

    # ── Step 3: single-pass sentiment + vectorisation ──
    all_news = sent_analyzer.transform(all_news)
    if vectorizer is not None:
        all_news = vectorizer.transform(all_news)

    # ── Free heavy models from memory after inference ──
    import gc
    for key in list(_MODEL_CACHE.keys()):
        del _MODEL_CACHE[key]
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("Models freed from memory")

    # ── Step 4: aggregate per ticker if requested ──
    if aggregate and "headline_sentiment" in all_news.columns:
        all_news = aggregate_daily_sentiment(all_news)

    all_news = all_news.sort_values(["ticker", "date"]).reset_index(drop=True)
    logger.info("Batch complete: %d tickers, %d total rows", total, len(all_news))

    # ── Save to result cache ──
    if cache_path is not None:
        try:
            all_news.to_parquet(cache_path, index=False)
            logger.info("Result cached to %s", cache_path)
        except Exception as e:
            logger.warning("Failed to write result cache: %s", e)

    return all_news


# ---------------------------------------------------------------------------
# Legacy alias (kept for backward compatibility)
# ---------------------------------------------------------------------------

def batch_get_news(
    tickers: List[str],
    start_date: str,
    end_date: str,
    sources: Optional[List[str]] = None,
    sentiment_method: Literal["vader", "finbert"] = "vader",
    vectorize_method: Optional[Literal["tfidf", "embedding", "finbert_embedding"]] = None,
    aggregate: bool = False,
    device: str = "cpu",
    use_cache: bool = True,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Scrape news + sentiment for **multiple tickers** in one call.

    .. deprecated::
        Use ``get_news(tickers, ...)`` directly – it now accepts a list.

    Parameters
    ----------
    tickers : list[str]
    start_date, end_date : str
    sources : list[str] or None
    sentiment_method : 'vader' | 'finbert'
    vectorize_method : None | 'tfidf' | 'embedding'
    aggregate : bool
    device : str
    use_cache : bool
    max_workers : int

    Returns
    -------
    pd.DataFrame  – all tickers concatenated, sorted by (ticker, date).
    """
    return get_news(
        ticker=tickers,
        start_date=start_date,
        end_date=end_date,
        sources=sources,
        sentiment_method=sentiment_method,
        vectorize_method=vectorize_method,
        aggregate=aggregate,
        device=device,
        use_cache=use_cache,
        max_workers=max_workers,
    )
