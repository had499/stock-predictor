"""
News sentiment analysis.

Transforms news headlines and descriptions into sentiment scores using
VADER (fast, rule-based) or FinBERT (accurate, financial-domain neural model).
Also provides aggregate_daily_sentiment() for collapsing per-article scores
to daily summaries.

All classes follow the sklearn BaseEstimator / TransformerMixin pattern
so they can be dropped into an sklearn Pipeline.

Exports: VaderSentimentAnalyzer, FinBERTSentimentAnalyzer,
         aggregate_daily_sentiment
"""

import logging
from typing import Any, Dict, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VADER-based sentiment (fast, no GPU required)
# ---------------------------------------------------------------------------

class VaderSentimentAnalyzer(BaseEstimator, TransformerMixin):
    """
    Compute VADER compound sentiment scores for news text.

    Adds the following columns to the DataFrame:
        - headline_sentiment   : compound score for headline
        - description_sentiment: compound score for description
        - sentiment_positive   : positive component
        - sentiment_negative   : negative component
        - sentiment_neutral    : neutral component

    Parameters
    ----------
    headline_col : str
        Column containing the headline text.
    description_col : str
        Column containing the description text.
    """

    def __init__(
        self,
        headline_col: str = "headline",
        description_col: str = "description",
    ):
        self.headline_col = headline_col
        self.description_col = description_col
        self._analyzer = None

    def fit(self, X: pd.DataFrame, y=None):
        """Import VADER lazily so it is only needed at runtime."""
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk

        # Download lexicon on first use
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        self._analyzer = SentimentIntensityAnalyzer()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score each row's headline and description."""
        if self._analyzer is None:
            raise RuntimeError("You must call .fit() before .transform()")

        df = X.copy()

        # Headline sentiment
        headline_scores = (
            df[self.headline_col]
            .fillna("")
            .apply(self._analyzer.polarity_scores)
        )
        df["headline_sentiment"] = headline_scores.apply(lambda s: s["compound"])
        df["sentiment_positive"] = headline_scores.apply(lambda s: s["pos"])
        df["sentiment_negative"] = headline_scores.apply(lambda s: s["neg"])
        df["sentiment_neutral"] = headline_scores.apply(lambda s: s["neu"])

        # Description sentiment
        df["description_sentiment"] = (
            df[self.description_col]
            .fillna("")
            .apply(lambda t: self._analyzer.polarity_scores(t)["compound"])
        )

        logger.info("VADER sentiment computed for %d rows", len(df))
        return df


# ---------------------------------------------------------------------------
# FinBERT-based sentiment (more accurate for financial text)
# ---------------------------------------------------------------------------

class FinBERTSentimentAnalyzer(BaseEstimator, TransformerMixin):
    """
    Compute FinBERT sentiment scores for news text.

    Uses the ProsusAI/finbert model from Hugging Face, which was
    fine-tuned on financial news. Requires `transformers` and `torch`.

    Adds the following columns:
        - headline_sentiment     : weighted compound  (-1 … +1)
        - description_sentiment  : weighted compound  (-1 … +1)
        - finbert_positive       : P(positive)
        - finbert_negative       : P(negative)
        - finbert_neutral        : P(neutral)

    Parameters
    ----------
    headline_col : str
        Column containing the headline text.
    description_col : str
        Column containing the description text.
    model_name : str
        Hugging Face model identifier.
    batch_size : int
        Inference batch size.
    device : str
        'cpu', 'cuda', or 'mps'.
    """

    def __init__(
        self,
        headline_col: str = "headline",
        description_col: str = "description",
        model_name: str = "ProsusAI/finbert",
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.headline_col = headline_col
        self.description_col = description_col
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._pipeline = None

    def fit(self, X: pd.DataFrame = None, y=None):
        """Load the FinBERT model (lazy import)."""
        from transformers import pipeline as hf_pipeline

        self._pipeline = hf_pipeline(
            "sentiment-analysis",
            model=self.model_name,
            tokenizer=self.model_name,
            device=self.device if self.device != "cpu" else -1,
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT model loaded (%s)", self.model_name)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._pipeline is None:
            raise RuntimeError("You must call .fit() before .transform()")

        df = X.copy()

        # --- Headlines ---
        headlines = df[self.headline_col].fillna("").tolist()
        h_scores = self._score_batch(headlines)
        df["headline_sentiment"] = [s["compound"] for s in h_scores]
        df["finbert_positive"] = [s["positive"] for s in h_scores]
        df["finbert_negative"] = [s["negative"] for s in h_scores]
        df["finbert_neutral"] = [s["neutral"] for s in h_scores]

        # --- Descriptions ---
        descriptions = df[self.description_col].fillna("").tolist()
        d_scores = self._score_batch(descriptions)
        df["description_sentiment"] = [s["compound"] for s in d_scores]

        logger.info("FinBERT sentiment computed for %d rows", len(df))
        return df

    # ---- internal helpers ---------------------------------------------------

    def _score_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Run inference in batches and return normalised scores."""
        results: List[Dict[str, float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Replace empty strings to avoid tokenizer issues
            batch = [t if t.strip() else "no content" for t in batch]
            preds = self._pipeline(batch)
            for pred in preds:
                results.append(self._normalise(pred))
        return results

    @staticmethod
    def _normalise(pred: Dict[str, Any]) -> Dict[str, float]:
        """Convert FinBERT output to a standardised dict with a compound score."""
        label = pred["label"].lower()
        score = pred["score"]
        pos = score if label == "positive" else 0.0
        neg = score if label == "negative" else 0.0
        neu = score if label == "neutral" else 0.0
        compound = pos - neg  # simple compound in range [-1, 1]
        return {
            "positive": round(pos, 4),
            "negative": round(neg, 4),
            "neutral": round(neu, 4),
            "compound": round(compound, 4),
        }


# ---------------------------------------------------------------------------
# Convenience function: compute daily aggregate sentiment
# ---------------------------------------------------------------------------

def aggregate_daily_sentiment(
    news_df: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    sentiment_col: str = "headline_sentiment",
) -> pd.DataFrame:
    """
    Aggregate per-article sentiment into a daily summary per ticker.

    Returns a DataFrame with columns:
        date, ticker,
        sentiment_mean, sentiment_median, sentiment_std,
        sentiment_min, sentiment_max, news_count,
        emb_0 … emb_N  (daily-mean embedding, if embedding cols exist)

    Parameters
    ----------
    news_df : DataFrame
        Output of a sentiment analyzer (must contain *sentiment_col*).
    date_col : str
    ticker_col : str
    sentiment_col : str
    """
    if news_df.empty or sentiment_col not in news_df.columns:
        logger.warning("Cannot aggregate – DataFrame is empty or missing '%s'", sentiment_col)
        return pd.DataFrame()

    group_keys = [date_col, ticker_col]

    # ── sentiment stats ──
    agg = (
        news_df.groupby(group_keys)[sentiment_col]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
    )
    agg.columns = [
        date_col,
        ticker_col,
        "sentiment_mean",
        "sentiment_median",
        "sentiment_std",
        "sentiment_min",
        "sentiment_max",
        "news_count",
    ]
    agg["sentiment_std"] = agg["sentiment_std"].fillna(0)

    # ── daily-mean embeddings (if present) ──
    emb_cols = [c for c in news_df.columns if c.startswith("emb_")]
    if emb_cols:
        emb_agg = news_df.groupby(group_keys)[emb_cols].mean().reset_index()
        agg = agg.merge(emb_agg, on=group_keys, how="left")

    return agg
