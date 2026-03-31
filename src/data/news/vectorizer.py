"""
News text vectorizers.

Converts news headlines and descriptions into numeric feature vectors
using TF-IDF or dense neural embeddings (Sentence-Transformers / FinBERT).

All classes follow the sklearn BaseEstimator / TransformerMixin pattern
so they can be dropped into an sklearn Pipeline.

Exports: NewsTfidfVectorizer, NewsEmbeddingVectorizer, FinBERTEmbeddingVectorizer
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TF-IDF vectoriser wrapper
# ---------------------------------------------------------------------------

class NewsTfidfVectorizer(BaseEstimator, TransformerMixin):
    """
    Convert news text into TF-IDF feature vectors.

    Concatenates headline + description, fits a TF-IDF model, and returns
    the original DataFrame with additional `tfidf_0 … tfidf_N` columns.

    Parameters
    ----------
    headline_col : str
    description_col : str
    max_features : int
        Number of TF-IDF dimensions.
    ngram_range : tuple
        n-gram range for TF-IDF.
    prefix : str
        Column prefix for the TF-IDF features.
    """

    def __init__(
        self,
        headline_col: str = "headline",
        description_col: str = "description",
        max_features: int = 100,
        ngram_range: tuple = (1, 2),
        prefix: str = "tfidf",
    ):
        self.headline_col = headline_col
        self.description_col = description_col
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.prefix = prefix
        self._vectorizer: Optional[TfidfVectorizer] = None

    def fit(self, X: pd.DataFrame, y=None):
        corpus = self._build_corpus(X)
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
        )
        self._vectorizer.fit(corpus)
        logger.info(
            "TF-IDF vectorizer fitted with %d features", self.max_features
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._vectorizer is None:
            raise RuntimeError("You must call .fit() before .transform()")

        corpus = self._build_corpus(X)
        matrix = self._vectorizer.transform(corpus).toarray()
        cols = [f"{self.prefix}_{i}" for i in range(matrix.shape[1])]
        tfidf_df = pd.DataFrame(matrix, columns=cols, index=X.index)

        df = pd.concat([X, tfidf_df], axis=1)
        logger.info("TF-IDF features added: %d dimensions", matrix.shape[1])
        return df

    def _build_corpus(self, X: pd.DataFrame) -> List[str]:
        return (
            X[self.headline_col].fillna("")
            + " "
            + X[self.description_col].fillna("")
        ).tolist()


# ---------------------------------------------------------------------------
# Sentence-Transformer embedding wrapper
# ---------------------------------------------------------------------------

class NewsEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Encode news text into dense vectors using a Sentence-Transformer model.

    The model is loaded lazily; requires the `sentence-transformers` package.

    Parameters
    ----------
    headline_col : str
    description_col : str
    model_name : str
        Any model from https://www.sbert.net/docs/pretrained_models.html
    prefix : str
        Column prefix for the embedding features.
    batch_size : int
        Encoding batch size.
    device : str
        'cpu', 'cuda', or 'mps'.
    """

    def __init__(
        self,
        headline_col: str = "headline",
        description_col: str = "description",
        model_name: str = "all-MiniLM-L6-v2",
        prefix: str = "emb",
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.headline_col = headline_col
        self.description_col = description_col
        self.model_name = model_name
        self.prefix = prefix
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def fit(self, X: pd.DataFrame = None, y=None):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Sentence-Transformer model loaded: %s", self.model_name)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("You must call .fit() before .transform()")

        corpus = (
            X[self.headline_col].fillna("")
            + " "
            + X[self.description_col].fillna("")
        ).tolist()

        embeddings = self._model.encode(
            corpus,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        dim = embeddings.shape[1]
        cols = [f"{self.prefix}_{i}" for i in range(dim)]
        emb_df = pd.DataFrame(embeddings, columns=cols, index=X.index)

        df = pd.concat([X, emb_df], axis=1)
        logger.info(
            "Sentence-Transformer embeddings added: %d dimensions", dim
        )
        return df


# ---------------------------------------------------------------------------
# FinBERT Embedding Vectorizer (financial-domain embeddings)
# ---------------------------------------------------------------------------

class FinBERTEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Extract dense CLS-token embeddings from the **FinBERT** model.

    Unlike generic sentence-transformer embeddings (all-MiniLM-L6-v2, 384-dim),
    FinBERT was fine-tuned on financial text.  The CLS hidden state (768-dim)
    captures financial semantics — earnings surprises, guidance changes,
    M&A language, regulatory risk — far better than a general-purpose encoder.

    Adds columns ``emb_0 … emb_767`` to the DataFrame.

    Parameters
    ----------
    headline_col : str
    description_col : str
    model_name : str
        HuggingFace model id.  Default ``ProsusAI/finbert``.
    prefix : str
        Column prefix for the embedding features.
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
        prefix: str = "emb",
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.headline_col = headline_col
        self.description_col = description_col
        self.model_name = model_name
        self.prefix = prefix
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._tokenizer = None

    def fit(self, X: pd.DataFrame = None, y=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(
            self.model_name, torch_dtype=torch.float16,
        )
        self._model.eval()

        # Move to device
        self._device = self.device
        if self._device == "mps" and not torch.backends.mps.is_available():
            self._device = "cpu"
        if self._device == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"
        self._model = self._model.to(self._device)

        logger.info("FinBERT embedding model loaded: %s (device=%s, dtype=float16)", self.model_name, self._device)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("You must call .fit() before .transform()")

        import gc
        import torch

        corpus = (
            X[self.headline_col].fillna("")
            + " "
            + X[self.description_col].fillna("")
        ).tolist()

        # Replace empty strings
        corpus = [t if t.strip() else "no content" for t in corpus]

        all_embeddings = []
        n_batches = (len(corpus) + self.batch_size - 1) // self.batch_size
        for i in range(0, len(corpus), self.batch_size):
            batch = corpus[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,           # shorter = less RAM (headlines are short)
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**encoded)

            # CLS token embedding → float32 numpy (from float16 tensors)
            cls_emb = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
            all_embeddings.append(cls_emb)

            # Free GPU/CPU tensors immediately
            del encoded, outputs
            if (i // self.batch_size) % 50 == 0:
                gc.collect()
                batch_num = i // self.batch_size + 1
                if batch_num % 100 == 0 or batch_num == 1:
                    logger.info("  FinBERT batch %d/%d", batch_num, n_batches)

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        del all_embeddings
        gc.collect()

        dim = embeddings.shape[1]
        cols = [f"{self.prefix}_{i}" for i in range(dim)]
        emb_df = pd.DataFrame(embeddings, columns=cols, index=X.index)
        del embeddings

        df = pd.concat([X, emb_df], axis=1)
        logger.info("FinBERT embeddings added: %d dimensions", dim)
        return df
