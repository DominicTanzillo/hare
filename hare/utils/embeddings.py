"""Text embedding utilities for HARE.

Provides a unified interface for encoding text into dense vectors,
with sentence-transformers as the primary backend and TF-IDF as a lightweight fallback.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class Embedder(Protocol):
    """Protocol for text embedding backends."""

    def encode(self, texts: list[str]) -> NDArray[np.floating]: ...

    @property
    def dim(self) -> int: ...


class TfidfEmbedder:
    """Lightweight TF-IDF embedder with optional SVD dimensionality reduction.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size.
    output_dim : int | None
        If set, reduce to this dimensionality via truncated SVD.
    """

    def __init__(self, max_features: int = 5000, output_dim: int | None = 128) -> None:
        self.max_features = max_features
        self.output_dim = output_dim
        self._vectorizer = TfidfVectorizer(max_features=max_features)
        self._svd = None
        self._fitted = False
        self._dim: int = output_dim or max_features

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, corpus: list[str]) -> TfidfEmbedder:
        """Fit the TF-IDF vocabulary (and optional SVD) on a corpus."""
        tfidf_matrix = self._vectorizer.fit_transform(corpus)

        if self.output_dim is not None:
            from sklearn.decomposition import TruncatedSVD

            n_components = min(self.output_dim, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
            self._svd = TruncatedSVD(n_components=n_components)
            self._svd.fit(tfidf_matrix)
            self._dim = n_components

        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> NDArray[np.floating]:
        """Encode texts into dense vectors.

        Returns
        -------
        array of shape (len(texts), dim)
            L2-normalized embeddings.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before encode()")

        tfidf_matrix = self._vectorizer.transform(texts)

        if self._svd is not None:
            embeddings = self._svd.transform(tfidf_matrix)
        else:
            embeddings = tfidf_matrix.toarray()

        return normalize(embeddings, norm="l2")


class SentenceTransformerEmbedder:
    """Sentence-transformer embedder using a pretrained model.

    Parameters
    ----------
    model_name : str
        HuggingFace model name for sentence-transformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        self._load()
        return self._dim  # type: ignore[return-value]

    def encode(self, texts: list[str]) -> NDArray[np.floating]:
        """Encode texts into dense vectors.

        Returns
        -------
        array of shape (len(texts), dim)
            L2-normalized embeddings.
        """
        self._load()
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)  # type: ignore[union-attr]
        return np.asarray(embeddings, dtype=np.float64)


def get_embedder(backend: str = "tfidf", **kwargs) -> Embedder:
    """Factory for embedding backends.

    Parameters
    ----------
    backend : str
        One of "tfidf" or "sentence-transformer".
    """
    if backend == "tfidf":
        return TfidfEmbedder(**kwargs)
    elif backend == "sentence-transformer":
        return SentenceTransformerEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'tfidf' or 'sentence-transformer'.")
