"""Embedding backends for retrieval.

* :class:`TFIDFEmbeddings` — pure-stdlib TF-IDF encoder. Always available;
  used when ``vector_database: tfidf`` (the default in ``config.yaml``).
* :class:`SentenceTransformerEmbeddings` — lazy wrapper around
  ``sentence_transformers``; imported only when instantiated.
* :func:`get_embedding_model` — factory that picks the backend from config,
  falling back to TF-IDF with a printed warning when the sentence-transformer
  stack is missing.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List

_STRIP_CHARS = ".,;:!?()\"'«»—–-…`´‘’“”[]{}|/\\@#%^&*+=~$:"


def _tokenize(text: str) -> List[str]:
    """Whitespace split + edge-punctuation strip (Unicode-safe).

    Keeps Indic combining marks (matras) attached to base letters so
    Devanagari words (e.g. 'रखने') survive as single tokens. Drops tokens
    shorter than 3 chars.
    """
    toks = []
    for raw in text.lower().split():
        tok = raw.strip(_STRIP_CHARS)
        if len(tok) >= 3:
            toks.append(tok)
    return toks


class TFIDFEmbeddings:
    """Sparse TF-IDF embeddings, stdlib only.

    ``fit(texts)`` builds the vocabulary and IDF weights; ``encode(texts)``
    returns L2-normalized sparse vectors as ``{term: weight}`` dicts.
    """

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._fitted = False

    def fit(self, texts: List[str]) -> "TFIDFEmbeddings":
        doc_freq: Dict[str, int] = {}
        n = 0
        for text in texts:
            n += 1
            for tok in set(_tokenize(text)):
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
        self.vocab = {tok: i for i, tok in enumerate(sorted(doc_freq))}
        # smoothed idf: log((1 + N) / (1 + df)) + 1
        self.idf = {tok: math.log((1 + n) / (1 + df)) + 1.0
                    for tok, df in doc_freq.items()}
        self._fitted = True
        return self

    def encode(self, texts: List[str]) -> List[Dict[str, float]]:
        if not self._fitted:
            raise RuntimeError("TFIDFEmbeddings.encode() called before fit()")
        vectors: List[Dict[str, float]] = []
        for text in texts:
            tf: Dict[str, int] = {}
            for tok in _tokenize(text):
                if tok in self.idf:
                    tf[tok] = tf.get(tok, 0) + 1
            vec = {tok: count * self.idf[tok] for tok, count in tf.items()}
            norm = math.sqrt(sum(w * w for w in vec.values()))
            if norm > 0:
                vec = {tok: w / norm for tok, w in vec.items()}
            vectors.append(vec)
        return vectors

    def dim(self) -> int:  # convenience: vocabulary size
        return len(self.vocab)


class SentenceTransformerEmbeddings:
    """Dense embeddings via ``sentence_transformers`` (optional dependency).

    Instantiation raises an informative ``ImportError`` when the package is
    not installed; ``get_embedding_model`` then falls back to TF-IDF.
    """

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed; cannot use "
                f"SentenceTransformerEmbeddings({model_name!r}). Install with: "
                "pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def fit(self, texts: List[str]) -> "SentenceTransformerEmbeddings":
        return self  # no fitting needed for pre-trained dense models

    def encode(self, texts: List[str]) -> List[Dict[str, float]]:
        import numpy as np

        arr = self._model.encode(texts, normalize_embeddings=True)
        arr = np.asarray(arr, dtype=float)
        return [{str(i): float(w) for i, w in enumerate(row)} for row in arr]

    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())


def get_embedding_model(cfg: dict):
    """Return the embedding backend selected by ``config.yaml``.

    ``vector_database: tfidf`` -> :class:`TFIDFEmbeddings`. Otherwise tries the
    sentence-transformer named by ``embedding_model`` and falls back to TF-IDF
    with a printed warning when it is unavailable.
    """
    if str(cfg.get("vector_database", "")).lower() == "tfidf":
        return TFIDFEmbeddings()
    model_name = cfg.get("embedding_model", "")
    try:
        return SentenceTransformerEmbeddings(model_name)
    except ImportError:
        print(f"[embeddings] WARNING: sentence-transformers unavailable; "
              f"falling back to TF-IDF (requested model: {model_name!r})")
        return TFIDFEmbeddings()
