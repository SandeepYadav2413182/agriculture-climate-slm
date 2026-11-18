"""In-memory cosine-similarity vector index with JSON persistence.

Vectors are sparse ``{term: weight}`` dicts (as produced by
:mod:`src.retrieval.embeddings`). ``numpy`` is used when importable for the
inner product, otherwise pure-Python dict math gives identical results.
An optional FAISS path is enabled only when ``faiss`` imports AND
``vector_database: faiss`` in the config.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class VectorIndex:
    """Simple vector store: add(ids, vectors), search, save/load (JSON)."""

    def __init__(self, use_faiss: bool = False) -> None:
        self.ids: List[str] = []
        self.vectors: List[Dict[str, float]] = []
        self._use_faiss = use_faiss
        self._faiss_index = None
        self._faiss_dim = 0

    # ------------------------------------------------------------------
    def add(self, ids: List[str], vectors: List[Dict[str, float]]) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        self.ids.extend(ids)
        self.vectors.extend([dict(v) for v in vectors])
        if self._use_faiss:
            self._rebuild_faiss()

    def search(self, query_vec: Dict[str, float], k: int = 5
               ) -> List[Tuple[str, float]]:
        """Top-k ``(id, cosine_score)`` pairs, sorted by score descending."""
        if not self.ids:
            return []
        if self._use_faiss and self._faiss_index is not None:
            return self._search_faiss(query_vec, k)
        return self._search_python(query_vec, k)

    # ------------------------- pure-python search ----------------------
    def _search_python(self, query_vec: Dict[str, float], k: int):
        try:
            import numpy as np

            return self._search_numpy(query_vec, k, np)
        except ImportError:
            pass
        scored = []
        qnorm = math.sqrt(sum(w * w for w in query_vec.values()))
        for doc_id, vec in zip(self.ids, self.vectors):
            dot = sum(query_vec.get(t, 0.0) * w for t, w in vec.items())
            dnorm = math.sqrt(sum(w * w for w in vec.values()))
            score = dot / (qnorm * dnorm) if qnorm and dnorm else 0.0
            scored.append((doc_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(k, 0)]

    def _search_numpy(self, query_vec, k, np):
        # True cosine: the doc vector must contribute its FULL norm, not the
        # norm of its projection onto the query terms (the latter wrongly
        # scores every doc containing all query terms as 1.0).
        terms = list(query_vec.keys())
        q = np.array([query_vec[t] for t in terms], dtype=float)
        qnorm = float(np.linalg.norm(q))
        scored = []
        for doc_id, vec in zip(self.ids, self.vectors):
            d = np.array([vec.get(t, 0.0) for t in terms], dtype=float)
            dnorm = float(np.linalg.norm(np.array(list(vec.values()), dtype=float)))
            denom = qnorm * dnorm
            score = float(np.dot(q, d) / denom) if denom else 0.0
            scored.append((doc_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(k, 0)]

    # ------------------------- persistence -----------------------------
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "ids": self.ids,
            "vectors": self.vectors,
            "use_faiss": self._use_faiss,
        }
        path = directory / "index.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex":
        path = Path(directory) / "index.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        index = cls(use_faiss=bool(payload.get("use_faiss", False)))
        index.ids = list(payload["ids"])
        index.vectors = [dict(v) for v in payload["vectors"]]
        if index._use_faiss:
            index._rebuild_faiss()
        return index

    # ------------------------- optional FAISS --------------------------
    def _rebuild_faiss(self) -> None:
        import faiss
        import numpy as np

        vocab = sorted({t for v in self.vectors for t in v})
        self._faiss_dim = len(vocab) or 1
        mat = np.zeros((len(self.vectors), self._faiss_dim), dtype="float32")
        for i, vec in enumerate(self.vectors):
            for t, w in vec.items():
                mat[i, vocab.index(t)] = w
        self._faiss_index = faiss.IndexFlatIP(self._faiss_dim)
        self._faiss_index.add(mat)
        self._faiss_vocab = vocab

    def _search_faiss(self, query_vec, k):
        import numpy as np

        q = np.zeros((1, self._faiss_dim), dtype="float32")
        for t, w in query_vec.items():
            if t in self._faiss_vocab:
                q[0, self._faiss_vocab.index(t)] = w
        scores, idx = self._faiss_index.search(q, min(k, len(self.ids)))
        return [(self.ids[int(i)], float(scores[0][j]))
                for j, i in enumerate(idx[0]) if int(i) >= 0]


def build_index(cfg: dict) -> VectorIndex:
    """Create a VectorIndex honoring the FAISS config switch."""
    use_faiss = str(cfg.get("vector_database", "")).lower() == "faiss"
    if use_faiss:
        try:
            import faiss  # noqa: F401
        except ImportError:
            print("[index] WARNING: faiss not installed; using pure-python index")
            use_faiss = False
    return VectorIndex(use_faiss=use_faiss)
