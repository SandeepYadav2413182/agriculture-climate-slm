"""High-level retriever: embeds Q&A records, indexes them, answers queries.

Each record becomes one document: ``question + " " + context + " " + answer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - path setup
    sys.path.insert(0, str(ROOT))


def _doc_text(record: Dict) -> str:
    return " ".join(
        str(record.get(field, "") or "")
        for field in ("question", "context", "answer")
    ).strip()


class Retriever:
    """Embed + index + search over agriculture Q&A records."""

    def __init__(self, cfg: dict) -> None:
        from src.retrieval.embeddings import get_embedding_model
        from src.retrieval.index import build_index

        self.cfg = cfg
        self.embeddings = get_embedding_model(cfg)
        self.index = build_index(cfg)
        self.records: Dict[str, Dict] = {}
        self.texts: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def build(self, records: List[Dict]) -> "Retriever":
        """Index a list of records (each must carry an ``id`` field)."""
        ids, texts = [], []
        for rec in records:
            rid = str(rec.get("id"))
            ids.append(rid)
            self.records[rid] = rec
            text = _doc_text(rec)
            self.texts[rid] = text
            texts.append(text)
        self.embeddings.fit(texts)
        self.index.add(ids, self.embeddings.encode(texts))
        return self

    def query(self, text: str, top_k: Optional[int] = None) -> List[Dict]:
        """Return top-k hits as ``{id, score, text, record}``, best first."""
        k = int(top_k) if top_k is not None else int(self.cfg.get("top_k", 5))
        qvec = self.embeddings.encode([text])[0]
        hits = []
        for rid, score in self.index.search(qvec, k=k):
            hits.append({
                "id": rid,
                "score": round(score, 4),
                "text": self.texts[rid],
                "record": self.records[rid],
            })
        return hits

    # ------------------------------------------------------------------
    def save(self, directory: Optional[str] = None) -> Path:
        """Persist the index to ``index_dir`` (from config by default)."""
        import json

        directory = Path(directory or self.cfg["index_dir"])
        self.index.save(directory)
        (directory / "records.json").write_text(
            json.dumps(self.records, ensure_ascii=False), encoding="utf-8")
        (directory / "texts.json").write_text(
            json.dumps(self.texts, ensure_ascii=False), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, cfg: dict, directory: Optional[str] = None) -> "Retriever":
        """Load a previously saved retriever (embeddings are re-fit lazily)."""
        import json

        from src.retrieval.index import VectorIndex

        directory = Path(directory or cfg["index_dir"])
        retriever = cls(cfg)
        retriever.index = VectorIndex.load(directory)
        retriever.records = json.loads((directory / "records.json").read_text(encoding="utf-8"))
        retriever.texts = json.loads((directory / "texts.json").read_text(encoding="utf-8"))
        # Re-fit TF-IDF on stored texts so new queries embed consistently.
        retriever.embeddings.fit(list(retriever.texts.values()))
        return retriever


if __name__ == "__main__":  # pragma: no cover - toy smoke test
    from src.data.config_loader import load_config

    cfg = load_config()
    docs = [
        {"id": "d1",
         "question": "How do I control yellow rust in wheat?",
         "context": "Wheat disease management during the rabi season.",
         "answer": "Spray propiconazole 25 EC at 0.1% at first sign of yellow stripes."},
        {"id": "d2",
         "question": "When should I irrigate rice at tillering?",
         "context": "Rice water management in kharif.",
         "answer": "Maintain 2-5 cm standing water during active tillering stage."},
        {"id": "d3",
         "question": "Which fertilizer dose suits mustard sowing?",
         "context": "Mustard nutrient management for rabi sowing.",
         "answer": "Apply 40 kg N, 20 kg P2O5 and 20 kg K2O per hectare at sowing."},
    ]
    retriever = Retriever(cfg).build(docs)
    for q in ["wheat yellow rust spray treatment",
              "how much water does rice need while tillering"]:
        print(f"\nQUERY: {q}")
        for hit in retriever.query(q, top_k=2):
            print(f"  [{hit['score']:.4f}] {hit['id']}: "
                  f"{hit['record']['question']}")
