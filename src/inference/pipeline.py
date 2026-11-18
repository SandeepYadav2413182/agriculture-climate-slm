"""End-to-end advisory pipeline: classify -> retrieve -> weather -> generate.

:class:`AdvisoryPipeline` is the single entry point used by the demo app::

    from src.inference.pipeline import AdvisoryPipeline
    pipe = AdvisoryPipeline()
    result = pipe.answer(question, location, crop, crop_stage, language)

The pipeline works with NO model weights: when :class:`ModelWrapper` reports
unavailable, it falls back to retrieval-extractive answering (the best
retrieved record's answer, clearly labelled ``(retrieval-only mode)``).

External integrations are defensive on purpose:

* ``src.retrieval`` may not exist yet (built by a sibling module). If its
  ``Retriever`` cannot be imported/instantiated, a small stdlib
  :class:`KeywordRetriever` is used instead.
* ``src.climate.weather.get_weather_provider`` is probed with a couple of
  likely signatures; failure degrades to a "weather unavailable" string.
* A missing/empty train JSONL degrades to an empty retriever with a warning
  instead of raising.
"""

import json
import math
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config, repo_root  # noqa: E402
from src.inference.classifier import classify, detect_language  # noqa: E402
from src.inference.prompt import build_prompt  # noqa: E402
from src.inference.model import ModelWrapper  # noqa: E402
from src.safety.validator import validate_response  # noqa: E402


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------
def load_records(path) -> list:
    """Load JSONL dataset records; return [] (with warning) when missing."""
    path = Path(path)
    if not path.exists():
        warnings.warn(f"train records not found at {path}; retriever will be empty")
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                warnings.warn(f"skipping bad JSONL line {lineno} in {path}: {exc}")
    return records


def _record_text(rec: dict) -> str:
    """Indexable text for a dataset record (question + context + answer)."""
    parts = [str(rec.get("question", "")), str(rec.get("context", "")),
             str(rec.get("answer", ""))]
    for key in ("crop", "location", "crop_stage", "topic"):
        if rec.get(key):
            parts.append(str(rec[key]))
    return " ".join(p for p in parts if p)


def _tokenize(text: str) -> list:
    return re.findall(r"[a-zA-Z\u0900-\u097f]+|\d+", text.lower())


# ---------------------------------------------------------------------------
# Stdlib fallback retriever (used when src.retrieval is unavailable)
# ---------------------------------------------------------------------------
class KeywordRetriever:
    """Tiny TF-IDF-ish keyword retriever; stdlib only.

    Intended as a graceful fallback, not a replacement for the real
    ``src.retrieval`` module. ``search`` returns normalized hits:
    ``[{"id", "score", "text"}]`` sorted by descending score.
    """

    def __init__(self, records: list):
        self.records = records
        self._by_id = {str(r.get("id", i)): r for i, r in enumerate(records)}
        # Document frequencies for idf weighting.
        df = Counter()
        self._doc_tokens = []
        for rec in records:
            toks = set(_tokenize(_record_text(rec)))
            self._doc_tokens.append(toks)
            df.update(toks)
        n = max(len(records), 1)
        self._idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}

    def search(self, query: str, top_k: int = 5) -> list:
        q_toks = Counter(_tokenize(query))
        scored = []
        for rec, toks in zip(self.records, self._doc_tokens):
            score = sum(
                cnt * self._idf.get(t, 0.0)
                for t, cnt in q_toks.items() if t in toks
            )
            if score > 0:
                rec_id = str(rec.get("id", ""))
                scored.append({
                    "id": rec_id,
                    "score": float(score),
                    "text": _record_text(rec),
                    "record": rec,
                })
        scored.sort(key=lambda h: h["score"], reverse=True)
        return scored[:top_k]

    def get_record(self, rec_id: str):
        return self._by_id.get(str(rec_id))


def _try_external_retriever(records: list, cfg: dict):
    """Instantiate src.retrieval's Retriever(cfg) and build it; else None.

    The real interface is ``Retriever(cfg).build(records)`` with
    ``.query(text, top_k) -> [{id, score, text, record}]``. Anything that
    deviates (older/newer sibling implementation) falls back to the stdlib
    :class:`KeywordRetriever` with a warning.
    """
    try:
        from src.retrieval.retriever import Retriever as ExternalRetriever
    except Exception as exc:  # module may not exist yet
        warnings.warn(f"src.retrieval unavailable ({exc}); using KeywordRetriever")
        return None
    try:
        inst = ExternalRetriever(cfg)
        if not records:
            # An unbuilt index cannot serve queries (embeddings are unfit);
            # the empty keyword fallback handles this case gracefully.
            warnings.warn("no train records; using empty KeywordRetriever")
            return None
        inst.build(records)
        if not callable(getattr(inst, "query", None)):
            raise AttributeError("external Retriever has no query() method")
        return inst
    except Exception as exc:
        warnings.warn(f"could not use external Retriever ({exc}); "
                      "using KeywordRetriever")
        return None


def _try_weather_provider(cfg: dict):
    """Build the weather provider via src.climate.weather; None on failure."""
    try:
        from src.climate.weather import get_weather_provider
    except Exception as exc:
        warnings.warn(f"src.climate.weather unavailable ({exc}); no weather context")
        return None
    for attempt in (
        lambda: get_weather_provider(cfg),
        lambda: get_weather_provider(cfg.get("weather_provider", "mock")),
    ):
        try:
            return attempt()
        except Exception:
            continue
    warnings.warn("could not build weather provider; no weather context")
    return None


def _weather_context(provider, location) -> str:
    """Render a human-readable weather/climate context string."""
    if provider is None:
        return "Weather data unavailable for this query."
    try:
        current = provider.get_current(location or "India")
    except Exception:
        current = None
    if isinstance(current, dict) and current:
        lines = [
            f"- temperature: {current.get('temperature_c', '?')} C",
            f"- humidity: {current.get('humidity_pct', '?')} %",
            f"- rainfall (24h): {current.get('rainfall_mm_24h', '?')} mm",
            f"- wind: {current.get('wind_kph', '?')} kph",
            f"- summary: {current.get('forecast_summary', 'n/a')}",
            f"- source: {current.get('source', 'unknown')}",
        ]
        if str(current.get("source", "")).startswith("mock"):
            lines.append("(mock data: synthetic, not a real observation)")
        return "Current weather for {}:\n{}".format(
            location or "the region", "\n".join(lines))
    # Fallback for providers with a different interface.
    for meth in ("get_context", "describe", "for_location", "context"):
        if hasattr(provider, meth):
            try:
                return str(getattr(provider, meth)(location))
            except Exception:
                continue
    try:
        return str(provider)
    except Exception:
        return "Weather data unavailable for this query."


def _normalize_hits(hits, records_by_id: dict, top_k: int) -> list:
    """Normalize heterogeneous retriever hits to [{"id","score","text","record"}]."""
    norm = []
    for hit in hits or []:
        if isinstance(hit, dict):
            rec_id = str(hit.get("id", ""))
            score = float(hit.get("score", 0.0))
            text = hit.get("text")
            record = hit.get("record") or records_by_id.get(rec_id)
        elif isinstance(hit, (tuple, list)) and len(hit) >= 2:
            rec_id, score = str(hit[0]), float(hit[1])
            text = hit[2] if len(hit) > 2 else None
            record = records_by_id.get(rec_id)
        else:  # duck-typed object
            rec_id = str(getattr(hit, "id", ""))
            score = float(getattr(hit, "score", 0.0))
            text = getattr(hit, "text", None)
            record = getattr(hit, "record", None) or records_by_id.get(rec_id)
        if record is None:
            continue
        norm.append({
            "id": rec_id,
            "score": score,
            "text": text or _record_text(record),
            "record": record,
        })
    norm.sort(key=lambda h: h["score"], reverse=True)
    return norm[:top_k]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class AdvisoryPipeline:
    """Question-answering pipeline with retrieval-only graceful degradation."""

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.top_k = int(self.cfg.get("top_k", 5))
        self.default_language = self.cfg.get("default_language", "en")

        train_path = repo_root() / self.cfg.get(
            "train_path", "data/train/train.jsonl")
        self.records = load_records(train_path)
        self._records_by_id = {
            str(r.get("id", i)): r for i, r in enumerate(self.records)
        }

        external = _try_external_retriever(self.records, self.cfg)
        self.retriever = external if external is not None else KeywordRetriever(self.records)
        self._external_retriever = external is not None

        self.weather_provider = _try_weather_provider(self.cfg)
        self.model = ModelWrapper(self.cfg)

    # -- retrieval ------------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None) -> list:
        """Return normalized hits: [{"id","score","text","record"}]."""
        k = top_k or self.top_k
        if self._external_retriever:
            try:
                raw = self.retriever.query(question, k)
            except Exception as exc:
                # The external index can break at query time (e.g. unfit
                # embeddings); degrade permanently to the stdlib fallback.
                warnings.warn(f"external retriever query failed ({exc}); "
                              "switching to KeywordRetriever")
                self.retriever = KeywordRetriever(self.records)
                self._external_retriever = False
                raw = self.retriever.search(question, k)
        else:
            raw = self.retriever.search(question, k)
        return _normalize_hits(raw, self._records_by_id, k)

    # -- answering ------------------------------------------------------
    def answer(self, question, location=None, crop=None, crop_stage=None,
               language=None) -> dict:
        """Answer a farmer's question.

        Returns ``{"answer", "sources", "weather_context", "query_meta",
        "safety"}`` where ``safety`` is the validator report
        ``{"passed", "flags", "notes", "risk_level"}``.
        """
        lang = (language or detect_language(question) or self.default_language)
        meta = classify(question)
        if crop:
            meta["crop"] = crop
        if location:
            meta["location"] = location
        if crop_stage:
            meta["crop_stage"] = crop_stage
        meta["language"] = lang

        weather_context = _weather_context(self.weather_provider,
                                           meta.get("location"))
        hits = self.retrieve(question, self.top_k)
        sources = [
            {"id": h["id"], "score": h["score"],
             "text": h["record"].get("answer", h["text"])[:600]}
            for h in hits
        ]

        if self.model.available():
            prompt = build_prompt(question, hits, weather_context, meta, lang)
            answer_text = self.model.generate(prompt)
        else:
            answer_text = self._retrieval_only_answer(question, hits, lang)

        safety = validate_response(answer_text, meta)
        return {
            "answer": answer_text,
            "sources": sources,
            "weather_context": weather_context,
            "query_meta": meta,
            "safety": safety,
        }

    def _retrieval_only_answer(self, question: str, hits: list, lang: str) -> str:
        header = "(retrieval-only mode)"
        if not hits:
            body = ("I could not find relevant information in the knowledge base "
                    "for this question. Please rephrase it with the crop name, "
                    "location and crop stage, or contact your local Krishi "
                    "Vigyan Kendra / agriculture extension officer.")
            if lang == "hi":
                body = ("इस प्रश्न के लिए ज्ञान-आधार (knowledge base) में प्रासंगिक "
                        "जानकारी नहीं मिली। कृपया फसल का नाम, स्थान और फसल अवस्था "
                        "बताकर दोबारा पूछें, या अपने स्थानीय कृषि विज्ञान केंद्र / "
                        "कृषि विस्तार अधिकारी से संपर्क करें।")
            return f"{header}\n\n{body}"

        best = hits[0]["record"]
        best_answer = str(best.get("answer", "")).strip() or "(record has no answer text)"
        note_en = (
            "\n\n—\nNote: the language model is not available in this environment, "
            f"so this answer is extracted from the most relevant knowledge-base "
            f"record (id={hits[0]['id']}, score={hits[0]['score']:.3f}). It may not "
            "cover your exact situation; for a tailored recommendation please "
            "consult your local agriculture extension officer."
        )
        note_hi = (
            "\n\n—\nनोट: इस परिवेश में भाषा मॉडल उपलब्ध नहीं है, इसलिए यह उत्तर "
            f"सबसे प्रासंगिक ज्ञान-आधार रिकॉर्ड (id={hits[0]['id']}) से लिया गया है। "
            "यह आपकी सटीक स्थिति पर लागू न भी हो; अनुकूल सलाह के लिए कृपया अपने "
            "स्थानीय कृषि विस्तार अधिकारी से संपर्क करें।"
        )
        return f"{header}\n\n{best_answer}{note_hi if lang == 'hi' else note_en}"


if __name__ == "__main__":
    pipe = AdvisoryPipeline()
    print(f"model available: {pipe.model.available()} "
          f"({pipe.model.availability_reason()})")
    print(f"records loaded: {len(pipe.records)}; "
          f"external retriever: {pipe._external_retriever}")
    demo_q = sys.argv[1] if len(sys.argv) > 1 else \
        "My wheat has yellow leaves at flowering in Punjab, what should I do?"
    import pprint
    out = pipe.answer(demo_q)
    print("META:", out["query_meta"])
    print("ANSWER:\n", out["answer"][:800])
    print("SAFETY:", {k: v for k, v in out["safety"].items() if k != "notes"})
