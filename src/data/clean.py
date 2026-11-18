"""Clean raw records: normalize text, enforce schema, drop empties.

Stdlib only. Every string field is stripped; whitespace inside
question/context/answer is collapsed to single spaces. Records with an
empty question or answer are dropped (they carry no training signal).
Language is normalized to ``"hi"`` when Devanagari script is detected,
otherwise the original value is kept. Crop names are lowercased for
consistent grouping. All 17 expected fields are guaranteed to exist,
with missing optional fields filled with ``""``.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPECTED_FIELDS = [
    "id", "question", "context", "answer", "crop", "crop_stage",
    "location", "soil_type", "climate_condition", "weather_condition",
    "language", "topic", "source", "source_type", "confidence",
    "safety_level",
]

_WS_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    return _WS_RE.sub(" ", text).strip()


def _is_devanagari(text: str) -> bool:
    """True if any character is in the Devanagari Unicode block."""
    return any("\u0900" <= ch <= "\u097f" for ch in text)


def clean_records(records: list[dict]) -> tuple[list[dict], dict]:
    """Clean records and return ``(cleaned_records, report)``.

    Report keys: ``n_input``, ``n_output``, ``n_dropped_empty_qa``,
    ``missing_counts`` (per-field counts of values that were missing or
    empty before cleaning).
    """
    cleaned: list[dict] = []
    missing_counts = {field: 0 for field in EXPECTED_FIELDS}
    n_dropped_empty_qa = 0

    for rec in records:
        row: dict = {}
        for field in EXPECTED_FIELDS:
            value = rec.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_counts[field] += 1
                row[field] = ""
            elif isinstance(value, str):
                row[field] = value.strip()
            else:
                row[field] = str(value)

        # Normalize whitespace in the free-text fields.
        for field in ("question", "context", "answer"):
            row[field] = _normalize_whitespace(row[field])

        # Drop records with no question or no answer.
        if not row["question"] or not row["answer"]:
            n_dropped_empty_qa += 1
            continue

        # Language: Devanagari script => Hindi, otherwise keep as-is.
        if _is_devanagari(row["question"] + row["answer"]):
            row["language"] = "hi"
        else:
            row["language"] = row["language"].strip().lower() or "en"

        # Canonicalize crop names.
        row["crop"] = row["crop"].lower()

        cleaned.append(row)

    report = {
        "n_input": len(records),
        "n_output": len(cleaned),
        "n_dropped_empty_qa": n_dropped_empty_qa,
        "missing_counts": missing_counts,
    }
    return cleaned, report
