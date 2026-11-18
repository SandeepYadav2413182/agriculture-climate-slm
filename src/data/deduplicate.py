"""Deduplicate records by exact and near-duplicate question text.

Two passes, both keep-first:
1. Exact duplicates: normalized question (lowercased, whitespace-collapsed)
   seen before is dropped.
2. Near duplicates: token Jaccard similarity on the question is computed
   against every surviving record; if similarity >= threshold the record
   is dropped and the matching pair is reported.

Tokenization: lowercase alphanumeric tokens of length >= 3. Stdlib only.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_STRIP_CHARS = ".,;:!?()\"'«»—–-…`´‘’“”[]{}|/\\@#%^&*+=~$:"

def _raw_tokenize(text: str) -> list[str]:
    """Split on whitespace and strip edge punctuation (Unicode-safe).

    Unlike ``\\w``-based matching this keeps Indic combining marks (matras)
    attached to their base letters, so Devanagari words stay intact.
    """
    toks = []
    for raw in text.lower().split():
        tok = raw.strip(_STRIP_CHARS)
        if len(tok) >= 3:
            toks.append(tok)
    return toks

# Public alias kept for backwards compatibility.
tokenize = _raw_tokenize


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity of the two strings' token sets (0..1)."""
    set_a, set_b = set(tokenize(a)), set(tokenize(b))
    if not set_a or not set_b:
        # No comparable tokens (e.g. empty strings): not duplicates.
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _norm_question(rec: dict) -> str:
    return re.sub(r"\s+", " ", rec.get("question", "").lower()).strip()


def deduplicate(
    records: list[dict], threshold: float
) -> tuple[list[dict], dict]:
    """Remove exact then near-duplicate records (keep-first).

    Returns ``(kept_records, report)`` with ``n_exact_dupes``,
    ``n_near_dupes``, ``near_duplicate_pairs`` (kept_id, dropped_id,
    similarity) and ``duplicate_rate``.
    """
    # Pass 1: exact duplicates on normalized question.
    seen: dict[str, dict] = {}
    survivors: list[dict] = []
    n_exact_dupes = 0
    for rec in records:
        key = _norm_question(rec)
        if key in seen:
            n_exact_dupes += 1
            continue
        seen[key] = rec
        survivors.append(rec)

    # Pass 2: near duplicates via token Jaccard on question text.
    # Token sets are precomputed once so the O(n^2) pairwise loop is pure
    # set arithmetic (identical results to calling jaccard() per pair).
    toksets = [frozenset(tokenize(rec.get("question", ""))) for rec in survivors]
    kept_idx: list[int] = []
    near_pairs: list[tuple[str, str, float]] = []
    for i, rec in enumerate(survivors):
        si = toksets[i]
        best_sim = 0.0
        best_keeper = None
        if si:
            for j in kept_idx:
                sj = toksets[j]
                if not sj:
                    continue
                inter = len(si & sj)
                if not inter:
                    continue
                sim = inter / (len(si) + len(sj) - inter)
                if sim > best_sim:
                    best_sim = sim
                    best_keeper = survivors[j]
        if best_sim >= threshold and best_keeper is not None:
            near_pairs.append(
                (best_keeper.get("id", "?"), rec.get("id", "?"),
                 round(best_sim, 4))
            )
            continue
        kept_idx.append(i)
    kept = [survivors[i] for i in kept_idx]

    n_near_dupes = len(near_pairs)
    n_input = len(records)
    report = {
        "n_input": n_input,
        "n_output": len(kept),
        "n_exact_dupes": n_exact_dupes,
        "n_near_dupes": n_near_dupes,
        "near_duplicate_pairs": near_pairs,
        "duplicate_rate": round(
            (n_exact_dupes + n_near_dupes) / n_input, 4
        ) if n_input else 0.0,
    }
    return kept, report
