"""Text-generation and ranking metrics — stdlib only.

Provided metrics:

* :func:`rouge_l_f1` — ROUGE-L F1 from the longest common subsequence of
  word tokens.
* :func:`bleu_score` — corpus-style BLEU over a single prediction/reference
  pair, with simple smoothing for zero-count n-grams and the standard
  brevity penalty.
* :func:`levenshtein_mean` — the project's competition metric: the mean over
  (prediction, reference) pairs of ``1 - lev_dist / max_len``, where the
  distance is computed on **word tokens** (cheaper and more meaningful than
  character-level edit distance for paragraph-length answers).
* :func:`recall_at_k`, :func:`precision_at_k`, :func:`mrr` — ranking metrics
  over ranked id lists.

All functions are pure and deterministic. The config is loaded so evaluation
hyper-parameters stay centralized; ``seed`` is honoured wherever randomness
could enter (currently none of these metrics sample).
"""

import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config  # noqa: E402

_CFG = load_config()
_SEED = _CFG.get("seed", 42)  # reserved: kept for reproducibility of any
                               # future stochastic metric variants.


def _tokens(text: str) -> list:
    return (text or "").lower().split()


# ---------------------------------------------------------------------------
# ROUGE-L (LCS based)
# ---------------------------------------------------------------------------
def _lcs_length(a: list, b: list) -> int:
    """Length of the longest common subsequence (O(min) space DP)."""
    if len(a) > len(b):
        a, b = b, a
    prev = [0] * (len(a) + 1)
    for bj in b:
        cur = [0]
        for i, ai in enumerate(a, 1):
            if ai == bj:
                cur.append(prev[i - 1] + 1)
            else:
                cur.append(max(prev[i], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    """ROUGE-L F1 between prediction and reference (word-level LCS)."""
    p_tok, r_tok = _tokens(pred), _tokens(ref)
    if not p_tok or not r_tok:
        return 0.0
    lcs = _lcs_length(p_tok, r_tok)
    precision = lcs / len(p_tok)
    recall = lcs / len(r_tok)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# BLEU (single pair, smoothed)
# ---------------------------------------------------------------------------
def _ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(pred: str, ref: str, max_n: int = 4) -> float:
    """Smoothed BLEU for one (pred, ref) pair.

    Smoothing: a zero-count n-gram precision is replaced by
    ``1 / (total_n + 1)`` instead of zeroing the whole geometric mean
    (otherwise any unseen 4-gram would force BLEU to 0). Standard brevity
    penalty applies.
    """
    p_tok, r_tok = _tokens(pred), _tokens(ref)
    if not p_tok or not r_tok:
        return 0.0

    log_sum = 0.0
    for n in range(1, max_n + 1):
        p_ng, r_ng = _ngrams(p_tok, n), _ngrams(r_tok, n)
        total = sum(p_ng.values())
        if total == 0:
            continue  # prediction shorter than n: skip this order
        clipped = sum(min(c, r_ng[g]) for g, c in p_ng.items())
        if clipped == 0:
            prec = 1.0 / (total + 1)  # smoothing
        else:
            prec = clipped / total
        log_sum += math.log(prec)
    # Effective n = number of orders actually evaluated.
    eff_n = sum(1 for n in range(1, max_n + 1)
                if len(p_tok) >= n)
    if eff_n == 0:
        return 0.0
    bp = 1.0 if len(p_tok) >= len(r_tok) else math.exp(1 - len(r_tok) / len(p_tok))
    return bp * math.exp(log_sum / eff_n)


# ---------------------------------------------------------------------------
# Levenshtein mean (competition metric)
# ---------------------------------------------------------------------------
def _levenshtein(a: list, b: list) -> int:
    """Word-token Levenshtein distance (O(min) space)."""
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, bj in enumerate(b, 1):
        cur = [j]
        for i, ai in enumerate(a, 1):
            cost = 0 if ai == bj else 1
            cur.append(min(prev[i] + 1, cur[-1] + 1, prev[i - 1] + cost))
        prev = cur
    return prev[-1]


def levenshtein_mean(preds: list, refs: list) -> float:
    """Mean over pairs of ``1 - lev_dist / max_len`` (word tokens).

    Identical strings score exactly 1.0; an empty-vs-nonempty pair scores 0.0
    and two empty strings score 1.0. This is the project's competition metric.
    """
    if len(preds) != len(refs):
        raise ValueError("preds and refs must have equal length")
    if not preds:
        return 0.0
    total = 0.0
    for pred, ref in zip(preds, refs):
        p_tok, r_tok = _tokens(pred), _tokens(ref)
        denom = max(len(p_tok), len(r_tok))
        if denom == 0:
            total += 1.0
        else:
            total += 1.0 - _levenshtein(p_tok, r_tok) / denom
    return total / len(preds)


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------
def recall_at_k(ranked_ids: list, relevant_ids, k: int) -> float:
    """|relevant ∩ top-k| / |relevant| (0.0 when no relevant ids)."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def precision_at_k(ranked_ids: list, relevant_ids, k: int) -> float:
    """|relevant ∩ top-k| / k (0.0 when k == 0)."""
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    return len(set(ranked_ids[:k]) & relevant) / k


def mrr(ranked_lists: list, relevant_lists: list) -> float:
    """Mean reciprocal rank over queries (0.0 when a query has no hit)."""
    if len(ranked_lists) != len(relevant_lists):
        raise ValueError("ranked_lists and relevant_lists must have equal length")
    if not ranked_lists:
        return 0.0
    total = 0.0
    for ranked, relevant in zip(ranked_lists, relevant_lists):
        relevant = set(relevant)
        rr = 0.0
        for rank, doc_id in enumerate(ranked, 1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        total += rr
    return total / len(ranked_lists)


if __name__ == "__main__":
    # Sanity checks.
    assert levenshtein_mean(["hello world"], ["hello world"]) == 1.0
    assert levenshtein_mean(["a b c"], ["a b c"]) == 1.0
    assert levenshtein_mean([""], [""]) == 1.0
    assert 0.0 <= levenshtein_mean(["totally different text here"],
                                   ["nothing in common at all"]) <= 1.0
    assert rouge_l_f1("the cat sat", "the cat sat") == 1.0
    assert rouge_l_f1("x", "the cat sat on the mat") < 0.5
    assert bleu_score("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert 0.0 < bleu_score("the cat sat", "the cat sat on the mat") < 1.0
    assert recall_at_k(["a", "b", "c"], {"b", "d"}, 2) == 0.5
    assert precision_at_k(["a", "b", "c"], {"b", "d"}, 2) == 0.5
    assert mrr([["a", "b"]], [{"b"}]) == 0.5
    print("metrics sanity checks passed")
