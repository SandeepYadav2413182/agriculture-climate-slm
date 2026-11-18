"""CPU baseline: multinomial Naive Bayes topic classifier (stdlib only).

Trains on the train-split questions -> topic labels and evaluates on the
test split. This complements the GPU QLoRA path (src/training/train.py):
a fast, transparent, zero-dependency baseline for the topic taxonomy.

Features: word unigrams + bigrams from the question, using the pipeline's
Unicode-safe tokenizer (Devanagari-safe).

Usage (from repo root):
    python3 -m src.training.baseline_nb
Saves the model to models/baseline_nb.json and prints test metrics.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config, repo_root  # noqa: E402
from src.data.deduplicate import tokenize  # noqa: E402


def featurize(text: str) -> list[str]:
    toks = tokenize(text)
    feats = list(toks)
    feats += [toks[i] + "\u0001" + toks[i + 1] for i in range(len(toks) - 1)]
    return feats


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def train(records: list[dict], alpha: float = 1.0) -> dict:
    topics = sorted({r["topic"] for r in records})
    doc_count: Counter = Counter()
    feat_count: dict[str, Counter] = {t: Counter() for t in topics}
    feat_total: Counter = Counter()
    vocab: set[str] = set()
    for r in records:
        t = r["topic"]
        doc_count[t] += 1
        for f in featurize(r["question"]):
            feat_count[t][f] += 1
            feat_total[t] += 1
            vocab.add(f)
    n_docs = len(records)
    n_vocab = len(vocab)
    log_prior = {t: math.log(doc_count[t] / n_docs) for t in topics}
    log_cond: dict[str, dict[str, float]] = {}
    for t in topics:
        denom = feat_total[t] + alpha * n_vocab
        log_cond[t] = {
            f: math.log((feat_count[t][f] + alpha) / denom) for f in vocab
        }
        log_cond[t]["<UNK>"] = math.log(alpha / denom)
    return {
        "topics": topics,
        "vocab": sorted(vocab),
        "alpha": alpha,
        "log_prior": log_prior,
        "log_cond": log_cond,
        "n_train": n_docs,
    }


def predict(model: dict, text: str, top_k: int = 1) -> list[tuple[str, float]]:
    feats = featurize(text)
    vocab = set(model["vocab"])
    scores = []
    for t in model["topics"]:
        s = model["log_prior"][t]
        lc = model["log_cond"][t]
        for f in feats:
            s += lc.get(f, lc["<UNK>"])
        scores.append((t, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def evaluate(model: dict, records: list[dict]) -> dict:
    topics = model["topics"]
    t2i = {t: i for i, t in enumerate(topics)}
    correct = 0
    top3_correct = 0
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    lang_correct: Counter = Counter()
    lang_total: Counter = Counter()
    for r in records:
        gold = r["topic"]
        ranked = predict(model, r["question"], top_k=3)
        pred = ranked[0][0]
        if pred == gold:
            correct += 1
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1
        if gold in [t for t, _ in ranked]:
            top3_correct += 1
        lang_total[r.get("language", "?")] += 1
        if pred == gold:
            lang_correct[r.get("language", "?")] += 1
    n = len(records)
    f1s = []
    for t in topics:
        denom = 2 * tp[t] + fp[t] + fn[t]
        f1s.append(2 * tp[t] / denom if denom else 0.0)
    return {
        "n_test": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "top3_accuracy": round(top3_correct / n, 4) if n else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "per_language_accuracy": {
            l: round(lang_correct[l] / lang_total[l], 4) for l in sorted(lang_total)
        },
    }


def main() -> None:
    cfg = load_config()
    root = repo_root()
    train_recs = load_jsonl(root / cfg["train_path"])
    test_recs = load_jsonl(root / cfg["test_path"])
    print(f"[baseline_nb] train={len(train_recs)} test={len(test_recs)}")

    t0 = __import__("time").time()
    model = train(train_recs)
    train_s = __import__("time").time() - t0
    print(f"[baseline_nb] trained on {model['n_train']} records "
          f"({len(model['topics'])} topics, {len(model['vocab'])} features) "
          f"in {train_s:.1f}s")

    metrics = evaluate(model, test_recs)
    print("[baseline_nb] test metrics:", json.dumps(metrics, indent=2))

    out = root / "models" / "baseline_nb.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    print(f"[baseline_nb] model saved to {out}")


if __name__ == "__main__":
    main()
