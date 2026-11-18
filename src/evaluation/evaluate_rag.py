"""RAG retrieval evaluation (CLI).

Builds the pipeline's retriever on the train records, queries it with every
test question, and scores the ranked lists. A train record counts as relevant
to a test question when both share the same ``topic`` field.

Reports Recall@K, Precision@K and MRR for K in (1, 3, 5) — the upper bound
for retrieval K comes from ``top_k`` in config.yaml (evaluated K values are
capped at it).

Usage (from repo root)::

    python3 src/evaluation/evaluate_rag.py

Prints a table and saves ``experiments/rag_eval.json``.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config, repo_root  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    mrr, precision_at_k, recall_at_k)
from src.inference.pipeline import AdvisoryPipeline  # noqa: E402

EVAL_KS = (1, 3, 5)


def main() -> int:
    cfg = load_config()
    root = repo_root()
    pipe = AdvisoryPipeline()  # retriever built on train records internally
    top_k = int(cfg.get("top_k", 5))

    test_path = root / cfg.get("test_path", "data/test/test.jsonl")
    from src.inference.pipeline import load_records
    test_records = load_records(test_path)
    if not test_records:
        print(f"no test records at {test_path}; nothing to evaluate")
        return 0
    if not pipe.records:
        print("no train records; retriever is empty, skipping")
        return 0

    # topic -> train ids index for relevance judgements.
    topic_to_ids: dict[str, list[str]] = {}
    for i, rec in enumerate(pipe.records):
        topic = rec.get("topic") or "__unknown__"
        topic_to_ids.setdefault(topic, []).append(str(rec.get("id", i)))

    ranked_all, relevant_all = [], []
    per_k_hits = {k: 0 for k in EVAL_KS}
    for rec in test_records:
        topic = rec.get("topic") or "__unknown__"
        relevant = topic_to_ids.get(topic, [])
        hits = pipe.retrieve(rec.get("question", ""), max(EVAL_KS))
        ranked = [h["id"] for h in hits]
        ranked_all.append(ranked)
        relevant_all.append(relevant)
        for k in EVAL_KS:
            if set(ranked[:k]) & set(relevant):
                per_k_hits[k] += 1

    n = len(test_records)
    results = {}
    for k in EVAL_KS:
        kk = min(k, top_k)
        results[f"recall@{k}"] = sum(
            recall_at_k(r, rel, kk) for r, rel in zip(ranked_all, relevant_all)
        ) / n
        results[f"precision@{k}"] = sum(
            precision_at_k(r, rel, kk) for r, rel in zip(ranked_all, relevant_all)
        ) / n
        # Hit-rate@K (fraction of queries with >=1 relevant in top-k).
        results[f"hit_rate@{k}"] = per_k_hits[k] / n
    results["mrr"] = mrr(ranked_all, relevant_all)
    results["n_queries"] = n
    results["n_train_docs"] = len(pipe.records)

    out_dir = root / Path(cfg.get("experiments_csv",
                                  "experiments/results.csv")).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rag_eval.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nRAG retrieval eval ({n} queries, {len(pipe.records)} train docs)")
    print(f"  {'K':>3}  {'recall':>8}  {'precision':>9}  {'hit_rate':>8}")
    for k in EVAL_KS:
        print(f"  {k:>3}  {results[f'recall@{k}']:>8.4f}  "
              f"{results[f'precision@{k}']:>9.4f}  {results[f'hit_rate@{k}']:>8.4f}")
    print(f"  MRR: {results['mrr']:.4f}")
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
