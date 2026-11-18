"""End-to-end answer-quality evaluation (CLI).

Runs :class:`src.inference.pipeline.AdvisoryPipeline` over every question in
the test split and scores the generated answers against the reference answers
with :func:`levenshtein_mean` (competition metric), mean ROUGE-L F1 and mean
BLEU. The pipeline runs in retrieval-only mode automatically when no model
weights are available, so this script needs no GPU.

Usage (from repo root)::

    python3 src/evaluation/evaluate.py

Writes ``experiments/eval_report.json`` (directory taken from
``experiments_csv`` in config.yaml) and prints a summary.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config, repo_root  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bleu_score, levenshtein_mean, rouge_l_f1)
from src.inference.pipeline import AdvisoryPipeline, load_records  # noqa: E402


def _experiments_dir(cfg: dict) -> Path:
    exp_csv = cfg.get("experiments_csv", "experiments/results.csv")
    d = repo_root() / Path(exp_csv).parent
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> int:
    cfg = load_config()
    root = repo_root()
    test_path = root / cfg.get("test_path", "data/test/test.jsonl")
    test_records = load_records(test_path)
    if not test_records:
        print(f"no test records at {test_path}; nothing to evaluate")
        report = {"n": 0, "note": "test set missing or empty",
                  "model_available": False}
        out = _experiments_dir(cfg) / "eval_report.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"wrote {out}")
        return 0

    pipe = AdvisoryPipeline()
    model_on = pipe.model.available()
    print(f"model available: {model_on} "
          f"({'generative' if model_on else 'retrieval-only'} mode)")
    print(f"evaluating {len(test_records)} test questions...")

    preds, refs = [], []
    rouge_scores, bleu_scores = [], []
    t0 = time.perf_counter()
    for i, rec in enumerate(test_records, 1):
        out = pipe.answer(rec.get("question", ""))
        pred, ref = out["answer"], rec.get("answer", "")
        preds.append(pred)
        refs.append(ref)
        rouge_scores.append(rouge_l_f1(pred, ref))
        bleu_scores.append(bleu_score(pred, ref))
        if i % 10 == 0 or i == len(test_records):
            print(f"  {i}/{len(test_records)} done")
    elapsed = time.perf_counter() - t0

    report = {
        "n": len(test_records),
        "model_available": model_on,
        "mode": "generative" if model_on else "retrieval-only",
        "levenshtein_mean": levenshtein_mean(preds, refs),
        "rouge_l_f1_mean": sum(rouge_scores) / len(rouge_scores),
        "bleu_mean": sum(bleu_scores) / len(bleu_scores),
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_question": round(elapsed / len(test_records), 3),
    }
    out_path = _experiments_dir(cfg) / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print("\n--- evaluation summary ---")
    print(f"  questions        : {report['n']}")
    print(f"  mode             : {report['mode']}")
    print(f"  levenshtein_mean : {report['levenshtein_mean']:.4f}")
    print(f"  rouge_l_f1_mean  : {report['rouge_l_f1_mean']:.4f}")
    print(f"  bleu_mean        : {report['bleu_mean']:.4f}")
    print(f"  time/question    : {report['seconds_per_question']}s")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
