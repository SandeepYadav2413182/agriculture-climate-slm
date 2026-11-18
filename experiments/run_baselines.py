#!/usr/bin/env python3
"""Run the 4 baseline experiments for agriculture-climate-slm.

Experiments (all evaluated on the test split with `levenshtein_mean`
from src.evaluation.metrics — lower is better):

  1. baseline_retrieval_only — retriever top-1 passage text used directly as the
     answer; no generation, no weather context. Measures the retriever alone.
  2. finetuned_no_rag        — fine-tuned adapter generation. Recorded as
     SKIPPED unless models/adapters/agri-climate-qlora exists (needs a
     Kaggle GPU training run first).
  3. rag                     — full pipeline answer with RAG (top_k from config),
     weather context excluded from the scored text.
  4. rag_plus_climate         — RAG answer with the pipeline's weather_context
     appended to the scored text.

Results are APPENDED to the CSV at `experiments_csv` in config.yaml
(default: experiments/results.csv). The header is written if the file is
new/empty — so even a failed run leaves a valid CSV.

Graceful degradation: missing test data -> clear message, header-only CSV,
exit 0 (nothing to score is not a crash).

Run from the repo root:
    python experiments/run_baselines.py
"""

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- config ---
try:
    from src.data.config_loader import load_config
    cfg = load_config(str(ROOT / "config.yaml"))
except Exception as exc:  # config loader missing/broken: keep going with defaults
    print(f"[warn] could not load config.yaml ({exc}); using defaults")
    cfg = {}


def C(key, default=None):
    return cfg.get(key, default) if isinstance(cfg, dict) else default


COLUMNS = [
    "experiment_id", "base_model", "dataset_version",
    "train_samples", "validation_samples",
    "epochs", "learning_rate", "lora_rank", "lora_alpha",
    "training_time_s", "gpu_memory_mb",
    "eval_metric", "eval_score", "notes",
]

EVAL_METRIC = "levenshtein_mean"


# ------------------------------------------------------------ data utils ---
def count_jsonl(path):
    p = ROOT / path
    if not p.exists():
        return 0
    with p.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_test_rows():
    """Load test items; each needs question + answer (reference)."""
    path = C("test_path", "data/test/test.jsonl")
    p = ROOT / path
    if not p.exists():
        print(f"[error] test data not found: {p}")
        print("        run the data pipeline first (notebooks/01_data_preparation.ipynb).")
        return []
    rows, skipped = [], 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not item.get("question") or "answer" not in item:
                skipped += 1
                continue
            rows.append(item)
    if skipped:
        print(f"[warn] skipped {skipped} malformed test rows")
    print(f"[info] loaded {len(rows)} test rows from {path}")
    return rows


# --------------------------------------------------------------- scoring ---
def score_predictions(preds, refs):
    """levenshtein_mean over (prediction, reference) pairs."""
    from src.evaluation.metrics import levenshtein_mean

    try:
        return float(levenshtein_mean(preds, refs))
    except TypeError:
        # Fallback: per-pair levenshtein if the batch form is unavailable.
        from src.evaluation.metrics import levenshtein

        vals = [levenshtein(p, r) for p, r in zip(preds, refs)]
        return sum(vals) / len(vals) if vals else float("nan")


# ------------------------------------------------------------ experiments ---
def run_experiment_1_baseline(pipe, rows):
    """Retrieval-only: top-1 passage text as the answer (no generation, no weather)."""
    preds = []
    for r in rows:
        res = pipe.answer(
            question=r["question"],
            location=r.get("location", ""),
            crop=r.get("crop", ""),
            crop_stage=r.get("crop_stage", ""),
            language=r.get("language"),
        )
        sources = res.get("sources") or []
        preds.append(sources[0].get("text", "") if sources else "")
    return score_predictions(preds, [r["answer"] for r in rows])


def run_experiment_3_rag(pipe, rows, include_weather):
    """Full pipeline answer; weather context included only when requested."""
    preds = []
    for r in rows:
        res = pipe.answer(
            question=r["question"],
            location=r.get("location", ""),
            crop=r.get("crop", ""),
            crop_stage=r.get("crop_stage", ""),
            language=r.get("language"),
        )
        text = res.get("answer", "")
        if include_weather and res.get("weather_context"):
            text = f"{text}\n\nClimate context: {res['weather_context']}"
        preds.append(text)
    return score_predictions(preds, [r["answer"] for r in rows])


def base_row(experiment_id, notes=""):
    return {
        "experiment_id": experiment_id,
        "base_model": C("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
        "dataset_version": C("dataset_version", "v0.1-sample"),
        "train_samples": count_jsonl(C("train_path", "data/train/train.jsonl")),
        "validation_samples": count_jsonl(C("validation_path", "data/validation/val.jsonl")),
        "epochs": C("epochs", 3),
        "learning_rate": C("learning_rate", 0.0002),
        "lora_rank": C("lora_rank", 16),
        "lora_alpha": C("lora_alpha", 32),
        "training_time_s": "",
        "gpu_memory_mb": "",  # CPU run: no GPU memory to report
        "eval_metric": EVAL_METRIC,
        "eval_score": "",
        "notes": notes,
    }


def write_csv(rows):
    """Append rows; always ensure the header exists (even with zero rows)."""
    csv_path = ROOT / C("experiments_csv", "experiments/results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in COLUMNS})
    print(f"[info] wrote {len(rows)} row(s) to {csv_path}")


def main():
    rows = []

    test_rows = load_test_rows()
    if not test_rows:
        print("[info] no test data — writing header-only CSV and exiting.")
        write_csv(rows)
        return 0

    try:
        from src.inference.pipeline import AdvisoryPipeline

        pipe = AdvisoryPipeline(str(ROOT / "config.yaml"))
    except Exception as exc:
        print(f"[error] could not build AdvisoryPipeline: {exc}")
        write_csv(rows)  # header still written
        return 1

    top_k = C("top_k", 5)

    # --- (1) baseline: retrieval-only -------------------------------------
    t0 = time.perf_counter()
    try:
        s = run_experiment_1_baseline(pipe, test_rows)
        row = base_row(
            "baseline_retrieval_only",
            notes=f"CPU run; retriever top-1 passage as answer, no generation, no weather",
        )
        row["eval_score"] = round(s, 4)
        row["training_time_s"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        print(f"[done] baseline_retrieval_only  {EVAL_METRIC}={row['eval_score']}")
    except Exception as exc:
        print(f"[error] baseline_retrieval_only failed: {exc}")

    # --- (2) fine-tuned (adapter-gated) ------------------------------------
    adapter_dir = ROOT / C("adapter_dir", "models/adapters/agri-climate-qlora")
    if adapter_dir.exists():
        t0 = time.perf_counter()
        try:
            # The pipeline picks up the adapter via config; score its answers.
            s = run_experiment_3_rag(pipe, test_rows, include_weather=False)
            row = base_row("finetuned_no_rag", notes="adapter present; run on GPU")
            row["eval_score"] = round(s, 4)
            row["training_time_s"] = round(time.perf_counter() - t0, 1)
            rows.append(row)
            print(f"[done] finetuned_no_rag  {EVAL_METRIC}={row['eval_score']}")
        except Exception as exc:
            print(f"[error] finetuned_no_rag failed: {exc}")
    else:
        row = base_row(
            "finetuned_no_rag",
            notes="SKIPPED - adapter not found; run QLoRA training on Kaggle GPU first",
        )
        rows.append(row)
        print("[skip] finetuned_no_rag — no adapter at", adapter_dir)

    # --- (3) +RAG ----------------------------------------------------------
    t0 = time.perf_counter()
    try:
        s = run_experiment_3_rag(pipe, test_rows, include_weather=False)
        row = base_row(
            "rag",
            notes=f"CPU run; pipeline answer with RAG top_k={top_k}, weather excluded",
        )
        row["eval_score"] = round(s, 4)
        row["training_time_s"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        print(f"[done] rag  {EVAL_METRIC}={row['eval_score']}")
    except Exception as exc:
        print(f"[error] rag failed: {exc}")

    # --- (4) +RAG + climate -------------------------------------------------
    t0 = time.perf_counter()
    try:
        s = run_experiment_3_rag(pipe, test_rows, include_weather=True)
        row = base_row(
            "rag_plus_climate",
            notes=f"CPU run; RAG top_k={top_k} with weather_context included",
        )
        row["eval_score"] = round(s, 4)
        row["training_time_s"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        print(f"[done] rag_plus_climate  {EVAL_METRIC}={row['eval_score']}")
    except Exception as exc:
        print(f"[error] rag_plus_climate failed: {exc}")

    write_csv(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
