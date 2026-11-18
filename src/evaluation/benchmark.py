"""Model-size / latency benchmark — no model weights required.

Two parts:

1. **Estimated**: parameter count for known models (lookup table, ``"unknown"``
   fallback), weight size in MB derived from the ``quantization`` setting in
   config.yaml (4bit -> 0.5 byte/param, 8bit -> 1, bf16/fp16 -> 2, fp32 -> 4),
   and a rough latency class (low < 2B params, medium < 8B, high otherwise).
   These are back-of-the-envelope figures for capacity planning, not measured
   timings.
2. **Measured**: real wall-clock timing of a stdlib TF-IDF fit + encode over
   sample agriculture texts (the retrieval path that actually runs on CPU
   here), reported as ms per document.

Usage (from repo root)::

    python3 src/evaluation/benchmark.py

Writes ``experiments/benchmark.json`` and prints a table.
"""

import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config, repo_root  # noqa: E402

# Known model -> parameter count. Extend as new base models are tried.
PARAM_COUNTS = {
    "Qwen/Qwen2.5-1.5B-Instruct": 1_540_000_000,
    "Qwen/Qwen2.5-0.5B-Instruct": 494_000_000,
    "Qwen/Qwen2.5-3B-Instruct": 3_090_000_000,
    "Qwen/Qwen2.5-7B-Instruct": 7_610_000_000,
    "Qwen/Qwen2.5-14B-Instruct": 14_800_000_000,
    "meta-llama/Llama-3.2-1B-Instruct": 1_240_000_000,
    "meta-llama/Llama-3.2-3B-Instruct": 3_210_000_000,
    "google/gemma-2-2b-it": 2_610_000_000,
}

BYTES_PER_PARAM = {
    "4bit": 0.5,
    "8bit": 1.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp32": 4.0,
}

# Sample texts for the measured TF-IDF benchmark.
SAMPLE_TEXTS = [
    "Yellow leaves in wheat at flowering often indicate nitrogen deficiency; "
    "apply urea only after a soil test and label guidance.",
    "धान की फसल में खरपतवार नियंत्रण के लिए समय पर निराई-गुड़ाई करें।",
    "Drip irrigation saves water for cotton in Maharashtra during dry spells.",
    "Whitefly on tomato spreads leaf curl virus; use yellow sticky traps.",
    "Monsoon onset in Kerala usually occurs in early June; plan paddy nursery.",
    "Soil pH below 5.5 reduces phosphorus availability for groundnut.",
    "Store harvested onions in ventilated sheds to reduce post-harvest losses.",
    "Frost in January can damage mustard at flowering; light irrigation helps.",
]


def _tok(text: str) -> list:
    return re.findall(r"[a-zA-Z\u0900-\u097f]+|\d+", text.lower())


def estimate_model(cfg: dict) -> dict:
    """Back-of-the-envelope size/latency estimate from config values."""
    model_name = cfg.get("model_name", "")
    quantization = str(cfg.get("quantization", "")).lower()
    params = PARAM_COUNTS.get(model_name, "unknown")
    bpp = BYTES_PER_PARAM.get(quantization)
    if isinstance(params, int) and bpp:
        size_mb = round(params * bpp / 1e6, 1)
    else:
        size_mb = "unknown"
    if isinstance(params, int):
        latency_class = ("low" if params < 2e9
                         else "medium" if params < 8e9 else "high")
    else:
        latency_class = "unknown"
    return {
        "model_name": model_name,
        "quantization": quantization,
        "params": params,
        "estimated_size_mb": size_mb,
        "latency_class": latency_class,  # rough: low<2B, medium<8B, high>=8B
        "note": "estimates only; not measured on hardware",
    }


def measure_tfidf(texts: list, dim: int = 4096, repeats: int = 5) -> dict:
    """Time a stdlib hashing-TF-IDF fit+encode; returns ms/doc stats."""
    def encode(doc: str):
        vec = {}
        for t in _tok(doc):
            h = hash(t) % dim
            vec[h] = vec.get(h, 0) + 1
        n = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / n for k, v in vec.items()}

    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        encoded = [encode(d) for d in texts]  # fit+encode in one pass
        timings.append((time.perf_counter() - t0) / len(texts) * 1000)
    assert all(len(v) > 0 for v in encoded)
    return {
        "n_docs": len(texts),
        "dim": dim,
        "repeats": repeats,
        "ms_per_doc_mean": round(sum(timings) / len(timings), 4),
        "ms_per_doc_min": round(min(timings), 4),
        "ms_per_doc_max": round(max(timings), 4),
    }


def main() -> int:
    cfg = load_config()
    root = repo_root()
    report = {
        "model_estimate": estimate_model(cfg),
        "tfidf_measured": measure_tfidf(SAMPLE_TEXTS),
    }
    out_dir = root / Path(cfg.get("experiments_csv",
                                  "experiments/results.csv")).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    est, tf = report["model_estimate"], report["tfidf_measured"]
    print("\n--- benchmark ---")
    print("model (estimated):")
    print(f"  name            : {est['model_name']}")
    print(f"  params          : {est['params']}")
    print(f"  quantization    : {est['quantization']}")
    print(f"  est. size       : {est['estimated_size_mb']} MB")
    print(f"  latency class   : {est['latency_class']} (rough)")
    print("TF-IDF fit+encode (measured, stdlib):")
    print(f"  docs/dim        : {tf['n_docs']} / {tf['dim']}")
    print(f"  ms/doc (mean)   : {tf['ms_per_doc_mean']}")
    print(f"  ms/doc (min-max): {tf['ms_per_doc_min']} - {tf['ms_per_doc_max']}")
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
