# Experiment Report

> **STATUS: TEMPLATE — TO BE FILLED AFTER KAGGLE GPU RUNS.**
> All values below are placeholders. Do not treat them as results.

## Setup

| Field | Value |
|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Dataset version | v0.1-sample |
| Training config | QLoRA r=16, α=32, dropout 0.05, 4-bit, lr 0.0002, 3 epochs, batch 4 × grad-accum 4 |
| Eval metric | `levenshtein_mean` (competition metric: Levenshtein Mean; lower is better) |
| Test split | `data/test/test.jsonl` (15% of cleaned data, seed 42) |
| Hardware | Kaggle GPU (T4) for training; CPU for retrieval-only baselines |

## Baseline comparisons

| Experiment | Description | Eval metric | Score | Notes |
|---|---|---|---|---|
| `baseline_retrieval_only` | Retriever top-1 passage as answer; no generation, no weather | levenshtein_mean | TBD | CPU run |
| `finetuned_no_rag` | Fine-tuned adapter, generation without retrieval | levenshtein_mean | TBD | Requires `models/adapters/agri-climate-qlora`; skipped if adapter missing |
| `rag` | Pipeline answer with RAG (top_k=5), no weather context | levenshtein_mean | TBD | CPU run; retrieval-only mode if no adapter |
| `rag_plus_climate` | RAG answer + weather context included | levenshtein_mean | TBD | CPU run; `weather_provider: mock` in this config |

Rows are produced by `experiments/run_baselines.py` and appended to
`experiments/results.csv` (path from `experiments_csv` in `config.yaml`).

## Per-experiment detail (fill after runs)

### baseline_retrieval_only
- Score: TBD
- Training time (s): n/a (no training)
- Observations: TBD

### finetuned_no_rag
- Score: TBD
- Training time (s): TBD
- GPU memory (MB): TBD
- Observations: TBD

### rag
- Score: TBD
- Training time (s): n/a (no training)
- Observations: TBD

### rag_plus_climate
- Score: TBD
- Training time (s): n/a (no training)
- Observations: TBD

## Conclusions
TBD — to be written after the Kaggle GPU runs complete and `experiments/results.csv`
contains real scores.
