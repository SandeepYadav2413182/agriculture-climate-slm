# Architecture

End-to-end system diagram for `agriculture-climate-slm`. All parameters live in
`config.yaml` (single source of truth); the boxes below are the `src/` modules that
read from it.

```mermaid
flowchart TD
    subgraph DATA["Data pipeline (stdlib-only, CPU)"]
        RAW[("data/raw<br/>sample_seed.jsonl")] --> CLEAN["clean / normalize"]
        CLEAN --> DEDUP["dedup<br/>similarity ≥ 0.85"]
        DEDUP --> SPLIT["split<br/>15% val · 15% test · seed 42"]
        SPLIT --> TRAIN[("data/train/train.jsonl")]
        SPLIT --> VAL[("data/validation/val.jsonl")]
        SPLIT --> TEST[("data/test/test.jsonl")]
    end

    subgraph TRAINING["Training (Kaggle GPU)"]
        TRAIN --> STAGEA["Stage A: continued pre-training<br/>(optional — disabled in config)"]
        STAGEA --> STAGEB
        VAL --> STAGEB
        STAGEB["Stage B: QLoRA SFT<br/>Qwen2.5-1.5B · r=16 · α=32 · 4-bit · 3 epochs"] --> ADAPTER[("models/adapters/<br/>agri-climate-qlora")]
        ADAPTER --> MERGE["merge → models/merged/<br/>agri-climate-slm"]
    end

    subgraph RAG["Retrieval (CPU: TF-IDF)"]
        TRAIN --> EMBED["embed passages"]
        EMBED --> INDEX[("vector_index<br/>TF-IDF (stdlib)<br/>dense: MiniLM-L6-v2 optional")]
    end

    subgraph SERVE["Inference: AdvisoryPipeline"]
        Q(["user query<br/>(location, crop, stage, language)"]) --> CLF["query classifier<br/>language detect (Devanagari heuristic)<br/>+ slot extraction"]
        CLF --> RET["retrieve<br/>top_k = 5"]
        INDEX --> RET
        RET --> CTX["weather provider<br/>mock (Open-Meteo optional)"]
        CTX --> PROMPT["prompt builder<br/>sources + climate + query"]
        ADAPTER -. "if present" .-> SLM
        PROMPT --> SLM["SLM generation<br/>fine-tuned adapter or base<br/>(retrieval-only on CPU)"]
        SLM --> SAFE["safety validator<br/>strict mode"]
        SAFE --> ANS(["answer + sources<br/>+ weather_context + query_meta + safety"])
    end

    subgraph EVAL["Evaluation + tracking (side path)"]
        TEST --> E1["evaluate<br/>answer quality · Levenshtein Mean"]
        TEST --> E2["evaluate_rag<br/>retrieval hit-rate"]
        ANS --> E3["benchmark<br/>latency / throughput"]
        E1 --> CSV[("experiments/results.csv")]
        E2 --> CSV
        E3 --> CSV
    end

    style STAGEA stroke-dasharray: 5 5
```

## Component notes

| Component | Module | Runs on |
|---|---|---|
| Config loader | `src/data/config_loader.py` | CPU (stdlib) |
| Data pipeline | `src/data/prepare.py` | CPU (stdlib) |
| QLoRA SFT | `src/training/train.py` | Kaggle GPU (T4+) |
| TF-IDF index / retriever | `src/retrieval/` | CPU (stdlib); dense embeddings optional |
| Weather provider | `src/climate/` | CPU (`mock`; Open-Meteo toggle in config) |
| Advisory pipeline | `src/inference/pipeline.py` | CPU retrieval-only; GPU for generation |
| Safety validator | `src/safety/` | CPU |
| Evaluation | `src/evaluation/` | CPU (pure-Python metric fallbacks) |
| Baseline runner | `experiments/run_baselines.py` | CPU |

## Key design decisions

- **Config-driven**: every hyper-parameter, path, and toggle is read from `config.yaml`.
  No hard-coded training or retrieval parameters in `src/`.
- **Two-stage training, one enabled**: Stage A (continued pre-training) is optional and
  disabled (`stage_a_enabled: false`); Stage B (QLoRA SFT) is the training path.
- **Retrieval-first inference**: the pipeline always grounds answers in retrieved
  passages and explicit weather context; the SLM generates on top, and the safety
  validator (`safety_strict: true`) gates the final answer.
- **Multilingual by detection**: Hindi is detected with a Devanagari heuristic;
  technical agronomy terms are preserved, not blindly translated.
