# agriculture-climate-slm

A small-language-model advisory assistant for agriculture under climate stress:
ask a farming question (English or Hindi) and get a recommendation grounded in
retrieved agronomy guidance plus local climate context.

Pipeline: config-driven data prep (stdlib-only) → optional QLoRA fine-tuning of
**Qwen/Qwen2.5-1.5B-Instruct** on a Kaggle GPU → TF-IDF retrieval (CPU) →
weather context → prompt building → generation → safety validation.

## Competition status (verified)

This project originated from the **"Agriculture & Climate SLM Challenge"**:

| Fact | Value |
|---|---|
| Status | **CLOSED — Sep 9, 2026** |
| Evaluation metric | **Levenshtein Mean** |
| Points / medals | **None awarded** |
| Submission format | **TBD** |

The competition is over and its submission format was never finalized, so this
repo is maintained as a **portfolio project**, not a live competition entry.

## Project structure

```
agriculture-climate-slm/
├── config.yaml                  # single source of truth (all hyper-params, paths, toggles)
├── requirements.txt             # stdlib-only core; heavy deps for Kaggle GPU / demo app
├── app/
│   └── app.py                   # Gradio demo (CPU-friendly, retrieval-only)
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_training.ipynb        # Kaggle GPU (T4)
│   ├── 03_evaluation.ipynb
│   └── 04_inference.ipynb       # CPU demo, English + Hindi
├── docs/
│   ├── architecture.md          # mermaid system diagram
│   └── experiment_report.md     # results template (filled after GPU runs)
├── experiments/
│   ├── run_baselines.py         # 4-experiment orchestrator
│   └── results.csv              # experiment tracking log
├── src/
│   ├── data/        # config loader + stdlib data pipeline
│   ├── training/    # QLoRA SFT (Stage B; Stage A optional/disabled)
│   ├── retrieval/   # TF-IDF index + retriever (dense embeddings optional)
│   ├── climate/     # weather provider (mock; Open-Meteo toggle)
│   ├── inference/   # AdvisoryPipeline: classify → retrieve → weather → prompt → SLM → safety
│   ├── safety/      # safety validator (strict mode)
│   └── evaluation/  # metrics incl. Levenshtein Mean, RAG eval, benchmark
├── data/            # raw / processed / train / validation / test splits
├── models/          # base / adapters / merged
└── scripts/
```

`docs/dataset_report.md` (dataset card) is maintained separately.

## Quick start

### 1. Data preparation (CPU, stdlib-only)

```bash
python -m src.data.prepare
# or walk through it:
#   jupyter notebook notebooks/01_data_preparation.ipynb
```

Reads `data/raw/sample_seed.jsonl`, cleans, dedups (threshold 0.85), and writes
train/validation/test splits (15% val, 15% test, seed 42) per `config.yaml`.

### 2. Training — QLoRA SFT (Kaggle GPU)

Needs a Kaggle notebook with a **GPU (T4 or better)**:

```bash
pip install -r requirements.txt
python -m src.training.train
# or follow notebooks/02_training.ipynb
```

Saves the LoRA adapter to `models/adapters/agri-climate-qlora` (`adapter_dir`).
Stage A (continued pre-training) is disabled in this config; only Stage B runs.

### 3. Evaluation (CPU)

```bash
python -m src.evaluation.evaluate
python -m src.evaluation.evaluate_rag
python -m src.evaluation.benchmark
# or: jupyter notebook notebooks/03_evaluation.ipynb
```

### 4. Baselines (CPU)

```bash
python experiments/run_baselines.py
```

Runs 4 experiments on the test split — retrieval-only baseline, fine-tuned
(skipped unless the adapter exists), +RAG, +RAG+climate — scored with
`levenshtein_mean`, appending rows to `experiments/results.csv`.

### 5. Demo app (CPU)

```bash
pip install 'gradio>=4.0'   # or: pip install -r requirements.txt
python app/app.py
```

Opens the "Agriculture & Climate Assistant" UI: location, crop, stage, question
→ recommendation + climate context + retrieved sources. Retrieval-only mode, no
weights needed.

### 6. Inference notebook (CPU)

```bash
jupyter notebook notebooks/04_inference.ipynb
```

English + Hindi `AdvisoryPipeline` demos with RAG sources and weather context.

## Kaggle setup notes

1. New Kaggle notebook → accelerator **GPU T4 x2** (P100 also works).
2. Get the repo in: upload as a dataset or `git clone` into `/kaggle/working`.
3. `%pip install -q -r requirements.txt` (notebook 02 does this).
4. Run `notebooks/01_data_preparation.ipynb` first (data), then `02_training.ipynb`.
5. Download `models/adapters/agri-climate-qlora/` from the Kaggle output when done —
   evaluation, baselines (experiment 2), and the demo pick it up automatically.

## What runs on CPU here vs what needs a Kaggle GPU

| Stage | CPU (this machine) | Kaggle GPU |
|---|---|---|
| Data prep (`src/data`) | ✅ stdlib-only | — |
| TF-IDF retrieval / query classifier / safety | ✅ | — |
| Evaluation + metric fallbacks | ✅ | — |
| Baselines 1, 3, 4 (retrieval-only) | ✅ | — |
| Gradio demo (retrieval-only, no weights) | ✅ (`gradio` install) | — |
| Dense embeddings (MiniLM) | ✅ (slow) / optional | faster |
| QLoRA fine-tuning (`src/training`) | ❌ not practical | ✅ T4+ |
| Baseline 2 (fine-tuned model) | ❌ needs adapter | ✅ after training |

## config.yaml reference

`config.yaml` is the single source of truth — every `src/` module reads its
parameters via `src/data/config_loader.py`. Key entries:

| Key | Value | Meaning |
|---|---|---|
| `model_name` | Qwen/Qwen2.5-1.5B-Instruct | base SLM for QLoRA |
| `max_seq_length` | 2048 | training/inference context |
| `learning_rate` / `epochs` | 0.0002 / 3 | SFT schedule |
| `lora_rank` / `lora_alpha` / `lora_dropout` | 16 / 32 / 0.05 | LoRA config |
| `quantization` | 4bit | QLoRA quantization |
| `stage_a_enabled` | false | continued pre-training off |
| `test_size` / `val_size` | 0.15 / 0.15 | split ratios |
| `dedup_similarity_threshold` | 0.85 | near-dup cutoff |
| `vector_database` / `top_k` | tfidf / 5 | retrieval backend + depth |
| `weather_provider` | mock | `open_meteo_enabled: false` |
| `supported_languages` | en, hi | English + Hindi |
| `safety_strict` | true | strict safety validation |
| `adapter_dir` | models/adapters/agri-climate-qlora | QLoRA output |
| `experiments_csv` | experiments/results.csv | tracking log |

## Safety

- The safety validator (`src/safety/`, `safety_strict: true`) checks every answer
  before it is returned: result carries `{"passed", "flags", "risk_level"}`.
- The demo app surfaces failed/flagged checks inline with a "verify with a local
  agronomist" notice.
- This is an advisory tool, not a substitute for professional agronomic advice.

## Multilingual notes

- Supported languages: English (`en`) and Hindi (`hi`).
- Language is auto-detected when `language=None` (the default in the app and
  notebooks); Hindi detection uses a **Devanagari heuristic** on the query text.
- **Technical agronomy terms are preserved, not blindly translated** — e.g. crop
  names, growth-stage terms, and chemical names stay as-is across languages to
  avoid dangerous mistranslations.
