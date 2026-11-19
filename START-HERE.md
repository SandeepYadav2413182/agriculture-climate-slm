# Agri-Climate SLM — Full Project (ML + Web UI)

This repo contains the complete project in one place:

- **ML project (repo root)** — the Python research project: dataset, training,
  retrieval, safety checks and evaluation.
  - `src/` — pipeline code (data, training, retrieval, inference, safety, evaluation)
  - `data/` — raw, cleaned and split datasets (train / validation / test)
  - `notebooks/`, `experiments/`, `scripts/`, `docs/` — exploration, runs and write-ups
  - `models/` — trained model files (see the note below about one missing file)
  - `config.yaml`, `requirements.txt` — configuration and Python dependencies
- **`app/`** — a small Gradio demo of the model.
- **`agri-climate-slm-app/`** — the product web UI (Node.js + TypeScript,
  glassmorphism site with login, advisor, dataset and evaluation pages).

## Run the Gradio demo

```bash
pip install gradio
python app/app.py
```

## Run the web UI

Needs Node.js 18 or newer.

```bash
cd agri-climate-slm-app
npm install
npm run dev
```

Then open http://localhost:3000 — login and the advisor only work while this
dev server is running; opening the HTML files directly will not work.

On Windows PowerShell, if `npm` is blocked by the execution policy, use
`npm.cmd` instead (`npm.cmd install`, `npm.cmd run dev`).

## Note: one model file is not in this repo

`models/baseline_nb.json` (158 MB) is **excluded on purpose** — it is over
GitHub's 100 MB single-file limit, so it cannot be pushed. Nothing in the demo
or the web UI needs it at runtime. If you ever need it back, regenerate it from
the repo root:

```bash
python -m src.training.baseline_nb
```

## Deploy on Render

**Web UI (the product site):**

- New Web Service from this repo, runtime **Docker**
- **Root Directory:** `agri-climate-slm-app` (Render finds the Dockerfile there)
- No build or start commands to set — the Dockerfile handles them, and Render
  sets the port.

**Gradio demo (the model):**

- New Web Service from this repo, runtime **Python 3**
- **Build Command:** `pip install 'gradio>=4.0'`
  (do not install the full `requirements.txt` — it carries heavy training-only
  packages the demo does not need)
- **Start Command:** `python app/app.py`
- Environment variables: `GRADIO_SERVER_NAME` = `0.0.0.0`,
  `GRADIO_SERVER_PORT` = `10000`

Free-tier note for both: the service sleeps after ~15 minutes idle, so open
the link a few minutes before you need to show it.
