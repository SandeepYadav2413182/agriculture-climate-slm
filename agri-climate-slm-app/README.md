# Agri-Climate SLM App

A retrieval-grounded agriculture & climate advisory web app: bilingual (English/Hindi) crop advice with backend authentication, per-user advisory history, dataset and evaluation views.

This is the runnable codebase behind the hosted demo. It runs in **retrieval-grounded mode** over a curated 295-record knowledge base — the honest, low-cost baseline. The companion ML pipeline (QLoRA fine-tuning of Qwen2.5-1.5B-Instruct) can later supply a trained adapter as the answer composer, with retrieval and safety unchanged.

## Quickstart

```bash
npm install
npm run dev        # watch mode on http://localhost:3000
# or
npm run build && npm start
```

Open http://localhost:3000, create an account, and ask the advisor.

## Features

- **Advisor** (`/advisor.html`, login required): classify → TF-IDF retrieve → weather context → safety gate → persisted advisory
- **Auth**: salted PBKDF2-SHA256 password hashing, random 256-bit expiring sessions, email-or-username login
- **History**: per-user advisory timeline backed by `/api/history`
- **Dataset** (`/dataset.html`): live stats, topics, crops, sample records from `/api/dataset/*`
- **Evaluation** (`/evaluation.html`): retrieval MRR / hit@5, safety and grounding results from `/api/evaluation`
- **Architecture** (`/architecture.html`): pipeline documentation
- **Safety**: exact pesticide/fertilizer dosages withheld without crop, formulation, region, and authoritative guidance; low-confidence queries get clarifying questions, not guesses

## Project structure

```
src/
  server.ts      Express app + API routes
  auth.ts        PBKDF2 hashing, sessions, username validation
  store.ts       JSON file store (users, sessions, advisories)
  retrieval.ts   TF-IDF retrieval, Unicode-safe tokenizer (EN + HI)
  classify.ts    23-topic keyword classifier, language detection
  weather.ts     Mock provider (default) + optional Open-Meteo live provider
  safety.ts      Dosage guard, grounding, disclaimers, answer composition
  data.ts        Knowledge-base / stats / evaluation loaders
data/
  records.json        295 curated advisory records (synthetic seed data)
  dataset_stats.json  Cleaning/dedup/split statistics
  evaluation.json     Measured retrieval, safety, and grounding results
  app.db.json        Runtime user data (created on first write, gitignored)
public/          Static frontend (7 pages, vanilla JS + CSS)
Dockerfile       Container build
```

## API reference

All auth endpoints accept the session token in the JSON body as `token` or as an `Authorization: Bearer` header.

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/register` | – | `{username, password}` → `{token, username}` |
| POST | `/api/login` | – | `{username, password}` → `{token, username}` |
| POST | `/api/logout` | token | Invalidates the session |
| GET | `/api/me` | token | Current user |
| POST | `/api/advise` | token | `{question, crop?, crop_stage?, location?}` → grounded advisory |
| GET | `/api/history` | token | Your past advisories (newest first) |
| GET | `/api/dataset/stats` | – | Cleaning/dedup/split statistics |
| GET | `/api/dataset/sample?n=8` | – | Sample records |
| GET | `/api/dataset/topics` | – | Per-topic counts |
| GET | `/api/dataset/crops` | – | Per-crop counts |
| GET | `/api/evaluation` | – | Measured evaluation results |
| GET | `/api/health` | – | Liveness probe |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3000` | HTTP port |
| `WEATHER_PROVIDER` | `mock` | Set to `open-meteo` for live weather (no API key needed) |

## Data notes

`data/records.json` is curated/synthetic seed data for pipeline development — not official competition data. Source fields on each record say so explicitly. Expand it with sourced agricultural-extension material before any production use.

## Security notes

- Passwords are salted PBKDF2 hashes; sessions are random expiring tokens. No demo accounts are seeded.
- `data/app.db.json` holds user data and is gitignored — back it up separately if you deploy.
- This is a demo-grade store (single JSON file). For multi-instance production, swap `src/store.ts` for a real database.

## Docker

```bash
docker build -t agri-climate-slm-app .
docker run -p 3000:3000 agri-climate-slm-app
```

## Push to GitHub

```bash
git init -b main
git add .
git commit -m "Initial commit: Agri-Climate SLM advisory app"
gh repo create agri-climate-slm-app --private --source=. --push
# without the GitHub CLI:
# git remote add origin https://github.com/<you>/agri-climate-slm-app.git
# git push -u origin main
```
