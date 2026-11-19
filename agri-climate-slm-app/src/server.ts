import crypto from "node:crypto";
import path from "node:path";
import express from "express";
import { RECORDS, DATASET_STATS, EVALUATION } from "./data";
import { store, Advisory } from "./store";
import {
  normalizeUsername,
  validateUsername,
  validatePassword,
  hashPassword,
  verifyPassword,
  createSession,
  getSessionUser,
  destroySession,
} from "./auth";
import { search } from "./retrieval";
import { classify } from "./classify";
import { resolveWeather } from "./weather";
import { buildAdvisory } from "./safety";

const app = express();
app.use(express.json({ limit: "256kb" }));
app.use(express.static(path.join(__dirname, "..", "public")));

function tokenFrom(req: express.Request): string | undefined {
  const body = req.body as { token?: string } | undefined;
  if (body?.token) return body.token;
  const h = req.headers.authorization;
  if (h?.startsWith("Bearer ")) return h.slice(7);
  const q = req.query.token;
  return typeof q === "string" ? q : undefined;
}

function requireUser(req: express.Request, res: express.Response): string | null {
  const username = getSessionUser(tokenFrom(req));
  if (!username) {
    res.status(401).json({ error: "Not signed in. Please log in first." });
    return null;
  }
  return username;
}

// ---------- Auth ----------
app.post("/api/register", (req, res) => {
  const { username = "", password = "" } = (req.body ?? {}) as { username?: string; password?: string };
  const uErr = validateUsername(username);
  if (uErr) return res.status(400).json({ error: uErr });
  const pErr = validatePassword(password);
  if (pErr) return res.status(400).json({ error: pErr });

  const key = normalizeUsername(username);
  const db = store.read();
  if (db.users[key]) return res.status(409).json({ error: "That username is already registered. Try signing in." });

  const { salt, hash, iterations } = hashPassword(password);
  db.users[key] = { username: username.trim(), salt, hash, iterations, createdAt: new Date().toISOString() };
  store.write(db);
  const token = createSession(key);
  res.json({ token, username: username.trim() });
});

app.post("/api/login", (req, res) => {
  const { username = "", password = "" } = (req.body ?? {}) as { username?: string; password?: string };
  const key = normalizeUsername(username);
  const db = store.read();
  const user = db.users[key];
  if (!user || !verifyPassword(password, user.salt, user.hash, user.iterations)) {
    return res.status(401).json({ error: "Invalid username or password." });
  }
  const token = createSession(key);
  res.json({ token, username: user.username });
});

app.post("/api/logout", (req, res) => {
  const t = tokenFrom(req);
  if (t) destroySession(t);
  res.json({ ok: true });
});

app.get("/api/me", (req, res) => {
  const username = requireUser(req, res);
  if (!username) return;
  const db = store.read();
  res.json({ username: db.users[username]?.username ?? username });
});

// ---------- Advisory ----------
app.post("/api/advise", async (req, res) => {
  const username = requireUser(req, res);
  if (!username) return;
  const { question = "", crop, crop_stage, location } = (req.body ?? {}) as {
    question?: string;
    crop?: string;
    crop_stage?: string;
    location?: string;
  };
  if (!question || question.trim().length < 8) {
    return res.status(400).json({ error: "Please describe your question in a few more words." });
  }
  if (question.length > 1000) {
    return res.status(400).json({ error: "Question is too long (max 1000 characters)." });
  }

  const classification = classify(question);
  const results = search(question, 5, crop);
  const weather = await resolveWeather(location);
  const advisory = buildAdvisory(question, results, classification, weather, {
    crop: crop?.trim(),
    crop_stage: crop_stage?.trim(),
    location: location?.trim(),
  });

  const record: Advisory = {
    id: crypto.randomUUID(),
    username,
    question: question.trim().slice(0, 1000),
    answer: advisory.answer.slice(0, 4000),
    topic: advisory.topic,
    crop: crop?.trim() || undefined,
    location: location?.trim() || undefined,
    confidence: advisory.confidence,
    sources: advisory.sources.map((s) => s.id),
    createdAt: new Date().toISOString(),
  };
  const db = store.read();
  db.advisories.push(record);
  store.write(db);

  res.json({ ...advisory, id: record.id, createdAt: record.createdAt });
});

app.get("/api/history", (req, res) => {
  const username = requireUser(req, res);
  if (!username) return;
  const db = store.read();
  const items = db.advisories
    .filter((a) => a.username === username)
    .sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1))
    .slice(0, 50);
  res.json({ items });
});

// ---------- Dataset ----------
app.get("/api/dataset/stats", (_req, res) => {
  res.json(DATASET_STATS);
});

app.get("/api/dataset/sample", (req, res) => {
  const n = Math.min(Math.max(parseInt(String(req.query.n ?? "8"), 10) || 8, 1), 50);
  const sample = RECORDS.slice(0, n).map((r) => ({
    id: r.id,
    question: r.question,
    crop: r.crop,
    topic: r.topic,
    language: r.language,
    location: r.location,
  }));
  res.json({ items: sample, total: RECORDS.length });
});

app.get("/api/dataset/topics", (_req, res) => {
  const stats = DATASET_STATS as { per_topic?: Record<string, number> };
  res.json({ topics: stats.per_topic ?? {} });
});

app.get("/api/dataset/crops", (_req, res) => {
  const counts = new Map<string, number>();
  for (const r of RECORDS) counts.set(r.crop, (counts.get(r.crop) ?? 0) + 1);
  res.json({ crops: Object.fromEntries([...counts.entries()].sort((a, b) => b[1] - a[1])) });
});

// ---------- Evaluation ----------
app.get("/api/evaluation", (_req, res) => {
  res.json(EVALUATION);
});

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, records: RECORDS.length, mode: "retrieval-grounded" });
});

const PORT = parseInt(process.env.PORT ?? "3000", 10);
app.listen(PORT, () => {
  console.log(`Agri-Climate SLM app listening on http://localhost:${PORT} (${RECORDS.length} KB records)`);
});
