import crypto from "node:crypto";
import { store } from "./store";

const ITERATIONS = 120000;
const KEYLEN = 32;
const DIGEST = "sha256";
const SESSION_TTL_MS = 30 * 24 * 3600 * 1000; // 30 days

export function normalizeUsername(u: string): string {
  return u.trim().toLowerCase();
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const USER_RE = /^[a-z0-9._-]{3,60}$/;

export function validateUsername(u: string): string | null {
  const n = u.trim();
  if (!n) return "Username is required.";
  if (EMAIL_RE.test(n)) return null;
  if (!USER_RE.test(n.toLowerCase())) {
    return "Use an email address, or 3-60 characters: letters, numbers, dot, dash, underscore.";
  }
  return null;
}

export function validatePassword(p: string): string | null {
  if (!p || p.length < 10) return "Use at least 10 characters.";
  if (p.length > 128) return "Password must be under 128 characters.";
  return null;
}

export function hashPassword(password: string): { salt: string; hash: string; iterations: number } {
  const salt = crypto.randomBytes(16).toString("hex");
  const hash = crypto.pbkdf2Sync(password, salt, ITERATIONS, KEYLEN, DIGEST).toString("hex");
  return { salt, hash, iterations: ITERATIONS };
}

export function verifyPassword(password: string, salt: string, hash: string, iterations: number): boolean {
  const h = crypto.pbkdf2Sync(password, salt, iterations, KEYLEN, DIGEST).toString("hex");
  const a = Buffer.from(h, "hex");
  const b = Buffer.from(hash, "hex");
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

export function createSession(username: string): string {
  const token = crypto.randomBytes(32).toString("hex");
  const db = store.read();
  db.sessions[token] = { username, expiresAt: Date.now() + SESSION_TTL_MS };
  store.write(db);
  return token;
}

export function getSessionUser(token: string | undefined): string | null {
  if (!token) return null;
  const db = store.read();
  const s = db.sessions[token];
  if (!s) return null;
  if (s.expiresAt < Date.now()) {
    delete db.sessions[token];
    store.write(db);
    return null;
  }
  return s.username;
}

export function destroySession(token: string): void {
  const db = store.read();
  delete db.sessions[token];
  store.write(db);
}
