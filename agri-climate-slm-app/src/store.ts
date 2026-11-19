import fs from "node:fs";
import path from "node:path";

const DB_PATH = path.join(__dirname, "..", "data", "app.db.json");

export interface UserRec {
  username: string;
  salt: string;
  hash: string;
  iterations: number;
  createdAt: string;
}

export interface SessionRec {
  username: string;
  expiresAt: number;
}

export interface Advisory {
  id: string;
  username: string;
  question: string;
  answer: string;
  topic: string;
  crop?: string;
  location?: string;
  confidence: string;
  sources: string[];
  createdAt: string;
}

interface Db {
  users: Record<string, UserRec>;
  sessions: Record<string, SessionRec>;
  advisories: Advisory[];
}

function blank(): Db {
  return { users: {}, sessions: {}, advisories: [] };
}

function load(): Db {
  try {
    const raw = JSON.parse(fs.readFileSync(DB_PATH, "utf8")) as Partial<Db>;
    return {
      users: raw.users ?? {},
      sessions: raw.sessions ?? {},
      advisories: raw.advisories ?? [],
    };
  } catch {
    return blank();
  }
}

function save(db: Db): void {
  fs.writeFileSync(DB_PATH, JSON.stringify(db, null, 2), "utf8");
}

export const store = { read: load, write: save };
