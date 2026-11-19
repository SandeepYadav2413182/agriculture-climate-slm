import fs from "node:fs";
import path from "node:path";

export interface KbRecord {
  id: string;
  question: string;
  answer: string;
  crop: string;
  crop_stage: string;
  location: string;
  soil_type: string;
  language: string;
  topic: string;
  source: string;
  source_type: string;
  confidence: string;
  safety_level: string;
}

const DATA_DIR = path.join(__dirname, "..", "data");

function readJson<T>(name: string): T {
  return JSON.parse(fs.readFileSync(path.join(DATA_DIR, name), "utf8")) as T;
}

export const RECORDS: KbRecord[] = readJson<KbRecord[]>("records.json");
export const DATASET_STATS: Record<string, unknown> = readJson("dataset_stats.json");
export const EVALUATION: Record<string, unknown> = readJson("evaluation.json");
