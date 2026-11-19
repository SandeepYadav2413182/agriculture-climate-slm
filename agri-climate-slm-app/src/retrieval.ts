import { RECORDS, KbRecord } from "./data";

/** Unicode-safe tokenizer: works for English and Hindi (Devanagari) alike. */
export function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
}

const STOP = new Set([
  // English
  "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "for",
  "on", "my", "what", "how", "should", "i", "do", "does", "it", "with", "by",
  "at", "from", "that", "this", "when", "which", "be", "as", "if", "we", "you",
  "your", "our", "their", "there", "have", "has", "had", "will", "would", "can",
  // Hindi
  "क्या", "है", "में", "की", "का", "को", "और", "से", "पर", "यह", "वह", "कैसे",
  "मेरे", "अपने", "लिए", "ने", "तो", "भी", "का", "एक", "कोई",
]);

function contentTokens(r: KbRecord): string[] {
  return tokenize(`${r.question} ${r.answer} ${r.crop} ${r.topic} ${r.crop_stage}`).filter(
    (t) => !STOP.has(t)
  );
}

interface DocVec {
  record: KbRecord;
  weights: Map<string, number>;
  norm: number;
}

const N = RECORDS.length;
const df = new Map<string, number>();
const docTermFreqs: Map<string, number>[] = RECORDS.map((r) => {
  const tf = new Map<string, number>();
  for (const t of contentTokens(r)) tf.set(t, (tf.get(t) ?? 0) + 1);
  for (const t of tf.keys()) df.set(t, (df.get(t) ?? 0) + 1);
  return tf;
});

function idf(term: string): number {
  return Math.log((N + 1) / ((df.get(term) ?? 0) + 1)) + 1;
}

const DOCS: DocVec[] = RECORDS.map((record, i) => {
  const weights = new Map<string, number>();
  let sumSq = 0;
  for (const [t, f] of docTermFreqs[i]) {
    const w = f * idf(t);
    weights.set(t, w);
    sumSq += w * w;
  }
  return { record, weights, norm: Math.sqrt(sumSq) || 1 };
});

export interface ScoredRecord {
  record: KbRecord;
  score: number;
}

/** Cosine-similarity TF-IDF search over the knowledge base. */
export function search(query: string, topK = 5, cropHint?: string): ScoredRecord[] {
  const qtf = new Map<string, number>();
  for (const t of tokenize(query).filter((t) => !STOP.has(t))) {
    qtf.set(t, (qtf.get(t) ?? 0) + 1);
  }
  const qweights = new Map<string, number>();
  let qsumSq = 0;
  for (const [t, f] of qtf) {
    const w = f * idf(t);
    qweights.set(t, w);
    qsumSq += w * w;
  }
  const qnorm = Math.sqrt(qsumSq) || 1;
  const hint = (cropHint ?? "").trim().toLowerCase();

  const scored: ScoredRecord[] = DOCS.map((d) => {
    let dot = 0;
    for (const [t, qw] of qweights) {
      const dw = d.weights.get(t);
      if (dw !== undefined) dot += qw * dw;
    }
    let score = dot / (qnorm * d.norm);
    if (hint && d.record.crop.toLowerCase() === hint) score *= 1.25;
    return { record: d.record, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}
