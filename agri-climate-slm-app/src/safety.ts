import { KbRecord } from "./data";
import { ScoredRecord } from "./retrieval";
import { Classification, detectLanguage } from "./classify";
import { WeatherCtx } from "./weather";

const DOSAGE_RE =
  /(\d+(\.\d+)?\s*(ml|g|kg|l|litre|liter|ppm)\s*(\/|per)\s*(l|litre|liter|ha|acre)|kg\s*\/\s*ha|ml\s*\/\s*l|exact\s+(dose|dosage))/i;

export function requestsExactDosage(question: string): boolean {
  return DOSAGE_RE.test(question);
}

const DISCLAIMER =
  "Advisory generated from a curated knowledge base (retrieval-grounded). " +
  "Verify critical decisions with your local agricultural extension officer.";

export interface AdvisoryResult {
  answer: string;
  topic: string;
  confidence: string;
  sources: Array<{ id: string; question: string; crop: string; topic: string }>;
  weather: WeatherCtx | null;
  safety: { level: string; notes: string[] };
}

function dosageGuidance(question: string, language: string): string {
  const hi = language === "hi" || detectLanguage(question) === "hi";
  const body = hi
    ? "सटीक कीटनाशक/उर्वरक मात्रा (dose) फसल, उत्पाद के formulation, क्षेत्र और मौसम पर निर्भर करती है, इसलिए मैं यहाँ कोई निश्चित मात्रा नहीं बता रहा। कृपया बताएं: (1) फसल और उसकी अवस्था, (2) उत्पाद का नाम/formulation, (3) क्षेत्र/राज्य, (4) समस्या के लक्षण। तब तक उत्पाद के लेबल और अपने कृषि विस्तार अधिकारी की सलाह का पालन करें।"
    : "Exact pesticide/fertilizer dosages depend on the crop, the product's formulation, your region, and current conditions, so I won't state a fixed dose here. Please share: (1) crop and growth stage, (2) product name/formulation, (3) your district/state, (4) the symptoms you see. Until then, follow the product label and your local agricultural extension officer's guidance.";
  return `${body}\n\n${DISCLAIMER}`;
}

function clarifyingResponse(topic: string): string {
  return (
    `I don't have a confident match for this in the knowledge base, so I won't guess. ` +
    `To give useful advice, please add: (1) crop name, (2) growth stage, (3) location/district, ` +
    `(4) symptoms or what you observe, and (5) soil/irrigation details if relevant. ` +
    `(Detected area: ${topic}.)\n\n${DISCLAIMER}`
  );
}

export function buildAdvisory(
  question: string,
  results: ScoredRecord[],
  classification: Classification,
  weather: WeatherCtx | null,
  hints: { crop?: string; crop_stage?: string; location?: string }
): AdvisoryResult {
  const language = detectLanguage(question);
  const sources = results.slice(0, 3).map((r) => ({
    id: r.record.id,
    question: r.record.question,
    crop: r.record.crop,
    topic: r.record.topic,
  }));

  if (requestsExactDosage(question)) {
    return {
      answer: dosageGuidance(question, language),
      topic: classification.topic,
      confidence: "withheld — needs crop, formulation, and region",
      sources,
      weather,
      safety: {
        level: "caution",
        notes: ["Exact dosage request: no fixed dose given without authoritative guidance."],
      },
    };
  }

  const top: ScoredRecord | undefined = results[0];
  const LOW_CUTOFF = 0.08;
  if (!top || top.score < LOW_CUTOFF) {
    return {
      answer: clarifyingResponse(classification.topic),
      topic: classification.topic,
      confidence: "low — asking for details",
      sources,
      weather,
      safety: { level: "safe", notes: ["Low retrieval confidence: answered with clarifying questions, not a guess."] },
    };
  }

  const rec: KbRecord = top.record;
  const parts: string[] = [rec.answer];
  if (weather) parts.push(`\nWeather context: ${weather.summary}. (${weather.note})`);
  if (hints.crop_stage && rec.crop_stage && hints.crop_stage.toLowerCase() !== rec.crop_stage.toLowerCase()) {
    parts.push(`\nNote: the closest record concerns the "${rec.crop_stage}" stage; adjust for your "${hints.crop_stage}" stage.`);
  }
  parts.push(`\n${DISCLAIMER}`);

  return {
    answer: parts.join("\n"),
    topic: rec.topic,
    confidence: rec.confidence,
    sources,
    weather,
    safety: { level: rec.safety_level, notes: [] },
  };
}
