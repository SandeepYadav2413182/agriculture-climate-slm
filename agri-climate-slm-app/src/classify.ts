import { tokenize } from "./retrieval";

const TOPIC_KEYWORDS: Record<string, string[]> = {
  irrigation: ["irrigat", "water", "पानी", "सिंचाई", "drip", "sprinkler", "moisture"],
  pest_management: ["pest", "insect", "aphid", "bollworm", "whitefly", "कीट", "pesticide", "spray"],
  disease_identification: ["disease", "yellow", "spot", "blight", "wilt", "rust", "रोग", "पत्त"],
  fertilizers: ["fertiliz", "fertiliser", "urea", "npk", "nitrogen", "phosphorus", "potash", "उर्वरक", "खाद"],
  soil: ["soil", "मिट्टी", "ph", "organic matter", "soil health", "salinity"],
  seeds: ["seed", "बीज", "sowing", "germination", "variety"],
  weed_management: ["weed", "खरपतवार", "herbicide"],
  harvesting: ["harvest", "कटाई", "threshing", "maturity"],
  post_harvest: ["storage", "store", "भंडारण", "post-harvest", "grading"],
  crop_rotation: ["rotation", "intercrop", "mixed crop"],
  organic_farming: ["organic", "जैविक", "compost", "vermicompost", "biofertilizer"],
  temperature_stress: ["temperature", "heat stress", "cold", "तापमान"],
  rainfall: ["rain", "बारिश", "monsoon", "precipitation"],
  drought: ["drought", "सूखा", "dry spell", "water scarcity"],
  flooding: ["flood", "बाढ़", "waterlog"],
  heat_waves: ["heat wave", "heatwave", "लू", "scorching"],
  frost: ["frost", "पाला", "freezing"],
  humidity: ["humid", "आर्द्रता", "moisture in air"],
  climate_change: ["climate change", "जलवायु परिवर्तन", "global warming"],
  extreme_weather: ["storm", "hail", "तूफान", "cyclone", "extreme"],
  seasonal_variability: ["season", "मौसम", "kharif", "rabi", "zaid", "sowing time"],
  crop_management: ["crop", "फसल", "field", "खेत", "cultivation", "farming"],
  decision_support: ["should i", "decide", "advice", "suggest", "what to do", "सलाह"],
};

export function detectLanguage(text: string): "hi" | "en" {
  return /[\u0900-\u097F]/.test(text) ? "hi" : "en";
}

export interface Classification {
  topic: string;
  confidence: "high" | "medium" | "low";
}

/** Keyword-based query classifier over the 23 knowledge-base topics. */
export function classify(query: string): Classification {
  const q = query.toLowerCase();
  let best = "general";
  let bestScore = 0;
  for (const [topic, keywords] of Object.entries(TOPIC_KEYWORDS)) {
    let score = 0;
    for (const kw of keywords) {
      if (q.includes(kw.toLowerCase())) score += kw.length > 4 ? 2 : 1;
    }
    if (score > bestScore) {
      bestScore = score;
      best = topic;
    }
  }
  const tokens = tokenize(query).length || 1;
  const confidence: Classification["confidence"] =
    bestScore >= 4 ? "high" : bestScore >= 2 ? "medium" : "low";
  void tokens;
  return { topic: best, confidence };
}
