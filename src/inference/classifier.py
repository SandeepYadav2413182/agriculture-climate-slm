"""Rule-based query classifier for the agriculture advisory pipeline.

Stdlib only. Provides:

* :data:`INTENTS` — the fixed intent taxonomy.
* :func:`detect_language` — Devanagari-script heuristic ("hi" vs "en").
* :func:`classify` — gazetteer-based extraction of intent, crop, location,
  crop stage, problem, climate factor and urgency from a farmer's question.

No ML is used here on purpose: the classifier must run on stock Python with
zero dependencies, and rule-based gazetteers are transparent and easy for
domain experts to extend.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config  # noqa: E402

_CFG = load_config()
# Fallback language when the input carries no detectable signal.
DEFAULT_LANGUAGE = _CFG.get("default_language", "en")

# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------
INTENTS = [
    "crop_management",
    "pest_management",
    "disease_identification",
    "soil_management",
    "irrigation",
    "fertilizer",
    "weather_advisory",
    "climate_adaptation",
    "harvest_post_harvest",
    "general",
]

# ---------------------------------------------------------------------------
# Gazetteers: canonical value -> list of aliases (English + Hindi).
# These are lexical data, not hyper-parameters, so they live with the code.
# ---------------------------------------------------------------------------
CROPS = {
    "rice": ["rice", "paddy", "dhaan", "dhan", "धान"],
    "wheat": ["wheat", "gehu", "gehun", "गेहूं", "गेहूँ"],
    "maize": ["maize", "corn", "makka", "makai", "मक्का", "मकई"],
    "cotton": ["cotton", "kapas", "कपास"],
    "sugarcane": ["sugarcane", "ganna", "cane", "गन्ना"],
    "soybean": ["soybean", "soya", "soyabean", "सोयाबीन"],
    "groundnut": ["groundnut", "peanut", "moongphali", "मूंगफली"],
    "mustard": ["mustard", "sarson", "rai", "सरसों"],
    "chickpea": ["chickpea", "gram", "chana", "चना"],
    "pigeonpea": ["pigeonpea", "arhar", "tur", "अरहर"],
    "tomato": ["tomato", "tamatar", "टमाटर"],
    "onion": ["onion", "pyaz", "pyaaz", "प्याज", "प्याज़"],
    "potato": ["potato", "aloo", "aaloo", "आलू"],
}

CROP_STAGES = {
    "sowing": ["sowing", "sow", "planting", "बुवाई", "बोआई"],
    "germination": ["germination", "germinate", "sprout", "अंकुरण"],
    "vegetative": ["vegetative", "tillering", "tiller", "वृद्धि"],
    "flowering": ["flowering", "flower", "bloom", "फूल", "पुष्पन"],
    "fruiting": ["fruiting", "boll formation", "pod formation", "फलन"],
    "grain_filling": ["grain filling", "grain-filling", "milky stage", "दाना भरना"],
    "maturity": ["maturity", "mature", "ripening", "परिपक्वता"],
    "harvesting": ["harvest", "harvesting", "कटाई", "फसल कटाई"],
}

PROBLEMS = {
    "pest": ["pest", "insect", "aphid", "bollworm", "stem borer", "whitefly",
             "thrips", "mite", "caterpillar", "कीट", "कीड़ा", "सुंडी"],
    "disease": ["disease", "blight", "wilt", "rust", "smut", "fungus", "fungal",
                "virus", "bacterial", "रोग", "बीमारी", "झुलसा"],
    "yellow_leaves": ["yellow leaves", "yellowing", "chlorosis", "पीली पत्तियां",
                      "पीली पत्तियाँ", "पीलापन"],
    "weeds": ["weed", "weeds", "खरपतवार"],
    "wilting": ["wilt", "wilting", "drooping", "मुरझाना"],
    "stunted_growth": ["stunted", "poor growth", "dwarf", "बौनापन", "वृद्धि रुकना"],
    "nutrient_deficiency": ["deficiency", "deficient", "कमी"],
}

CLIMATE_FACTORS = {
    "drought": ["drought", "dry spell", "water scarcity", "सूखा", "अकाल"],
    "flood": ["flood", "waterlogging", "water logging", "बाढ़", "जलभराव"],
    "heat": ["heat", "heatwave", "heat wave", "high temperature", "गर्मी", "लू"],
    "cold": ["cold", "frost", "chill", "पाला", "ठंड"],
    "rainfall": ["rainfall", "rain", "precipitation", "बारिश", "वर्षा"],
    "monsoon": ["monsoon", "मानसून"],
    "hailstorm": ["hailstorm", "hail", "ओलावृष्टि", "ओले"],
    "cyclone": ["cyclone", "storm", "चक्रवात", "तूफान"],
}

# Indian states / UTs (English; a few common Hindi forms).
STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya",
    "mizoram", "nagaland", "odisha", "orissa", "punjab", "rajasthan",
    "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "west bengal", "jammu and kashmir", "ladakh", "delhi",
    "उत्तर प्रदेश", "मध्य प्रदेश", "महाराष्ट्र", "पंजाब", "बिहार", "गुजरात",
    "राजस्थान", "हरियाणा", "कर्नाटक", "तमिलनाडु", "पश्चिम बंगाल",
]

URGENCY_HIGH = ["today", "urgent", "immediately", "asap", "right now",
                "at once", "emergency", "तुरंत", "आज", "फौरन"]
URGENCY_MEDIUM = ["tomorrow", "expected", "soon", "upcoming", "forecast",
                  "कल", "आने वाले", "जल्द"]

# Intent keyword rules, evaluated in priority order. The first intent whose
# keywords match wins; this ordering resolves e.g. "pest and disease" queries
# deterministically (disease_identification is checked before pest_management
# because symptom description usually dominates such questions).
INTENT_RULES = [
    ("disease_identification",
     ["disease", "blight", "wilt", "rust", "smut", "fungus", "fungal",
      "virus", "bacterial", "symptom", "spot on leaf", "yellow leaves",
      "yellowing", "रोग", "बीमारी", "झुलसा", "धब्बा", "पीली पत्तिय"]),
    ("pest_management",
     ["pest", "insect", "aphid", "bollworm", "borer", "whitefly", "thrips",
      "mite", "caterpillar", "larva", "spray insecticide", "कीट", "कीड़ा",
      "सुंडी", "इल्ली"]),
    ("soil_management",
     ["soil", "mitti", "ph", "soil health", "soil test", "मिट्टी", "मृदा"]),
    ("irrigation",
     ["irrigation", "irrigate", "watering", "water management", "drip",
      "sprinkler", "सिंचाई", "पानी देना"]),
    ("fertilizer",
     ["fertilizer", "fertiliser", "urea", "dap", "npk", "manure", "compost",
      "nutrient", "खाद", "उर्वरक", "यूरिया"]),
    ("weather_advisory",
     ["weather", "forecast", "rain expected", "temperature today", "मौसम",
      "मौसम पूर्वानुमान"]),
    ("climate_adaptation",
     ["climate change", "adaptation", "resilient", "drought resistant",
      "drought-resistant", "climate", "जलवायु", "जलवायु परिवर्तन"]),
    ("harvest_post_harvest",
     ["harvest", "storage", "store grain", "mandi", "sell", "threshing",
      "post-harvest", "कटाई", "भंडारण", "मंडी", "बेचना"]),
    ("crop_management",
     ["how to grow", "cultivation", "sowing time", "variety", "seed rate",
      "crop management", "खेती", "उगाना", "बुवाई का समय", "किस्म"]),
]


def _contains_any(text: str, phrases) -> bool:
    """True if any phrase occurs in text (word-boundary aware for ASCII)."""
    for phrase in phrases:
        if not phrase.strip():
            continue
        if re.search(r"[\u0900-\u097f]", phrase):
            # Devanagari: plain substring match (no word boundaries).
            if phrase in text:
                return True
        else:
            if re.search(r"\b" + re.escape(phrase) + r"\b", text):
                return True
    return False


def _match_gazetteer(text: str, gazetteer: dict) -> str | None:
    for canonical, aliases in gazetteer.items():
        if _contains_any(text, aliases):
            return canonical
    return None


def detect_language(text: str) -> str:
    """Return ``"hi"`` if the text contains Devanagari characters, else ``"en"``.

    This is a deliberate heuristic, not a full language identifier: it detects
    *script*, so Hindi written in Latin script ("meri fasal") is classified as
    "en". That trade-off keeps the function dependency-free and predictable;
    romanized Hindi handling would require a trained identifier.
    """
    if not text or not text.strip():
        return DEFAULT_LANGUAGE
    if re.search(r"[\u0900-\u097f]", text):
        return "hi"
    return "en"


def classify(question: str) -> dict:
    """Classify a farmer's question into structured query metadata.

    Returns a dict with keys: ``intent``, ``crop``, ``location``,
    ``crop_stage``, ``problem``, ``climate_factor``, ``urgency``.
    Missing slots are ``None`` (``urgency`` defaults to ``"low"``).
    """
    text = (question or "").lower()

    intent = "general"
    for candidate, keywords in INTENT_RULES:
        if _contains_any(text, keywords):
            intent = candidate
            break
    # A named crop with no stronger signal is still crop management.
    if intent == "general" and _match_gazetteer(text, CROPS):
        intent = "crop_management"

    crop = _match_gazetteer(text, CROPS)
    crop_stage = _match_gazetteer(text, CROP_STAGES)
    problem = _match_gazetteer(text, PROBLEMS)
    climate_factor = _match_gazetteer(text, CLIMATE_FACTORS)

    location = None
    for state in STATES:
        if state in text:
            location = state
            break

    urgency = "low"
    if _contains_any(text, URGENCY_HIGH):
        urgency = "high"
    elif _contains_any(text, URGENCY_MEDIUM):
        urgency = "medium"

    return {
        "intent": intent,
        "crop": crop,
        "location": location,
        "crop_stage": crop_stage,
        "problem": problem,
        "climate_factor": climate_factor,
        "urgency": urgency,
    }


if __name__ == "__main__":
    samples = [
        "My wheat crop has yellow leaves during flowering stage in Punjab, what should I do today?",
        "गेहूं में कीट लग गए हैं, तुरंत उपाय बताएं",
        "When will the monsoon arrive in Maharashtra? Is rain expected tomorrow?",
        "Best fertilizer dose for paddy in Bihar during tillering?",
        "How to store harvested onions safely?",
    ]
    for s in samples:
        print(f"Q: {s}")
        print(f"   lang={detect_language(s)} meta={classify(s)}")
