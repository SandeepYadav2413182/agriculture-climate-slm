"""Prompt construction for the agriculture & climate advisory assistant.

:func:`build_prompt` assembles a single prompt string from the user's
question, retrieved knowledge-base passages, weather/climate context and the
structured query metadata produced by :mod:`src.inference.classifier`.

Design choices (documented here because they are deliberate):

* The assistant is instructed to be *evidence-grounded*: it may only use the
  supplied knowledge and climate context, must not invent facts, and must
  state explicitly which information is missing.
* Exact pesticide/fertilizer dosages are gated: they may only appear when
  crop + product + formulation + region + authoritative guidance are ALL
  present. Otherwise the model must refuse the exact number and point at the
  product label / local extension officer. (The safety validator enforces the
  same rule on the output side.)
* General advice and location-specific advice must be distinguished, because
  a recommendation valid in Punjab's rabi season can be harmful in Kerala.
* For Hindi (``language="hi"``) the system preamble is written in Hindi, but
  technical terms (pesticide names, units, crop-growth-stage names) are kept
  in English/Latin script. Farmers and extension material in India routinely
  code-switch this way; transliterating "imidacloprid" into Devanagari would
  make the product unidentifiable at the agro-dealer.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config  # noqa: E402

_CFG = load_config()
# When safety_strict is on, the system prompt carries an extra hard rule
# restating the dosage policy, so even a weak base model stays cautious.
_SAFETY_STRICT = bool(_CFG.get("safety_strict", True))

_SYSTEM_EN = """SYSTEM: You are an agriculture and climate advisory assistant for Indian farmers.
Provide practical, evidence-grounded agricultural information.

Rules you must follow:
- Use ONLY the knowledge supplied in AGRICULTURAL CONTEXT and WEATHER/CLIMATE CONTEXT below, plus the STRUCTURED QUERY. Do NOT invent facts, figures, product names, or research findings.
- If the supplied context does not contain the information needed, say so explicitly and state what information is missing.
- For pesticide, fertilizer, or disease-control advice: do NOT give exact dosages (ml/litre, g/acre, kg/ha, etc.) unless ALL of the following are present: the crop, the product name, the formulation, the region, and authoritative guidance. If any are missing, refuse the exact number and direct the farmer to the product label and the local agriculture extension officer / Krishi Vigyan Kendra.
- Clearly distinguish GENERAL advice from LOCATION-SPECIFIC advice. Never present region-specific guidance as universally applicable.
- Be concise and actionable. Prefer numbered steps a farmer can follow in the field."""

_SYSTEM_HI = """SYSTEM: आप भारतीय किसानों के लिए कृषि और जलवायु सलाहकार सहायक हैं।
व्यावहारिक, साक्ष्य-आधारित कृषि जानकारी दें।

आपको इन नियमों का पालन करना होगा:
- नीचे दिए गए AGRICULTURAL CONTEXT और WEATHER/CLIMATE CONTEXT तथा STRUCTURED QUERY में दी गई जानकारी का ही उपयोग करें। तथ्य, आंकड़े, उत्पादों के नाम या शोध निष्कर्ष न गढ़ें।
- यदि दिए गए संदर्भ में आवश्यक जानकारी नहीं है, तो स्पष्ट रूप से बताएं और लिखें कि कौन-सी जानकारी missing है।
- कीटनाशक, उर्वरक या रोग-नियंत्रण सलाह के लिए: सटीक मात्रा (ml/litre, g/acre, kg/ha आदि) तभी बताएं जब ये सभी उपलब्ध हों: फसल, उत्पाद का नाम, formulation, क्षेत्र और आधिकारिक मार्गदर्शन। यदि कोई भी missing हो, तो सटीक मात्रा देने से मना करें और किसान को उत्पाद के label तथा स्थानीय कृषि विस्तार अधिकारी / कृषि विज्ञान केंद्र से संपर्क करने को कहें।
- सामान्य (GENERAL) सलाह और स्थान-विशिष्ट (LOCATION-SPECIFIC) सलाह में स्पष्ट अंतर करें। किसी क्षेत्र-विशिष्ट मार्गदर्शन को सार्वभौमिक न बताएं।
- संक्षिप्त और कार्यात्मक उत्तर दें। खेत में अपनाए जा सकने वाले क्रमबद्ध चरणों को प्राथमिकता दें।
नोट: तकनीकी शब्द जैसे pesticide/fertilizer के नाम, इकाइयां (ml, kg, litre) और फसल अवस्थाओं के नाम English में ही रखें ताकि उत्पाद की पहचान बनी रहे।"""

_EXTRA_SAFETY_EN = (
    "\n- SAFETY: never state an exact chemical dosage without complete "
    "crop/product/formulation/region/authoritative-guidance context."
)
_EXTRA_SAFETY_HI = (
    "\n- सुरक्षा: फसल/उत्पाद/formulation/क्षेत्र/आधिकारिक-मार्गदर्शन की पूर्ण "
    "जानकारी के बिना रसायन की सटीक मात्रा कभी न बताएं।"
)


def _format_retrieved(retrieved_docs) -> str:
    if not retrieved_docs:
        return "(no knowledge-base passages retrieved)"
    parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        doc_id = doc.get("id", f"doc{i}")
        score = doc.get("score", 0.0)
        text = doc.get("text", "")
        parts.append(f"[{i}] (id={doc_id}, score={score:.3f})\n{text}")
    return "\n\n".join(parts)


def build_prompt(question, retrieved_docs, weather_context, query_meta,
                 language="en") -> str:
    """Build the full prompt string for the advisory model.

    Parameters
    ----------
    question: str
        The farmer's raw question.
    retrieved_docs: list of dict
        Each dict has ``id``, ``score`` and ``text`` keys.
    weather_context: str
        Rendered weather/climate context (may note unavailability).
    query_meta: dict
        Structured metadata from :func:`src.inference.classifier.classify`.
    language: str
        ``"en"`` or ``"hi"``; selects the system preamble language.

    Returns
    -------
    str
        ``SYSTEM: ...`` block followed by the ``USER:`` block with the
        question, contexts, structured query and the required answer format.
    """
    lang = (language or "en").lower()
    system = _SYSTEM_HI if lang == "hi" else _SYSTEM_EN
    if _SAFETY_STRICT:
        system += _EXTRA_SAFETY_HI if lang == "hi" else _EXTRA_SAFETY_EN

    meta_lines = []
    for key in ("intent", "crop", "location", "crop_stage", "problem",
                "climate_factor", "urgency", "language"):
        if query_meta and query_meta.get(key) is not None:
            meta_lines.append(f"- {key}: {query_meta[key]}")
    meta_str = "\n".join(meta_lines) if meta_lines else "(none extracted)"

    provide_en = ("Provide:\n"
                  "1. Direct answer\n"
                  "2. Recommended actions\n"
                  "3. Conditions/cautions\n"
                  "4. Info needed for a more precise recommendation")
    provide_hi = ("उत्तर इस प्रारूप में दें:\n"
                  "1. सीधा उत्तर (Direct answer)\n"
                  "2. अनुशंसित कार्य (Recommended actions)\n"
                  "3. शर्तें/सावधानियां (Conditions/cautions)\n"
                  "4. अधिक सटीक सिफारिश के लिए आवश्यक जानकारी (Info needed)")
    provide = provide_hi if lang == "hi" else provide_en

    return (
        f"{system}\n\n"
        f"USER: {question}\n\n"
        f"AGRICULTURAL CONTEXT:\n{_format_retrieved(retrieved_docs)}\n\n"
        f"WEATHER/CLIMATE CONTEXT:\n{weather_context or '(not available)'}\n\n"
        f"STRUCTURED QUERY:\n{meta_str}\n\n"
        f"{provide}"
    )


if __name__ == "__main__":
    from src.inference.classifier import classify, detect_language
    q = "गेहूं में पीली पत्तियां दिख रही हैं, क्या करें?"
    lang = detect_language(q)
    docs = [{"id": "r1", "score": 0.82, "text": "Wheat yellow leaves are often caused by nitrogen deficiency."}]
    print(build_prompt(q, docs, "No weather data available.", classify(q), lang))
    print("\n" + "=" * 60 + "\n")
    q2 = "How much urea for paddy at tillering in Bihar?"
    print(build_prompt(q2, [], "Rain expected tomorrow.", classify(q2), "en")[:400], "...")
