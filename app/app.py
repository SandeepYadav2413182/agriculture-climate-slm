"""Gradio demo for the agriculture-climate-slm portfolio project.

Retrieval-only mode works on CPU — no model weights or GPU needed.
Run from the repo root:  python app/app.py
Requires: pip install -r requirements.txt   (gradio>=4.0)
"""

import sys
from pathlib import Path

# Make `src.` importable whether launched as `python app/app.py` or from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import gradio as gr
except ImportError as exc:  # clear, actionable error instead of a traceback
    raise SystemExit(
        "gradio is not installed. Install it with:\n"
        "    pip install -r requirements.txt\n"
        "or:\n"
        "    pip install 'gradio>=4.0'"
    ) from exc

from src.inference.pipeline import AdvisoryPipeline

CROPS = ["rice", "wheat", "maize", "cotton", "sugarcane", "soybean", "tomato"]
STAGES = ["sowing", "vegetative", "flowering", "maturity", "harvest"]

# Retrieval-only pipeline — works on CPU, no weights needed.
pipe = AdvisoryPipeline(str(ROOT / "config.yaml"))


def format_sources(sources):
    """Render retrieved passages as markdown: id + score per source."""
    lines = []
    for s in sources or []:
        sid = s.get("id", "?")
        score = s.get("score", 0.0)
        text = (s.get("text") or "").strip().replace("\n", " ")
        lines.append(f"- **{sid}** · score `{score:.3f}`\n  > {text[:600]}")
    return "\n\n".join(lines) if lines else "_No sources retrieved._"


def get_advice(location, crop, stage, question):
    """Call the advisory pipeline and format the three outputs."""
    if not (question or "").strip():
        return (
            "Please enter a question first.",
            "",
            "_No sources retrieved._",
        )
    result = pipe.answer(
        question=question.strip(),
        location=(location or "").strip(),
        crop=crop,
        crop_stage=stage,
        language=None,  # auto-detect (Devanagari heuristic for Hindi)
    )

    answer = result.get("answer", "")
    safety = result.get("safety", {}) or {}

    # Surface safety outcome in the recommendation pane.
    if not safety.get("passed", True) or safety.get("flags"):
        risk = safety.get("risk_level", "unknown")
        flags = ", ".join(safety.get("flags", [])) or "none listed"
        answer += (
            f"\n\n---\n⚠️ **Safety check:** {risk} risk "
            f"(passed={safety.get('passed')}, flags: {flags}). "
            "Verify with a local agronomist before acting."
        )

    weather = result.get("weather_context", "") or ""
    sources_md = format_sources(result.get("sources"))
    return answer, weather, sources_md


with gr.Blocks(title="Agriculture & Climate Assistant") as demo:
    gr.Markdown("# 🌾 Agriculture & Climate Assistant")
    gr.Markdown(
        "Ask a farming question — get advice grounded in retrieved agronomy guidance "
        "plus local climate context. English and Hindi (हिन्दी) supported; "
        "language is auto-detected."
    )
    with gr.Row():
        location = gr.Textbox(label="Location", placeholder="e.g. Ludhiana, Punjab")
        crop = gr.Dropdown(choices=CROPS, value="rice", label="Crop")
        stage = gr.Dropdown(choices=STAGES, value="vegetative", label="Stage")
    question = gr.Textbox(
        label="Question",
        placeholder="e.g. My rice leaves are yellowing at tillering stage — what should I do?",
        lines=3,
    )
    btn = gr.Button("Get Advice", variant="primary")
    with gr.Row():
        recommendation = gr.Markdown(label="Recommendation")
    climate = gr.Textbox(label="Climate Context", lines=4)
    sources = gr.Markdown(label="Sources")

    btn.click(
        fn=get_advice,
        inputs=[location, crop, stage, question],
        outputs=[recommendation, climate, sources],
    )


if __name__ == "__main__":
    demo.launch()
