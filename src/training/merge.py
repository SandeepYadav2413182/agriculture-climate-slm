"""Merge a trained LoRA adapter back into the base model.

Loads the base model (``model_name``), attaches the adapter from
``adapter_dir``, runs ``merge_and_unload()`` and saves the standalone
merged model + tokenizer to ``merged_dir``.

Run from the repo root::

    python src/training/merge.py

All heavy imports are guarded so the module imports safely without torch.
"""

from __future__ import annotations

from pathlib import Path


import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - path setup
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.data.config_loader import load_config, repo_root

    cfg = load_config()
    root = repo_root()

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.training.qlora import build_bnb_config
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Merging needs 'torch', 'transformers' and 'peft'. "
            "Install with: pip install torch transformers peft"
        ) from exc

    adapter_dir = root / cfg["adapter_dir"]
    merged_dir = root / cfg["merged_dir"]
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    print(f"[merge] base model : {cfg['model_name']}")
    print(f"[merge] adapter    : {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=build_bnb_config(cfg),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"[merge] merged model saved to {merged_dir}")


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
