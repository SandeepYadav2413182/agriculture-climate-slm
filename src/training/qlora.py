"""QLoRA building blocks: 4-bit quantization config + LoRA adapter config.

All hyper-parameters come from ``config.yaml`` (via ``src.data.config_loader``);
nothing is hard-coded here. Heavy third-party imports (``transformers``,
``peft``) are done inside the functions so that importing this module on a
machine without them raises no error — a clear ``ImportError`` is raised only
when the builder is actually called.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_bnb_config(cfg: Dict[str, Any]) -> Optional[Any]:
    """Build a bitsandbytes 4-bit quantization config (QLoRA style).

    Returns ``None`` when ``cfg["quantization"] != "4bit"`` (i.e. full-precision
    training is wanted). When requested, uses 4-bit NF4, bfloat16 compute dtype
    and double quantization — the standard QLoRA recipe.

    Raises:
        ImportError: if ``transformers`` is not installed and 4-bit was asked for.
    """
    if str(cfg.get("quantization", "")).lower() != "4bit":
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "4-bit quantization requested (quantization: 4bit) but the required "
            "packages are missing. Install with: pip install torch transformers "
            "bitsandbytes"
        ) from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(cfg: Dict[str, Any]) -> Any:
    """Build the PEFT LoRA config for causal-LM fine-tuning.

    Rank / alpha / dropout are read from ``cfg`` (``lora_rank``,
    ``lora_alpha``, ``lora_dropout``).

    Raises:
        ImportError: if ``peft`` is not installed.
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "LoRA training requested but 'peft' is not installed. "
            "Install with: pip install peft"
        ) from exc

    return LoraConfig(
        r=int(cfg["lora_rank"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
