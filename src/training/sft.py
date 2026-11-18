"""Supervised fine-tuning (SFT) data preparation + TRL SFTTrainer factory.

* :func:`prepare_datasets` reads the train/validation JSONL files, formats each
  record into instruction-style text and tokenizes it with ``tokenizer_name``
  (truncated to ``max_seq_length``).
* :func:`create_sft_trainer` wires a ``trl.SFTTrainer`` with ``SFTConfig``
  built from ``config.yaml``.

Heavy imports (``datasets``, ``transformers``, ``trl``) are guarded so this
module imports safely on machines without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_example(record: Dict[str, Any]) -> str:
    """Render one JSONL record as instruction-style chat text.

    Accepted field variants:
        instruction/question, input (optional context), output/answer.
    """
    instruction = record.get("instruction") or record.get("question") or ""
    context = record.get("input") or record.get("context") or ""
    answer = record.get("output") or record.get("answer") or ""

    parts = [f"### Instruction:\n{instruction.strip()}"]
    if str(context).strip():
        parts.append(f"### Input:\n{str(context).strip()}")
    parts.append(f"### Response:\n{str(answer).strip()}")
    return "\n\n".join(parts)


def prepare_datasets(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    """Load, format and tokenize the train/validation splits.

    Returns:
        (train_hf_dataset, val_hf_dataset) — tokenized ``datasets.Dataset``
        objects with ``input_ids``/``attention_mask`` (``labels`` are derived
        from ``input_ids`` by the SFTTrainer).

    Raises:
        ImportError: if ``datasets``/``transformers`` are not installed.
        FileNotFoundError: if a split file does not exist.
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Dataset preparation needs 'datasets' and 'transformers'. "
            "Install with: pip install datasets transformers"
        ) from exc

    from src.data.config_loader import repo_root

    root = repo_root()
    train_path = root / cfg["train_path"]
    val_path = root / cfg["validation_path"]
    for p in (train_path, val_path):
        if not p.exists():
            raise FileNotFoundError(f"Split file not found: {p}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_len = int(cfg["max_seq_length"])

    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    val_ds = load_dataset("json", data_files=str(val_path), split="train")

    def _tokenize(batch):
        texts = [format_example(dict(zip(batch.keys(), vals)))
                 for vals in zip(*batch.values())]
        enc = tokenizer(texts, truncation=True, max_length=max_len)
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"]}

    keep = {"input_ids", "attention_mask"}
    train_ds = train_ds.map(_tokenize, batched=True,
                            remove_columns=[c for c in train_ds.column_names
                                            if c not in keep])
    val_ds = val_ds.map(_tokenize, batched=True,
                        remove_columns=[c for c in val_ds.column_names
                                        if c not in keep])
    return train_ds, val_ds


def create_sft_trainer(model, tokenizer, train_ds, val_ds, cfg: Dict[str, Any]):
    """Create a ``trl.SFTTrainer`` from config hyper-parameters.

    Raises:
        ImportError: if ``torch``/``trl`` are not installed.
    """
    try:
        import torch
        from trl import SFTTrainer, SFTConfig
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "SFT training needs 'torch' and 'trl'. "
            "Install with: pip install torch trl"
        ) from exc

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    sft_config = SFTConfig(
        output_dir=str(cfg["adapter_dir"]),
        num_train_epochs=float(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=int(cfg["batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation"]),
        bf16=use_bf16,
        gradient_checkpointing=bool(cfg["use_gradient_checkpointing"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        seed=int(cfg.get("seed", 42)),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        args=sft_config,
    )
    return trainer
