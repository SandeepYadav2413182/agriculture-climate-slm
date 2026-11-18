"""End-to-end QLoRA training entry point.

Stage A (optional, ``stage_a_enabled: true``): continued pre-training of the
base model on the ``context`` field corpus (causal LM, 1 epoch).
Stage B: QLoRA supervised fine-tuning of the base model on the SFT splits,
saving the adapter to ``adapter_dir`` and appending one row to
``experiments_csv``.

Run from the repo root::

    python -m src.training.train        # or: python src/training/train.py

All heavy imports live inside functions so ``import`` of this module works on
machines without torch/transformers/peft/trl.
"""

from __future__ import annotations

import csv
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


# --- repo-root preamble (only needed when run as a script, not via -m) -----
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - path setup
    sys.path.insert(0, str(ROOT))


def set_seed(seed: int) -> None:
    """Seed Python (and torch/numpy when importable) for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_tokenizer_and_base(cfg: Dict[str, Any]):
    """Load tokenizer + base model (4-bit NF4 when configured)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.training.qlora import build_bnb_config

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=build_bnb_config(cfg),
        device_map="auto",
    )
    return model, tokenizer


def stage_a_continued_pretraining(cfg: Dict[str, Any]) -> None:
    """Stage A: continued pre-training on the ``context`` corpus (1 epoch).

    Only called when ``stage_a_enabled`` is true. Trains a plain causal-LM
    head on the ``context`` fields of the training split so the model absorbs
    domain text before instruction tuning.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    from src.data.config_loader import repo_root

    train_path = repo_root() / cfg["train_path"]
    if not train_path.exists():
        raise FileNotFoundError(f"Training split not found: {train_path}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    corpus = load_dataset("json", data_files=str(train_path), split="train")
    max_len = int(cfg["max_seq_length"])

    def _tok(batch):
        texts = [str(batch["context"][i] or "") for i in range(len(batch["context"]))]
        return tokenizer(texts, truncation=True, max_length=max_len)

    corpus = corpus.map(_tok, batched=True,
                        remove_columns=[c for c in corpus.column_names
                                        if c not in {"input_ids", "attention_mask"}])

    model, _ = _load_tokenizer_and_base(cfg)

    args = TrainingArguments(
        output_dir=str(Path(cfg["adapter_dir"]).parent / "stage_a"),
        num_train_epochs=1,
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=int(cfg["batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation"]),
        bf16=True,
        save_strategy="no",
        logging_steps=25,
        report_to="none",
        seed=int(cfg.get("seed", 42)),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=corpus,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(Path(cfg["adapter_dir"]).parent / "stage_a"))


def _experiments_row(cfg, train_n, val_n, training_time_s, eval_score, notes) -> Dict[str, Any]:
    return {
        "experiment_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "base_model": cfg["model_name"],
        "dataset_version": cfg.get("dataset_version", ""),
        "train_samples": train_n,
        "validation_samples": val_n,
        "epochs": cfg["epochs"],
        "learning_rate": cfg["learning_rate"],
        "lora_rank": cfg["lora_rank"],
        "lora_alpha": cfg["lora_alpha"],
        "training_time_s": round(training_time_s, 1),
        "gpu_memory_mb": _gpu_memory_mb(),
        "eval_metric": "eval_loss",
        "eval_score": eval_score,
        "notes": notes,
    }


def _gpu_memory_mb():
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    except ImportError:
        pass
    return ""


def _append_experiment(cfg: Dict[str, Any], row: Dict[str, Any]) -> None:
    from src.data.config_loader import repo_root

    csv_path = repo_root() / cfg["experiments_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["experiment_id", "base_model", "dataset_version", "train_samples",
               "validation_samples", "epochs", "learning_rate", "lora_rank",
               "lora_alpha", "training_time_s", "gpu_memory_mb", "eval_metric",
               "eval_score", "notes"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in columns})


def main() -> None:
    from src.data.config_loader import load_config, repo_root
    from src.training.qlora import build_lora_config
    from src.training.sft import create_sft_trainer, prepare_datasets

    cfg = load_config()
    root = repo_root()
    set_seed(int(cfg.get("seed", 42)))
    print(f"[train] project={cfg['project_name']} base={cfg['model_name']}")

    # ---- Stage A (optional continued pre-training) ----------------------
    if bool(cfg.get("stage_a_enabled", False)):
        print("[train] Stage A: continued pre-training on context corpus (1 epoch)")
        stage_a_continued_pretraining(cfg)
    else:
        print("[train] Stage A skipped (stage_a_enabled: false)")

    # ---- Stage B (QLoRA SFT) --------------------------------------------
    print("[train] Stage B: loading base model + tokenizer")
    model, tokenizer = _load_tokenizer_and_base(cfg)

    from peft import get_peft_model

    model = get_peft_model(model, build_lora_config(cfg))
    model.print_trainable_parameters()

    print("[train] preparing SFT datasets")
    train_ds, val_ds = prepare_datasets(cfg)
    trainer = create_sft_trainer(model, tokenizer, train_ds, val_ds, cfg)

    print("[train] starting SFT")
    t0 = time.time()
    train_result = trainer.train()
    training_time_s = time.time() - t0

    adapter_dir = root / cfg["adapter_dir"]
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[train] adapter saved to {adapter_dir}")

    metrics = trainer.evaluate()
    eval_score = metrics.get("eval_loss", "")
    print(f"[train] eval_loss={eval_score} time_s={training_time_s:.1f}")

    _append_experiment(cfg, _experiments_row(
        cfg, len(train_ds), len(val_ds), training_time_s, eval_score,
        notes=f"qlora sft; quantization={cfg.get('quantization')}"))
    print(f"[train] logged to {cfg['experiments_csv']}")


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
