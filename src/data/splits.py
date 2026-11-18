"""Stratified train/val/test splits with a leakage guard.

Stratification key: (topic, language). Within each stratum the records
are shuffled (seeded RNG) and allocated so that ``test_size`` and
``val_size`` fractions go to test/val while at least one record per
stratum stays in train.

Leakage guard: every val/test record whose question has token-Jaccard
similarity >= dedup_similarity_threshold with ANY train record is moved
back to train, so near-duplicates can never inflate eval scores.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config, repo_root  # noqa: E402
from src.data.deduplicate import jaccard  # noqa: E402


def _distributions(records: list[dict]) -> dict:
    """Per-split language and topic counts (plain dicts for JSON)."""
    return {
        "language": dict(Counter(r.get("language", "") for r in records)),
        "topic": dict(Counter(r.get("topic", "") for r in records)),
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def stratified_split(records: list[dict], cfg: dict) -> dict:
    """Split records into train/val/test; write JSONL files; return stats."""
    seed = int(cfg.get("seed", 42))
    test_size = float(cfg.get("test_size", 0.15))
    val_size = float(cfg.get("val_size", 0.15))
    threshold = float(cfg.get("dedup_similarity_threshold", 0.85))

    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)

    # Group by (topic, language), preserving shuffled order within strata.
    strata: dict[tuple[str, str], list[dict]] = {}
    for rec in records:
        key = (rec.get("topic", ""), rec.get("language", ""))
        strata.setdefault(key, []).append(rec)

    train, val, test = [], [], []
    for key in sorted(strata):
        group = strata[key]
        n = len(group)
        n_test = round(n * test_size)
        n_val = round(n * val_size)
        # Guarantee at least one record stays in train.
        while n - n_test - n_val < 1 and (n_test + n_val) > 0:
            if n_val > n_test:
                n_val -= 1
            else:
                n_test -= 1
        rng.shuffle(group)
        test.extend(group[:n_test])
        val.extend(group[n_test:n_test + n_val])
        train.extend(group[n_test + n_val:])

    # Leakage guard: move val/test records too similar to any train
    # question back into train.
    n_moved = 0
    train_questions = [r.get("question", "") for r in train]
    for split in (val, test):
        survivors = []
        for rec in split:
            q = rec.get("question", "")
            if any(jaccard(q, tq) >= threshold for tq in train_questions):
                train.append(rec)
                train_questions.append(q)
                n_moved += 1
            else:
                survivors.append(rec)
        split[:] = survivors

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    paths = {
        "train": repo_root() / cfg["train_path"],
        "val": repo_root() / cfg["validation_path"],
        "test": repo_root() / cfg["test_path"],
    }
    _write_jsonl(paths["train"], train)
    _write_jsonl(paths["val"], val)
    _write_jsonl(paths["test"], test)

    return {
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "n_moved_by_leakage_guard": n_moved,
        "train_path": str(paths["train"]),
        "validation_path": str(paths["val"]),
        "test_path": str(paths["test"]),
        "distributions": {
            "train": _distributions(train),
            "val": _distributions(val),
            "test": _distributions(test),
        },
    }


def main() -> None:
    """Re-split the cleaned dataset (expects data/processed/cleaned.jsonl)."""
    cfg = load_config()
    cleaned_path = repo_root() / cfg["processed_data_path"]
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {cleaned_path}; "
            "run `python3 -m src.data.prepare` first."
        )
    records = [
        json.loads(line)
        for line in cleaned_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stats = stratified_split(records, cfg)
    print(json.dumps(stats["sizes"], indent=2))


if __name__ == "__main__":
    main()
