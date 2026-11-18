"""End-to-end data pipeline: collect -> clean -> deduplicate -> split.

Reads all parameters from config.yaml (paths, seed, test/val sizes,
dedup similarity threshold). Writes the final cleaned dataset to
``processed_data_path``, the train/val/test splits to their configured
paths, and ``data/processed/dataset_stats.json``. Prints a statistics
table to stdout.

Run from the repo root with: ``python3 -m src.data.prepare``
or via ``bash scripts/prepare_data.sh``.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.clean import EXPECTED_FIELDS  # noqa: E402
from src.data.clean import clean_records  # noqa: E402
from src.data.collect import collect  # noqa: E402
from src.data.config_loader import load_config, repo_root  # noqa: E402
from src.data.deduplicate import deduplicate  # noqa: E402
from src.data.splits import stratified_split  # noqa: E402


def _top(counter: Counter, n: int = 15) -> list[tuple[str, int]]:
    return counter.most_common(n)


def _fmt_table(rows: list[tuple[str, str]]) -> str:
    width = max(len(r[0]) for r in rows)
    return "\n".join(f"  {k:<{width}} : {v}" for k, v in rows)


def main() -> None:
    cfg = load_config()
    seed = cfg.get("seed")
    threshold = cfg["dedup_similarity_threshold"]
    root = repo_root()

    # 1) collect
    raw_records = collect(cfg)

    # 2) clean
    cleaned, clean_report = clean_records(raw_records)

    # 3) deduplicate
    deduped, dedup_report = deduplicate(cleaned, float(threshold))

    # 4) persist the final cleaned dataset
    cleaned_path = root / cfg["processed_data_path"]
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    with cleaned_path.open("w", encoding="utf-8") as fh:
        for rec in deduped:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 5) stratified splits
    split_stats = stratified_split(deduped, cfg)

    # 6) dataset-level statistics (computed on the final dataset)
    lang_counts = Counter(r.get("language", "") for r in deduped)
    topic_counts = Counter(r.get("topic", "") for r in deduped)
    crop_counts = Counter(r.get("crop", "") for r in deduped)

    q_lens = [len(r.get("question", "").split()) for r in deduped]
    a_lens = [len(r.get("answer", "").split()) for r in deduped]
    avg_q = sum(q_lens) / len(q_lens) if q_lens else 0.0
    avg_a = sum(a_lens) / len(a_lens) if a_lens else 0.0

    missing_summary = {
        field: count
        for field, count in clean_report["missing_counts"].items()
        if count > 0
    }

    stats = {
        "dataset_version": cfg.get("dataset_version"),
        "seed": seed,
        "n_raw": len(raw_records),
        "n_after_clean": clean_report["n_output"],
        "n_dropped_empty_qa": clean_report["n_dropped_empty_qa"],
        "n_exact_dupes": dedup_report["n_exact_dupes"],
        "n_near_dupes": dedup_report["n_near_dupes"],
        "duplicate_rate": dedup_report["duplicate_rate"],
        "n_final": len(deduped),
        "per_language": dict(lang_counts),
        "per_topic": dict(topic_counts),
        "per_crop_top15": _top(crop_counts),
        "missing_values": missing_summary,
        "avg_question_tokens": round(avg_q, 2),
        "avg_answer_tokens": round(avg_a, 2),
        "splits": split_stats,
        "schema_fields": EXPECTED_FIELDS,
    }

    stats_path = root / "data/processed/dataset_stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 7) printed statistics table
    dist = split_stats["distributions"]
    lines = [
        "=== agriculture-climate-slm dataset statistics ===",
        _fmt_table([
            ("raw records", str(len(raw_records))),
            ("after clean", str(clean_report["n_output"])),
            ("dropped (empty Q/A)", str(clean_report["n_dropped_empty_qa"])),
            ("exact duplicates", str(dedup_report["n_exact_dupes"])),
            ("near duplicates", str(dedup_report["n_near_dupes"])),
            ("duplicate rate", f"{dedup_report['duplicate_rate']:.2%}"),
            ("final records", str(len(deduped))),
            ("avg question tokens", f"{avg_q:.2f}"),
            ("avg answer tokens", f"{avg_a:.2f}"),
        ]),
        "",
        "--- per-language counts ---",
        _fmt_table([(k, str(v)) for k, v in lang_counts.most_common()]),
        "",
        "--- per-topic counts ---",
        _fmt_table([(k, str(v)) for k, v in topic_counts.most_common()]),
        "",
        "--- per-crop counts (top 15) ---",
        _fmt_table([(k, str(v)) for k, v in _top(crop_counts)]),
        "",
        "--- missing values (pre-fill) ---",
        (_fmt_table([(k, str(v)) for k, v in missing_summary.items()])
         if missing_summary else "  none"),
        "",
        "--- splits ---",
        _fmt_table([
            ("train", str(split_stats["sizes"]["train"])),
            ("val", str(split_stats["sizes"]["val"])),
            ("test", str(split_stats["sizes"]["test"])),
            ("moved by leakage guard",
             str(split_stats["n_moved_by_leakage_guard"])),
        ]),
        "",
        "--- train split distribution (language / topic) ---",
        _fmt_table([
            ("language", json.dumps(dist["train"]["language"],
                                    ensure_ascii=False)),
            ("topic", json.dumps(dist["train"]["topic"], ensure_ascii=False)),
        ]),
        "",
        "--- val split distribution (language / topic) ---",
        _fmt_table([
            ("language", json.dumps(dist["val"]["language"],
                                    ensure_ascii=False)),
            ("topic", json.dumps(dist["val"]["topic"], ensure_ascii=False)),
        ]),
        "",
        "--- test split distribution (language / topic) ---",
        _fmt_table([
            ("language", json.dumps(dist["test"]["language"],
                                    ensure_ascii=False)),
            ("topic", json.dumps(dist["test"]["topic"], ensure_ascii=False)),
        ]),
        "",
        f"wrote: {cfg['processed_data_path']}, "
        f"{cfg['train_path']}, {cfg['validation_path']}, "
        f"{cfg['test_path']}, data/processed/dataset_stats.json",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
