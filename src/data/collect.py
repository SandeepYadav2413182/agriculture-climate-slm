"""Collect raw records from the bundled JSONL dataset (stdlib only).

Reads the raw-data path from config and returns every JSONL record as a
list of dicts. Any missing file or malformed line raises a clear error so
the pipeline fails fast instead of silently training on partial data.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config, repo_root  # noqa: E402


def collect(cfg: dict) -> list[dict]:
    """Load raw records from ``cfg["raw_data_path"]``.

    Raises:
        FileNotFoundError: if the raw data file does not exist.
        ValueError: if a non-empty line is not valid JSON or not an object.
    """
    raw_path = Path(cfg["raw_data_path"])
    if not raw_path.is_absolute():
        raw_path = repo_root() / raw_path
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_path}. "
            f"Check 'raw_data_path' in config.yaml (repo root: {repo_root()})."
        )

    records: list[dict] = []
    with raw_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue  # tolerate blank lines
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {raw_path}:{lineno}: {exc}"
                ) from exc
            if not isinstance(rec, dict):
                raise ValueError(
                    f"Line {lineno} of {raw_path} is not a JSON object."
                )
            records.append(rec)
    return records


def main() -> None:
    cfg = load_config()
    records = collect(cfg)
    print(f"Collected {len(records)} records from {cfg['raw_data_path']}")


if __name__ == "__main__":
    main()
