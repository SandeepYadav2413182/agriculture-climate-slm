"""Dataset download stub.

The Kaggle competition this project was seeded from closed on
2026-09-09, so the official competition dataset is no longer
downloadable through the Kaggle API. This stub documents that fact
and points at the bundled sample instead.

Use the bundled sample seed dataset at ``data/raw/sample_seed.jsonl``
(~295 curated agriculture/climate QA records, 17 fields) as the raw
input for the pipeline: ``bash scripts/prepare_data.sh``.

If an official dataset becomes available again, replace this stub with
a downloader that fetches it (e.g. via the Kaggle API) and writes to
``data/raw/``, then re-run the pipeline.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    print(
        "Dataset download: the Kaggle competition closed on 2026-09-09, "
        "so the official dataset is unavailable."
    )
    print(
        "Using the bundled sample seed dataset instead: "
        "data/raw/sample_seed.jsonl"
    )
    print("Run `bash scripts/prepare_data.sh` to build the pipeline outputs.")


if __name__ == "__main__":
    main()
