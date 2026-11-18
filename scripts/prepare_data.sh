#!/bin/bash -e
# Run the full data pipeline: collect -> clean -> deduplicate -> split.
# All parameters are read from config.yaml (via src/data/config_loader.py).
cd "$(dirname "$0")/.."   # repo root
python3 -m src.data.prepare
