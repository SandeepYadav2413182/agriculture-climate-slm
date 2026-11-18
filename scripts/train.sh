#!/bin/bash -e
# Run model training (src/training/train.py).
cd "$(dirname "$0")/.."   # repo root
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU detected:"
  nvidia-smi --query-gpu=name,memory.total --format=csv || true
else
  echo "WARNING: nvidia-smi not found - no GPU detected. Continuing on CPU (slow)."
fi
python3 -m src.training.train "$@"
