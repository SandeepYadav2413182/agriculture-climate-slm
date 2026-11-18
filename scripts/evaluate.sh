#!/bin/bash -e
# Run the full evaluation suite:
#   - src/evaluation/evaluate.py      (base model / fine-tuned eval)
#   - src/evaluation/evaluate_rag.py  (RAG pipeline eval)
#   - src/evaluation/benchmark.py     (benchmarks + hallucination/safety tests)
cd "$(dirname "$0")/.."   # repo root
echo "=== evaluate ==="
python3 -m src.evaluation.evaluate "$@"
echo "=== evaluate_rag ==="
python3 -m src.evaluation.evaluate_rag "$@"
echo "=== benchmark ==="
python3 -m src.evaluation.benchmark "$@"
