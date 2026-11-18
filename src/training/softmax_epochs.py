"""Iterative CPU baseline: multinomial softmax (logistic) regression trained
with SGD for a configurable number of epochs (numpy only).

Unlike the one-pass Naive Bayes baseline, this model trains iteratively, so
an epoch count is meaningful here. Default 500 epochs per project config.

Features: word unigrams + bigrams from the question (same Unicode-safe
tokenizer as the rest of the pipeline).

Usage (from repo root):
    python3 -m src.training.softmax_epochs            # 500 epochs (config)
    python3 -m src.training.softmax_epochs --epochs 50
Saves weights to models/softmax_<epochs>/ and prints test metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.data.config_loader import load_config, repo_root  # noqa: E402
from src.data.deduplicate import tokenize  # noqa: E402
from src.training.baseline_nb import evaluate as nb_evaluate  # noqa: E402


def featurize(text: str) -> list[str]:
    toks = tokenize(text)
    return toks + [toks[i] + "\u0001" + toks[i + 1] for i in range(len(toks) - 1)]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def train_sgd(train, topics, vocab_index, epochs=500, lr=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n_classes = len(topics)
    n_feat = len(vocab_index)
    t2i = {t: i for i, t in enumerate(topics)}
    W = np.zeros((n_classes, n_feat), dtype=np.float64)
    b = np.zeros(n_classes, dtype=np.float64)

    # sparse index lists per training record
    X = []
    y = []
    for r in train:
        idx = [vocab_index[f] for f in featurize(r["question"]) if f in vocab_index]
        if not idx:
            continue
        X.append(np.array(idx, dtype=np.int64))
        y.append(t2i[r["topic"]])
    y = np.array(y, dtype=np.int64)
    n = len(X)
    order = np.arange(n)

    for epoch in range(1, epochs + 1):
        lr_e = lr / (1.0 + 0.002 * epoch)
        rng.shuffle(order)
        total_loss = 0.0
        for i in order:
            idx = X[i]
            scores = W[:, idx].sum(axis=1) + b
            p = softmax(scores)
            total_loss += -np.log(p[y[i]] + 1e-12)
            g = p
            g[y[i]] -= 1.0
            W[:, idx] -= lr_e * g[:, None]
            b -= lr_e * g
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"[softmax] epoch {epoch}/{epochs}  mean_loss={total_loss / n:.4f}  lr={lr_e:.4f}",
                  flush=True)
    return W, b


class SoftmaxModel:
    """Thin wrapper so baseline_nb.evaluate can score this model."""

    def __init__(self, topics, vocab, W, b):
        self.topics = topics
        self.vocab = vocab
        self.W = W
        self.b = b
        self.v2i = {v: i for i, v in enumerate(vocab)}

    def predict_topk(self, text, top_k=3):
        idx = [self.v2i[f] for f in featurize(text) if f in self.v2i]
        if idx:
            scores = self.W[:, idx].sum(axis=1) + self.b
        else:
            scores = self.b.copy()
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.topics[i], float(scores[i])) for i in order]


def evaluate(model: SoftmaxModel, records) -> dict:
    # Same metric formulas as baseline_nb.evaluate.
    from collections import Counter
    t2i = {t: i for i, t in enumerate(model.topics)}
    correct = top3 = 0
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    lang_c: Counter = Counter()
    lang_t: Counter = Counter()
    n = len(records)
    for r in records:
        gold = r["topic"]
        ranked = model.predict_topk(r["question"], 3)
        pred = ranked[0][0]
        if pred == gold:
            correct += 1
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1
        if gold in [t for t, _ in ranked]:
            top3 += 1
        lang = r.get("language", "?")
        lang_t[lang] += 1
        if pred == gold:
            lang_c[lang] += 1
    f1s = []
    for t in model.topics:
        d = 2 * tp[t] + fp[t] + fn[t]
        f1s.append(2 * tp[t] / d if d else 0.0)
    return {
        "n_test": n,
        "accuracy": round(correct / n, 4),
        "top3_accuracy": round(top3 / n, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "per_language_accuracy": {l: round(lang_c[l] / lang_t[l], 4) for l in sorted(lang_t)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    epochs = args.epochs or int(cfg.get("epochs", 500))
    root = repo_root()
    train = load_jsonl(root / cfg["train_path"])
    test = load_jsonl(root / cfg["test_path"])
    topics = sorted({r["topic"] for r in train})
    vocab = sorted({f for r in train for f in featurize(r["question"])})
    v2i = {v: i for i, v in enumerate(vocab)}
    print(f"[softmax] train={len(train)} test={len(test)} topics={len(topics)} "
          f"features={len(vocab)} epochs={epochs}", flush=True)

    t0 = time.time()
    W, b = train_sgd(train, topics, v2i, epochs=epochs)
    print(f"[softmax] training done in {time.time() - t0:.1f}s", flush=True)

    model = SoftmaxModel(topics, vocab, W, b)
    metrics = evaluate(model, test)
    print("[softmax] test metrics:", json.dumps(metrics, indent=2))

    outdir = root / "models" / f"softmax_{epochs}"
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "W.npy", W)
    np.save(outdir / "b.npy", b)
    (outdir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (outdir / "topics.json").write_text(json.dumps(topics, ensure_ascii=False), encoding="utf-8")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[softmax] model saved to {outdir}")


if __name__ == "__main__":
    main()
