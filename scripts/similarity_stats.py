#!/usr/bin/env python3
"""Summarise how separable each retrieval model's embedding space is.

For every model: the cosine between a narrative and its own clip (the positive
pairs) against the cosine between a narrative and every other clip (the
negatives). Retrieval only works when the positives sit above the negatives, so
the gap between those two distributions is the quantity of interest — and it is
what exposed Qwen3-Omni, whose positives land *below* its negatives.

Emits summary statistics plus fixed-bin histograms, so the figure never has to
carry a million raw negative pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BINS = np.linspace(-1.0, 1.0, 81)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=Path("embeddings"))
    parser.add_argument("--out", type=Path, default=Path("retrieval/similarity_stats.json"))
    args = parser.parse_args()

    payload: dict[str, dict] = {}
    for path in sorted(args.embeddings.glob("*.jsonl")):
        # `*.rankN.jsonl` are shards of a merged file; `*.bak*.jsonl` are
        # snapshots kept while re-embedding. Neither is a model.
        if "rank" in path.name or ".bak" in path.name:
            continue
        queries, corpus = [], []
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            queries.append(row["query_embedding"])
            corpus.append(row["corpus_embedding"])
        q = np.asarray(queries, dtype=np.float32)
        c = np.asarray(corpus, dtype=np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        sims = q @ c.T
        n = len(q)
        positives = np.diag(sims).copy()
        off = ~np.eye(n, dtype=bool)
        negatives = sims[off]

        payload[path.stem] = {
            "n": n,
            "positive": {
                "mean": float(positives.mean()),
                "std": float(positives.std()),
                "median": float(np.median(positives)),
                "hist": np.histogram(positives, bins=BINS, density=True)[0].round(4).tolist(),
            },
            "negative": {
                "mean": float(negatives.mean()),
                "std": float(negatives.std()),
                "median": float(np.median(negatives)),
                "hist": np.histogram(negatives, bins=BINS, density=True)[0].round(4).tolist(),
            },
            "gap": float(positives.mean() - negatives.mean()),
            "recall_at_1": float((sims.argmax(1) == np.arange(n)).mean()),
        }
        print(f"  {path.stem:34} n={n:4} pos={positives.mean():.4f} "
              f"neg={negatives.mean():.4f} gap={payload[path.stem]['gap']:+.4f} "
              f"R@1={100 * payload[path.stem]['recall_at_1']:.1f}%", flush=True)
        del sims, negatives

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"bin_edges": BINS.round(4).tolist(), "models": payload}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.out} ({len(payload)} models)")


if __name__ == "__main__":
    main()
