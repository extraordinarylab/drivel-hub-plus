#!/usr/bin/env python3
"""Per-query gold ranks for two retrieval models, in both directions.

Backs the case study in Appendix "Direction-Specific Retrieval Failures":
aggregate R@1 hides that two checkpoints fail on different clips, and that a
given failure often appears in only one retrieval direction. Ranking follows
``scripts/retrieval.py`` -- cosine similarity over L2-normalised vectors, with
the qrels supplying any extra relevant documents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.linalg.norm(left, axis=1, keepdims=True)
    right = right / np.linalg.norm(right, axis=1, keepdims=True)
    return left @ right.T


def gold_ranks(
    scores: np.ndarray, file_ids: list[str], qrels: dict[str, dict[str, int]],
) -> np.ndarray:
    """Best rank over the relevant documents, one entry per query."""
    index = {file_id: j for j, file_id in enumerate(file_ids)}
    ranks = []
    for i, query_id in enumerate(file_ids):
        relevant = qrels.get(query_id, {query_id: 1})
        documents = [
            document_id
            for document_id, relevance in relevant.items()
            if document_id in index and int(relevance) > 0
        ] or [query_id]
        row = scores[i]
        ranks.append(min(int((row > row[index[d]]).sum()) + 1 for d in documents))
    return np.asarray(ranks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=Path("embeddings"))
    parser.add_argument("--qrels", type=Path, default=Path("qrels.json"))
    parser.add_argument("--models", nargs=2, metavar="MODEL",
                        default=["Qwen2.5-Omni-3B", "Qwen2.5-Omni-7B"])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--good-rank", type=int, default=3,
                        help="a case is 'solved' at this rank or better")
    parser.add_argument("--bad-rank", type=int, default=300,
                        help="a case is 'missed' at this rank or worse")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qrels = json.loads(args.qrels.read_text(encoding="utf-8"))
    first, second = args.models

    per_model: dict[str, dict[str, np.ndarray]] = {}
    file_ids: list[str] | None = None
    annotations: dict[str, str] = {}
    for model in args.models:
        rows = read_jsonl(args.embeddings / f"{model}.jsonl")
        ids = [str(row["file"]) for row in rows]
        if file_ids is None:
            file_ids = ids
            annotations = {str(r["file"]): r.get("annotation", "") for r in rows}
        elif ids != file_ids:
            raise ValueError(f"{model} does not share the row order of {args.models[0]}")
        text = np.asarray([row["query_embedding"] for row in rows], dtype=np.float32)
        video = np.asarray([row["corpus_embedding"] for row in rows], dtype=np.float32)
        per_model[model] = {
            "t2v": gold_ranks(cosine_matrix(text, video), ids, qrels),
            "v2t": gold_ranks(cosine_matrix(video, text), ids, qrels),
        }

    assert file_ids is not None
    records = []
    for i, file_id in enumerate(file_ids):
        record = {"file": file_id, "annotation": annotations.get(file_id, "")}
        for direction in ("t2v", "v2t"):
            for model in args.models:
                record[f"{direction}_{model}"] = int(per_model[model][direction][i])
        records.append(record)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )

    for direction in ("t2v", "v2t"):
        for winner, loser in ((first, second), (second, first)):
            cases = [
                r for r in records
                if r[f"{direction}_{winner}"] <= args.good_rank
                and r[f"{direction}_{loser}"] >= args.bad_rank
            ]
            cases.sort(key=lambda r: (r[f"{direction}_{winner}"], -r[f"{direction}_{loser}"]))
            print(f"\n=== {direction}: {winner} solves, {loser} misses "
                  f"({len(cases)} cases) ===")
            for record in cases:
                print(f"  {record[f'{direction}_{winner}']:>4} vs "
                      f"{record[f'{direction}_{loser}']:>4}  {record['file']}")
                print(f"        {record['annotation'][:150]}")


if __name__ == "__main__":
    main()
