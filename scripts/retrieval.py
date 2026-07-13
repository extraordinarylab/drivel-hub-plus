#!/usr/bin/env python3
"""Evaluate Table 3 text/video retrieval from paired embedding JSONL files.

Every JSONL row must contain ``file``, ``query_embedding`` (annotation text),
and ``corpus_embedding`` (video). The implementation intentionally uses the
same shared file IDs and qrels in both directions as the original experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ranx import Qrels, Run, evaluate


METRICS = [
    "recall@1", "recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10",
]
DISPLAY_NAMES = {
    "recall@1": "R@1",
    "recall@5": "R@5",
    "recall@10": "R@10",
    "mrr": "MRR",
    "ndcg@5": "N@5",
    "ndcg@10": "N@10",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def validate(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("embedding JSONL is empty")
    required = {"file", "query_embedding", "corpus_embedding"}
    seen: set[str] = set()
    query_dim = corpus_dim = None
    for index, row in enumerate(rows, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {index} is missing: {', '.join(sorted(missing))}")
        file_id = str(row["file"])
        if file_id in seen:
            raise ValueError(f"duplicate file ID: {file_id}")
        seen.add(file_id)
        qdim = len(row["query_embedding"])
        cdim = len(row["corpus_embedding"])
        query_dim = qdim if query_dim is None else query_dim
        corpus_dim = cdim if corpus_dim is None else corpus_dim
        if qdim != query_dim or cdim != corpus_dim:
            raise ValueError("embedding dimensions are inconsistent across rows")
    if query_dim != corpus_dim:
        raise ValueError(f"text/video dimensions differ: {query_dim} vs {corpus_dim}")


def cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    if np.any(left_norm == 0) or np.any(right_norm == 0):
        raise ValueError("zero-norm embedding found")
    return (left / left_norm) @ (right / right_norm).T


def make_run(
    query_vectors: np.ndarray,
    corpus_vectors: np.ndarray,
    file_ids: list[str],
    qrels: dict[str, dict[str, int]],
    name: str,
) -> tuple[Qrels, Run]:
    scores = cosine_matrix(query_vectors, corpus_vectors)
    available = set(file_ids)
    qrels_for_run: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for i, query_id in enumerate(file_ids):
        relevant = qrels.get(query_id, {query_id: 1})
        qrels_for_run[query_id] = {
            document_id: int(relevance)
            for document_id, relevance in relevant.items()
            if document_id in available and int(relevance) > 0
        }
        if not qrels_for_run[query_id]:
            qrels_for_run[query_id] = {query_id: 1}
        run[query_id] = {
            document_id: float(scores[i, j])
            for j, document_id in enumerate(file_ids)
        }
    return Qrels(qrels_for_run), Run(run, name=name)


def report(label: str, results: dict[str, float]) -> None:
    print(label)
    print("  " + "  ".join(
        f"{DISPLAY_NAMES.get(metric, metric)}={100 * float(results[metric]):.1f}"
        for metric in METRICS
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, default=Path("qrels.json"))
    parser.add_argument("--direction", choices=("t2v", "v2t", "both"), default="both")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.embeddings)
    validate(rows)
    qrels = json.loads(args.qrels.read_text(encoding="utf-8"))
    file_ids = [str(row["file"]) for row in rows]
    text = np.asarray([row["query_embedding"] for row in rows], dtype=np.float32)
    video = np.asarray([row["corpus_embedding"] for row in rows], dtype=np.float32)
    all_results: dict[str, dict[str, float]] = {}

    if args.direction in {"t2v", "both"}:
        t2v_qrels, t2v_run = make_run(text, video, file_ids, qrels, "T2V")
        all_results["text_to_video"] = evaluate(t2v_qrels, t2v_run, METRICS)
        report("Text-to-Video", all_results["text_to_video"])
    if args.direction in {"v2t", "both"}:
        v2t_qrels, v2t_run = make_run(video, text, file_ids, qrels, "V2T")
        all_results["video_to_text"] = evaluate(v2t_qrels, v2t_run, METRICS)
        report("Video-to-Text", all_results["video_to_text"])

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(all_results, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )


if __name__ == "__main__":
    main()
