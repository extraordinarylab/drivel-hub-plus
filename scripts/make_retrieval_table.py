#!/usr/bin/env python3
"""Render retrieval results in the paper's Table 3 layout.

Reads the per-model JSON that retrieval.py writes and emits the same two-block
table: a \\multirow header spanning Text-to-Video and Video-to-Text, one row per
model, best/worst three shaded per column.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

METRICS = ["R@1", "R@5", "R@10", "MRR", "N@5", "N@10"]
METRIC_KEYS = {
    "R@1": "recall@1", "R@5": "recall@5", "R@10": "recall@10",
    "MRR": "mrr", "N@5": "ndcg@5", "N@10": "ndcg@10",
}

# out-name -> (display, group). Order here is the order in the table.
DISPLAY = {
    "jina-v5-omni-nano-retrieval":  ("jina-v5-omni-nano", "native"),
    "jina-v5-omni-small-retrieval": ("jina-v5-omni-small", "native"),
    "omni-embed-nemotron-3b":       ("omni-embed-nemotron-3b", "native"),
    "e5-omni-7B":                   ("e5-omni-7B", "native"),
    "LCO-Omni-3B":                  ("LCO-Omni-3B", "native"),
    "LCO-Omni-7B":                  ("LCO-Omni-7B", "native"),
    "Qwen2.5-Omni-3B":              ("Qwen2.5-Omni-3B", "generative"),
    "Qwen2.5-Omni-7B":              ("Qwen2.5-Omni-7B", "generative"),
    "Qwen3-Omni-30B-A3B-Thinking":  ("Qwen3-Omni-30B-A3B", "generative"),
}
GROUPS = [
    ("native", "Embedding-native retrieval models"),
    ("generative", "Representative generative models adapted for retrieval"),
]


def pick(payload: dict, direction: str, metric: str) -> float | None:
    """retrieval.py nests by direction; accept a few spellings of each key."""
    alias = {"t2v": "text_to_video", "v2t": "video_to_text"}
    block = payload.get(alias[direction]) or payload.get(direction) or {}
    key = METRIC_KEYS[metric]
    if key not in block:
        return None
    value = float(block[key])
    # retrieval.py reports fractions; the paper's table is in percent.
    return value * 100 if value <= 1.0 else value


def rank_marks(values: list[float | None]) -> dict[int, str]:
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    if not present:
        return {}
    ordered = sorted({round(v, 4) for _, v in present}, reverse=True)
    marks: dict[int, str] = {}
    for rank, letter in zip(ordered[:3], ("bestA", "bestB", "bestC")):
        for i, v in present:
            if round(v, 4) == rank:
                marks[i] = letter
    for rank, letter in zip(list(reversed(ordered))[:3], ("worstA", "worstB", "worstC")):
        for i, v in present:
            if round(v, 4) == rank and i not in marks:
                marks[i] = letter
    return marks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=Path("retrieval"))
    parser.add_argument("--out", type=Path, default=Path("latex/retrieval_updated.tex"))
    args = parser.parse_args()

    rows = []
    for name, (display, group) in DISPLAY.items():
        path = args.retrieval / f"{name}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores = {d: [pick(payload, d, m) for m in METRICS] for d in ("t2v", "v2t")}
        rows.append((display, group, scores))

    if not rows:
        raise SystemExit(f"no retrieval results under {args.retrieval}")

    marks = {}
    for direction in ("t2v", "v2t"):
        for col in range(len(METRICS)):
            marks[(direction, col)] = rank_marks([r[2][direction][col] for r in rows])

    def cell(value: float | None, mark: str | None) -> str:
        if value is None:
            return "--"
        text = f"{value:.1f}"
        return f"\\{mark}{{{text}}}" if mark else text

    lines = [
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\resizebox{\\linewidth}{!}{",
        "\\begin{tabular}{lcccccccccccc}", "\\toprule",
        "\\multirow{2}{*}{\\textbf{Model}}",
        "& \\multicolumn{6}{c}{\\textbf{Text-to-Video}}",
        "& \\multicolumn{6}{c}{\\textbf{Video-to-Text}} \\\\",
        "\\cmidrule(lr){2-7} \\cmidrule(lr){8-13}",
        "& " + " & ".join(METRICS) + "",
        "& " + " & ".join(METRICS) + " \\\\",
        "\\midrule",
    ]
    first = True
    for key, title in GROUPS:
        members = [(i, r) for i, r in enumerate(rows) if r[1] == key]
        if not members:
            continue
        if not first:
            lines.append("\\midrule")
        first = False
        lines += ["", f"\\multicolumn{{13}}{{l}}{{\\textit{{{title}}}}} \\\\", ""]
        for i, (display, _, scores) in members:
            t2v = " & ".join(cell(scores["t2v"][c], marks[("t2v", c)].get(i))
                             for c in range(len(METRICS)))
            v2t = " & ".join(cell(scores["v2t"][c], marks[("v2t", c)].get(i))
                             for c in range(len(METRICS)))
            lines += [display, f"& {t2v}", f"& {v2t} \\\\", ""]

    lines += [
        "\\bottomrule", "\\end{tabular}", "}",
        "\\caption{Text-to-video and video-to-text retrieval performance on the revised "
        "DrivelHub+ split. Scores are percentages. Green shading marks the top three results "
        "in each column, and purple shading marks the bottom three.}",
        "\\label{tab:retrieval}", "\\end{table*}",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(rows)} model row(s)")


if __name__ == "__main__":
    main()
