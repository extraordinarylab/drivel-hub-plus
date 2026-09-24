#!/usr/bin/env python3
"""Per-system totals under every judge, for the judge-robustness appendix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Mode labels must match scripts/make_latex_tables.py exactly. Verified against
# the checkpoints: both Qwen3-Omni rows are the SAME Thinking checkpoint run
# with thinking on and off (Qwen3-Omni-30B-A3B-Instruct is a different
# checkpoint and is the open judge, not a graded system); Gemma 4 has
# enable_thinking defaulting off, so its runs are a real No-thinking choice;
# Qwen2.5-Omni, MiniCPM-o and EchoInk-R1 have no thinking mode at all.
NAMES = {
    "human-baseline": ("Human baseline", "--"),
    "Qwen/Qwen3-Omni-30B-A3B-Thinking": ("Qwen3-Omni 30B-A3B", "Thinking"),
    "Qwen/Qwen3-Omni-30B-A3B-No-Thinking": ("Qwen3-Omni 30B-A3B", "No-thinking"),
    "openbmb/MiniCPM-o-2_6": ("MiniCPM-o-2.6 9B", "Instruct"),
    "harryhsing/EchoInk-R1-7B": ("EchoInk-R1 7B", "Instruct"),
    "Qwen/Qwen2.5-Omni-7B": ("Qwen2.5-Omni 7B", "Instruct"),
    "Qwen/Qwen2.5-Omni-3B": ("Qwen2.5-Omni 3B", "Instruct"),
    "google/gemma-4-E4B-it": ("Gemma 4 E4B", "No-thinking"),
    "google/gemma-4-E2B-it": ("Gemma 4 E2B", "No-thinking"),
}


def load(judgments: Path, judge: str) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    root = judgments / judge
    for path in sorted(root.rglob("judgments.jsonl")):
        rows = {}
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                if "judge_score_total" in row:
                    rows[row["file"]] = row
        out[str(path.parent.relative_to(root))] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judgments", type=Path, default=Path("judgments"))
    ap.add_argument("--judges", nargs="+", required=True,
                    help="judge directory names, primary first")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    loaded = [load(args.judgments, j) for j in args.judges]
    systems = [s for s in NAMES if all(s in l for l in loaded)]
    # Score every judge on the rows all of them managed to grade, so the
    # columns are comparable rather than each judge's own best subset.
    common = {s: set.intersection(*[set(l[s]) for l in loaded]) for s in systems}
    paired = sum(len(v) for v in common.values())

    stats = []
    for s in systems:
        files = sorted(common[s])
        row = {"system": s, "n": len(files)}
        for i, l in enumerate(loaded):
            row[f"t{i}"] = sum(l[s][f]["judge_score_total"] for f in files) / len(files)
            row[f"a{i}"] = 100 * sum(bool(l[s][f]["judge_aligned"]) for f in files) / len(files)
        stats.append(row)

    ranks = [{r["system"]: k + 1 for k, r in
              enumerate(sorted(stats, key=lambda r: -r[f"t{i}"]))}
             for i in range(len(loaded))]

    cols = "ll" + "ccc" * len(loaded)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\color{revcolor}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Setting}}"]
    for label in args.labels:
        lines.append(rf"& \multicolumn{{3}}{{c}}{{\textbf{{{label}}}}}")
    lines[-1] += r" \\"
    lines.append(" ".join(rf"\cmidrule(lr){{{3+3*i}-{5+3*i}}}" for i in range(len(loaded))))
    lines.append("& & " + " & ".join(
        [r"Total/12 & Aligned\% & Rank"] * len(loaded)) + r" \\")
    lines.append(r"\midrule")
    for r in sorted(stats, key=lambda r: -r["t0"]):
        name, setting = NAMES[r["system"]]
        cells = []
        for i in range(len(loaded)):
            cells.append(f"{r[f't{i}']:.2f} & {r[f'a{i}']:.1f} & {ranks[i][r['system']]}")
        lines.append(f"{name} & {setting} & " + " & ".join(cells) + r" \\")
        if r["system"] == "human-baseline":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    # The caption is text mode, so a plain comma is the separator; the count is
    # spelled out as clips x systems because "9,000" on its own reads as a
    # mismatch with the 1,000-clip corpus.
    paired_tex = f"{paired:,}"
    judges = " and ".join(args.labels) if len(args.labels) == 2 else \
        ", ".join(args.labels[:-1]) + " and " + args.labels[-1]
    lines.append(
        rf"\caption{{The same systems graded by {judges}. Each judge scores one "
        rf"explanation per clip for every system in the table, {paired_tex} "
        rf"judgements per judge. The first column pair is the proprietary "
        rf"grading used in Table~\ref{{tab:generation}}; the human baseline is "
        rf"the reference ceiling.}}")
    lines += [r"\label{tab:judge-agreement}", r"\end{table*}"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"paired items: {paired}")
    for r in sorted(stats, key=lambda r: -r["t0"]):
        print(f"  {r['system']:42} " + "  ".join(
            f"{r[f't{i}']:6.2f}/{r[f'a{i}']:5.1f}%" for i in range(len(loaded))))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
