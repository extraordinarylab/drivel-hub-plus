#!/usr/bin/env python3
"""LaTeX table for the cascaded text-conversion baseline, under both judges.

The primary judge carries the argument and the open judge is the check, so both
belong in the table: the absolute scales differ by about a factor of two while
the proportion recovered barely moves, and that is the point the appendix makes.
Column groups follow tables/judge_agreement.tex.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The paper already has symbols for the three evidence sources, and the
# dataset-statistics table writes the combinations this way. Spelled-out
# headers were the widest thing in an 11-column table.
SYMBOL = {
    "Audio": r"\audiomod",
    "Audio + vision": r"\audiomod+\videomod",
    "On-screen text": r"\textmod",
    "Text + vision": r"\textmod+\videomod",
    "Vision only": r"\videomod",
}

CONDS = ("transcript-only", "caption-only", "ocr-asr")
LABEL = {"transcript-only": "ASR only", "caption-only": "OCR only",
         "ocr-asr": "OCR + ASR"}


def row_values(a: dict, cond: str | None) -> list[str]:
    """One judge's cells: the All column then one per modality group."""
    groups = a["by_group"]
    if cond is None:
        return [f"{a['multimodal']:.2f}"] + [f"{g['multimodal']:.2f}" for g in groups]
    overall = next(r["total"] for r in a["overall"] if r["condition"] == LABEL[cond])
    return [f"{overall:.2f}"] + [f"{g[cond]:.2f}" for g in groups]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--analysis-secondary", type=Path, required=True)
    ap.add_argument("--label", default="Gemini 3.8 Flash")
    ap.add_argument("--label-secondary", default="Qwen3-Omni-30B-A3B")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    a = json.loads(args.analysis.read_text(encoding="utf-8"))
    b = json.loads(args.analysis_secondary.read_text(encoding="utf-8"))
    groups = a["by_group"]
    n = len(groups) + 1

    heads = ["All"] + [SYMBOL.get(g["group"], g["group"]) for g in groups]
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\color{revcolor}",
             r"\resizebox{\linewidth}{!}{",
             rf"\begin{{tabular}}{{l{'r' * (2 * n)}}}", r"\toprule",
             rf"\multirow{{2}}{{*}}{{\textbf{{Input to the model}}}}"
             rf" & \multicolumn{{{n}}}{{c}}{{\textbf{{{args.label}}}}}"
             rf" & \multicolumn{{{n}}}{{c}}{{\textbf{{{args.label_secondary}}}}} \\",
             rf"\cmidrule(lr){{2-{n + 1}}} \cmidrule(lr){{{n + 2}-{2 * n + 1}}}",
             " & " + " & ".join(heads + heads) + r" \\",
             r"\midrule",
             "Video with soundtrack (multimodal) & "
             + " & ".join(row_values(a, None) + row_values(b, None)) + r" \\",
             r"\midrule"]
    for c in CONDS:
        lines.append(f"{LABEL[c]} & "
                     + " & ".join(row_values(a, c) + row_values(b, c)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}"]
    lines.append(
        r"\caption{The cascaded text-conversion baseline under both judges, over "
        + f"the {a['n']:,}"
        + r" clips graded in every setting. Columns after the first group the "
        r"clips by the stream the drivelological point rests on, using the "
        r"modality annotation of Appendix~\ref{app:modality-annotation}: "
        r"\audiomod\ audio, \textmod\ on-screen text, \videomod\ vision, and "
        r"a combination where the point needs both. The "
        r"groups shown are those holding at least 15 clips, and their sizes are "
        r"given in the text. \revb{The two judges disagree on the scale and "
        r"agree on the finding: the open judge scores everything roughly twice "
        r"as high, but a condition tracks the multimodal model where its own "
        r"modality carries the meaning and collapses where it does not under "
        r"both.} Negative totals arise where the rubric's penalties exceed its "
        r"positive dimensions, as when a model confabulates a reading from a "
        r"transcript that does not contain the drivelology.}")
    lines += [r"\label{tab:cascade}", r"\end{table*}"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
