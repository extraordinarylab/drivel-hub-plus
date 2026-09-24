#!/usr/bin/env python3
"""Render judgment results as the paper's LaTeX table, one file per judge.

Each judge gets its own table because scores from different judges are not
comparable: the rubric is the same but the grader is not. The human baseline
heads every table as the ceiling, and is excluded from the best/worst shading —
it is the reference the models are measured against, not a competitor, and
letting it take \\bestA would erase the ranking among the models.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

# Table 2 column order, with the direction that counts as better.
COLUMNS = [
    ("judge_aligned", "Aligned\\%", "up"),
    ("judge_core_intent", "Core/5", "up"),
    ("judge_rhetorical_signal", "Rhet./3", "up"),
    ("judge_affective_or_social_meaning", "Social/2", "up"),
    ("judge_grounding", "Ground./2", "up"),
    ("judge_hallucination_penalty", "Halluc./3", "down"),
    ("judge_literal_only_penalty", "Literal/3", "down"),
    ("judge_vague_or_overgeneralized_penalty", "Vague/2", "down"),
    ("judge_score_total", "Total/12", "up"),
]

# prediction dir -> (model, size, setting, group)
DISPLAY = {
    # The third field is the decoding mode, and it only means something where a
    # thinking mode exists. Verified against the model cards:
    #   Qwen3-Omni   one checkpoint (the Thinking one) run both ways -- both
    #                rows below come from .../Qwen3-Omni-30B-A3B-Thinking.
    #   Gemma 4      has enable_thinking, default off; we use the default, so
    #                "No-thinking" is a real choice we made.
    #   Qwen2.5-Omni, MiniCPM-o, EchoInk-R1  have no thinking mode at all --
    #                their chat templates contain no thinking branch -- so
    #                "No-thinking" would imply a setting that does not exist.
    #                EchoInk-R1 is an RL-tuned finetune of Qwen2.5-Omni-7B:
    #                its reasoning is trained in, not switchable.
    "Qwen/Qwen3-Omni-30B-A3B-Thinking":    ("Qwen3-Omni", "30B-A3B", "Thinking", "Qwen-Omni models"),
    "Qwen/Qwen3-Omni-30B-A3B-No-Thinking": ("Qwen3-Omni", "30B-A3B", "No-thinking", "Qwen-Omni models"),
    "Qwen/Qwen2.5-Omni-7B":                ("Qwen2.5-Omni", "7B", "Instruct", "Qwen-Omni models"),
    "Qwen/Qwen2.5-Omni-3B":                ("Qwen2.5-Omni", "3B", "Instruct", "Qwen-Omni models"),
    "google/gemma-4-E2B-it":               ("Gemma 4", "E2B", "No-thinking", "Gemma models"),
    "google/gemma-4-E4B-it":               ("Gemma 4", "E4B", "No-thinking", "Gemma models"),
    "harryhsing/EchoInk-R1-7B":            ("EchoInk-R1", "7B", "Instruct", "Other audio-visual models"),
    # The prose calls this MiniCPM-o-2.6 throughout, so the table carries the
    # version too; "MiniCPM-o" alone left a reader unable to find the row.
    "openbmb/MiniCPM-o-2_6":               ("MiniCPM-o-2.6", "9B", "Instruct", "Other audio-visual models"),
}
HUMAN = "human-baseline"
# The cascaded text-only conditions. They are graded on the same rubric by the
# same judge, so they belong on the same scale, but they are diagnostics rather
# than entries in the ranking: each is an off-the-shelf extractor feeding a
# text model, not a system anyone would deploy. Rendered greyed and excluded
# from the shading, exactly as the human ceiling is.
CASCADE = {
    "cascade/transcript-only": "ASR only",
    "cascade/caption-only": "OCR only",
    "cascade/ocr-asr": "OCR + ASR",
}


# The paper cites the primary judge's table as tab:generation; a second judge's
# table is an appendix table and needs its own label.
# Overridable: which judge the paper treats as its reference grading.
PRIMARY_JUDGE = os.environ.get("PRIMARY_JUDGE", "Qwen3-Omni-30B-A3B-Instruct")


def label_for(judge: str) -> str:
    if slug(judge) == slug(PRIMARY_JUDGE):
        return "tab:generation"
    return f"tab:generation-{slug(judge)}"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower().rstrip("/").rsplit("/", 1)[-1]).strip("-")


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("judge_aligned") is not None]
    out: dict = {"n": len(scored)}
    for key, _, _ in COLUMNS:
        vals = [r[key] for r in scored if r.get(key) is not None]
        out[key] = (sum(float(v) for v in vals) / len(vals)) if vals else None
    return out


def rank_marks(values: list[float | None], direction: str) -> dict[int, str]:
    """bestA/B/C to the three best distinct values, worstA/B/C to the three worst."""
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    if not present:
        return {}
    ordered = sorted({round(v, 6) for _, v in present}, reverse=(direction == "up"))
    marks: dict[int, str] = {}
    for rank, letter in zip(ordered[:3], ("bestA", "bestB", "bestC")):
        for i, v in present:
            if round(v, 6) == rank:
                marks[i] = letter
    for rank, letter in zip(list(reversed(ordered))[:3], ("worstA", "worstB", "worstC")):
        for i, v in present:
            if round(v, 6) == rank and i not in marks:
                marks[i] = letter
    return marks


def grey(cell: str) -> str:
    return f"\\textcolor{{gray}}{{{cell}}}"


def fmt(value: float | None, mark: str | None) -> str:
    if value is None:
        return "--"
    cell = f"{value:.3f}"
    return f"\\{mark}{{{cell}}}" if mark else cell


def render(judge: str, entries: list[tuple[str, dict]], human: dict | None,
           cascade: list[tuple[str, dict]] | None = None) -> str:
    models = [(DISPLAY[k], agg) for k, agg in entries if k in DISPLAY]
    marks = {}
    for col, (key, _, direction) in enumerate(COLUMNS):
        marks[col] = rank_marks([a[key] for _, a in models], direction)

    lines = [
        "\\begin{table*}[t]", "\\centering", "\\resizebox{\\linewidth}{!}{",
        "\\begin{tabular}{lllccccccccc}", "\\toprule",
        "\\textbf{Model} ", "& \\textbf{Size}", "& \\textbf{Mode}",
    ]
    for _, label, direction in COLUMNS:
        arrow = "\\uparrow" if direction == "up" else "\\downarrow"
        lines.append(f"& \\textbf{{{label}}} ${arrow}$")
    lines[-1] += " \\\\"
    lines.append("\\midrule")

    if human:
        # Greyed out and excluded from the shading: the row is the ceiling the
        # models are read against, not another entry in the ranking. A rule, not
        # a heading, keeps it separate without adding a section label.
        cells = " & ".join(grey(fmt(human[k], None)) for k, _, _ in COLUMNS)
        lines.append(f"{grey('Human baseline')} & {grey('--')} & {grey('--')}")
        lines.append(f"& {cells} \\\\")
        lines.append("\\midrule")

    lines.append("")
    for i, ((name, size, setting, _), agg) in enumerate(models):
        cells = " & ".join(
            fmt(agg[key], marks[col].get(i)) for col, (key, _, _) in enumerate(COLUMNS)
        )
        lines.append(f"{name} & {size} & {setting}")
        lines.append(f"& {cells} \\\\")

    if cascade:
        lines.append("\\midrule")
        lines.append("\\multicolumn{12}{l}{\\textit{"
                     + grey("Text-only cascade: the clip converted to text, no video")
                     + "}} \\\\")
        for name, agg in cascade:
            cells = " & ".join(grey(fmt(agg[k], None)) for k, _, _ in COLUMNS)
            lines.append(f"{grey(name)} & {grey('--')} & {grey('--')}")
            lines.append(f"& {cells} \\\\")

    lines += [
        "", "\\bottomrule", "\\end{tabular}", "}", "\\caption{",
        f"Video LLM-as-a-judge results for implicit-meaning understanding on 1,000 drivelological "
        f"videos, graded by {judge}. Green shading marks the top three results in each column "
        f"and purple shading the bottom three, computed over the models only; the human "
        f"baseline is the reference ceiling and is excluded from the shading. The "
        f"cascade rows convert each clip to text with off-the-shelf speech and "
        f"character recognition and give a text-only model nothing else; they are "
        f"diagnostics on the same scale rather than competing systems, and are also "
        f"excluded from the shading.",
        "}", f"\\label{{{label_for(judge)}}}", "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, default=Path("judgments"))
    parser.add_argument("--out-dir", type=Path, default=Path("latex"))
    args = parser.parse_args()

    per_judge: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(args.judgments.rglob("judgments.jsonl")):
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if not rows:
            continue
        # Results may sit at judgments/<model>/ (single-judge layout) or at
        # judgments/<judge>/<model>/ once a second judge is added. Match the
        # deepest suffix that names a known model so both layouts work.
        rel = path.parent.relative_to(args.judgments)
        parts = rel.parts
        key = str(rel)
        for depth in range(1, len(parts) + 1):
            candidate = "/".join(parts[-depth:])
            if candidate in DISPLAY or candidate == HUMAN or candidate in CASCADE:
                key = candidate
                break
        for judge in {r.get("judge_model", "unknown") for r in rows}:
            subset = [r for r in rows if r.get("judge_model") == judge]
            slug_name = judge.rstrip("/").rsplit("/", 1)[-1]
            # Two directories can carry the same judge_model -- a pilot subset
            # and the full run both say "gemini-3.8-flash" -- and whichever is
            # walked last would silently win. A 200-row pilot overwrote the
            # 1,000-row run this way and the table looked entirely plausible.
            # Keep the larger set and say so.
            existing = per_judge[slug_name].get(key)
            if existing and existing.get("n", 0) >= len(subset):
                print(f"  skipping {path.parent} ({len(subset)} rows): already have "
                      f"{existing['n']} rows for {slug_name}/{key}")
                continue
            if existing:
                print(f"  replacing {slug_name}/{key}: {existing['n']} rows -> "
                      f"{len(subset)} rows from {path.parent}")
            entry = aggregate(subset)
            entry["n"] = len(subset)
            per_judge[slug_name][key] = entry

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for judge, entries in sorted(per_judge.items()):
        human = entries.get(HUMAN)
        cascade = [(CASCADE[k], v) for k, v in entries.items() if k in CASCADE]
        cascade.sort(key=lambda kv: -(kv[1].get("judge_score_total") or -1e9))
        models = [(k, v) for k, v in entries.items()
                  if k != HUMAN and k not in CASCADE]
        # One block, no family grouping: rank by total score so the table reads
        # top to bottom as a single comparison.
        models.sort(key=lambda kv: -(kv[1].get("judge_score_total") or -1e9))
        out = args.out_dir / f"generation_{slug(judge)}.tex"
        out.write_text(render(judge, models, human, cascade), encoding="utf-8")
        skipped = [k for k, _ in models if k not in DISPLAY]
        print(f"{out}: {len(models)} model row(s)"
              f"{', human baseline' if human else ', NO human baseline'}"
              f"{f', skipped (no display name): {skipped}' if skipped else ''}")


if __name__ == "__main__":
    main()
