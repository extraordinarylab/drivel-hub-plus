#!/usr/bin/env python3
"""Evaluation cost of the reference judge, at the published token prices.

Priced from token counts here rather than from the stored judge_cost_usd,
because that field carries whatever BILLING_CALIBRATION was set to when the row
was written, and the constant changed mid-project: the same 2,800-token
judgement is recorded at $0.0027 in the main grading and $0.0051 in the
cascade. Token counts do not drift, so they are what the table is built from.

What this reports is therefore the published-rate cost, which is the figure a
reader can predict and reproduce. The real invoice ran about a quarter higher
(GBP 31.77 for 12,000 judgements against a listed $32.16), and the appendix
says so; but that gap carries an exchange rate and whatever tax the account
attracts, neither of which transfers to anyone else. Readers
budgeting a reproduction need the per-item figure; the totals show what a full
9,000-item pass comes to.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Both Qwen3-Omni generation runs use the SAME checkpoint,
# Qwen3-Omni-30B-A3B-Thinking, with thinking enabled and disabled. Do not call
# the disabled run "Instruct": that is a different checkpoint
# (Qwen3-Omni-30B-A3B-Instruct) and it is the open judge, not a graded system.
# USD per million tokens for gemini-3.8-flash, from the published price list.
PRICE_IN, PRICE_OUT = 0.75, 3.75

DISPLAY = {
    "Qwen/Qwen3-Omni-30B-A3B-Thinking": "Qwen3-Omni 30B-A3B (Thinking)",
    "Qwen/Qwen3-Omni-30B-A3B-No-Thinking": "Qwen3-Omni 30B-A3B (No-thinking)",
    "Qwen/Qwen2.5-Omni-7B": "Qwen2.5-Omni 7B",
    "Qwen/Qwen2.5-Omni-3B": "Qwen2.5-Omni 3B",
    "openbmb/MiniCPM-o-2_6": "MiniCPM-o-2.6 9B",
    "google/gemma-4-E4B-it": "Gemma 4 E4B",
    "google/gemma-4-E2B-it": "Gemma 4 E2B",
    "harryhsing/EchoInk-R1-7B": "EchoInk-R1 7B",
    "human-baseline": "Human baseline",
}
# The cascade was excluded while it was a separate appendix experiment. It is
# now a block in Table 1, so a reader reproducing the paper's Gemini grading
# pays for all 12,000 judgements, not 9,000; leaving it out understates the
# cost of reproduction. Kept in its own block so the 9,000-item figure for one
# full pass over the benchmark is still readable on its own.
CASCADE = {
    "cascade/ocr-asr": "Cascade: OCR + ASR",
    "cascade/transcript-only": "Cascade: ASR only",
    "cascade/caption-only": "Cascade: OCR only",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judgments", type=Path,
                    default=Path("judgments/gemini-3-8-flash"))
    ap.add_argument("--judge-label", default="Gemini 3.8 Flash")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    rows_by_system: dict[str, list[dict]] = {}
    for path in sorted(args.judgments.rglob("judgments.jsonl")):
        system = str(path.parent.relative_to(args.judgments))
        # Only the evaluated systems. Other gradings under the same judge --
        # the cascaded baseline, pilot subsets -- are separate experiments, and
        # folding them in silently inflates both the row count and the total.
        if system not in DISPLAY and system not in CASCADE:
            print(f"  excluded from the cost table: {system}")
            continue
        rows_by_system[system] = [
            json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    entries = []
    for system, rows in rows_by_system.items():
        priced = [r for r in rows if "judge_prompt_tokens" in r]
        if not priced:
            continue
        entries.append({
            "system": system,
            "n": len(priced),
            "in_tok": sum(r["judge_prompt_tokens"] for r in priced) / len(priced),
            "out_tok": sum(r["judge_output_tokens"] for r in priced) / len(priced),
            "cost": sum(r["judge_prompt_tokens"] for r in priced) / 1e6 * PRICE_IN
                    + sum(r.get("judge_output_tokens", 0) for r in priced)
                    / 1e6 * PRICE_OUT,
            "words": sum(len(str(r.get("implicit_meaning", "")).split())
                         for r in priced) / len(priced),
        })
    main = sorted((e for e in entries if e["system"] in DISPLAY),
                  key=lambda e: -e["cost"])
    casc = sorted((e for e in entries if e["system"] in CASCADE),
                  key=lambda e: -e["cost"])
    main_cost, main_n = sum(e["cost"] for e in main), sum(e["n"] for e in main)
    casc_cost, casc_n = sum(e["cost"] for e in casc), sum(e["n"] for e in casc)
    total_cost, total_n = main_cost + casc_cost, main_n + casc_n
    entries = main + casc

    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\color{revcolor}",
             r"\resizebox{\linewidth}{!}{",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"\textbf{Graded system} & \textbf{In} & \textbf{Out} & \textbf{USD} \\",
             r"\midrule"]
    names = {**DISPLAY, **CASCADE}
    for e in main:
        # Plain commas: these cells are text mode, not math, so {,} would only
        # add noise. It is needed inside $...$, which is why the caption and
        # the prose keep it there.
        lines.append(f"{names[e['system']]} & {e['in_tok']:,.0f} & "
                     f"{e['out_tok']:,.0f} & {e['cost']:.2f} \\\\")
    lines.append(r"\midrule")
    lines.append(f"\\textit{{One full pass, {main_n:,} judgements}} & & & "
                 f"\\textit{{{main_cost:.2f}}} \\\\")
    lines.append(r"\midrule")
    for e in casc:
        lines.append(f"{names[e['system']]} & {e['in_tok']:,.0f} & "
                     f"{e['out_tok']:,.0f} & {e['cost']:.2f} \\\\")
    lines.append(r"\midrule")
    lines.append(f"\\textit{{Cascaded baseline, {casc_n:,} judgements}} & & & "
                 f"\\textit{{{casc_cost:.2f}}} \\\\")
    lines += [r"\midrule",
              f"\\textbf{{All {total_n:,} judgements}} & & & "
              f"\\textbf{{{total_cost:.2f}}} \\\\",
              r"\bottomrule", r"\end{tabular}", r"}"]
    per_item = main_cost / main_n
    # Format the numeral only: a blanket replace would rewrite the commas in
    # the sentence as well, which is how "Flash{,} from the API" got shipped.
    # Text-mode caption, so a plain comma is correct here.
    total_tex = f"{total_n:,}"
    lines.append(
        rf"\caption{{Cost of every {args.judge_label} grading in this paper, at "
        rf"the published token prices: the {main_n:,} explanations of "
        rf"Table~\ref{{tab:generation}}, and the {casc_n:,} cascade responses of "
        rf"Table~\ref{{tab:cascade}}. \textbf{{In}} and "
        rf"\textbf{{Out}} are mean input and output tokens per judgement; input is "
        rf"dominated by the video, which the judge receives in full with its "
        rf"soundtrack. At \${per_item:.4f} per judgement, re-grading the benchmark "
        rf"costs about \${per_item * 9000:.0f}. The open judge of "
        rf"Appendix~\ref{{app:judge-robustness}} costs nothing but GPU time. "
        rf"Billing for this project ran about a quarter above the listed rate, "
rf"which is the usual gap for video input.}}")
    lines += [r"\label{tab:judge-cost}", r"\end{table}"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{total_n} judgements | ${total_cost:.2f} | ${per_item:.4f}/item")
    for e in entries:
        print(f"  {DISPLAY.get(e['system'], e['system']):32} "
              f"{e['in_tok']:7,.0f} in {e['out_tok']:5,.0f} out  "
              f"${e['cost']:5.2f}  ({e['words']:5.1f} words)")
    print(f"wrote {args.out}")

    if args.json_out:
        args.json_out.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
