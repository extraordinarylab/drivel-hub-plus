#!/usr/bin/env python
"""Every system loses accuracy when vision carries the meaning; humans do not.

A paired-dot ("dumbbell") row per system: the left dot is the alignment rate on
the clips whose modality annotation says vision carries the implicit meaning,
the right dot the rate on the rest. The human baseline is the control -- its two
dots coincide, so the gap below is a property of the models and not of the
clips.

Colour follows Okabe-Ito, choosing the sky-blue/vermillion pair, which holds the
largest separation of that palette under red-green deficiency. Marker shape
repeats the distinction so colour is never the only channel. Only the left and
bottom spines are drawn, matching ``scripts/plot_similarity.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]

OKABE_ITO = {
    "orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
VISION = OKABE_ITO["vermillion"]
OTHER = OKABE_ITO["sky_blue"]
INK = "#1a1a19"
MUTED = "#6b6b68"
RULE = "#d9d8d4"

TEXTWIDTH_IN = 6.299
BODY_PT = 8.0
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "font.size": BODY_PT,
        "axes.labelsize": BODY_PT,
        "xtick.labelsize": BODY_PT,
        "ytick.labelsize": BODY_PT,
        "legend.fontsize": BODY_PT,
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Worst first, so the rows read bottom-to-top as the leaderboard does.
SYSTEMS = [
    ("openbmb/MiniCPM-o-2_6", "MiniCPM-o-2.6"),
    ("google/gemma-4-E2B-it", "Gemma 4 E2B"),
    ("Qwen/Qwen2.5-Omni-3B", "Qwen2.5-Omni 3B"),
    ("google/gemma-4-E4B-it", "Gemma 4 E4B"),
    ("Qwen/Qwen2.5-Omni-7B", "Qwen2.5-Omni 7B"),
    ("harryhsing/EchoInk-R1-7B", "EchoInk-R1 7B"),
    ("Qwen/Qwen3-Omni-30B-A3B-No-Thinking", "Qwen3-Omni (no-think)"),
    ("Qwen/Qwen3-Omni-30B-A3B-Thinking", "Qwen3-Omni (thinking)"),
    ("human-baseline", "Human baseline"),
]
# Name the judges rather than describing them: "proprietary" and "open" is the
# paper's framing, but a reader looking at the panel wants to know which model
# produced the numbers. The open judge is written as the agreement table writes
# it, without the -Instruct suffix.
# "graded by" is not padding: Qwen3-Omni-30B-A3B is also two of the rows, so a
# bare model name as a panel title reads as another evaluated system.
JUDGES = [("gemini-3-8-flash", "graded by Gemini 3.8 Flash"),
          ("qwen3-omni-30b-a3b-instruct", "graded by Qwen3-Omni-30B-A3B")]


def bare_axes(ax: plt.Axes) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.75, length=3)


def rates(judge: str) -> list[tuple[str, float, float]]:
    out = []
    for system, label in SYSTEMS:
        path = REPO / "judgments" / judge / system / "judgments.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        buckets: dict[bool, list[bool]] = defaultdict(list)
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            buckets["vision" in row["modalities"].strip()].append(
                bool(row["judge_aligned"]))
        v, o = buckets[True], buckets[False]
        out.append((label, 100 * sum(v) / len(v), 100 * sum(o) / len(o)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "figures" / "vision_gap.pdf")
    args = ap.parse_args()

    panels = [(title, rates(judge)) for judge, title in JUDGES]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.36),
                             sharey=True)

    for ax, (title, data) in zip(axes, panels):
        y = np.arange(len(data))
        for i, (label, vision, other) in enumerate(data):
            human = label.startswith("Human")
            ax.plot([vision, other], [i, i], color=RULE if human else MUTED,
                    linewidth=1.0, zorder=1, solid_capstyle="round")
            ax.plot(vision, i, marker="o", markersize=4.2, color=VISION,
                    zorder=3, clip_on=False)
            ax.plot(other, i, marker="s", markersize=3.8, color=OTHER,
                    zorder=3, clip_on=False)
            # The control needs its zero printed too: a blank row would read as
            # missing data rather than as the absence of a gap.
            gap = other - vision
            ax.annotate("0" if abs(gap) < 0.5 else f"{gap:.0f}",
                        xy=(max(vision, other), i),
                        xytext=(max(vision, other) + 3.0, i),
                        color=MUTED, ha="left", va="center")
        # Separate the control from the systems it controls for.
        ax.axhline(len(data) - 1.5, color=RULE, linewidth=0.75, zorder=0)
        bare_axes(ax)
        ax.set_yticks(y)
        ax.set_yticklabels([d[0] for d in data])
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(-0.7, len(data) - 0.3)
        # One shared label below both panels: repeating it per panel makes the
        # two collide at this width.
        ax.set_title(title, color=INK, fontsize=BODY_PT, pad=4.0)
        ax.grid(axis="x", color=RULE, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    handles = [
        Line2D([], [], marker="o", markersize=4.2, color=VISION, linestyle="",
               label="vision needed\n(235 clips)"),
        Line2D([], [], marker="s", markersize=3.8, color=OTHER, linestyle="",
               label="vision not needed\n(765 clips)"),
    ]
    leg = fig.legend(handles=handles, loc="center left", frameon=False,
                     bbox_to_anchor=(0.795, 0.56), handletextpad=0.5,
                     labelspacing=1.2, borderpad=0.0)
    for text in leg.get_texts():
        text.set_color(INK)

    fig.tight_layout(pad=0.3, w_pad=1.0, rect=(0, 0.045, 0.79, 1))
    # "Aligned" is the rubric's word, not something a reader can decode from an
    # axis, so the label says what the judge actually decided.
    fig.supxlabel("explanations matching the human reading (%)", color=INK,
                  fontsize=BODY_PT, x=0.435, y=0.015)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    for title, data in panels:
        print(title)
        for label, vision, other in data:
            print(f"  {label:24}{vision:7.1f}{other:7.1f}{other - vision:7.1f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
