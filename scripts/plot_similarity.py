#!/usr/bin/env python
"""Plot positive- and mismatched-pair cosine scores per retrieval model.

The input is ``retrieval/similarity_stats.json`` (written by
``scripts/similarity_stats.py``).  The script writes two complementary views:

* ``similarity_densities.{pdf,png}``: probability densities on one shared
  y-scale, so peak heights remain comparable across models.
* ``similarity_cdfs.{pdf,png}``: binned empirical CDFs on a fixed [0, 1]
  y-scale.  This is the recommended main-paper view because it does not rely
  on per-panel vertical rescaling.
* ``retrieval/similarity_stats.csv``: summary statistics, random-pair AUC,
  distribution overlap, and text-to-video R@1.

Important limitation
--------------------
These plots compare positives with *all mismatched pairs*.  They diagnose
score-space separation, but they do not explain R@1: retrieval is controlled
by each query's highest-scoring mismatched item.  A query-level margin plot,

    positive_score - max(mismatched_scores_for_that_query),

requires per-query scores and cannot be reconstructed from aggregate
histograms.  Do not describe mean separation as causing R@1.

Colours use the Okabe-Ito colour-blind-safe palette.  Only the left and bottom
spines are drawn.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox

REPO = Path(__file__).resolve().parents[1]

# Okabe & Ito (2008), "Color Universal Design".
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
POS = OKABE_ITO["blue"]
NEG = OKABE_ITO["orange"]

INK = "#1a1a19"
MUTED = "#6b6b68"
SURFACE = "#ffffff"
GROUP_TINT = "#f0efec"

# ACL-style A4 two-column text width (16 cm).  The figure is saved at exactly
# this width, so do not use bbox_inches="tight" when including it at \textwidth.
TEXTWIDTH_IN = 6.299
BODY_PT = 11.0
# Every label is body-sized: the figure is included at \textwidth with no
# rescaling, so 11 pt here is 11 pt on the page.
AXIS_LABEL_PT = BODY_PT
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "font.size": BODY_PT,
        "axes.labelsize": AXIS_LABEL_PT,
        "xtick.labelsize": BODY_PT,
        "ytick.labelsize": BODY_PT,
        "legend.fontsize": BODY_PT,
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# The generic jina checkpoints duplicate their retrieval-specific twins.
# LCO-Omni-3B-2605 is omitted because it is not reported in the paper.
EXCLUDE = {"jina-v5-omni-nano", "jina-v5-omni-small", "LCO-Omni-3B-2605"}

GENERATIVE = {
    "Qwen2.5-Omni-3B",
    "Qwen2.5-Omni-7B",
    "Qwen3-Omni-30B-A3B-Thinking",
}
SHORT = {
    "omni-embed-nemotron-3b": "omni-embed-nemotron-3B",
    "jina-v5-omni-nano-retrieval": "jina-v5-omni-nano",
    "jina-v5-omni-small-retrieval": "jina-v5-omni-small",
    "Qwen3-Omni-30B-A3B-Thinking": "Qwen3-Omni-30B-A3B",
}


def bare_axes(ax: plt.Axes) -> None:
    """Draw only quiet left and bottom spines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.75)
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.75, length=3)


def label_of(name: str) -> str:
    return SHORT.get(name, name)


def load_recall(name: str, results_dir: Path) -> float:
    """Load text-to-video R@1 as a percentage."""
    path = results_dir / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"No retrieval result for {name!r} at {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    try:
        return float(result["text_to_video"]["recall@1"]) * 100.0
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid text_to_video.recall@1 in {path}") from exc


def load_stats(path: Path):
    stats = json.loads(path.read_text(encoding="utf-8"))
    try:
        edges = np.asarray(stats["bin_edges"], dtype=float)
        raw_models = stats["models"]
    except KeyError as exc:
        raise SystemExit(f"Missing key {exc.args[0]!r} in {path}") from exc

    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise SystemExit("bin_edges must be a strictly increasing 1-D array")

    widths = np.diff(edges)
    if not np.allclose(widths, widths[0], rtol=1e-6, atol=1e-12):
        raise SystemExit(
            "This script expects equal-width histogram bins. "
            "For unequal bins, store per-bin probability mass explicitly."
        )

    models = {name: value for name, value in raw_models.items() if name not in EXCLUDE}
    if not models:
        raise SystemExit(f"No models remain after exclusions in {path}")

    expected = len(edges) - 1
    for name, model in models.items():
        for key in ("positive", "negative"):
            try:
                hist = np.asarray(model[key]["hist"], dtype=float)
            except KeyError as exc:
                raise SystemExit(f"Missing {name}.{key}.{exc.args[0]} in {path}") from exc
            if hist.shape != (expected,):
                raise SystemExit(
                    f"{name}.{key}.hist has {hist.size} bins; expected {expected}"
                )
            if np.any(~np.isfinite(hist)) or np.any(hist < 0) or hist.sum() <= 0:
                raise SystemExit(f"{name}.{key}.hist must be finite, non-negative, and non-empty")
    return edges, models


def probability_mass(hist) -> np.ndarray:
    """Convert equal-width histogram heights/counts to probability mass."""
    values = np.asarray(hist, dtype=float)
    return values / values.sum()


def density(hist, edges: np.ndarray) -> np.ndarray:
    return probability_mass(hist) / np.diff(edges)


def separation_metrics(model) -> tuple[float, float]:
    """Return binned random-pair AUC and overlap coefficient.

    AUC estimates P(score_positive > score_random), with half credit for pairs
    that fall in the same bin.  It is invariant to monotone score scaling and
    is therefore more comparable across embedding spaces than a raw mean gap.
    """
    pos = probability_mass(model["positive"]["hist"])
    neg = probability_mass(model["negative"]["hist"])
    neg_below = np.cumsum(neg) - neg
    auc = float(np.sum(pos * (neg_below + 0.5 * neg)))
    overlap = float(np.sum(np.minimum(pos, neg)))
    return auc, overlap


def group_models(models, recall, sort_by: str):
    def score(name: str):
        model = models[name]
        if sort_by == "recall":
            return recall[name]
        if sort_by == "auc":
            return separation_metrics(model)[0]
        if sort_by == "gap":
            return float(model["positive"]["mean"] - model["negative"]["mean"])
        return label_of(name).casefold()

    def ordered(names):
        if sort_by == "name":
            return sorted(names, key=score)
        return sorted(names, key=score, reverse=True)

    return (
        ordered([name for name in models if name not in GENERATIVE]),
        ordered([name for name in models if name in GENERATIVE]),
    )


# Two groups side by side, with every panel the same size.  The left family has
# two columns, the right family one column.
NROWS = 3
GROUP_NCOLS = (2, 1)
GROUP_NAMES = ("Embedding-native", "Generative-adapted")
LEFT_EDGE, RIGHT_EDGE = 0.120, 0.972
GROUP_GAP = 0.060
COL_GAP_FRAC = 0.17
ROW_GAP_FRAC = 0.42
BAND_TOP, BAND_BOTTOM = 0.820, 0.185

# All distances from content to a group frame, and from an external axis title
# back to that frame, use the same physical value.  Keeping this in points
# makes the spacing independent of figure DPI and robust to label-length
# changes.
FRAME_PAD_PT = 8.0
OUTSIDE_LABEL_GAP_PT = FRAME_PAD_PT


def panel_geometry():
    intra = sum(n - 1 for n in GROUP_NCOLS)
    span = RIGHT_EDGE - LEFT_EDGE - GROUP_GAP * (len(GROUP_NCOLS) - 1)
    width = span / (sum(GROUP_NCOLS) + COL_GAP_FRAC * intra)
    col_gap = COL_GAP_FRAC * width
    height = (BAND_TOP - BAND_BOTTOM) / (NROWS + ROW_GAP_FRAC * (NROWS - 1))
    rows = [BAND_TOP - r * height * (1 + ROW_GAP_FRAC) - height for r in range(NROWS)]

    x0 = LEFT_EDGE
    groups = []
    for ncols in GROUP_NCOLS:
        groups.append([x0 + c * (width + col_gap) for c in range(ncols)])
        x0 += ncols * width + (ncols - 1) * col_gap + GROUP_GAP
    return width, height, rows, groups


def occupied_xlim(edges: np.ndarray, models, full_range: bool) -> tuple[float, float]:
    if full_range:
        return -1.0, 1.0

    occupied = np.zeros(len(edges) - 1, dtype=bool)
    for model in models.values():
        occupied |= np.asarray(model["positive"]["hist"], dtype=float) > 0
        occupied |= np.asarray(model["negative"]["hist"], dtype=float) > 0
    indices = np.flatnonzero(occupied)
    if not len(indices):
        return -1.0, 1.0

    # Include every occupied bin plus one empty bin of visual padding.  Unlike
    # the old >0.01 rule, this never discards a low-probability observed tail.
    left = max(0, int(indices[0]) - 1)
    right = min(len(edges) - 1, int(indices[-1]) + 2)
    return max(-1.0, float(edges[left])), min(1.0, float(edges[right]))


def x_ticks(xlo: float, xhi: float) -> list[float]:
    candidates = np.arange(-1.0, 1.001, 0.5)
    ticks = [float(x) for x in candidates if xlo - 1e-9 <= x <= xhi + 1e-9]
    return ticks if len(ticks) >= 2 else [xlo, xhi]


def _content_bboxes(axes, renderer) -> tuple[Bbox, Bbox]:
    """Return structural-content and model-title boxes.

    Axis titles are deliberately excluded: they sit outside the grey family
    frames.  Titles are kept separate so a long model name cannot add spurious
    left padding between the y tick labels and the family frame.  Both boxes
    are in display coordinates.
    """
    structural_boxes = []
    title_boxes = []
    for ax in axes:
        structural_boxes.append(ax.get_window_extent(renderer))
        text_artists = [*ax.get_xticklabels(), *ax.get_yticklabels()]
        for artist in text_artists:
            if artist.get_visible() and artist.get_text().strip():
                structural_boxes.append(artist.get_window_extent(renderer))
        if ax.title.get_visible() and ax.title.get_text().strip():
            title_boxes.append(ax.title.get_window_extent(renderer))
    return Bbox.union(structural_boxes), Bbox.union(title_boxes)


def add_group_backgrounds(fig, axes_by_group) -> list[Bbox]:
    """Frame each family from measured content, using equal padding."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    measured = [_content_bboxes(axes, renderer) for axes in axes_by_group]
    pad_px = FRAME_PAD_PT * fig.dpi / 72.0

    # The two family frames share their top and bottom edges.  Their horizontal
    # bounds remain content-aware because only the left family has y tick text.
    shared_y0 = min(min(body.y0, titles.y0) for body, titles in measured) - pad_px
    shared_y1 = max(max(body.y1, titles.y1) for body, titles in measured) + pad_px
    frame_boxes = []
    for body, titles in measured:
        x0 = body.x0 - pad_px
        x1 = body.x1 + pad_px
        # Expand only when a title would actually cross the already padded
        # boundary; otherwise the y ticks retain exactly FRAME_PAD_PT.
        if titles.x0 < x0:
            x0 = titles.x0 - pad_px
        if titles.x1 > x1:
            x1 = titles.x1 + pad_px
        display_box = Bbox.from_extents(
            x0,
            shared_y0,
            x1,
            shared_y1,
        )
        figure_box = display_box.transformed(fig.transFigure.inverted())
        frame_boxes.append(figure_box)
        fig.add_artist(
            FancyBboxPatch(
                (figure_box.x0, figure_box.y0),
                figure_box.width,
                figure_box.height,
                boxstyle="round,pad=0,rounding_size=0.008",
                transform=fig.transFigure,
                facecolor=GROUP_TINT,
                edgecolor="none",
                zorder=-10,
            )
        )
    return frame_boxes


def add_external_axis_titles(fig, bottom_axes, frame_boxes, ylabel: str) -> None:
    """Place axis titles outside the frames at the same physical distance."""
    gap_y = OUTSIDE_LABEL_GAP_PT / 72.0 / fig.get_figheight()
    gap_x = OUTSIDE_LABEL_GAP_PT / 72.0 / fig.get_figwidth()
    frame_bottom = min(box.y0 for box in frame_boxes)

    for ax in bottom_axes:
        position = ax.get_position()
        fig.text(
            position.x0 + position.width / 2.0,
            frame_bottom - gap_y,
            "Cosine similarity",
            ha="center",
            va="top",
            color=INK,
            fontsize=AXIS_LABEL_PT,
        )

    # Measure the rotated text once, then place its right edge exactly one
    # OUTSIDE_LABEL_GAP_PT to the left of the embedding-native frame.
    frame_mid_y = (frame_boxes[0].y0 + frame_boxes[0].y1) / 2.0
    y_artist = fig.text(
        0.0,
        frame_mid_y,
        ylabel,
        rotation=90,
        ha="center",
        va="center",
        color=INK,
        fontsize=AXIS_LABEL_PT,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_box = y_artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    y_artist.set_x(frame_boxes[0].x0 - gap_x - text_box.width / 2.0)


def plot_distributions(
    edges: np.ndarray,
    models,
    groups,
    recall,
    out: Path,
    view: Literal["density", "cdf"],
    full_range: bool,
) -> None:
    fig = plt.figure(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 9 / 16))
    fig.patch.set_facecolor(SURFACE)
    width, height, row_bottoms, col_lefts = panel_geometry()

    xlo, xhi = occupied_xlim(edges, models, full_range)
    ticks = x_ticks(xlo, xhi)

    if view == "density":
        all_density = [
            density(model[key]["hist"], edges)
            for model in models.values()
            for key in ("negative", "positive")
        ]
        ymax = max(float(curve.max()) for curve in all_density) * 1.07
        yticks = [0.0, ymax / 2.0, ymax]
        ylabel = "Probability density"
    else:
        ymax = 1.02  # small headroom keeps the CDF plateau visible
        yticks = [0.0, 0.5, 1.0]
        ylabel = "Cumulative probability"

    axes_by_group = []
    bottom_axes = []
    for group_index, (names, ncols, lefts) in enumerate(zip(groups, GROUP_NCOLS, col_lefts)):
        if len(names) > ncols * NROWS:
            raise SystemExit(f"{len(names)} models do not fit {NROWS}x{ncols}")

        placed = []
        for i, name in enumerate(names):
            row, col = divmod(i, ncols)
            ax = fig.add_axes([lefts[col], row_bottoms[row], width, height])
            ax.patch.set_visible(False)
            bare_axes(ax)
            model = models[name]

            for stats, colour in (
                (model["negative"], NEG),
                (model["positive"], POS),
            ):
                mass = probability_mass(stats["hist"])
                if view == "density":
                    curve = mass / np.diff(edges)
                    ax.stairs(
                        curve,
                        edges,
                        baseline=0,
                        fill=True,
                        facecolor=colour,
                        alpha=0.20,
                        edgecolor="none",
                        zorder=2,
                    )
                    ax.stairs(curve, edges, color=colour, linewidth=1.15, zorder=3)
                    mean_bin = int(np.clip(np.searchsorted(edges, stats["mean"]) - 1, 0, len(curve) - 1))
                    line_top = float(curve[mean_bin])
                else:
                    curve = np.r_[0.0, np.cumsum(mass)]
                    ax.fill_between(
                        edges,
                        curve,
                        step="post",
                        color=colour,
                        alpha=0.08,
                        linewidth=0,
                        zorder=2,
                    )
                    ax.step(edges, curve, where="post", color=colour, linewidth=1.2, zorder=3)
                    line_top = 1.0

                ax.vlines(
                    stats["mean"],
                    0,
                    line_top,
                    color=colour,
                    linewidth=0.85,
                    linestyle=(0, (2, 2)),
                    zorder=4,
                )

            # No group titles, and no metric text on the density view: the
            # figure carries only the model names there. The CDF view is the
            # main-paper figure and its caption refers to the AUC, so that one
            # number is printed in each panel. Overlap and the rest stay in
            # retrieval/similarity_stats.csv.
            ax.set_title(label_of(name), color=INK, fontsize=BODY_PT, pad=2.0)

            # The AUC sits in the bottom-left corner of the CDF panels, on a
            # patch in the block's own tint so it masks the flat start of the
            # curves rather than tangling with them. The density view carries
            # no metric text; everything is in similarity_stats.csv.
            if view == "cdf":
                auc, _ = separation_metrics(model)
                ax.text(
                    0.03,
                    0.05,
                    f"AUC {auc:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    color=MUTED,
                    fontsize=BODY_PT,
                    zorder=5,
                    bbox={
                        "boxstyle": "square,pad=0.18",
                        "facecolor": GROUP_TINT,
                        "edgecolor": "none",
                        # Translucent: the flat start of a curve stays visible
                        # under the label instead of being cut out of the plot.
                        "alpha": 0.78,
                    },
                )

            ax.set_xlim(xlo, xhi)
            ax.set_ylim(0, ymax)
            ax.set_xticks(ticks)
            ax.set_yticks(yticks)
            # A single set of y tick labels is enough because every panel uses
            # the same scale.  Repeating them beside the narrow right block
            # wastes the inter-group gutter and can look like concatenated
            # numbers at paper size.
            if group_index != 0 or col != 0:
                ax.set_yticklabels([])
            elif view == "density":
                ax.set_yticklabels(["0", f"{yticks[1]:.1f}", f"{yticks[2]:.1f}"])
            else:
                ax.set_yticklabels(["0", ".5", "1"])
            placed.append((row, col, ax))

        occupied = {(row, col) for row, col, _ in placed}
        for row, col, ax in placed:
            if (row + 1, col) in occupied:
                ax.set_xticklabels([])
            else:
                bottom_axes.append(ax)
        axes_by_group.append([ax for _, _, ax in placed])

    frame_boxes = add_group_backgrounds(fig, axes_by_group)
    add_external_axis_titles(fig, bottom_axes, frame_boxes, ylabel)

    legend_handles = [
        Line2D([0], [0], color=NEG, linewidth=1.5, label="Mismatched pairs"),
        Line2D([0], [0], color=POS, linewidth=1.5, label="Positive pairs"),
        Line2D([0], [0], color=MUTED, linewidth=0.9, linestyle=(0, (2, 2)), label="Mean"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.003),
        frameon=False,
        labelcolor=INK,
        fontsize=BODY_PT - 1.0,
        handlelength=1.8,
        ncols=3,
        columnspacing=1.9,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(
            out.with_suffix(suffix),
            dpi=300,
            facecolor=SURFACE,
            metadata={"Creator": Path(__file__).name},
        )
    plt.close(fig)


def write_table(models, groups, recall, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "model",
                "family",
                "n_queries",
                "positive_mean",
                "positive_std",
                "positive_median",
                "mismatched_mean",
                "mismatched_std",
                "mismatched_median",
                "mean_gap",
                "random_pair_auc",
                "overlap_coefficient",
                "text_to_video_recall_at_1",
            ]
        )
        for names, family in zip(groups, ("embedding-native", "generative-adapted")):
            for name in names:
                model = models[name]
                pos, neg = model["positive"], model["negative"]
                auc, overlap = separation_metrics(model)
                writer.writerow(
                    [
                        name,
                        family,
                        model["n"],
                        f"{pos['mean']:.4f}",
                        f"{pos['std']:.4f}",
                        f"{pos['median']:.4f}",
                        f"{neg['mean']:.4f}",
                        f"{neg['std']:.4f}",
                        f"{neg['median']:.4f}",
                        f"{pos['mean'] - neg['mean']:.4f}",
                        f"{auc:.4f}",
                        f"{overlap:.4f}",
                        f"{recall[name]:.1f}",
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats",
        type=Path,
        default=REPO / "retrieval" / "similarity_stats.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "latex" / "figures",
    )
    parser.add_argument(
        "--view",
        choices=("both", "density", "cdf"),
        default="both",
        help="Write both views (default), or only one.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("recall", "auc", "gap", "name"),
        default="auc",
        help="Panel order within each model family (default: random-pair AUC).",
    )
    parser.add_argument(
        "--full-cosine-range",
        action="store_true",
        help="Show [-1, 1]; by default use the common occupied range without dropping nonzero bins.",
    )
    args = parser.parse_args()

    edges, models = load_stats(args.stats)
    recall = {name: load_recall(name, args.stats.parent) for name in models}
    groups = group_models(models, recall, args.sort_by)

    if args.view in ("both", "density"):
        plot_distributions(
            edges,
            models,
            groups,
            recall,
            args.out_dir / "similarity_densities",
            "density",
            args.full_cosine_range,
        )
    if args.view in ("both", "cdf"):
        plot_distributions(
            edges,
            models,
            groups,
            recall,
            args.out_dir / "similarity_cdfs",
            "cdf",
            args.full_cosine_range,
        )

    write_table(models, groups, recall, args.stats.with_suffix(".csv"))

    for names, family in zip(groups, ("embedding-native", "generative-adapted")):
        print(f"[{family}; ordered by {args.sort_by}]")
        for name in names:
            model = models[name]
            auc, overlap = separation_metrics(model)
            print(
                f"  {name:<32} n={model['n']:<5} "
                f"R@1={recall[name]:5.1f}% AUC={auc:.4f} overlap={overlap:.4f} "
                f"gap={model['positive']['mean'] - model['negative']['mean']:+.4f}"
            )


if __name__ == "__main__":
    main()
