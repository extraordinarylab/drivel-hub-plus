#!/usr/bin/env python
"""Two StoryScope-style views of explanation style, each a single panel.

rarity.pdf   after StoryScope Figure 5. For each explanation, the mean distance
             to its 25 nearest neighbours written by the same source, in a space
             of length-normalised function-word and punctuation rates. Sources
             all have 877 texts, so the measure is balanced by construction.
             High means the wording is unlike the source's other attempts.

lda.pdf      after StoryScope Figure 2, with the circularity removed: the
             discriminant is fitted on half the clips and only the held-out half
             is plotted, so the separation is a prediction rather than a
             restatement of the labels it was given.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

REPO = Path(__file__).resolve().parents[1]
JUDGE = "gemini-3-8-flash"
K = 25

HUMAN_C = "#0072B2"      # Okabe-Ito blue
MODEL_C = "#D55E00"      # Okabe-Ito vermillion
INK = "#1a1a19"
MUTED = "#6b6b68"
RULE = "#d9d8d4"

TEXTWIDTH_IN = 6.299
COLWIDTH_IN = 3.04
BODY_PT = 8.0
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": BODY_PT, "axes.labelsize": BODY_PT,
    "xtick.labelsize": BODY_PT, "ytick.labelsize": BODY_PT,
    "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42, "ps.fonttype": 42,
})

SYSTEMS = [
    ("Qwen/Qwen3-Omni-30B-A3B-Thinking", "Qwen3-Omni"),
    ("Qwen/Qwen3-Omni-30B-A3B-No-Thinking", "Qwen3-Omni (no-think)"),
    ("harryhsing/EchoInk-R1-7B", "EchoInk-R1"),
    ("Qwen/Qwen2.5-Omni-7B", "Qwen2.5-Omni 7B"),
    ("google/gemma-4-E4B-it", "Gemma 4 E4B"),
    ("google/gemma-4-E2B-it", "Gemma 4 E2B"),
    ("Qwen/Qwen2.5-Omni-3B", "Qwen2.5-Omni 3B"),
    ("openbmb/MiniCPM-o-2_6", "MiniCPM-o-2.6"),
]
FUNCTION = """the of and a to in is it that was he for on are as with his they i
at be this have from or one had by word but not what all were we when your can
said there use an each which she do how their if will up other about out many
then them these so some her would make like him into time has look two more
write go see number no way could people my than first been call who oil its now
find long down day did get come made may part over new sound take only little
work know place year live me back give most very after thing our just name good
sentence man think say great where help through much before line right too mean
old any same tell boy follow came want show also around form three small set put
end does another well large must big even such because turn here why ask went
men read need land different home us move try kind hand picture again change off
play spell air away animal house point page letter mother answer found study
still learn should america world""".split()
PUNCT = list(".,;:!?—-\"'()")
WORD = re.compile(r"[A-Za-z']+")


def bare_axes(ax, left=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    for side in (("left", "bottom") if left else ("bottom",)):
        ax.spines[side].set_linewidth(0.75)
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.75, length=3)


def features(text):
    toks = [w.lower() for w in WORD.findall(text)]
    n = max(len(toks), 1)
    counts = dict.fromkeys(FUNCTION, 0)
    for w in toks:
        if w in counts:
            counts[w] += 1
    vec = [counts[w] / n for w in FUNCTION]
    vec += [text.count(p) / n for p in PUNCT]
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    vec += [len(toks) / sents / 40.0, len(set(toks)) / n,
            sum(len(w) for w in toks) / n / 10.0,
            sum(1 for w in toks if w in counts) / n]
    return np.asarray(vec, dtype=np.float64)


def load(system):
    path = REPO / "judgments" / JUDGE / system / "judgments.jsonl"
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if row.get("implicit_meaning"):
                out[row["file"]] = row["implicit_meaning"]
    return out


def knn_mean(block, k):
    out = np.empty(len(block))
    for i in range(0, len(block), 256):
        chunk = block[i:i + 256]
        d = np.linalg.norm(chunk[:, None, :] - block[None, :, :], axis=2)
        d.sort(axis=1)
        out[i:i + len(chunk)] = d[:, 1:1 + k].mean(axis=1)
    return out


def build():
    meta = {r["file"]: r for r in csv.DictReader(
        (REPO / "metadata.csv").open(newline="", encoding="utf-8"))}
    english = sorted(f for f, r in meta.items()
                     if sum(c.isascii() for c in r["annotation"]) / max(
                         len(r["annotation"]), 1) > 0.95)
    keep = set(english)
    corpora = {"Human reference": {f: meta[f]["annotation"] for f in english},
               "Human baseline": {f: v for f, v in load("human-baseline").items()
                                  if f in keep}}
    for system, label in SYSTEMS:
        corpora[label] = {f: v for f, v in load(system).items() if f in keep}
    labels = [l for l in corpora if len(corpora[l]) >= 800]
    blocks = {l: np.asarray([features(corpora[l][f]) for f in english
                             if f in corpora[l]]) for l in labels}
    X = np.vstack([blocks[l] for l in labels])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    for l in labels:
        blocks[l] = (blocks[l] - mu) / sd
    return labels, blocks, {l: [corpora[l][f] for f in english
                                if f in corpora[l]] for l in labels}


PALETTE = {
    "Human reference": "#000000",
    "Human baseline": "#0072B2",
    "Qwen3-Omni": "#D55E00",
    "Qwen3-Omni (no-think)": "#E69F00",
    "EchoInk-R1": "#009E73",
    "Qwen2.5-Omni 7B": "#56B4E9",
    "Gemma 4 E4B": "#CC79A7",
    "Gemma 4 E2B": "#882255",
    "Qwen2.5-Omni 3B": "#44AA99",
    "MiniCPM-o-2.6": "#777777",
}



def recycled_share(texts, budget=10000, n=3, seed=20260923, draws=20):
    """Share of each text's n-word phrases the same source uses elsewhere.

    A distance in a 214-dimensional function-word space is not something a
    reader can interpret, so the axis is a plain percentage instead: of the
    trigrams in this explanation, how many does its writer also use in other
    explanations?

    The reference is capped at a fixed word budget per source. Without that cap
    a verbose source would win by volume alone: MiniCPM-o-2.6 writes ten times
    as many words as a human, so any given phrase would have ten times the
    chance of turning up again regardless of how templated the writing is.

    Averaged over `draws` random references, which is not optional. A verbose
    source fills the budget from only ~32 documents, and a single draw put
    MiniCPM-o-2.6 at 10.4% and another at 24.4% -- a swing wider than most gaps
    between models. One draw cannot order the models; twenty can.

    n=3 is chosen, not assumed. The source ordering barely moves with n
    (Spearman rho >= 0.945 for n in 2..5), but the separation does: at n=2
    bigrams are common enough that every writer reuses them and the humans stop
    being lowest, and by n=5 several medians sit at 0% and the measure no
    longer discriminates. Only n=3 and n=4 put both humans below every model,
    and n=4 already floors two sources at 0%. See
    scripts/check_ngram_sensitivity.py.
    """
    grams_of = []
    for t in texts:
        toks = [w.lower() for w in WORD.findall(t)]
        grams_of.append([tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)])

    per_text = [[] for _ in texts]
    for draw in range(draws):
        rng = np.random.default_rng(seed + draw)
        reference, used, words = set(), set(), 0
        for i in rng.permutation(len(texts)):
            if words >= budget:
                break
            reference.update(grams_of[i])
            used.add(int(i))
            words += len(grams_of[i]) + n - 1
        for i, grams in enumerate(grams_of):
            if i in used or not grams:
                continue
            per_text[i].append(100.0 * sum(g in reference for g in grams)
                               / len(grams))
    return np.asarray([float(np.mean(v)) for v in per_text if v])


def plot_rarity(labels, texts, out):
    """A ridgeline, one row per source, ordered by median."""
    rec = {l: recycled_share(texts[l]) for l in labels}
    order = sorted(labels, key=lambda l: np.median(rec[l]))
    hi = min(100.0, np.percentile(np.concatenate(list(rec.values())), 99.5))
    grid = np.linspace(0, hi, 400)

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.38))
    for i, l in enumerate(order):
        human = l.startswith("Human")
        dens = gaussian_kde(rec[l])(grid)
        dens = dens / dens.max() * 0.92
        ax.fill_between(grid, i, i + dens, color=PALETTE[l],
                        alpha=0.55 if human else 0.30, linewidth=0, zorder=i + 2)
        ax.plot(grid, i + dens, color=PALETTE[l],
                linewidth=1.3 if human else 0.9, zorder=i + 2)
        ax.plot([np.median(rec[l])] * 2, [i, i + 0.30], color=PALETTE[l],
                linewidth=1.1, zorder=i + 2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.75, length=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    for tick, l in zip(ax.get_yticklabels(), order):
        tick.set_color(INK if l.startswith("Human") else MUTED)
    ax.set_xlim(0, hi)
    ax.set_ylim(-0.25, len(order) + 0.15)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("wording repeated across its own explanations", color=INK)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    for l in order:
        print(f"    {l:24}{np.median(rec[l]):6.1f}%  mean {rec[l].mean():5.1f}%"
              f"  n={len(rec[l])}")


def plot_lda(labels, blocks, out, bundle=25):
    """Each point is a BUNDLE of explanations, not a single one.

    A 33-word text carries too little signal to place on its own, so plotting
    one point per text shows a shift buried in overlap. The claim is about
    sources, so the unit of analysis is a bundle: the mean feature vector of
    `bundle` explanations from one source. That is a sample mean, and its
    scatter is the sampling distribution of a source's style.
    """
    X = np.vstack([blocks[l] for l in labels])
    y = np.concatenate([[i] * len(blocks[l]) for i, l in enumerate(labels)])
    n = len(blocks[labels[0]])
    rng = np.random.default_rng(20260923)
    is_train = np.zeros(n, dtype=bool)
    is_train[rng.permutation(n)[: n // 2]] = True
    mask = np.tile(is_train, len(labels))

    pca = PCA(n_components=50, random_state=0).fit(X[mask])
    lda = LinearDiscriminantAnalysis(
        n_components=2, solver="eigen", shrinkage="auto").fit(
            pca.transform(X[mask]), y[mask])

    pts, owner = [], []
    for i, l in enumerate(labels):
        held = blocks[l][~is_train]
        idx = rng.permutation(len(held))
        for j in range(0, len(idx) - bundle + 1, bundle):
            pts.append(held[idx[j:j + bundle]].mean(0))
            owner.append(i)
    pts = np.asarray(pts)
    owner = np.asarray(owner)
    Z = lda.transform(pca.transform(pts))

    is_human = np.isin(y, [labels.index(l) for l in labels
                           if l.startswith("Human")]).astype(int)
    f1 = f1_score(is_human, cross_val_predict(
        LogisticRegression(max_iter=2000), X, is_human, cv=5), average="macro")

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.34))
    for i, l in enumerate(labels):
        sel = owner == i
        ax.scatter(Z[sel, 0], Z[sel, 1], s=11, c=PALETTE[l], alpha=0.45,
                   linewidths=0, zorder=2, label=l)
    for i, l in enumerate(labels):
        c = Z[owner == i].mean(0)
        ax.scatter(*c, marker="D", s=46, c=PALETTE[l], edgecolors="white",
                   linewidths=0.9, zorder=5)
    bare_axes(ax)
    ax.set_xlabel("discriminant 1", color=INK)
    ax.set_ylabel("discriminant 2", color=INK)
    handles = [plt.Line2D([], [], marker="D", linestyle="", markersize=5,
                          markerfacecolor=PALETTE[l], markeredgecolor="white",
                          markeredgewidth=0.7, label=l) for l in labels]
    leg = ax.legend(handles=handles, loc="center left", frameon=False,
                    bbox_to_anchor=(1.01, 0.5), handletextpad=0.4,
                    labelspacing=0.55, borderpad=0.0)
    for text, l in zip(leg.get_texts(), labels):
        text.set_color(INK if l.startswith("Human") else MUTED)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"  lda: {len(pts)} bundles of {bundle}, human-vs-model F1 {f1:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=REPO / "figures")
    args = ap.parse_args()
    labels, blocks, raw = build()
    print(f"sources {len(labels)}, texts each {len(blocks[labels[0]])}")
    plot_rarity(labels, raw, args.outdir / "rarity.pdf")
    plot_lda(labels, blocks, args.outdir / "lda.pdf")


if __name__ == "__main__":
    main()
