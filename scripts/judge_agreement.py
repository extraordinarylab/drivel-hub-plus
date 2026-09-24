#!/usr/bin/env python3
"""Compare two Video-LM judges over the systems they both graded.

Backs the "Judge Robustness" appendix: how far the benchmark's conclusions
depend on which judge produced them. Reports system-level and item-level
agreement, per-dimension agreement, the human-baseline headroom each judge
reports, a same-family self-preference check, and the answer-length
correlation. With --latex it also writes the appendix table.

Correlations are computed here rather than with SciPy, which this env does not
carry; ties use average ranks and Kendall is tau-b.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DIMENSIONS = [
    "core_intent", "rhetorical_signal", "affective_or_social_meaning", "grounding",
    "hallucination_penalty", "literal_only_penalty", "vague_or_overgeneralized_penalty",
]
# Display names for the appendix table, keyed by the system's directory.
NAMES = {
    "human-baseline": ("Human baseline", "--"),
    "Qwen/Qwen3-Omni-30B-A3B-Thinking": ("Qwen3-Omni 30B-A3B", "Thinking"),
    "Qwen/Qwen3-Omni-30B-A3B-No-Thinking": ("Qwen3-Omni 30B-A3B", "No-thinking"),
    "openbmb/MiniCPM-o-2_6": ("MiniCPM-o-2.6 9B", "No-thinking"),
    "harryhsing/EchoInk-R1-7B": ("EchoInk-R1 7B", "No-thinking"),
    "Qwen/Qwen2.5-Omni-7B": ("Qwen2.5-Omni 7B", "No-thinking"),
    "Qwen/Qwen2.5-Omni-3B": ("Qwen2.5-Omni 3B", "No-thinking"),
    "google/gemma-4-E4B-it": ("Gemma 4 E4B", "Instruct"),
    "google/gemma-4-E2B-it": ("Gemma 4 E2B", "Instruct"),
}


def load(judgments: Path, judge: str) -> dict[str, dict[str, dict]]:
    """system -> filename -> judgement row, dropping rows with no score."""
    root = judgments / judge
    if not root.is_dir():
        raise SystemExit(f"no judgements for {judge} under {judgments}")
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(root.rglob("judgments.jsonl")):
        rows = {}
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                if "judge_score_total" in row:
                    rows[row["file"]] = row
        out[str(path.parent.relative_to(root))] = rows
    return out


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = average
        i = j + 1
    return out


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    concordant = discordant = tied_x = tied_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            a, b = x[i] - x[j], y[i] - y[j]
            if a == 0 and b == 0:
                tied_x += 1
                tied_y += 1
            elif a == 0:
                tied_x += 1
            elif b == 0:
                tied_y += 1
            elif a * b > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = ((concordant + discordant + tied_x)
                   * (concordant + discordant + tied_y)) ** 0.5
    return (concordant - discordant) / denominator if denominator else float("nan")


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def latex_table(stats, rank_a, rank_b, judge_a, judge_b, paired, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small", r"\color{revcolor}",
        r"\begin{tabular}{llcccccc}", r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Setting}}",
        rf"& \multicolumn{{3}}{{c}}{{\textbf{{{judge_a} judge}}}}",
        rf"& \multicolumn{{3}}{{c}}{{\textbf{{{judge_b} judge}}}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r"& & Total/12 & Aligned\% & Rank & Total/12 & Aligned\% & Rank \\",
        r"\midrule",
    ]
    for row in sorted(stats, key=lambda r: -r["a"]):
        name, setting = NAMES.get(row["system"], (row["system"], "--"))
        lines.append(
            f"{name} & {setting} & {row['a']:.2f} & {row['aligned_a']:.1f} "
            f"& {rank_a[row['system']]} & {row['b']:.2f} & {row['aligned_b']:.1f} "
            f"& {rank_b[row['system']]} \\\\"
        )
        if row["system"] == "human-baseline":
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{The same systems graded by two independent Video-LM judges, "
        rf"over the ${paired:,}$ items both judges scored.}}".replace(",", "{,}"),
        r"\label{tab:judge-agreement}", r"\end{table*}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, default=Path("judgments"))
    parser.add_argument("--judge-a", default="qwen3-omni-30b-a3b-instruct")
    parser.add_argument("--judge-b", required=True)
    parser.add_argument("--label-a", default="Qwen3-Omni-30B-A3B")
    parser.add_argument("--label-b")
    parser.add_argument("--latex", type=Path, help="write the appendix table here")
    args = parser.parse_args()
    label_b = args.label_b or args.judge_b

    A = load(args.judgments, args.judge_a)
    B = load(args.judgments, args.judge_b)
    systems = sorted(set(A) & set(B))
    if not systems:
        raise SystemExit("the two judges share no systems")

    stats, pairs = [], []
    for system in systems:
        common = sorted(set(A[system]) & set(B[system]))
        if not common:
            print(f"warning: no shared items for {system}")
            continue
        rows_a = [A[system][f] for f in common]
        rows_b = [B[system][f] for f in common]
        pairs += list(zip(rows_a, rows_b))
        stats.append({
            "system": system,
            "n": len(common),
            "a": sum(r["judge_score_total"] for r in rows_a) / len(common),
            "b": sum(r["judge_score_total"] for r in rows_b) / len(common),
            "aligned_a": 100 * sum(bool(r["judge_aligned"]) for r in rows_a) / len(common),
            "aligned_b": 100 * sum(bool(r["judge_aligned"]) for r in rows_b) / len(common),
            "words": sum(len((r.get("implicit_meaning") or "").split())
                         for r in rows_a) / len(common),
        })

    xa = [r["a"] for r in stats]
    xb = [r["b"] for r in stats]
    rank_a = {r["system"]: i + 1 for i, r in enumerate(sorted(stats, key=lambda r: -r["a"]))}
    rank_b = {r["system"]: i + 1 for i, r in enumerate(sorted(stats, key=lambda r: -r["b"]))}

    print(f"judge A = {args.judge_a}\njudge B = {args.judge_b}")
    print(f"\n=== per system ({len(stats)} systems, {len(pairs)} paired items) ===")
    print(f"  {'system':42} {'A':>7} {'B':>7} {'rank':>9} {'words':>7} {'n':>6}")
    for row in sorted(stats, key=lambda r: -r["a"]):
        movement = f"{rank_a[row['system']]}->{rank_b[row['system']]}"
        print(f"  {row['system']:42} {row['a']:7.3f} {row['b']:7.3f} "
              f"{movement:>9} {row['words']:7.1f} {row['n']:6}")

    print("\n=== system level ===")
    print(f"  pearson  {pearson(xa, xb):.4f}")
    print(f"  spearman {spearman(xa, xb):.4f}")
    print(f"  kendall  {kendall_tau_b(xa, xb):.4f}")

    ta = [p[0]["judge_score_total"] for p in pairs]
    tb = [p[1]["judge_score_total"] for p in pairs]
    la = [bool(p[0]["judge_aligned"]) for p in pairs]
    lb = [bool(p[1]["judge_aligned"]) for p in pairs]
    print(f"\n=== item level (n={len(pairs)}) ===")
    print(f"  pearson  {pearson(ta, tb):.4f} | spearman {spearman(ta, tb):.4f}")
    print(f"  exact agreement on the total: "
          f"{100 * sum(x == y for x, y in zip(ta, tb)) / len(ta):.2f}%")
    print(f"  mean |A-B| {sum(abs(x - y) for x, y in zip(ta, tb)) / len(ta):.3f} of 12")
    print(f"  mean  A-B  {sum(x - y for x, y in zip(ta, tb)) / len(ta):+.3f}")
    print(f"  aligned rate: A {100 * sum(la) / len(la):.2f}% | B {100 * sum(lb) / len(lb):.2f}%")
    print(f"  aligned label: agreement "
          f"{100 * sum(x == y for x, y in zip(la, lb)) / len(la):.2f}% | "
          f"cohen kappa {cohen_kappa(la, lb):.4f}")

    print("\n=== per dimension (item level) ===")
    for dimension in DIMENSIONS:
        da = [p[0][f"judge_{dimension}"] for p in pairs]
        db = [p[1][f"judge_{dimension}"] for p in pairs]
        print(f"  {dimension:36} r {pearson(da, db):6.3f} | "
              f"A {sum(da) / len(da):.3f} | B {sum(db) / len(db):.3f}")

    human = next((r for r in stats if r["system"] == "human-baseline"), None)
    if human:
        models = [r for r in stats if r["system"] != "human-baseline"]
        best_a = max(models, key=lambda r: r["a"])
        best_b = max(models, key=lambda r: r["b"])
        print("\n=== human-baseline headroom ===")
        print(f"  A: human {human['a']:.3f} vs best model {best_a['a']:.3f} "
              f"({best_a['system']}) -> {human['a'] - best_a['a']:+.3f}")
        print(f"  B: human {human['b']:.3f} vs best model {best_b['b']:.3f} "
              f"({best_b['system']}) -> {human['b'] - best_b['b']:+.3f}")

    print("\n=== self-preference: delta relative to the mean delta ===")
    mean_delta = sum(r["a"] - r["b"] for r in stats) / len(stats)
    for row in sorted(stats, key=lambda r: -(r["a"] - r["b"])):
        relative = (row["a"] - row["b"]) - mean_delta
        print(f"  {row['system']:42} delta {row['a'] - row['b']:+.3f} "
              f"relative {relative:+.3f}")

    words = [len((p[0].get("implicit_meaning") or "").split()) for p in pairs]
    print("\n=== answer length vs score (item level) ===")
    print(f"  A total vs length: spearman {spearman(list(map(float, words)), ta):+.4f}")
    print(f"  B total vs length: spearman {spearman(list(map(float, words)), tb):+.4f}")
    print(f"  (A-B)   vs length: spearman "
          f"{spearman(list(map(float, words)), [x - y for x, y in zip(ta, tb)]):+.4f}")

    if args.latex:
        latex_table(stats, rank_a, rank_b, args.label_a, label_b, len(pairs), args.latex)


if __name__ == "__main__":
    main()
