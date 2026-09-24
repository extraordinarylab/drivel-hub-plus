#!/usr/bin/env python3
"""Render the dataset-statistics tables from metadata.csv.

Emits three files, all counted from the released metadata so they cannot drift
from the dataset again:

  statistics.tex  Table 1, the top labels per category with an "Other" bucket
  datasets.tex    Table 6, the full language breakdown
  remarks.tex     Table 7, the sensitive-content remarks

The raw columns are free text written by the annotators, so they are normalised
here: case is folded, the spelling slip "mardarin" is mapped to Mandarin,
"taiwanese" is reported as Taiwanese Hokkien, an empty field or "-" becomes
None, and a multi-language cell is sorted alphabetically so that
"english+mandarin" and "mandarin+english" count as one label.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

LANGUAGE = {
    "english": "English",
    "mandarin": "Mandarin",
    "mardarin": "Mandarin",  # annotator typo, one row
    "taiwanese": "Taiwanese Hokkien",
    "cantonese": "Cantonese",
    "korean": "Korean",
    "japanese": "Japanese",
    "french": "French",
    "spanish": "Spanish",
    "arabic": "Arabic",
    "icelandic": "Icelandic",
    "thai": "Thai",
}
REMARK = {"": "None", "none": "None", "racist": "Racist",
          "dark": "Dark humour", "sex": "Sexual content"}
MODALITY = {"text": "\\textmod", "audio": "\\audiomod", "vision": "\\videomod"}
NONE = "\\textit{None}"
TOP_N = 5


def normalise_language(value: str) -> str:
    value = (value or "").strip().lower()
    if value in ("", "-", "none"):
        return NONE
    parts = sorted(LANGUAGE.get(p.strip(), p.strip().title()) for p in value.split("+") if p.strip())
    return "+".join(parts) or NONE


def normalise_modality(value: str) -> str:
    value = (value or "").strip().lower()
    order = ["text", "audio", "vision"]
    parts = [p.strip() for p in value.split("+") if p.strip()]
    parts.sort(key=lambda p: order.index(p) if p in order else len(order))
    return "+".join(MODALITY.get(p, p) for p in parts) or NONE


def normalise_remark(value: str) -> str:
    return REMARK.get((value or "").strip().lower(), (value or "").strip().title())


def ordered(counts: Counter) -> list[tuple[str, int]]:
    """Most frequent first; None last within its count so it reads as a floor."""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0] == NONE, kv[0]))


def block(name: str, counts: Counter, limit: int | None) -> list[str]:
    rows = ordered(counts)
    if limit is not None and len(rows) > limit:
        head, tail = rows[:limit], rows[limit:]
        rows = head + [("Other", sum(v for _, v in tail))]
    lines = [f"\\multirow{{{len(rows)}}}{{*}}{{{name}}}"]
    lines += [f"& {label} & {count} \\\\" for label, count in rows]
    return lines


def table(body: list[str], caption: str, label: str) -> str:
    return "\n".join([
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{llr}", "\\toprule",
        "\\textbf{Category} & \\textbf{Label} & \\textbf{Count} \\\\",
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}",
        "\\caption{", caption, "}", f"\\label{{{label}}}", "\\end{table}", "",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-csv", type=Path, default=REPO / "metadata.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO / "latex")
    args = ap.parse_args()

    with args.metadata_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    total = len(rows)

    speech = Counter(normalise_language(r.get("speech", "")) for r in rows)
    caption = Counter(normalise_language(r.get("caption", "")) for r in rows)
    modality = Counter(normalise_modality(r.get("modalities", "")) for r in rows)
    remark = Counter(normalise_remark(r.get("remark", "")) for r in rows)
    for name, counts in (("speech", speech), ("caption", caption),
                         ("modality", modality), ("remark", remark)):
        assert sum(counts.values()) == total, f"{name}: {sum(counts.values())} != {total}"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Table 1. The modality block is kept, commented out, because the paper no
    # longer reports modality-conditioned results; uncomment to restore it.
    summary = ["% " + line for line in block("Modality", modality, None)] + ["% \\midrule"]
    summary += block("Speech", speech, TOP_N) + ["\\midrule"]
    summary += block("Caption", caption, TOP_N)
    (args.out_dir / "statistics.tex").write_text(table(
        summary,
        "Dataset distributions by speech language and caption language. "
        "A full breakdown of the language distribution underlying ``Other'' is "
        "provided in Table~\\ref{tab:dataset-language-full}.",
        "tab:dataset-distributions",
    ), encoding="utf-8")

    full = block("Speech", speech, None) + ["", "\\midrule"] + block("Caption", caption, None)
    (args.out_dir / "datasets.tex").write_text(table(
        full,
        "Dataset distributions by speech language and caption language. "
        "For language metadata, multiple languages are recorded when multiple "
        "linguistic signals contribute to interpretation. The label "
        "\\textit{None} indicates that no corresponding linguistic signal is present.",
        "tab:dataset-language-full",
    ), encoding="utf-8")

    remark_rows = ordered(remark)
    (args.out_dir / "remarks.tex").write_text("\n".join([
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\begin{tabular}{lr}", "\\toprule",
        "\\textbf{Category} & \\textbf{Count} \\\\", "\\midrule",
        *[f"{label} & {count} \\\\" for label, count in remark_rows],
        "\\bottomrule", "\\end{tabular}", "\\caption{",
        "Distribution of sensitive-content remark categories in the dataset. "
        "The label \\textit{None} denotes examples without an additional "
        "sensitive-content remark. These categories are included to document "
        "potentially sensitive material and support transparent use of the "
        "benchmark; they do not reflect the views of the authors.",
        "}", "\\label{tab:sensitive-content-remarks}", "\\end{table}", "",
    ]), encoding="utf-8")

    for name, counts in (("speech", speech), ("caption", caption),
                         ("modality", modality), ("remark", remark)):
        print(f"{name}: {len(counts)} labels over {total} rows")
    print(f"wrote statistics.tex, datasets.tex, remarks.tex to {args.out_dir}")


if __name__ == "__main__":
    main()
