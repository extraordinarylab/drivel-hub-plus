#!/usr/bin/env python3
"""Cascade baseline against multimodal, overall and by carrying modality.

The point of the cascade is not its score but what the score attributes. A
condition that matches multimodal where its modality carries the meaning, and
collapses where it does not, separates modality extraction from pragmatic
reasoning -- which is what the text-conversion diagnostic was asked to do.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CONDS = ("transcript-only", "caption-only", "ocr-asr")
LABEL = {"transcript-only": "ASR only", "caption-only": "OCR only",
         "ocr-asr": "OCR + ASR"}
# Printable names for the modality groups, in reporting order.
GROUPS = [("audio", "Audio"), ("audio+vision", "Audio + vision"),
          ("text", "On-screen text"), ("text+vision", "Text + vision"),
          ("vision", "Vision only")]


def load(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["file"]: r for r in
            (json.loads(l) for l in path.open(encoding="utf-8") if l.strip())
            if "judge_score_total" in r}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", default="qwen3-omni-30b-a3b-instruct")
    ap.add_argument("--reference", default="Qwen/Qwen3-Omni-30B-A3B-No-Thinking",
                    help="multimodal system the cascade is compared against")
    ap.add_argument("--metadata-csv", type=Path, default=Path("metadata.csv"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    meta = {r["file"]: r for r in csv.DictReader(
        args.metadata_csv.open(newline="", encoding="utf-8"))}
    root = Path("judgments") / args.judge
    mm = load(root / args.reference / "judgments.jsonl")
    conds = {c: load(root / "cascade" / c / "judgments.jsonl") for c in CONDS}
    common = sorted(set(mm) & set.intersection(*(set(d) for d in conds.values())))
    print(f"judge {args.judge} | reference {args.reference}")
    print(f"paired clips scored in every setting: {len(common)}\n")

    def stats(files, rows):
        if not files:
            return 0.0, 0.0
        return (sum(rows[f]["judge_score_total"] for f in files) / len(files),
                100 * sum(bool(rows[f]["judge_aligned"]) for f in files) / len(files))

    rows_out = []
    mm_all, mm_al = stats(common, mm)
    print(f"  {'condition':14} {'total/12':>9} {'aligned%':>9} {'% of multimodal':>16}")
    print(f"  {'multimodal':14} {mm_all:9.2f} {mm_al:9.1f} {100.0:15.0f}%")
    for c in CONDS:
        tot, al = stats(common, conds[c])
        print(f"  {LABEL[c]:14} {tot:9.2f} {al:9.1f} {100*tot/mm_all:15.0f}%")
        rows_out.append({"condition": LABEL[c], "total": tot, "aligned": al})

    print(f"\n  {'meaning carried by':18} {'n':>5} {'multimodal':>11}"
          + "".join(f"{LABEL[c]:>13}" for c in CONDS))
    by_group = []
    for key, name in GROUPS:
        files = [f for f in common if meta[f]["modalities"].strip() == key]
        if len(files) < 15:
            continue
        m, _ = stats(files, mm)
        line = f"  {name:18} {len(files):5} {m:11.2f}"
        entry = {"group": name, "n": len(files), "multimodal": m}
        for c in CONDS:
            t, _ = stats(files, conds[c])
            line += f"{t:13.2f}"
            entry[c] = t
        print(line)
        by_group.append(entry)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"n": len(common), "multimodal": mm_all, "multimodal_aligned": mm_al,
             "overall": rows_out, "by_group": by_group}, indent=2) + "\n",
            encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
