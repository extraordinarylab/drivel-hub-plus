#!/usr/bin/env python3
"""Emit the human baseline in the shape video_llm_judge.py consumes.

`metadata.csv` already holds a second annotator's reading of every clip in
`human_baseline`, so this needs no model and no GPU: it just re-labels that
column as `implicit_meaning` and keeps `annotation` as the reference. Judging
the result with the same judge used for the models gives the human ceiling on
the same rubric.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=Path("metadata.csv"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--column",
        default="human_baseline",
        help="Column holding the human explanation to score (default: human_baseline).",
    )
    args = parser.parse_args()

    with args.metadata_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for row in rows:
            text = (row.get(args.column) or "").strip()
            if not text:
                skipped += 1
                continue
            record = {
                **row,
                "video_path": str(args.data_dir / row["file"]),
                "model": f"human:{args.column}",
                "input_mode": "full",
                "remove_audio": False,
                "remove_vision": False,
                "use_audio_in_video": False,
                "implicit_meaning": text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} rows to {args.output_jsonl} (skipped {skipped} empty)")


if __name__ == "__main__":
    main()
