#!/usr/bin/env python3
"""Compute paired input-stream ablation deltas by evidence label.

Each ``--run`` supplies one model/setting and its full, without-audio, and
without-vision judged JSONL files. Deltas are always ablated minus full, as in
Figure 2 of the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


MODALITY_ORDER = [
    "text", "audio", "vision", "text+audio", "text+vision", "audio+vision",
    "text+audio+vision",
]


def canonical_modalities(value: object) -> str:
    aliases = {"caption": "text", "captions": "text", "visual": "vision"}
    if isinstance(value, list):
        parts = [str(item).strip().lower() for item in value]
    elif isinstance(value, dict):
        parts = [str(key).strip().lower() for key, enabled in value.items() if enabled]
    else:
        text = str(value or "").strip().lower()
        for separator in (",", "/", "|"):
            text = text.replace(separator, "+")
        parts = [item.strip() for item in text.split("+") if item.strip()]
    parts = [aliases.get(item, item) for item in parts]
    known = [item for item in ("text", "audio", "vision") if item in parts]
    unknown = [item for item in parts if item not in known]
    return "+".join(known + unknown) or "unknown"


def read_judged(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            file_id = row.get("file")
            if not file_id:
                raise ValueError(f"missing file ID at {path}:{line_number}")
            if file_id in rows:
                raise ValueError(f"duplicate file ID {file_id!r} in {path}")
            rows[file_id] = row
    return rows


def numeric_alignment(value: object) -> float:
    if type(value) is bool:
        return float(value)
    if value in (0, 1):
        return float(value)
    raise ValueError(f"invalid judge_aligned value: {value!r}")


def aggregate_run(
    label: str,
    full_path: Path,
    without_audio_path: Path,
    without_vision_path: Path,
) -> list[dict]:
    full = read_judged(full_path)
    variants = {
        "without_audio": read_judged(without_audio_path),
        "without_vision": read_judged(without_vision_path),
    }
    accumulators: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0, "score": 0.0, "aligned": 0.0}
    )

    for variant_name, ablated in variants.items():
        for file_id in sorted(full.keys() & ablated.keys()):
            baseline = full[file_id]
            variant = ablated[file_id]
            if baseline.get("judge_score_total") is None or variant.get("judge_score_total") is None:
                continue
            if baseline.get("judge_aligned") is None or variant.get("judge_aligned") is None:
                continue
            modality = canonical_modalities(baseline.get("modalities"))
            bucket = accumulators[(variant_name, modality)]
            bucket["n"] += 1
            bucket["score"] += float(variant["judge_score_total"]) - float(baseline["judge_score_total"])
            bucket["aligned"] += (
                numeric_alignment(variant["judge_aligned"])
                - numeric_alignment(baseline["judge_aligned"])
            )

    rows = []
    for (variant_name, modality), values in accumulators.items():
        n = int(values["n"])
        rows.append({
            "model_setting": label,
            "ablation": variant_name,
            "modalities": modality,
            "n_paired": n,
            "delta_score_total": values["score"] / n,
            "delta_aligned_rate": values["aligned"] / n,
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", nargs=4, required=True,
        metavar=("LABEL", "FULL", "WITHOUT_AUDIO", "WITHOUT_VISION"),
        help="May be repeated for each model/setting row in the figure.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for label, full, without_audio, without_vision in args.run:
        rows.extend(aggregate_run(
            label, Path(full), Path(without_audio), Path(without_vision),
        ))
    modality_index = {name: index for index, name in enumerate(MODALITY_ORDER)}
    rows.sort(key=lambda row: (
        row["model_setting"], row["ablation"],
        modality_index.get(row["modalities"], len(modality_index)), row["modalities"],
    ))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_setting", "ablation", "modalities", "n_paired",
        "delta_score_total", "delta_aligned_rate",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} grouped rows to {args.output_csv}")


if __name__ == "__main__":
    main()
