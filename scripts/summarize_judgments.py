#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import argparse
import json
import numbers
import os

from rich.console import Console
from rich.table import Table


parser = argparse.ArgumentParser()
parser.add_argument(
    "root",
    nargs="?",
    default="judgments",
    help="Root folder, e.g. judgments",
)
parser.add_argument(
    "--rubrics",
    nargs="*",
    default=None,
    help=(
        "Optional rubric names to show. "
        "Can use names with or without judge_ prefix."
    ),
)
args = parser.parse_args()

root = Path(args.root)
files = sorted(root.rglob("judgments.jsonl"))

if not files:
    raise SystemExit(f"No judgments.jsonl found under {root}")


# ------------------------------------------------------------
# First pass: read files + automatically discover judge_* fields
# ------------------------------------------------------------

datasets = []
field_types = defaultdict(set)

for path in files:
    records = []

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: {path}:{lineno}: {e}")
                continue

            records.append(obj)

            for key, value in obj.items():
                if not key.startswith("judge_"):
                    continue

                if isinstance(value, bool):
                    field_types[key].add("bool")
                elif isinstance(value, numbers.Number):
                    field_types[key].add("number")
                elif isinstance(value, str):
                    field_types[key].add("str")

    datasets.append((path, records))


# ------------------------------------------------------------
# Select rubrics
# ------------------------------------------------------------

# Table 2 of the paper, in its column order. Anything discovered outside this
# map is appended after it, so a new rubric still shows up.
TABLE2_COLUMNS = [
    ("judge_aligned", "Aligned\u2191"),
    ("judge_core_intent", "Core/5\u2191"),
    ("judge_rhetorical_signal", "Rhet./3\u2191"),
    ("judge_affective_or_social_meaning", "Social/2\u2191"),
    ("judge_grounding", "Ground./2\u2191"),
    ("judge_hallucination_penalty", "Halluc./3\u2193"),
    ("judge_literal_only_penalty", "Literal/3\u2193"),
    ("judge_vague_or_overgeneralized_penalty", "Vague/2\u2193"),
    ("judge_score_total", "Total/12\u2191"),
]
TABLE2_ORDER = [key for key, _ in TABLE2_COLUMNS]
LABELS = dict(TABLE2_COLUMNS)

# judge_* fields that record how a run was produced, not how it scored. They
# are booleans, so without this they would be averaged into the table.
PROVENANCE_FIELDS = {"judge_use_audio_in_video"}

# Only fields that can sensibly be aggregated
all_rubrics = [
    key
    for key in field_types
    if ("bool" in field_types[key] or "number" in field_types[key])
    and key not in PROVENANCE_FIELDS
]

# Table 2 order first, then anything new that was discovered.
ordered = [key for key in TABLE2_ORDER if key in all_rubrics]
ordered += [key for key in all_rubrics if key not in TABLE2_ORDER]
all_rubrics = ordered


if args.rubrics:
    wanted = []

    for name in args.rubrics:
        key = name if name.startswith("judge_") else f"judge_{name}"

        if key not in all_rubrics:
            print(f"Warning: rubric not found: {key}")
            continue

        wanted.append(key)

    rubrics = wanted
else:
    rubrics = all_rubrics


# ------------------------------------------------------------
# Aggregate
# ------------------------------------------------------------

rows = []

for path, records in datasets:
    relative = path.relative_to(root)

    # judgments/<org>/<model>/judgments.jsonl, or with a judge level in front
    # once a second judge is added. The org is the component directly above the
    # model, when there is one.
    parts = relative.parts
    family = parts[-2] if len(parts) >= 3 else ""
    model = path.parent.name

    judges = {
        record["judge_model"].rstrip("/").rsplit("/", 1)[-1]
        for record in records
        if record.get("judge_model")
    }
    row = {
        "family": family,
        "model": model,
        "judge": ", ".join(sorted(judges)) if judges else "-",
        "n": len(records),
    }

    for key in rubrics:
        values = [
            record[key]
            for record in records
            if key in record and record[key] is not None
        ]

        if not values:
            row[key] = None
            continue

        # bool rubric -> rate in [0, 1], as Table 2 reports it
        if all(isinstance(v, bool) for v in values):
            row[key] = sum(values) / len(values)

        # numeric rubric -> arithmetic mean
        else:
            nums = [
                float(v)
                for v in values
                if isinstance(v, numbers.Number)
                and not isinstance(v, bool)
            ]
            row[key] = sum(nums) / len(nums) if nums else None

    rows.append(row)


# ------------------------------------------------------------
# Print Rich table
# ------------------------------------------------------------

table = Table(
    show_header=True,
    header_style="bold",
    show_lines=True,
)

table.add_column("Family", no_wrap=True)
table.add_column("Prediction", no_wrap=True)
table.add_column("Judge", no_wrap=True)
table.add_column("N", justify="right")

for key in rubrics:
    # prettier heading:
    # judge_core_intent -> Core intent
    label = LABELS.get(key) or key.removeprefix("judge_").replace("_", " ").title()
    table.add_column(label, justify="right")


for row in rows:
    cells = [
        row["family"],
        row["model"],
        row["judge"],
        str(row["n"]),
    ]

    for key in rubrics:
        value = row[key]

        if value is None:
            cells.append("-")
        else:
            # Table 2 prints every cell, including the alignment rate, to 3dp.
            cells.append(f"{value:.3f}")

    table.add_row(*cells)


# Redirected output (a Slurm log) has no terminal width, and Rich then falls
# back to 80 columns and squeezes the 13 columns down to nothing.
Console(width=int(os.environ.get("SUMMARY_WIDTH", "200"))).print(table)