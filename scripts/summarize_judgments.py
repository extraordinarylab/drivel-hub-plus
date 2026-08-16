#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import argparse
import json
import numbers

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

# Only fields that can sensibly be aggregated
all_rubrics = [
    key
    for key in field_types
    if "bool" in field_types[key] or "number" in field_types[key]
]

# Keep aligned first, total last, others in file/discovery order
ordered = []

if "judge_aligned" in all_rubrics:
    ordered.append("judge_aligned")

for key in all_rubrics:
    if key not in {"judge_aligned", "judge_score_total"}:
        ordered.append(key)

if "judge_score_total" in all_rubrics:
    ordered.append("judge_score_total")

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

    # judgments/Qwen/Model/judgments.jsonl
    # family = Qwen
    # model  = Model
    family = relative.parts[0] if len(relative.parts) >= 3 else ""
    model = path.parent.name

    row = {
        "family": family,
        "model": model,
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

        # bool rubric -> percentage true
        if all(isinstance(v, bool) for v in values):
            row[key] = 100 * sum(values) / len(values)

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
table.add_column("N", justify="right")

for key in rubrics:
    # prettier heading:
    # judge_core_intent -> Core intent
    label = key.removeprefix("judge_").replace("_", " ").title()
    table.add_column(label, justify="right")


for row in rows:
    cells = [
        row["family"],
        row["model"],
        str(row["n"]),
    ]

    for key in rubrics:
        value = row[key]

        if value is None:
            cells.append("-")
        elif "bool" in field_types[key]:
            cells.append(f"{value:.1f}%")
        else:
            cells.append(f"{value:.3f}")

    table.add_row(*cells)


Console().print(table)