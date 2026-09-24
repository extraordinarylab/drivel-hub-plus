#!/usr/bin/env python3
"""Grade predictions with a hosted Gemini model.

Same rubric, prompt, bounds and output fields as scripts/video_llm_judge.py, so
the gradings stay comparable; the difference is that the judge is an API rather
than a served checkpoint.

Cost control, because this one spends real money:
  * every call's usage_metadata is recorded on the row, so the run's true cost
    is auditable afterwards rather than estimated,
  * --max-retries defaults to 3, not 10: a judge that cannot produce valid JSON
    in three attempts is better recorded as an error than paid for ten times,
  * --budget-usd stops the run when the measured spend crosses a ceiling,
  * output is append-only and resumable, so a stop is never lost work.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.video_llm_judge import (
    JUDGE_PROMPT,
    append_jsonl,
    completed_files,
    parse_json_object,
    read_jsonl,
    remove_error_rows,
)

# USD per million tokens, input/output, from the published price list.
#
# CALIBRATION. Token counts times the published prices under-predict what
# Google bills for video input, so the running total used by --budget-usd is
# scaled. The factor is now measured, not fitted: 12,000 judgements listed at
# $32.16 were billed GBP 31.77, about $40, giving ~1.26.
#
# The earlier 2.33 was wrong and doubled the recorded cost of every cascade
# row. It came from inferring the factor from a balance that ran out, but that
# inference assumed the tracked spend was already scaled by 1.29 when it was in
# fact unscaled, so the correction got applied twice.
#
# Nothing the paper prints depends on this: scripts/make_cost_table.py prices
# from raw token counts, precisely so a change here cannot move a published
# number again.
BILLING_CALIBRATION = 1.26
PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (0.30, 2.50),
}


def load_key(env_path: Path) -> str:
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit(f"GEMINI_API_KEY not found in {env_path}")


def cost_of(model: str, prompt_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, (0.75, 3.75))
    listed = prompt_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out
    return listed * BILLING_CALIBRATION


def judge_one(client, types, model: str, video: Path, annotation: str,
              answer: str, max_retries: int) -> tuple[dict, dict]:
    prompt = (f"{JUDGE_PROMPT}\n\nAnnotation: {annotation}\n\n"
              f"Implicit Meaning: {answer}\n\nReturn only valid JSON.")
    blob = video.read_bytes()
    usage_total = {"prompt": 0, "output": 0}
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=blob, mime_type="video/mp4"),
                          prompt],
            )
            usage = response.usage_metadata
            usage_total["prompt"] += usage.prompt_token_count or 0
            usage_total["output"] += usage.candidates_token_count or 0
            return parse_json_object((response.text or "").strip()), usage_total
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            text = str(exc)
            # RESOURCE_EXHAUSTED means two very different things and the
            # distinction matters: 402 is an empty prepaid balance, 429 is a
            # quota. Reporting a quota as "out of credit" sends people to buy
            # credit they do not need -- which is exactly what happened here on
            # the 10,000 requests/model/day Tier-1 cap.
            if "402" in text:
                raise SystemExit(
                    "[gemini] STOPPING: prepaid credit is exhausted (HTTP 402). "
                    "Rows already written are kept; top up and re-run to resume.")
            if "PerDay" in text or "per_day" in text:
                retry = re.search(r"retry in ([0-9hms.]+)", text)
                raise SystemExit(
                    "[gemini] STOPPING: DAILY REQUEST QUOTA reached, not a billing "
                    "problem -- your credit balance is untouched. "
                    + (f"Resets in {retry.group(1)}. " if retry else "")
                    + "Rows already written are kept; re-run after the reset.")
            if attempt < max_retries:
                # Back off on transient quota errors, retry parse errors fast.
                time.sleep(min(2 ** attempt, 30) if "429" in text else 1)
    raise RuntimeError(f"failed after {max_retries} attempts: {last_error}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--env", type=Path, default=Path(".env"))
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample", type=int,
                    help="judge this many clips, sampled deterministically")
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=load_key(args.env))
    rows = read_jsonl(args.input_jsonl)
    if args.sample:
        # Same clips for every system, so the judges can be compared per item.
        rows = sorted(rows, key=lambda r: r["file"])
        random.Random(args.seed).shuffle(rows)
        rows = sorted(rows[:args.sample], key=lambda r: r["file"])
    # A row that failed is still written, with judge_error and no score, and
    # completed_files counts any row carrying a "file" key. Without this the
    # clips that errored on an earlier run are skipped forever and the grading
    # quietly finishes short -- 14 clips were already in that state when a
    # daily-quota stop wrote error rows for them.
    retried = remove_error_rows(args.output_jsonl)
    if retried:
        print(f"[gemini] retrying {retried} rows that previously errored",
              flush=True)
    done = completed_files(args.output_jsonl)
    rows = [r for r in rows if r["file"] not in done]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[gemini] {args.model}: {len(rows)} rows to grade "
          f"({len(done)} already done), budget ${args.budget_usd:.2f}", flush=True)
    if not rows:
        return

    spent = 0.0
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        record = dict(row)
        record["judge_model"] = args.model
        record["judge_use_audio_in_video"] = True
        try:
            judgment, usage = judge_one(
                client, types, args.model, args.data_dir / row["file"],
                str(row["annotation"]).strip(), str(row["implicit_meaning"]).strip(),
                args.max_retries)
            for key, value in judgment.items():
                out_key = "judge_reason" if key == "reasoning_short" else f"judge_{key}"
                record[out_key] = value
            call_cost = cost_of(args.model, usage["prompt"], usage["output"])
            spent += call_cost
            record["judge_prompt_tokens"] = usage["prompt"]
            record["judge_output_tokens"] = usage["output"]
            record["judge_cost_usd"] = round(call_cost, 6)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            record["judge_aligned"] = None
            record["judge_error"] = str(exc)[:400]
            print(f"[gemini] error {row['file']}: {str(exc)[:160]}", flush=True)
        append_jsonl(args.output_jsonl, record)

        if index % 25 == 0 or index == len(rows):
            rate = (time.perf_counter() - started) / index
            print(f"[gemini] {index}/{len(rows)} | ${spent:.3f} spent "
                  f"(${spent/index:.4f}/item) | {rate:.1f}s/item", flush=True)
        if spent >= args.budget_usd:
            print(f"[gemini] BUDGET REACHED (${spent:.2f}); stopping after "
                  f"{index} rows. Re-run with a higher --budget-usd to continue.",
                  flush=True)
            break

    print(f"[gemini] done: ${spent:.3f} over {index} rows "
          f"(${spent/max(index,1):.4f}/item)", flush=True)


if __name__ == "__main__":
    main()
