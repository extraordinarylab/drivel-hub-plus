#!/usr/bin/env python3
"""Apply the paper's structured VideoLLM judge to inference JSONL output."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


JUDGE_PROMPT = """You are grading answers in a benchmark for implicit multimodal Drivelology understanding.
Your job is to judge semantic understanding, not style.

You will be shown a video along with:
- Annotation: a human-provided description of the video's intended implicit narrative or underlying meaning.
- Implicit Meaning: a model-generated interpretation of the video's core meaning.

Scoring dimensions:
- core_intent: Did the model capture the main intended implicit narrative or core message?
- rhetorical_signal: Did it recognize the crucial rhetorical structure, such as misdirection, inversion, paradox, switchbait, irony, satire, metaphor, wordplay, contrast, or another Drivelological mechanism?
- affective_or_social_meaning: Did it identify the relevant emotional implication, social implication, cultural meaning, target, group, institution, or value judgment when supported?
- grounding: Did the interpretation correctly use the relevant visual, textual, speech, audio, timing, editing, or cross-modal evidence?
- hallucination_penalty: Penalize invented claims not grounded by the video, annotation, or available evidence.
- literal_only_penalty: Penalize answers that remain at surface description and miss the implicit meaning.
- vague_or_overgeneralized_penalty: Penalize answers that are too generic, vague, or non-committal to demonstrate understanding of the specific video.

Score bounds:
- core_intent must be one of 0, 1, 2, 3, 4, 5
- rhetorical_signal must be one of 0, 1, 2, 3
- affective_or_social_meaning must be one of 0, 1, 2
- grounding must be one of 0, 1, 2
- hallucination_penalty must be one of 0, 1, 2, 3
- literal_only_penalty must be one of 0, 1, 2, 3
- vague_or_overgeneralized_penalty must be one of 0, 1, 2

Scoring rule:
score_total = core_intent + rhetorical_signal + affective_or_social_meaning + grounding - hallucination_penalty - literal_only_penalty - vague_or_overgeneralized_penalty
The maximum possible score is 12.

Interpretation guide:
- Judge semantic meaning, not wording similarity.
- A partially correct answer that captures the video's implicit point should score much higher than a polished but purely literal answer.
- Do not require exact wording match.
- Minor differences in phrasing, detail level, or style are acceptable if the core meaning is preserved.
- If the Annotation is vague or short, judge whether the Implicit Meaning is a reasonable elaboration of it given the video.
- Be strict with hallucinations.
- Penalize interpretations that substitute a different implicit meaning, even if they sound plausible.
- Penalize answers that only describe visible actions, objects, captions, speech, or surface events.
- Only reward affective_or_social_meaning when that dimension is genuinely relevant and correctly captured.
- Use the video evidence as grounding support, not as extra hidden labels to overfit.
- Keep reasoning_short concise.

Alignment rule:
- aligned should be true only when the Implicit Meaning substantially captures the same overall implicit meaning as the Annotation.
- If core_intent is 0, aligned must be false.
- If the Implicit Meaning is empty, irrelevant, or purely surface-level, aligned must be false.
- If hallucination_penalty is 3, aligned should usually be false unless the hallucination is unrelated to the core interpretation.
- If literal_only_penalty is 3, aligned should usually be false unless the Annotation itself is mostly literal.

Return only valid JSON with exactly these fields:
{{
  "aligned": true or false,
  "core_intent": 0,
  "rhetorical_signal": 0,
  "affective_or_social_meaning": 0,
  "grounding": 0,
  "hallucination_penalty": 0,
  "literal_only_penalty": 0,
  "vague_or_overgeneralized_penalty": 0,
  "score_total": 0,
  "reasoning_short": "one concise English sentence"
}}

The reasoning_short field must be written in English regardless of the video's language.

Annotation: {annotation}
Implicit Meaning: {implicit_meaning}

Return only valid JSON."""

BOUNDS = {
    "core_intent": (0, 5),
    "rhetorical_signal": (0, 3),
    "affective_or_social_meaning": (0, 2),
    "grounding": (0, 2),
    "hallucination_penalty": (0, 3),
    "literal_only_penalty": (0, 3),
    "vague_or_overgeneralized_penalty": (0, 2),
}


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].removeprefix("json").strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("judge response is not a JSON object")
    if type(value.get("aligned")) is not bool:
        raise ValueError("aligned must be a JSON boolean")
    for field, (minimum, maximum) in BOUNDS.items():
        score = value.get(field)
        if type(score) is not int or not minimum <= score <= maximum:
            raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    if not isinstance(value.get("reasoning_short", ""), str):
        raise ValueError("reasoning_short must be a string")
    value["score_total"] = (
        value["core_intent"]
        + value["rhetorical_signal"]
        + value["affective_or_social_meaning"]
        + value["grounding"]
        - value["hallucination_penalty"]
        - value["literal_only_penalty"]
        - value["vague_or_overgeneralized_penalty"]
    )
    return value


def judge_one(
    client: OpenAI,
    model: str,
    video_path: Path,
    annotation: str,
    implicit_meaning: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
    enable_thinking: bool,
    max_retries: int,
) -> dict:
    encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
    content = [
        {"type": "text", "text": JUDGE_PROMPT.format(
            annotation=annotation, implicit_meaning=implicit_meaning,
        )},
        {"type": "video_url", "video_url": {
            "url": f"data:video/mp4;base64,{encoded}",
        }},
    ]
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            request_content = content
            if last_error is not None:
                request_content = content + [{
                    "type": "text",
                    "text": (
                        f"Your previous response was invalid: {last_error}. "
                        "Return a new JSON object and strictly obey every integer "
                        "range stated above; do not reuse the invalid value."
                    ),
                }]
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": request_content}],
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                extra_body={
                    "top_k": top_k,
                    "min_p": min_p,
                    "repetition_penalty": repetition_penalty,
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking,
                        "thinking": enable_thinking,
                    },
                },
            )
            answer = response.choices[0].message.content
            if not answer:
                raise RuntimeError("empty judge response")
            return parse_json_object(answer)
        except Exception as exc:
            last_error = exc
            print(f"retry {attempt}/{max_retries} for {video_path.name}: {exc}")
            if attempt < max_retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"failed after {max_retries} attempts: {last_error}")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def completed_files(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["file"] for row in read_jsonl(path) if "file" in row}


def remove_error_rows(path: Path) -> int:
    """Atomically remove explicit judge failures so resume can retry them."""
    if not path.exists():
        return 0
    rows = read_jsonl(path)
    kept = [row for row in rows if not row.get("judge_error")]
    removed = len(rows) - len(kept)
    if not removed:
        return 0
    temporary = path.with_name(f".{path.name}.retry-errors.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return removed


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def judge_row(row: dict, args: argparse.Namespace, client: OpenAI) -> dict:
    filename = row["file"]
    video_path = args.data_dir / filename
    record = dict(row)
    try:
        judgment = judge_one(
            client, args.model, video_path, str(row["annotation"]).strip(),
            str(row["implicit_meaning"]).strip(), args.max_tokens,
            args.temperature, args.top_p, args.top_k, args.min_p,
            args.presence_penalty, args.repetition_penalty,
            args.enable_thinking, args.max_retries,
        )
        for key, value in judgment.items():
            output_key = (
                "judge_reason" if key == "reasoning_short" else f"judge_{key}"
            )
            record[output_key] = value
    except Exception as exc:
        record["judge_aligned"] = None
        record["judge_error"] = str(exc)
        print(f"error: {filename}: {exc}", flush=True)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Judge model exposed by vLLM.")
    parser.add_argument(
        "--base-url",
        action="append",
        dest="base_urls",
        help=(
            "OpenAI-compatible endpoint. Repeat to distribute concurrent "
            "requests across multiple local replicas (default: "
            "http://localhost:8000/v1)."
        ),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retry-error-rows",
        action="store_true",
        help=(
            "Remove rows containing judge_error from the existing output and "
            "retry only those rows during normal resume."
        ),
    )
    parser.add_argument("--include-error-rows", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()
    if args.retry_error_rows:
        removed = remove_error_rows(args.output_jsonl)
        print(f"retrying {removed} explicit error rows")
    completed = completed_files(args.output_jsonl)
    base_urls = args.base_urls or ["http://localhost:8000/v1"]
    clients = [
        OpenAI(base_url=url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
        for url in base_urls
    ]

    pending = []
    for row in read_jsonl(args.input_jsonl):
        filename = row.get("file", "")
        if not filename.endswith(".mp4") or filename in completed:
            continue
        annotation = str(row.get("annotation", "")).strip()
        implicit_meaning = str(row.get("implicit_meaning", "")).strip()
        if not annotation:
            print(f"skipping {filename}: missing annotation")
            continue
        if not args.include_error_rows and (not implicit_meaning or row.get("error")):
            print(f"skipping {filename}: missing generation or inference error")
            continue
        video_path = args.data_dir / filename
        if not video_path.is_file():
            print(f"missing: {video_path}")
            continue
        pending.append(row)
        if args.limit is not None and len(pending) >= args.limit:
            break

    processed = 0
    if args.workers == 1:
        for row in pending:
            print(f"judging: {row['file']}", flush=True)
            append_jsonl(args.output_jsonl, judge_row(row, args, clients[0]))
            processed += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for index, row in enumerate(pending):
                print(f"judging: {row['file']}", flush=True)
                future = executor.submit(
                    judge_row, row, args, clients[index % len(clients)]
                )
                futures[future] = row["file"]
            for future in as_completed(futures):
                append_jsonl(args.output_jsonl, future.result())
                processed += 1
                print(
                    f"judged: {futures[future]} ({processed}/{len(pending)})",
                    flush=True,
                )

    judged = [r for r in read_jsonl(args.output_jsonl) if r.get("judge_aligned") is not None]
    if judged:
        alignment = sum(bool(r["judge_aligned"]) for r in judged) / len(judged)
        mean_score = sum(float(r["judge_score_total"]) for r in judged) / len(judged)
        print(f"alignment: {100 * alignment:.1f}% ({sum(bool(r['judge_aligned']) for r in judged)}/{len(judged)})")
        print(f"mean total score: {mean_score:.3f}/12")
    print(f"done: wrote {processed} new rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
