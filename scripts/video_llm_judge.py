#!/usr/bin/env python3
"""Apply the paper's structured VideoLLM judge to inference JSONL output."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

from openai import OpenAI


JUDGE_PROMPT = """You are grading answers in a benchmark for implicit multimodal Drivelology understanding.
Your job is to judge semantic understanding, not style.

You will be shown a video along with:
- Annotation: a human-provided description of the video's intended implicit narrative or underlying meaning.
- Latent Meaning: a model-generated interpretation of the video's core meaning.

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
- If the Annotation is vague or short, judge whether the Latent Meaning is a reasonable elaboration of it given the video.
- Be strict with hallucinations.
- Penalize interpretations that substitute a different implicit meaning, even if they sound plausible.
- Penalize answers that only describe visible actions, objects, captions, speech, or surface events.
- Only reward affective_or_social_meaning when that dimension is genuinely relevant and correctly captured.
- Use the video evidence as grounding support, not as extra hidden labels to overfit.
- Keep reasoning_short concise.

Alignment rule:
- aligned should be true only when the Latent Meaning substantially captures the same overall implicit meaning as the Annotation.
- If core_intent is 0, aligned must be false.
- If the Latent Meaning is empty, irrelevant, or purely surface-level, aligned must be false.
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
Latent Meaning: {latent_meaning}

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
    latent_meaning: str,
    max_retries: int,
) -> dict:
    encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
    content = [
        {"type": "text", "text": JUDGE_PROMPT.format(
            annotation=annotation, latent_meaning=latent_meaning,
        )},
        {"type": "video_url", "video_url": {
            "url": f"data:video/mp4;base64,{encoded}",
        }},
    ]
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                max_tokens=16384,
                temperature=0.7,
                top_p=0.8,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
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


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Judge model exposed by vLLM.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-error-rows", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()
    completed = completed_files(args.output_jsonl)
    client = OpenAI(base_url=args.base_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
    processed = 0

    for row in read_jsonl(args.input_jsonl):
        filename = row.get("file", "")
        if not filename.endswith(".mp4") or filename in completed:
            continue
        annotation = str(row.get("annotation", "")).strip()
        latent_meaning = str(row.get("latent_meaning", "")).strip()
        if not annotation:
            print(f"skipping {filename}: missing annotation")
            continue
        if not args.include_error_rows and (not latent_meaning or row.get("error")):
            print(f"skipping {filename}: missing generation or inference error")
            continue
        video_path = args.data_dir / filename
        if not video_path.is_file():
            print(f"missing: {video_path}")
            continue

        record = dict(row)
        print(f"judging: {filename}")
        try:
            judgment = judge_one(
                client, args.model, video_path, annotation, latent_meaning,
                args.max_retries,
            )
            for key, value in judgment.items():
                output_key = "judge_reason" if key == "reasoning_short" else f"judge_{key}"
                record[output_key] = value
        except Exception as exc:
            record["judge_aligned"] = None
            record["judge_error"] = str(exc)
            print(f"error: {filename}: {exc}")
        append_jsonl(args.output_jsonl, record)
        processed += 1
        if args.limit is not None and processed >= args.limit:
            break

    judged = [r for r in read_jsonl(args.output_jsonl) if r.get("judge_aligned") is not None]
    if judged:
        alignment = sum(bool(r["judge_aligned"]) for r in judged) / len(judged)
        mean_score = sum(float(r["judge_score_total"]) for r in judged) / len(judged)
        print(f"alignment: {100 * alignment:.1f}% ({sum(bool(r['judge_aligned']) for r in judged)}/{len(judged)})")
        print(f"mean total score: {mean_score:.3f}/12")
    print(f"done: wrote {processed} new rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
