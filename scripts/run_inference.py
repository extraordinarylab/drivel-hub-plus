#!/usr/bin/env python3
"""Generate DrivelHub+ implicit-meaning explanations through a vLLM server.

The three input modes reproduce the full-input explanation setting and the two
input-stream ablations used in the paper. Output is append-only JSONL and can
be resumed safely by filename.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from openai import OpenAI


FULL_PROMPT = """Output only a short explanation of the video's core meaning or implied message. Do not add titles, quotation marks, introductions, bullet points, numbering, or extra formatting.

[Language Rule — Highest Priority]
- Detect the video's primary language from spoken audio and on-screen text.
- The response MUST be written entirely in that same language.
- Do not translate, mix languages, or add explanations in another language.
- If the video is mainly English, respond fully in English. If mainly Chinese, respond fully in Chinese, etc.

[Content Rule]
- Explain the underlying meaning, emotion, joke, sarcasm, irony, or social implication of the video.
- Do NOT simply describe or summarize what is happening scene by scene.
- Avoid repeating the video's exact words unless absolutely necessary.
- Focus on interpretation rather than transcription.

[Style Rule]
- Keep it concise: usually 1-2 sentences, maximum 3 short sentences.
- Write naturally as a single paragraph only.
- No bullet points, lists, headers, labels, or markdown.
- No opening phrases like “This video shows...” or “The video means...”.
- No unnecessary analysis, background context, or moral commentary.

If the output language does not match the video's primary language, the response is considered incorrect."""

VISION_ONLY_PROMPT = """Output only a short explanation of the video's core meaning or implied message based on visual information only. Do not add titles, quotation marks, introductions, bullet points, numbering, or extra formatting.

[Modality Rule — Highest Priority]
- You are given the original visual stream without audio.
- Use only visual frames, visible actions, objects, facial expressions, body language, scene context, captions, subtitles, and on-screen text.
- Do NOT infer from spoken words, music, sound effects, tone of voice, or any other audio cues.
- Do NOT say that audio is missing or unavailable.

[Language Rule]
- Detect the primary language from visible on-screen text, captions, subtitles, signs, or other written content.
- The response MUST be written entirely in that same language.
- Do not translate, mix languages, or add explanations in another language.
- If the visual text is mainly English, respond fully in English. If mainly Chinese, respond fully in Chinese, etc.
- If there is no readable text, respond in the most likely language suggested by the visual context. If no language can be inferred, respond in English.

[Content Rule]
- Explain the underlying meaning, emotion, joke, sarcasm, irony, or social implication conveyed by the visual content.
- Do NOT simply describe or summarize what is happening scene by scene.
- Avoid repeating on-screen text unless absolutely necessary.
- Focus on interpretation rather than description.
- If the visual information alone is insufficient to infer a hidden meaning, give the best concise interpretation supported by visual evidence only.

[Style Rule]
- Keep it concise: usually 1-2 sentences, maximum 3 short sentences.
- Write naturally as a single paragraph only.
- No bullet points, lists, headers, labels, or markdown.
- No opening phrases like “The video shows...” or “This clip means...”.
- No unnecessary analysis, background context, or moral commentary."""

AUDIO_ONLY_PROMPT = """Output only a short explanation of the audio's core meaning or implied message. Do not add titles, quotation marks, introductions, bullet points, numbering, or extra formatting.

[Modality Rule — Highest Priority]
- You are given only the audio extracted from the original video.
- Use only spoken words, music, sound effects, tone, emotion, rhythm, silence, and other audible cues.
- Do NOT refer to visual frames, objects, actions, facial expressions, captions, subtitles, or on-screen text.
- Do NOT say that visual information is missing or unavailable.

[Language Rule]
- Detect the primary language from the spoken audio.
- The response MUST be written entirely in that same language.
- Do not translate, mix languages, or add explanations in another language.
- If the audio is mainly English, respond fully in English. If mainly Chinese, respond fully in Chinese, etc.
- If there is no intelligible speech, respond in the most likely language suggested by the audible context. If no language can be inferred, respond in English.

[Content Rule]
- Explain the underlying meaning, emotion, joke, sarcasm, irony, or social implication conveyed by the audio.
- Do NOT simply transcribe or summarize the audio.
- Avoid repeating exact words unless absolutely necessary.
- Focus on interpretation rather than transcription.
- If the audio alone is insufficient to infer a hidden meaning, give the best concise interpretation supported by the audible evidence only.

[Style Rule]
- Keep it concise: usually 1-2 sentences, maximum 3 short sentences.
- Write naturally as a single paragraph only.
- No bullet points, lists, headers, labels, or markdown.
- No opening phrases like “The audio says...” or “This audio means...”.
- No unnecessary analysis, background context, or moral commentary."""

PROMPTS = {
    "full": FULL_PROMPT,
    "without-audio": VISION_ONLY_PROMPT,
    "without-vision": AUDIO_ONLY_PROMPT,
}


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{exc.stderr}") from exc


def has_audio(path: Path) -> bool:
    result = run_checked([
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
    ])
    return bool(result.stdout.strip())


@contextmanager
def prepared_media(video_path: Path, mode: str) -> Iterator[tuple[Path, str, str]]:
    if mode == "full":
        yield video_path, "video", "video/mp4"
        return

    if mode == "without-audio":
        with tempfile.TemporaryDirectory(prefix="drivelhub-") as tmp:
            output = Path(tmp) / "vision_only.mp4"
            run_checked([
                "ffmpeg", "-y", "-i", str(video_path), "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(output),
            ])
            yield output, "video", "video/mp4"
        return

    if not has_audio(video_path):
        raise ValueError("video has no audio stream")

    with tempfile.TemporaryDirectory(prefix="drivelhub-") as tmp:
        output = Path(tmp) / "audio_only.mp3"
        run_checked([
            "ffmpeg", "-y", "-i", str(video_path), "-vn", "-map", "0:a:0",
            "-ac", "1", "-ar", "16000", "-codec:a", "libmp3lame",
            "-b:a", "64k", str(output),
        ])
        yield output, "audio", "audio/mpeg"


def data_url(path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def generate(
    client: OpenAI,
    model: str,
    video_path: Path,
    mode: str,
    enable_thinking: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_retries: int,
) -> str:
    with prepared_media(video_path, mode) as (media_path, media_kind, mime_type):
        media = {
            "type": f"{media_kind}_url",
            f"{media_kind}_url": {"url": data_url(media_path, mime_type)},
        }
        content = [media, {"type": "text", "text": PROMPTS[mode]}]

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body={
                        "top_k": top_k,
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    },
                )
                answer = response.choices[0].message.content
                if not answer or not answer.strip():
                    raise RuntimeError("empty model response")
                return answer.strip()
            except Exception as exc:  # API clients expose backend-specific errors.
                last_error = exc
                print(f"retry {attempt}/{max_retries} for {video_path.name}: {exc}")
                if attempt < max_retries:
                    time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"failed after {max_retries} attempts: {last_error}")


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                completed.add(json.loads(line)["file"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name exposed by vLLM.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=Path("metadata.csv"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(PROMPTS), default="full")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()
    completed = load_completed(args.output_jsonl)
    client = OpenAI(base_url=args.base_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))

    with args.metadata_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    processed = 0
    for row in rows:
        filename = row.get("file", "")
        if not filename.endswith(".mp4") or filename in completed:
            continue
        video_path = args.data_dir / filename
        if not video_path.is_file():
            print(f"missing: {video_path}")
            continue

        record = {
            **row,
            "video_path": str(video_path),
            "input_mode": args.mode,
            "remove_audio": args.mode == "without-audio",
            "remove_vision": args.mode == "without-vision",
        }
        print(f"processing: {filename}")
        try:
            record["latent_meaning"] = generate(
                client, args.model, video_path, args.mode, args.enable_thinking,
                args.max_tokens, args.temperature, args.top_p, args.top_k,
                args.max_retries,
            )
        except Exception as exc:
            record["latent_meaning"] = ""
            record["error"] = str(exc)
            print(f"error: {filename}: {exc}")
        append_jsonl(args.output_jsonl, record)
        processed += 1
        if args.limit is not None and processed >= args.limit:
            break

    print(f"done: wrote {processed} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
