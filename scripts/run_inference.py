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
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import av
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


def final_answer(text: str) -> str:
    """Remove reasoning leaked by models that emit only a closing think tag."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def has_audio(path: Path) -> bool:
    with av.open(str(path)) as container:
        return bool(container.streams.audio)


def extract_audio(video_path: Path, output_path: Path) -> None:
    with av.open(str(video_path)) as container:
        if not container.streams.audio:
            raise ValueError("video has no audio stream")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        with wave.open(str(output_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    output.writeframes(resampled.to_ndarray().tobytes())
            for resampled in resampler.resample(None):
                output.writeframes(resampled.to_ndarray().tobytes())


def strip_audio(video_path: Path, output_path: Path) -> None:
    """Remux the first video stream into a new MP4 without audio streams."""
    with av.open(str(video_path)) as source:
        if not source.streams.video:
            raise ValueError("input has no video stream")
        source_stream = source.streams.video[0]
        with av.open(str(output_path), mode="w") as destination:
            destination_stream = destination.add_stream_from_template(source_stream)
            for packet in source.demux(source_stream):
                if packet.dts is None:
                    continue
                packet.stream = destination_stream
                destination.mux(packet)


@contextmanager
def prepared_media(
    video_path: Path, mode: str, use_audio_in_video: bool = False
) -> Iterator[list[tuple[Path, str, str]]]:
    if mode == "full":
        if not use_audio_in_video:
            yield [(video_path, "video", "video/mp4")]
            return

        if not has_audio(video_path):
            yield [(video_path, "video", "video/mp4")]
            return
        with tempfile.TemporaryDirectory(prefix="drivelhub-") as tmp:
            audio_path = Path(tmp) / "audio.wav"
            extract_audio(video_path, audio_path)
            # vLLM serve does not interleave embedded video audio for Qwen3-Omni.
            # The model's official serving example sends video and audio separately.
            yield [
                (video_path, "video", "video/mp4"),
                (audio_path, "audio", "audio/wav"),
            ]
        return

    if mode == "without-audio":
        with tempfile.TemporaryDirectory(prefix="drivelhub-") as tmp:
            output = Path(tmp) / "vision_only.mp4"
            strip_audio(video_path, output)
            yield [(output, "video", "video/mp4")]
        return

    if not has_audio(video_path):
        raise ValueError("video has no audio stream")

    with tempfile.TemporaryDirectory(prefix="drivelhub-") as tmp:
        output = Path(tmp) / "audio_only.wav"
        extract_audio(video_path, output)
        yield [(output, "audio", "audio/wav")]


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
    top_k: int | None,
    use_audio_in_video: bool,
    max_retries: int,
) -> str:
    with prepared_media(video_path, mode, use_audio_in_video) as prepared:
        media = [
            {
                "type": f"{media_kind}_url",
                f"{media_kind}_url": {"url": data_url(media_path, mime_type)},
            }
            for media_path, media_kind, mime_type in prepared
        ]
        content = [*media, {"type": "text", "text": PROMPTS[mode]}]

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                extra_body: dict = {
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking,
                        "thinking": enable_thinking,
                    },
                }
                if top_k is not None:
                    extra_body["top_k"] = top_k
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body=extra_body,
                )
                answer = response.choices[0].message.content
                if not answer or not answer.strip():
                    raise RuntimeError("empty model response")
                answer = final_answer(answer)
                if not answer:
                    raise RuntimeError("empty model response after removing reasoning")
                return answer
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
    parser.add_argument(
        "--base-url",
        action="append",
        dest="base_urls",
        help=(
            "OpenAI-compatible endpoint. Repeat to distribute concurrent "
            "requests across multiple local vLLM replicas (default: "
            "http://localhost:8000/v1)."
        ),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=Path("metadata.csv"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(PROMPTS), default="full")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument(
        "--top-k",
        type=int,
        help="Optional override; by default use the checkpoint generation config.",
    )
    parser.add_argument(
        "--use-audio-in-video",
        action="store_true",
        help=(
            "Extract and attach the video's audio track as a separate audio "
            "input. Required for Qwen3-Omni full-video inference via vLLM serve."
        ),
    )
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent requests sent to vLLM (default: 1).",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def infer_row(
    row: dict[str, str], args: argparse.Namespace, client: OpenAI
) -> dict:
    filename = row["file"]
    video_path = args.data_dir / filename
    record = {
        **row,
        "video_path": str(video_path),
        "input_mode": args.mode,
        "remove_audio": args.mode == "without-audio",
        "remove_vision": args.mode == "without-vision",
    }
    try:
        record["implicit_meaning"] = generate(
            client, args.model, video_path, args.mode, args.enable_thinking,
            args.max_tokens, args.temperature, args.top_p, args.top_k,
            args.use_audio_in_video,
            args.max_retries,
        )
    except Exception as exc:
        record["implicit_meaning"] = ""
        record["error"] = str(exc)
        print(f"error: {filename}: {exc}", flush=True)
    return record


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.use_audio_in_video and args.mode != "full":
        raise ValueError("--use-audio-in-video is only valid with --mode full")
    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()
    completed = load_completed(args.output_jsonl)
    base_urls = args.base_urls or ["http://localhost:8000/v1"]
    clients = [
        OpenAI(base_url=url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
        for url in base_urls
    ]

    with args.metadata_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    pending: list[dict[str, str]] = []
    for row in rows:
        filename = row.get("file", "")
        if not filename.endswith(".mp4") or filename in completed:
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
            print(f"processing: {row['file']}", flush=True)
            append_jsonl(args.output_jsonl, infer_row(row, args, clients[0]))
            processed += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for index, row in enumerate(pending):
                print(f"processing: {row['file']}", flush=True)
                future = executor.submit(
                    infer_row, row, args, clients[index % len(clients)]
                )
                futures[future] = row["file"]
            for future in as_completed(futures):
                append_jsonl(args.output_jsonl, future.result())
                processed += 1
                print(
                    f"completed: {futures[future]} ({processed}/{len(pending)})",
                    flush=True,
                )

    print(f"done: wrote {processed} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
