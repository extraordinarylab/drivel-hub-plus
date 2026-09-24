#!/usr/bin/env python3
"""Convert each clip to text for the cascaded (OCR + ASR -> text LLM) baseline.

Reviewers asked for a text-conversion diagnostic that separates failures of
perception, modality extraction, and pragmatic reasoning. This script only
produces the text; generation and judging reuse the normal pipeline.

Stages:
  asr      Whisper large-v3 over the clip's soundtrack.
  ocr      EasyOCR over evenly sampled frames. The reader's language set comes
           from the clip's annotated ``caption`` language, so a Mandarin clip is
           not read with an English-only recogniser.
  ocr-vlm  The same frames transcribed verbatim by a vision LM served through
           vLLM. A dedicated OCR engine is the honest cascade; this is the
           upper bound on how much on-screen text a text-only pipeline could
           possibly receive, so a low score under both cannot be blamed on the
           recogniser.
  caption  A video captioner describes the clip, and the caption alone is the
           text handed to the LLM (the video-captioning-then-LLM baseline).

Output is append-only JSONL keyed by filename and can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import time
import wave
from pathlib import Path

import av
import numpy as np

from scripts.run_inference import extract_audio, has_audio


# EasyOCR keeps most recognisers single-script: a Chinese model may be paired
# with English, but not with Korean or Japanese. Map the annotated caption
# language onto one legal reader set.
LANGUAGE_SETS: dict[str, tuple[str, ...]] = {
    "english": ("en",),
    "mandarin": ("ch_tra", "en"),
    "cantonese": ("ch_tra", "en"),
    "taiwanese": ("ch_tra", "en"),
    "japanese": ("ja", "en"),
    "korean": ("ko", "en"),
    "thai": ("th", "en"),
    "french": ("fr", "en"),
    "spanish": ("es", "en"),
    "icelandic": ("is", "en"),
    "arabic": ("ar", "en"),
}
DEFAULT_LANGUAGES = ("en",)


def normalise_language(value: str) -> str:
    """First annotated language of a possibly multi-language field."""
    token = re.split(r"[+/,]", (value or "").strip().lower())[0].strip()
    return {"mardarin": "mandarin", "": "-"}.get(token, token)


def reader_languages(caption_language: str) -> tuple[str, ...]:
    return LANGUAGE_SETS.get(normalise_language(caption_language), DEFAULT_LANGUAGES)


def sample_frames(video_path: Path, count: int) -> list[np.ndarray]:
    """Evenly spaced RGB frames, tolerating a missing frame count in the header."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        total = int(stream.frames or 0)
        if total <= 0 and stream.duration is not None and stream.time_base and fps:
            total = int(round(float(stream.duration * stream.time_base) * fps))
        if total <= 0 and container.duration and fps:
            total = int(round(container.duration / av.time_base * fps))

        if total > 0:
            wanted = {int(i) for i in np.linspace(0, total - 1, min(count, total))}
            frames = [
                frame.to_ndarray(format="rgb24")
                for index, frame in enumerate(container.decode(video=0))
                if index in wanted
            ]
        else:
            # No usable count: decode everything and thin it out afterwards.
            decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            if not decoded:
                return []
            wanted = {int(i) for i in np.linspace(0, len(decoded) - 1, min(count, len(decoded)))}
            frames = [decoded[i] for i in sorted(wanted)]
    return frames


def dedupe_lines(lines: list[str]) -> list[str]:
    """Drop repeats: the same subtitle is burned into many sampled frames."""
    seen: set[str] = set()
    kept = []
    for line in lines:
        key = re.sub(r"\s+", "", line).lower()
        if key and key not in seen:
            seen.add(key)
            kept.append(line)
    return kept


def read_wav(path: Path) -> np.ndarray:
    """16 kHz mono PCM as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    done.add(json.loads(line)["file"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_asr(rows: list[dict], args: argparse.Namespace, output: Path) -> None:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    # transformers' ASR pipeline imports torchcodec inside preprocess() whenever
    # it thinks torchcodec is available. Importing it pulls in
    # libtorchcodec_image.so, which does not load under this env's CUDA stack,
    # and the whole call dies even though we hand over decoded samples and need
    # no decoder at all. Tell the pipeline it is unavailable.
    import transformers.pipelines.automatic_speech_recognition as asr_pipeline

    asr_pipeline.is_torchcodec_available = lambda: False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.asr_model, dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.asr_model)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=args.batch_size,
        dtype=dtype,
        device=device,
    )

    for index, row in enumerate(rows, 1):
        video = args.data_dir / row["file"]
        started = time.perf_counter()
        record = {"file": row["file"], "text": "", "language": None}
        try:
            if not has_audio(video):
                record["note"] = "no audio stream"
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", dir=args.tmp_dir) as handle:
                    extract_audio(video, Path(handle.name))
                    # Hand the pipeline the samples, not the path: given a path
                    # it shells out to ffmpeg, which is not on PATH here, and
                    # extract_audio has already produced 16 kHz mono PCM.
                    result = asr(
                        {"raw": read_wav(Path(handle.name)), "sampling_rate": 16000},
                        return_timestamps=False,
                        generate_kwargs={"task": "transcribe"},
                    )
                record["text"] = (result.get("text") or "").strip()
        except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the run
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = round(time.perf_counter() - started, 2)
        append_jsonl(output, record)
        print(f"[asr] {index}/{len(rows)} {row['file']} {record['seconds']}s "
              f"{len(record['text'])} chars", flush=True)


def run_ocr(rows: list[dict], args: argparse.Namespace, output: Path) -> None:
    import easyocr

    readers: dict[tuple[str, ...], easyocr.Reader] = {}

    def reader_for(languages: tuple[str, ...]) -> easyocr.Reader:
        if languages not in readers:
            readers[languages] = easyocr.Reader(
                list(languages), gpu=True, model_storage_directory=str(args.ocr_cache),
                download_enabled=True, verbose=False,
            )
        return readers[languages]

    # Group by reader so each recogniser is built once.
    rows = sorted(rows, key=lambda r: reader_languages(r.get("caption", "")))
    for index, row in enumerate(rows, 1):
        video = args.data_dir / row["file"]
        languages = reader_languages(row.get("caption", ""))
        started = time.perf_counter()
        record = {"file": row["file"], "text": "", "languages": list(languages)}
        try:
            frames = sample_frames(video, args.frames)
            lines: list[str] = []
            if frames:
                reader = reader_for(languages)
                for frame in frames:
                    for _, text, confidence in reader.readtext(frame):
                        if confidence >= args.min_confidence and text.strip():
                            lines.append(text.strip())
            record["text"] = "\n".join(dedupe_lines(lines))
            record["frames"] = len(frames)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = round(time.perf_counter() - started, 2)
        append_jsonl(output, record)
        print(f"[ocr] {index}/{len(rows)} {row['file']} {'+'.join(languages)} "
              f"{record['seconds']}s {len(record['text'])} chars", flush=True)


OCR_VLM_PROMPT = """Transcribe every piece of text that appears in these frames, exactly as written.

Rules:
- Output the text only. Do not describe the scene, the people, the actions, or the setting.
- Do not translate, explain, correct, or interpret anything.
- Keep the original language and spelling, including deliberate misspellings.
- One line per distinct piece of text. Skip a line you have already written.
- Ignore platform watermarks, usernames, follower counts, and interface elements.
- If the frames contain no text at all, output exactly: (none)"""

CAPTION_PROMPT = """Describe this video in detail: the setting, the people, what they do, in what order, and any text or speech that appears. Write a single factual paragraph. Do not interpret the meaning, explain a joke, or speculate about intent."""


def encode_frame(frame: np.ndarray) -> str:
    """PNG data URL for one RGB frame."""
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def call_endpoint(client, model: str, content: list[dict], args) -> str:
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                raise RuntimeError("empty response")
            return answer
        except Exception as exc:  # backend-specific errors
            last_error = exc
            print(f"  retry {attempt}/{args.max_retries}: {exc}", flush=True)
            if attempt < args.max_retries:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"failed after {args.max_retries} attempts: {last_error}")


def run_ocr_vlm(rows: list[dict], args: argparse.Namespace, output: Path) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    model = args.served_model or client.models.list().data[0].id

    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        record = {"file": row["file"], "text": ""}
        try:
            frames = sample_frames(args.data_dir / row["file"], args.frames)
            content = [{"type": "image_url", "image_url": {"url": encode_frame(f)}}
                       for f in frames]
            content.append({"type": "text", "text": OCR_VLM_PROMPT})
            answer = call_endpoint(client, model, content, args)
            record["text"] = "" if answer.strip() == "(none)" else answer
            record["frames"] = len(frames)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = round(time.perf_counter() - started, 2)
        append_jsonl(output, record)
        print(f"[ocr-vlm] {index}/{len(rows)} {row['file']} {record['seconds']}s "
              f"{len(record['text'])} chars", flush=True)


def run_caption(rows: list[dict], args: argparse.Namespace, output: Path) -> None:
    from openai import OpenAI

    from scripts.run_inference import data_url

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    model = args.served_model or client.models.list().data[0].id

    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        record = {"file": row["file"], "text": ""}
        try:
            video = args.data_dir / row["file"]
            content = [
                {"type": "video_url",
                 "video_url": {"url": data_url(video, "video/mp4")}},
                {"type": "text", "text": CAPTION_PROMPT},
            ]
            record["text"] = call_endpoint(client, model, content, args)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = round(time.perf_counter() - started, 2)
        append_jsonl(output, record)
        print(f"[caption] {index}/{len(rows)} {row['file']} {record['seconds']}s "
              f"{len(record['text'])} chars", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("asr", "ocr", "ocr-vlm", "caption"),
                        required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=Path("metadata.csv"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--asr-model", default="openai/whisper-large-v3")
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="vLLM endpoint for the ocr-vlm and caption stages.")
    parser.add_argument("--served-model", help="model name the endpoint exposes")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames", type=int, default=12,
                        help="frames sampled per clip for OCR")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--ocr-cache", type=Path,
                        default=Path("/scratch/u6sn/yangw.u6sn/tmp/drivel-hub-plus/easyocr"))
    parser.add_argument("--tmp-dir", type=Path,
                        default=Path("/scratch/u6sn/yangw.u6sn/tmp/drivel-hub-plus"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.ocr_cache.mkdir(parents=True, exist_ok=True)

    with args.metadata_csv.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if (args.data_dir / row["file"]).exists()]

    if args.shard_count > 1:
        rows = rows[args.shard_index::args.shard_count]
    done = load_completed(args.output_jsonl)
    rows = [row for row in rows if row["file"] not in done]
    if args.limit:
        rows = rows[:args.limit]

    print(f"[{args.stage}] {len(rows)} clips to do "
          f"({len(done)} already in {args.output_jsonl})", flush=True)
    if not rows:
        return
    {"asr": run_asr, "ocr": run_ocr, "ocr-vlm": run_ocr_vlm,
     "caption": run_caption}[args.stage](rows, args, args.output_jsonl)


if __name__ == "__main__":
    main()
