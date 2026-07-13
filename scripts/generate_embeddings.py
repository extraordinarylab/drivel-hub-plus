from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


QUERY_TEMPLATE = "{}\nSummarize the above text in one word:"
VIDEO_PROMPT = "\nSummarize the above video in one word:"
VLLM_VIDEO_PROMPT = "Summarize the attached video in one word:"
VISION_TOKEN = "<|vision_start|><|video_pad|><|vision_end|>"
QWEN_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)


class QwenHiddenStateBackend:
    """The final-token Qwen2.5-Omni."""

    def __init__(self, model_path: str, device: str, fps: float, max_pixels: int):
        import torch
        from qwen_omni_utils import process_mm_info
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        self.torch = torch
        self.process_mm_info = process_mm_info
        self.device = torch.device(device)
        self.fps = fps
        self.max_pixels = max_pixels
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def _encode(self, messages: list[list[dict[str, Any]]]) -> list[float]:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        audio, images, videos = self.process_mm_info(
            messages, use_audio_in_video=False,
        )
        inputs = self.processor(
            text=text,
            audio=audio,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device) if isinstance(value, self.torch.Tensor) else value
            for key, value in inputs.items()
        }
        with self.torch.no_grad():
            outputs = self.model(
                **inputs, output_hidden_states=True, return_dict=True,
            )
        embedding = (
            outputs.hidden_states[-1][:, -1, :]
            .squeeze(0)
            .to(self.torch.float16)
            .cpu()
            .tolist()
        )
        del inputs, outputs
        self.torch.cuda.empty_cache()
        return embedding

    def embed_text(self, annotation: str) -> list[float]:
        messages = [[
            {
                "role": "system",
                "content": [{"type": "text", "text": QWEN_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": QUERY_TEMPLATE.format(annotation),
                }],
            },
        ]]
        return self._encode(messages)

    def embed_video(self, video_path: Path) -> list[float]:
        messages = [[
            {
                "role": "system",
                "content": [{"type": "text", "text": QWEN_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path.resolve()),
                        "max_pixels": self.max_pixels,
                        "fps": self.fps,
                    },
                    {"type": "text", "text": VIDEO_PROMPT},
                ],
            },
        ]]
        return self._encode(messages)


class NativeEmbedBackend:

    def __init__(self, model_path: str, device: str, max_length: int):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            default_task="retrieval",
        ).to(torch.device(device))
        self.model.eval()

    def _as_list(self, outputs: Any) -> list[float]:
        embedding = outputs.squeeze(0).to(self.torch.float16).cpu().tolist()
        del outputs
        self.torch.cuda.empty_cache()
        return embedding

    def embed_text(self, annotation: str) -> list[float]:
        inputs = self.processor(
            text=annotation,
            return_tensors="pt",
            truncation=False,
            max_length=self.max_length,
        ).to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model.embed(**inputs)
        return self._as_list(outputs)

    def embed_video(self, video_path: Path) -> list[float]:
        inputs = self.processor(
            videos=str(video_path.resolve()),
            text=VISION_TOKEN,
            return_tensors="pt",
            truncation=False,
            max_length=self.max_length,
        ).to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model.embed(**inputs)
        return self._as_list(outputs)


class VllmPoolingBackend:
    """The OpenAI-compatible vLLM pooling"""

    def __init__(self, model: str, base_url: str, timeout: float):
        import requests

        self.model = model
        self.endpoint = f"{base_url.rstrip('/')}/embeddings"
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, payload: dict[str, Any]) -> list[float]:
        response = self.session.post(
            self.endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        try:
            embedding = body["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"embedding endpoint returned an unexpected response: {body}"
            ) from exc
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("embedding endpoint returned an empty embedding")
        return embedding

    def embed_text(self, annotation: str) -> list[float]:
        return self._request({
            "model": self.model,
            "input": QUERY_TEMPLATE.format(annotation),
        })

    def embed_video(self, video_path: Path) -> list[float]:
        return self._request({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": VLLM_VIDEO_PROMPT},
                    {
                        "type": "video_url",
                        "video_url": {"url": video_path.resolve().as_uri()},
                    },
                ],
            }],
        })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("qwen-hidden-state", "native-embed", "vllm"),
        required=True,
    )
    parser.add_argument("--model", required=True, help="Checkpoint path or served model name.")
    parser.add_argument("--metadata-csv", type=Path, default=Path("metadata.csv"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=360 * 420)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def rank_output_path(path: Path, rank: int, world_size: int) -> Path:
    if world_size == 1:
        return path
    return path.with_name(f"{path.stem}.rank{rank}{path.suffix}")


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                completed.add(str(json.loads(line)["file"]))
            except (json.JSONDecodeError, KeyError) as exc:
                print(f"warning: ignoring malformed line {path}:{line_number}: {exc}")
    return completed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_rank_outputs(output: Path, world_size: int) -> None:
    seen: set[str] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for rank in range(world_size):
            shard = rank_output_path(output, rank, world_size)
            if not shard.exists():
                raise FileNotFoundError(f"missing rank output: {shard}")
            with shard.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    file_id = str(json.loads(line)["file"])
                    if file_id in seen:
                        raise ValueError(f"duplicate file ID while merging: {file_id}")
                    seen.add(file_id)
                    destination.write(line if line.endswith("\n") else line + "\n")
    print(f"merged {world_size} rank files ({len(seen)} rows) into {output}")


def load_metadata(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows if limit is None else rows[:limit]


def make_backend(args: argparse.Namespace, device: str) -> Any:
    if args.backend == "qwen-hidden-state":
        return QwenHiddenStateBackend(
            args.model, device, args.fps, args.max_pixels,
        )
    if args.backend == "native-embed":
        return NativeEmbedBackend(args.model, device, args.max_length)
    return VllmPoolingBackend(args.model, args.base_url, args.timeout)


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if args.backend == "vllm" and distributed:
        raise ValueError("the vllm backend should be run as a single client process")

    torch = None
    if args.backend != "vllm":
        import torch as torch_module

        torch = torch_module
        if not torch.cuda.is_available():
            raise RuntimeError("local embedding backends require CUDA, as in the original scripts")
        torch.cuda.set_device(local_rank)
        if distributed:
            torch.distributed.init_process_group(backend="nccl")

    output_path = rank_output_path(args.output_jsonl, rank, world_size)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch(exist_ok=True)
    completed = load_completed(output_path)

    rows = load_metadata(args.metadata_csv, args.limit)
    rows = rows[rank::world_size]
    backend = make_backend(args, f"cuda:{local_rank}")
    print(f"[rank {rank}] processing {len(rows)} metadata rows with {args.backend}")

    written = 0
    failed = 0
    for row in rows:
        filename = str(row.get("file", ""))
        if not filename.endswith(".mp4") or filename in completed:
            continue
        annotation = str(row.get("annotation", "") or "")
        video_path = args.data_dir / filename
        if not video_path.is_file():
            print(f"[rank {rank}] missing: {video_path}")
            failed += 1
            if args.fail_fast:
                raise FileNotFoundError(video_path)
            continue
        try:
            print(f"[rank {rank}] embedding: {filename}")
            result = {
                "file": filename,
                "annotation": annotation,
                "query_embedding": backend.embed_text(annotation),
                "corpus_embedding": backend.embed_video(video_path),
            }
            append_jsonl(output_path, result)
            written += 1
        except Exception as exc:
            failed += 1
            print(f"[rank {rank}] error for {filename}: {exc}")
            if args.fail_fast:
                raise

    print(f"[rank {rank}] wrote {written} rows to {output_path}; failures: {failed}")

    if distributed:
        assert torch is not None
        torch.distributed.barrier()
        if rank == 0:
            merge_rank_outputs(args.output_jsonl, world_size)
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
