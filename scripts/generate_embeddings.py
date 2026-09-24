from __future__ import annotations

import argparse
import csv
import sys
import tempfile
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


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_inference import extract_audio, has_audio


class QwenHiddenStateBackend:
    """The final-token Qwen2.5-Omni."""

    def __init__(self, model_path: str, device: str, fps: float, max_pixels: int,
                 use_audio_in_video: bool = False):
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
        # Off by default keeps the original behaviour; on, the clip's audio is
        # interleaved, which matters because most of this benchmark's meaning is
        # carried by sound rather than by the frames.
        self.use_audio_in_video = use_audio_in_video
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def _encode(self, messages: list[list[dict[str, Any]]],
                use_audio_in_video: bool | None = None) -> list[float]:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        audio, images, videos = self.process_mm_info(
            messages,
            use_audio_in_video=(
                self.use_audio_in_video if use_audio_in_video is None
                else use_audio_in_video
            ),
        )
        inputs = self.processor(
            text=text,
            audio=audio,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
        )
        model_dtype = next(self.model.parameters()).dtype
        moved = {}
        for key, value in inputs.items():
            if not isinstance(value, self.torch.Tensor):
                moved[key] = value
                continue
            value = value.to(self.device)
            if value.is_floating_point() and value.dtype != model_dtype:
                value = value.to(model_dtype)
            moved[key] = value
        inputs = moved
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

    def _sample_frames(self, video_path: Path, max_frames: int = 768) -> list:
        """Decode at most `max_frames` evenly spaced frames, streaming.

        Never holds more than the sampled frames, unlike the reader
        qwen_omni_utils falls back to.
        """
        import av
        from PIL import Image

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            total = stream.frames or 0
            rate = float(stream.average_rate or 30.0)
            duration = (container.duration or 0) / av.time_base
            wanted = max(2, min(max_frames, int(duration * self.fps) or 2))
            # Matches the library's fps-then-cap behaviour: one frame per
            # second by default, truncated at max_frames for very long clips.
            step = max(1, total // wanted) if total else max(1, int(rate / self.fps))
            frames = []
            for index, frame in enumerate(container.decode(video=0)):
                if index % step:
                    continue
                frames.append(frame.to_image().convert("RGB"))
                if len(frames) >= wanted:
                    break
        if not frames:
            raise ValueError(f"no frames decoded from {video_path.name}")
        return frames

    def embed_video(self, video_path: Path) -> list[float]:
        video_part = {
            "type": "video",
            "video": self._sample_frames(video_path),
            "max_pixels": self.max_pixels,
        }
        if not self.use_audio_in_video or not has_audio(video_path):
            return self._encode(self._messages([video_part]))
        with tempfile.TemporaryDirectory(prefix="drivelhub-emb-") as tmp:
            wav = Path(tmp) / "audio.wav"
            extract_audio(video_path, wav)
            parts = [video_part, {"type": "audio", "audio": str(wav)}]
            return self._encode(self._messages(parts), use_audio_in_video=False)

    def _messages(self, parts: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        return [[
            {
                "role": "system",
                "content": [{"type": "text", "text": QWEN_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [*parts, {"type": "text", "text": VIDEO_PROMPT}],
            },
        ]]


class Qwen3OmniHiddenStateBackend:
    """Final-token hidden state from the Qwen3-Omni thinker.

    Same idea as QwenHiddenStateBackend but against the Qwen3 classes, so the
    Table 2 leaders can be scored on retrieval with the same recipe used for
    their predecessors.
    """

    def __init__(self, model_path: str, device: str, fps: float, max_pixels: int,
                 use_audio_in_video: bool = False):
        import torch
        from qwen_omni_utils import process_mm_info
        from transformers import (
            Qwen3OmniMoeProcessor,
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )

        self.torch = torch
        self.process_mm_info = process_mm_info
        self.device = torch.device(device)
        self.fps = fps
        self.max_pixels = max_pixels
        self.use_audio_in_video = use_audio_in_video
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    _encode = QwenHiddenStateBackend._encode
    _messages = QwenHiddenStateBackend._messages
    _sample_frames = QwenHiddenStateBackend._sample_frames
    embed_text = QwenHiddenStateBackend.embed_text
    embed_video = QwenHiddenStateBackend.embed_video


class MiniCPMOHiddenStateBackend:
    """Final-token hidden state from MiniCPM-o, loaded through its remote code."""

    def __init__(self, model_path: str, device: str, max_length: int):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True,
            init_vision=True, init_audio=True, init_tts=False,
        ).to(self.device)
        self.model.eval()

    def _last_hidden(self, text: str) -> list[float]:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length,
        ).to(self.device)
        with self.torch.no_grad():
            llm = getattr(self.model, "llm", self.model)
            outputs = llm(**inputs, output_hidden_states=True, return_dict=True)
        embedding = (
            outputs.hidden_states[-1][:, -1, :]
            .squeeze(0).to(self.torch.float16).cpu().tolist()
        )
        del inputs, outputs
        self.torch.cuda.empty_cache()
        return embedding

    def embed_text(self, annotation: str) -> list[float]:
        return self._last_hidden(QUERY_TEMPLATE.format(annotation))

    def embed_video(self, video_path: Path) -> list[float]:
        # MiniCPM-o takes frames through its own chat interface rather than a
        # processor call, so this backend is text-side only until that path is
        # wired up; embed_video is what retrieval.py needs, so fail loudly
        # rather than silently returning a text embedding for a video.
        raise NotImplementedError(
            "MiniCPM-o video embedding is not wired up yet; see model card's "
            "chat() interface for the frame-sampling contract."
        )


class NativeEmbedBackend:

    def __init__(self, model_path: str, device: str, max_length: int):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                default_task="retrieval",
            ).to(torch.device(device))
        except TypeError as exc:
            if "default_task" not in str(exc):
                raise
            self.model = AutoModel.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                trust_remote_code=True,
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


class JinaOmniBackend:
    """jina-embeddings-v5-omni through its SentenceTransformers module.

    The AutoProcessor path takes `videos=` and contributes frames only — its
    __call__ has no audio argument at all. The module's own docstring gives the
    fusing form instead: "A video contributes its frames only; pass the
    soundtrack as its own part to fuse both, e.g. encode(("narration.wav",
    "clip.mp4"))". A tuple is fused into one embedding in a single forward pass.
    """

    def __init__(self, model_path: str, device: str, use_audio_in_video: bool = True):
        from sentence_transformers import SentenceTransformer

        kwargs = {"trust_remote_code": True, "device": device}
        try:
            self.model = SentenceTransformer(
                model_path, model_kwargs={"default_task": "retrieval"}, **kwargs,
            )
        except TypeError as exc:
            # The *-retrieval checkpoints have the adapter merged in and reject
            # default_task.
            if "default_task" not in str(exc):
                raise
            self.model = SentenceTransformer(model_path, **kwargs)
        self.use_audio_in_video = use_audio_in_video

    def embed_text(self, annotation: str) -> list[float]:
        vec = self.model.encode([QUERY_TEMPLATE.format(annotation)], prompt_name="query")
        return vec[0].astype("float32").tolist()

    def embed_video(self, video_path: Path) -> list[float]:
        path = str(video_path.resolve())
        if not self.use_audio_in_video or not has_audio(video_path):
            parts = (path,)
        else:
            with tempfile.TemporaryDirectory(prefix="drivelhub-jina-") as tmp:
                wav = Path(tmp) / "audio.wav"
                extract_audio(video_path, wav)
                vec = self.model.encode([(str(wav), path)], prompt_name="document")
                return vec[0].astype("float32").tolist()
        vec = self.model.encode([parts], prompt_name="document")
        return vec[0].astype("float32").tolist()


class SentenceTransformerOmniBackend:
    """Omni-Embed-Nemotron, which ships as a SentenceTransformers module.

    It exposes `encode_query` / `encode_document` rather than the `.embed()`
    that NativeEmbedBackend expects, and takes a document as a dict naming the
    media files. Video and audio are passed as the same MP4: the model decodes
    the two streams separately, which its technical report reports as better for
    retrieval than interleaving them.
    """

    def __init__(self, model_path: str, device: str, fps: float):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_path, trust_remote_code=True, device=device,
        )
        self.model[0].processing_kwargs.update({
            "video": {
                "min_pixels": 32 * 14 * 14,
                "max_pixels": 64 * 28 * 28,
                "do_sample_frames": True,
                "fps": fps,
            },
            "audio": {"max_length": 2048000},
        })

    def embed_text(self, annotation: str) -> list[float]:
        return self.model.encode_query([annotation])[0].astype("float32").tolist()

    def embed_video(self, video_path: Path) -> list[float]:
        path = str(video_path.resolve())
        if not has_audio(video_path):
            document = {"video": path}
            return self.model.encode_document([document])[0].astype("float32").tolist()
        with tempfile.TemporaryDirectory(prefix="drivelhub-nv-") as tmp:
            wav = Path(tmp) / "audio.wav"
            extract_audio(video_path, wav)
            document = {"video": path, "audio": str(wav)}
            return self.model.encode_document([document])[0].astype("float32").tolist()


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
        choices=("qwen-hidden-state", "qwen3-omni-hidden-state",
                 "minicpmo-hidden-state", "native-embed", "jina-omni",
                 "st-omni", "vllm"),
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
    parser.add_argument(
        "--use-audio-in-video",
        action="store_true",
        help=(
            "Interleave the clip's audio into the video input. Off by default, "
            "which is how the original retrieval numbers were produced."
        ),
    )
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
            args.model, device, args.fps, args.max_pixels, args.use_audio_in_video,
        )
    if args.backend == "qwen3-omni-hidden-state":
        return Qwen3OmniHiddenStateBackend(
            args.model, device, args.fps, args.max_pixels, args.use_audio_in_video,
        )
    if args.backend == "minicpmo-hidden-state":
        return MiniCPMOHiddenStateBackend(args.model, device, args.max_length)
    if args.backend == "native-embed":
        return NativeEmbedBackend(args.model, device, args.max_length)
    if args.backend == "jina-omni":
        return JinaOmniBackend(args.model, device, args.use_audio_in_video)
    if args.backend == "st-omni":
        return SentenceTransformerOmniBackend(args.model, device, args.fps)
    return VllmPoolingBackend(args.model, args.base_url, args.timeout)


def pin_media_decoders() -> None:
    """Force transformers onto decoders that work on this node.

    Two hazards, both silent:

    * `audio_utils.load_audio` prefers torchcodec whenever the package is
      merely importable *by name*, with no try/except. The aarch64 wheel here
      fails to dlopen ("libtorchcodec_image.so: undefined symbol:
      torch_from_blob"), so once torchcodec entered the lock every audio load
      raised OSError. Probe it and, if broken, tell transformers it is absent
      so load_audio falls back to librosa.
    * `BaseVideoProcessor.fetch_videos` hardcodes torchcodec and falls back to
      *torchvision*, whose reader calls `torchvision.io.read_video` -- it
      decodes the whole clip into one tensor before sampling. On a 235 s 1080p
      clip that is tens of GB, and FFmpeg then fails with "[swscaler] Failed
      initializing scaling graph (Resource temporarily unavailable)"; it also
      indexes that tensor with the sampler's indices, which is where "index
      467 is out of bounds for dimension 0 with size 467" came from. PyAV's
      reader streams and stops at the last wanted index, so pin it.
    """
    try:
        import torchcodec.decoders  # noqa: F401
    except Exception as exc:
        from transformers import audio_utils

        audio_utils.is_torchcodec_available = lambda: False
        print(f"warning: torchcodec unusable ({type(exc).__name__}: {exc}); "
              f"transformers will load audio with librosa")

    from transformers import video_processing_utils

    def fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
        if isinstance(video_url_or_urls, list):
            return list(zip(*[
                fetch_videos(self, url, sample_indices_fn=sample_indices_fn)
                for url in video_url_or_urls
            ]))
        return read_video_pyav_counted(video_url_or_urls, sample_indices_fn)

    video_processing_utils.BaseVideoProcessor.fetch_videos = fetch_videos


def read_video_pyav_counted(video_path: str, sample_indices_fn: Any, **kwargs: Any):
    """transformers' `read_video_pyav`, with the frame count repaired.

    It takes `total_num_frames` from `stream.frames`, which is 0 for an MP4
    that carries no frame count in its header. Qwen's `sample_frames` then
    computes `num_frames = 0` and dies in
    `torch.arange(0, total, total / num_frames)` with ZeroDivisionError. Fall
    back to duration x rate, then to counting packets (no decoding), before
    streaming out the wanted frames.
    """
    import av
    import numpy as np
    from transformers.video_utils import VideoMetadata

    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    total = int(stream.frames or 0)
    if total <= 0:
        seconds = None
        if stream.duration is not None and stream.time_base:
            seconds = float(stream.duration * stream.time_base)
        elif container.duration:
            seconds = container.duration / av.time_base
        if seconds and fps:
            total = int(round(seconds * fps))
    if total <= 0:
        total = sum(1 for _ in container.demux(video=0))
        container.close()
        container = av.open(video_path)
        stream = container.streams.video[0]
    if total <= 0:
        raise ValueError(f"no video frames in {video_path}")

    metadata = VideoMetadata(
        total_num_frames=int(total),
        fps=fps,
        duration=float(total / fps) if fps else 0.0,
        video_backend="pyav",
        height=stream.height,
        width=stream.width,
    )
    indices = sample_indices_fn(metadata=metadata, **kwargs)

    frames = []
    container.seek(0)
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i in indices:
            frames.append(frame)
    container.close()
    video = np.stack([frame.to_ndarray(format="rgb24") for frame in frames])
    metadata.frames_indices = indices
    return video, metadata


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if args.backend == "vllm" and distributed:
        raise ValueError("the vllm backend should be run as a single client process")

    import av as _av
    _av.logging.set_level(_av.logging.PANIC)
    # FFmpeg sizes its thread pools from av_cpu_count(), which on these nodes
    # reports 288. Four ranks each decoding a 1080p clip then failed with
    # "[swscaler] Failed initializing scaling graph (Resource temporarily
    # unavailable)". av_cpu_count() honours AV_CPU_COUNT, so cap it here unless
    # the caller has already set it.
    os.environ.setdefault("AV_CPU_COUNT", "4")

    pin_media_decoders()

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
            print(f"[rank {rank}] error for {filename}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            if failed <= 10:
                import traceback
                traceback.print_exc()
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
