#!/usr/bin/env python3
"""Transcode HEVC clips to H.264, equivalent to `ffmpeg -c:v libx264 -c:a copy`.

This cluster image ships no ffmpeg binary (see INSTALL.md), but the pinned PyAV
wheel bundles libx264, so the transcode runs through PyAV instead. Audio is
remuxed packet-for-packet rather than re-encoded, so the audio track stays
bit-identical to the source.

Output goes to a separate directory: the live data directory is read by running
inference jobs, and overwriting a clip underneath them yields a half-written
file and an untrustworthy prediction.
"""

from __future__ import annotations

import argparse
import warnings
from fractions import Fraction
from pathlib import Path

import av

warnings.filterwarnings("ignore")
av.logging.set_level(av.logging.PANIC)


def source_codec(path: Path) -> str:
    with av.open(str(path)) as container:
        if not container.streams.video:
            return "<none>"
        return container.streams.video[0].codec_context.name


def transcode(src: Path, dst: Path, crf: int, preset: str, threads: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(src)) as inp, av.open(str(dst), mode="w") as out:
        in_v = inp.streams.video[0]
        in_a = inp.streams.audio[0] if inp.streams.audio else None

        rate = in_v.average_rate or Fraction(30, 1)
        out_v = out.add_stream("libx264", rate=rate)
        out_v.width = in_v.codec_context.width
        out_v.height = in_v.codec_context.height
        out_v.pix_fmt = "yuv420p"
        out_v.time_base = in_v.time_base
        out_v.options = {"crf": str(crf), "preset": preset, "threads": str(threads)}

        out_a = out.add_stream_from_template(in_a) if in_a is not None else None

        streams = [in_v] + ([in_a] if in_a is not None else [])
        for packet in inp.demux(streams):
            if packet.stream is in_v:
                # The final packet of a stream has dts None and is the flush
                # packet: decoding it drains the frames the decoder still holds
                # in its reorder buffer. Skipping it silently truncates the clip
                # by a few frames.
                for frame in packet.decode():
                    for encoded in out_v.encode(frame):
                        out.mux(encoded)
            elif out_a is not None and packet.dts is not None:
                packet.stream = out_a
                out.mux(packet)

        for encoded in out_v.encode():
            out.mux(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--codec", default="hevc", help="Source codec to convert.")
    parser.add_argument("--crf", type=int, default=18, help="Lower is higher quality.")
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    targets = [p for p in sorted(args.data_dir.glob("*.mp4"))
               if source_codec(p) == args.codec]
    print(f"{len(targets)} file(s) with codec {args.codec}")

    for i, src in enumerate(targets, 1):
        dst = args.out_dir / src.name
        transcode(src, dst, args.crf, args.preset, args.threads)
        print(f"  [{i}/{len(targets)}] {src.name}  "
              f"{src.stat().st_size/1e6:.1f}MB -> {dst.stat().st_size/1e6:.1f}MB",
              flush=True)
    print(f"wrote {len(targets)} file(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
