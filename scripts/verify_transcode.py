#!/usr/bin/env python3
"""Verify transcoded clips against their sources, then optionally swap them in.

Checks each pair by fully decoding both files: codec, exact frame count,
duration, resolution, and a SHA-256 over the audio packets, which must be
byte-identical because the audio is stream-copied rather than re-encoded.

Nothing is swapped unless every file passes. With --apply the originals are
moved to --backup-dir first, so the swap is reversible.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import warnings
from pathlib import Path

import av

warnings.filterwarnings("ignore")
av.logging.set_level(av.logging.PANIC)


def probe(path: Path) -> dict:
    info: dict = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        info["vcodec"] = stream.codec_context.name
        info["res"] = (stream.codec_context.width, stream.codec_context.height)
        info["dur"] = (container.duration or 0) / av.time_base
        info["frames"] = sum(1 for _ in container.decode(video=0))
    with av.open(str(path)) as container:
        if container.streams.audio:
            audio = container.streams.audio[0]
            digest = hashlib.sha256()
            packets = 0
            for packet in container.demux(audio):
                if packet.dts is None:
                    continue
                digest.update(bytes(packet))
                packets += 1
            info["acodec"] = audio.codec_context.name
            info["apkts"] = packets
            info["ahash"] = digest.hexdigest()
        else:
            info["acodec"] = None
            info["ahash"] = None
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, required=True, help="Live data dir.")
    parser.add_argument("--new-dir", type=Path, required=True, help="Transcoded files.")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--expect-codec", default="h264")
    parser.add_argument("--apply", action="store_true", help="Swap in after all checks pass.")
    args = parser.parse_args()

    new_files = sorted(args.new_dir.glob("*.mp4"))
    print(f"verifying {len(new_files)} file(s)\n")
    print(f"{'file':42} {'codec':6} {'frames':>14} {'dur':>8} {'res':>5} {'audio':>6}")
    print("-" * 92)

    problems: list[tuple[str, str]] = []
    for new in new_files:
        old = args.src_dir / new.name
        if not old.is_file():
            problems.append((new.name, "no matching source")); continue
        try:
            a, b = probe(old), probe(new)
        except Exception as exc:
            problems.append((new.name, f"decode failed: {type(exc).__name__}: {exc}"))
            print(f"{new.name:42} DECODE FAILED")
            continue

        issues = []
        if b["vcodec"] != args.expect_codec:
            issues.append(f"codec {b['vcodec']}")
        if a["frames"] != b["frames"]:
            issues.append(f"frames {a['frames']}->{b['frames']}")
        if abs(a["dur"] - b["dur"]) > 0.15:
            issues.append(f"duration {a['dur']:.2f}->{b['dur']:.2f}")
        if a["res"] != b["res"]:
            issues.append(f"res {a['res']}->{b['res']}")
        if a["ahash"] != b["ahash"]:
            issues.append(f"audio differs ({a['acodec']}/{a.get('apkts')} vs {b['acodec']}/{b.get('apkts')})")
        problems += [(new.name, i) for i in issues]

        print(f"{new.name:42} {b['vcodec']:6} "
              f"{('OK' if a['frames'] == b['frames'] else f'{a['frames']}->{b['frames']}'):>14} "
              f"{('OK' if abs(a['dur'] - b['dur']) <= 0.15 else 'BAD'):>8} "
              f"{('OK' if a['res'] == b['res'] else 'BAD'):>5} "
              f"{('OK' if a['ahash'] == b['ahash'] else 'DIFF'):>6}")

    print()
    if problems:
        print(f"FAILED: {len(problems)} problem(s); nothing was swapped.")
        for name, msg in problems:
            print(f"  {name}: {msg}")
        sys.exit(1)

    print(f"All {len(new_files)} file(s) passed: {args.expect_codec}, every frame decodes, "
          f"frame count / duration / resolution match, audio byte-identical.")

    if not args.apply:
        print("\n(dry run: pass --apply to back up the originals and swap these in)")
        return

    if not args.backup_dir:
        sys.exit("--apply requires --backup-dir")
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    for new in new_files:
        old = args.src_dir / new.name
        shutil.copy2(old, args.backup_dir / new.name)
    print(f"\nbacked up {len(new_files)} original(s) to {args.backup_dir}")

    for new in new_files:
        shutil.copy2(new, args.src_dir / new.name)
    print(f"swapped {len(new_files)} file(s) into {args.src_dir}")


if __name__ == "__main__":
    main()
