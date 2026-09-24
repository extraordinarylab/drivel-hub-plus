#!/usr/bin/env python3
"""Sync the local DrivelHub+ dataset to the HuggingFace hub.

Uploads the 1,000 clips, the metadata and the retrieval relevance judgments.
`upload_folder` computes each file's hash locally and skips what the hub
already stores, so a re-run only sends what actually changed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[1]
DATA = Path("/scratch/u6sn/yangw.u6sn/huggingface_data/extraordinarylab/drivel-hub-plus/data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="extraordinarylab/drivel-hub-plus")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    print("user:", api.whoami().get("name"))

    with (REPO / "metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    listed = {r["file"] for r in rows}
    on_disk = {p.name for p in args.data_dir.glob("*.mp4")}
    missing = listed - on_disk
    extra = on_disk - listed
    print(f"metadata rows: {len(rows)} | clips on disk: {len(on_disk)}")
    if missing:
        raise SystemExit(f"{len(missing)} clips named in metadata.csv are missing: {sorted(missing)[:5]}")
    if extra:
        raise SystemExit(f"{len(extra)} clips on disk are not in metadata.csv: {sorted(extra)[:5]}")

    remote = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
    print(f"hub currently holds {sum(1 for f in remote if f.endswith('.mp4'))} clips")
    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    print("\nuploading clips ...")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.data_dir),
        path_in_repo="data",
        allow_patterns=["*.mp4"],
        commit_message="Re-encode HEVC clips to H.264 and replace the silent ones",
    )
    print("\nuploading metadata and qrels ...")
    for name in ("metadata.csv", "qrels.json"):
        path = REPO / name
        if not path.is_file():
            print(f"  skip {name}: not found")
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Update {name}",
        )
        print(f"  uploaded {name}")

    after = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
    print(f"\nhub now holds {sum(1 for f in after if f.endswith('.mp4'))} clips, "
          f"{len(after)} files total")
    for name in sorted(after - remote):
        print("  new:", name)


if __name__ == "__main__":
    main()
