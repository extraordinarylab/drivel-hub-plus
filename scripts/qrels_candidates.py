#!/usr/bin/env python3
"""Propose additional qrels pairs with EmbeddingGemma-300M, for human review.

Reproduces the paper's procedure: embed every narrative, take each sample's
top-k nearest neighbours, and keep pairs above a cosine threshold. Those pairs
are candidates only -- whether two narratives really express the same implicit
meaning is a judgement call, so nothing is written into qrels.json here.

sentence-transformers is not in the pinned environment, so the model's module
stack (mean pooling -> Dense 768x3072 -> Dense 3072x768 -> L2 normalize) is
applied directly on top of plain transformers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer


class EmbeddingGemma:
    def __init__(self, path: Path, device: str, dtype=torch.float32):
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModel.from_pretrained(str(path), dtype=dtype).to(device).eval()
        self.dense = []
        for sub in ("2_Dense", "3_Dense"):
            cfg = json.loads((path / sub / "config.json").read_text())
            weights = load_file(str(path / sub / "model.safetensors"))
            weight = next(iter(weights.values())).to(device=device, dtype=dtype)
            assert tuple(weight.shape) == (cfg["out_features"], cfg["in_features"]), \
                f"{sub}: unexpected weight shape {tuple(weight.shape)}"
            self.dense.append(weight)
        self.device = device

    @torch.no_grad()
    def encode(self, texts: list[str], prompt: str, batch_size: int = 16) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [prompt + t for t in texts[i:i + batch_size]]
            enc = self.tok(batch, padding=True, truncation=True, max_length=2048,
                           return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            for weight in self.dense:
                pooled = pooled @ weight.T
            out.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu())
        return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=Path("metadata.csv"))
    parser.add_argument("--qrels", type=Path, default=Path("qrels.json"))
    parser.add_argument("--column", default="annotation")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--focus", default="2026-09-21",
                        help="Substring marking the rows that still need pairs.")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.metadata_csv.open(encoding="utf-8-sig", newline="")))
    files = [r["file"] for r in rows]
    texts = [r[args.column].strip() for r in rows]
    qrels = json.loads(args.qrels.read_text(encoding="utf-8"))
    existing = {
        tuple(sorted((q, d))) for q, rel in qrels.items() for d in rel if d != q
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} rows={len(rows)} column={args.column!r}", flush=True)
    model = EmbeddingGemma(args.model, device)

    # The paper does not state which task prompt was used; score both against the
    # pairs already in qrels and report, so the choice is evidence-based.
    prompts = {
        "STS": "task: sentence similarity | query: ",
        "document": "title: none | text: ",
    }
    report: dict = {"existing_pairs": len(existing), "prompts": {}}
    best_name, best_recall, best_sim = None, -1.0, None

    for name, prompt in prompts.items():
        emb = model.encode(texts, prompt)
        sim = emb @ emb.T
        sim.fill_diagonal_(-1.0)
        hit = [float(sim[files.index(a), files.index(b)]) for a, b in existing]
        recall = sum(s >= args.threshold for s in hit) / max(len(hit), 1)
        report["prompts"][name] = {
            "existing_pair_cosine_min": min(hit) if hit else None,
            "existing_pair_cosine_median": sorted(hit)[len(hit) // 2] if hit else None,
            "recall_at_threshold": recall,
        }
        print(f"  prompt={name:9} existing pairs >= {args.threshold}: "
              f"{recall:.1%}  (median cosine {sorted(hit)[len(hit)//2]:.3f})", flush=True)
        if recall > best_recall:
            best_name, best_recall, best_sim = name, recall, sim

    report["chosen_prompt"] = best_name
    idx = {f: i for i, f in enumerate(files)}
    focus = [f for f in files if args.focus in f]
    candidates = []
    for f in focus:
        i = idx[f]
        scores, order = torch.topk(best_sim[i], args.top_k)
        for s, j in zip(scores.tolist(), order.tolist()):
            if s < args.threshold:
                continue
            pair = tuple(sorted((f, files[j])))
            candidates.append({
                "cosine": round(s, 4),
                "a": f, "b": files[j],
                "already_in_qrels": pair in existing,
                "a_text": texts[i], "b_text": texts[j],
            })
    candidates.sort(key=lambda c: -c["cosine"])
    report["focus_rows"] = len(focus)
    report["candidates"] = candidates
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nprompt chosen: {best_name}")
    print(f"{len(candidates)} candidate pair(s) >= {args.threshold} for {len(focus)} new row(s)")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
