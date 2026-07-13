# DrivelHub+

Code for evaluating implicit and non-literal meaning understanding in social media videos, accompanying the paper *Reading Between the Frames: Interpreting Implicit and Non-literal Meaning in Social Media Videos*.

## Setup

The scripts expect `metadata.csv`, `qrels.json`, and a local directory containing the corresponding MP4 files.

## Usage

Start an OpenAI-compatible vLLM server:

```bash
PORT=8000 MAX_MODEL_LEN=262144 REASONING_PARSER=qwen3 \
  bash scripts/serve_vllm.sh Qwen/Qwen3.5-27B
```

Generate implicit-meaning explanations:

```bash
python scripts/run_inference.py \
  --model Qwen/Qwen3.5-27B \
  --data-dir /path/to/videos \
  --output-jsonl outputs/predictions.jsonl \
  --mode full
```

Evaluate them with the VideoLLM judge:

```bash
python scripts/video_llm_judge.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --base-url http://localhost:8001/v1 \
  --data-dir /path/to/videos \
  --input-jsonl outputs/predictions.jsonl \
  --output-jsonl outputs/judged.jsonl
```

Generate paired text/video embeddings, then evaluate retrieval:

```bash
torchrun --nproc-per-node 4 scripts/generate_embeddings.py \
  --backend qwen-hidden-state \
  --model Qwen/Qwen2.5-Omni-7B \
  --data-dir /path/to/videos \
  --output-jsonl embeddings/Qwen2.5-Omni-7B.jsonl

python scripts/retrieval.py \
  --embeddings embeddings/Qwen2.5-Omni-7B.jsonl \
  --qrels qrels.json \
  --direction both
```

For modality ablations, run inference with `--mode full`,
`--mode without-audio`, and `--mode without-vision`, judge all three outputs,
then aggregate paired effects:

```bash
python scripts/ablation.py \
  --run MODEL judged/full.jsonl judged/without-audio.jsonl judged/without-vision.jsonl \
  --output-csv outputs/ablation_by_evidence.csv
```
