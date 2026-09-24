# DrivelHub+

Code for *Reading Between the Frames: Interpreting Implicit and Non-literal Meaning in Social Media Videos*. The benchmark asks whether a model can explain **why** an apparently trivial or nonsensical short-form video is meaningful, not describe what happens in it.

[Dataset](https://huggingface.co/datasets/extraordinarylab/drivel-hub-plus) · [Website](https://extraordinarylab.github.io/drivel-hub-plus)

## Setup

Install the environment with [INSTALL.md](INSTALL.md), then download the dataset from HuggingFace. It is gated, research-use only, and provides the 1,000 clips plus `metadata.csv` and `qrels.json`. Every command below expects those.

Model outputs and judge gradings are not in this repository; running the pipeline regenerates them.

In `metadata.csv`, `annotation` is the ground truth everything is scored against, and `human_baseline` is a second annotator's independent reading, scored the same way to give a human reference point. `modalities` records which streams the meaning *depends on*, not which streams the clip contains.

## 1. Generate explanations

```bash
MAX_MODEL_LEN=65536 TENSOR_PARALLEL_SIZE=4 bash scripts/serve_vllm.sh "$model"

python -m scripts.run_inference --model "$model" \
  --data-dir /path/to/videos --metadata-csv metadata.csv \
  --output-jsonl outputs/<model>/predictions.jsonl \
  --mode full --temperature 0.6 --top-p 0.95
```

`--mode` picks the input: `full`, `without-audio`, `without-vision`, or `text-cascade` for the text-only baseline. Add `--enable-thinking` for a thinking run.

The human baseline is scored like a system, so give it a predictions file too:

```bash
python -m scripts.make_human_baseline --metadata-csv metadata.csv \
  --data-dir /path/to/videos --output-jsonl outputs/human-baseline/predictions.jsonl
```

## 2. Judge

Both judges see the clip with its soundtrack, the human annotation and the model's explanation, and return a score out of 12 plus a binary alignment label. Both are resumable and skip clips already graded.

**Gemini 3.8 Flash** (primary). Reads `GEMINI_API_KEY` from `.env` and stops at `--budget-usd`. A full pass over 9,000 explanations costs about $24.

```bash
python -m scripts.judge_gemini --model gemini-3.8-flash --data-dir /path/to/videos \
  --input-jsonl outputs/<model>/predictions.jsonl \
  --output-jsonl judgments/gemini-3-8-flash/<model>/judgments.jsonl --budget-usd 5.00
```

**Qwen3-Omni-30B-A3B-Instruct** (open). Served locally, reproduces the same ranking without an API key. Use one `--base-url` per replica.

```bash
python -m scripts.video_llm_judge --model "$judge" \
  --base-url http://127.0.0.1:8000/v1 --data-dir /path/to/videos \
  --input-jsonl outputs/<model>/predictions.jsonl \
  --output-jsonl judgments/qwen3-omni-30b-a3b-instruct/<model>/judgments.jsonl --workers 16
```

If a finished run left `judge_error` rows, rerun `video_llm_judge` with `--retry-error-rows`. `judge_gemini` retries them automatically.

## 3. Cascaded text-only baseline

Converts each clip to text and gives a language model nothing else, which separates failures of modality extraction from failures of pragmatic inference. Speech goes through Whisper large-v3, on-screen text through EasyOCR over 12 sampled frames.

```bash
python -m scripts.extract_cascade_text --stage asr --data-dir /path/to/videos \
  --metadata-csv metadata.csv --output-jsonl cascade/asr.jsonl
python -m scripts.extract_cascade_text --stage ocr --data-dir /path/to/videos \
  --metadata-csv metadata.csv --output-jsonl cascade/ocr.jsonl
```

Then run inference with `--mode text-cascade`, judge as usual, and aggregate:

```bash
python -m scripts.analyse_cascade --judge gemini-3-8-flash --out cascade/analysis_gemini.json
```

## 4. Retrieval

```bash
torchrun --nproc-per-node 4 -m scripts.generate_embeddings --model Qwen/Qwen2.5-Omni-7B \
  --metadata-csv metadata.csv --data-dir /path/to/videos \
  --output-jsonl embeddings/Qwen2.5-Omni-7B.jsonl

python -m scripts.retrieval --embeddings embeddings/Qwen2.5-Omni-7B.jsonl \
  --qrels qrels.json --direction both --output-json retrieval/Qwen2.5-Omni-7B.json
```

Report both directions rather than averaging: clips that sit at rank 1 one way can sit past rank 150 the other.

## 5. Tables and figures

| Script | Produces |
| --- | --- |
| `make_latex_tables.py` | main generation table, one per judge |
| `make_retrieval_table.py` | retrieval table |
| `make_cascade_table.py` | cascade table, both judges side by side |
| `make_three_judge_table.py` | judge-agreement table |
| `make_cost_table.py` | API cost table |
| `make_dataset_tables.py` | dataset statistics |
| `plot_storyscope.py` | writing-style figures |
| `plot_vision_gap.py` | accuracy split by whether vision carries the meaning |
| `plot_similarity.py` | score-separation CDFs (needs `similarity_stats.py` first) |
| `retrieval_direction_cases.py` | per-query ranks behind the direction case study |
| `judge_agreement.py` | correlation statistics between two judges |

Set `PRIMARY_JUDGE=gemini-3.8-flash` so `make_latex_tables.py` labels the Gemini table as the main one.

Dataset construction lives in `hevc_to_h264.py` and `verify_transcode.py` (standardise to H.264), `qrels_candidates.py` (propose relevance pairs for review) and `upload_dataset.py`.

## Notes

Run scripts as modules (`python -m scripts.foo`), not `python scripts/foo.py`, which puts `scripts/` on `sys.path` instead of the repository root.

vLLM wheels from 0.20.0 need an NVIDIA driver of at least 580; on older drivers the ceiling is 0.19.1. vLLM also caps MiniCPM-o at 30 one-second audio chunks, so longer clips need `--audio-chunk-seconds`.

The Gemini API allows 10,000 requests per model per day. One full grading is 9,000, so a second pass the same day fails, and the 429 mentions billing even though the credit balance is untouched.
