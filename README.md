# DrivelHub+

Code for evaluating implicit and non-literal meaning understanding in social media videos, accompanying the paper *Reading Between the Frames: Interpreting Implicit and Non-literal Meaning in Social Media Videos*.

## Environment

The scripts expect `metadata.csv`, `qrels.json`, and a local directory containing the corresponding MP4 files.

Install the validated Python environment by following [INSTALL.md](INSTALL.md).

## Run vLLM and inference on Slurm

Request one interactive node. The vLLM server and inference client run together
on this node; none of the commands below requests another node:

```bash
srun --partition=interactive \
  --reservation=interactive \
  --cpus-per-task=64 \
  --gres=gpu:4 \
  --time=8:00:00 \
  --pty bash
```

Inside the allocated shell:

```bash
conda activate drivelology
cd ~/workspace/drivel-hub-plus
mkdir -p logs
```

The serving wrapper binds to `0.0.0.0` because these compute nodes cannot bind
vLLM directly to `127.0.0.1`; clients on the same node should still use the
loopback URLs shown below.

Run one checkpoint at a time. The table below records every paper checkpoint
and every additional checkpoint considered for this benchmark, whether run or
not. Paper settings are copied from Table 4. A dash means that a sampling value
was not specified. `not applicable` means that the experiment does not select
between thinking and no-thinking modes; it does not imply that the model never
performs internal reasoning. Inclusion here does not imply that the checkpoint
is compatible with the current vLLM environment.

| Model checkpoint (Hugging Face repo ID) | Thinking | Temperature | Top-p | Top-k |
| --- | --- | ---: | ---: | ---: |
| `AVoCaDO-Captioner/AVoCaDO` | not applicable | 0.7 | 0.90 | — |
| `harryhsing/EchoInk-R1-7B` | not applicable | 0.7 | 0.95 | — |
| `zai-org/GLM-4.1V-9B-Thinking` | not applicable | 0.6 | 0.95 | — |
| `Hcompany/Holo2-30B-A3B` | no thinking | 0.7 | 0.80 | — |
| `internlm/Intern-S1-mini` | not applicable | 0.8 | 1.00 | — |
| `OpenGVLab/InternVL3_5-8B-Instruct` | no thinking | 0.7 | 0.80 | — |
| `OpenGVLab/InternVL3_5-14B-Instruct` | no thinking | 0.7 | 0.80 | — |
| `openbmb/MiniCPM-o-2_6` | not applicable | 0.7 | 0.70 | — |
| `Qwen/QVQ-72B-Preview` | not applicable | 0.6 | 0.95 | — |
| `Qwen/Qwen2.5-VL-7B-Instruct` | not applicable | 0.7 | 0.80 | — |
| `Qwen/Qwen2.5-VL-32B-Instruct` | not applicable | 0.7 | 0.80 | — |
| `Qwen/Qwen2.5-VL-72B-Instruct` | not applicable | 0.7 | 0.80 | — |
| `Qwen/Qwen2.5-Omni-3B` | not applicable | 0.7 | 0.80 | — |
| `Qwen/Qwen2.5-Omni-7B` | not applicable | 0.7 | 0.80 | — |
| `Qwen/Qwen3-VL-8B-Thinking` | thinking | 1.0 | 0.95 | — |
| `Qwen/Qwen3-VL-8B-Instruct` | no thinking | 0.7 | 0.80 | — |
| `Qwen/Qwen3-VL-30B-A3B-Thinking` | thinking | 0.6 | 0.95 | — |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | no thinking | 0.7 | 0.80 | — |
| `Qwen/Qwen3-Omni-30B-A3B-Thinking` | thinking | 0.6 | 0.95 | — |
| `Qwen/Qwen3-Omni-30B-A3B-Thinking` | no thinking | 0.7 | 0.80 | — |
| `Qwen/Qwen3.5-9B` | thinking | 1.0 | 0.95 | — |
| `Qwen/Qwen3.5-9B` | no thinking | 0.7 | 0.80 | — |
| `Qwen/Qwen3.5-27B` | thinking | 1.0 | 0.95 | — |
| `Qwen/Qwen3.5-27B` | no thinking | 0.7 | 0.80 | — |
| `Qwen/Qwen3.5-35B-A3B` | thinking | 1.0 | 0.95 | — |
| `Qwen/Qwen3.5-35B-A3B` | no thinking | 0.7 | 0.80 | — |
| `Qwen/Qwen3.6-27B` | thinking | 1.0 | 0.95 | — |
| `Qwen/Qwen3.6-27B` | no thinking | 0.7 | 0.80 | — |
| `Qwen/Qwen3.6-35B-A3B` | thinking | 1.0 | 0.95 | — |
| `Qwen/Qwen3.6-35B-A3B` | no thinking | 0.7 | 0.80 | — |
| `Qwen/Qwen3.8-27B` | no thinking | 0.7 | 0.80 | 20 |
| `google/gemma-4-E2B-it` | not applicable | 1.0 | 0.95 | 64 |
| `google/gemma-4-12B-it` | not applicable | 1.0 | 0.95 | 64 |
| `zai-org/GLM-4.5V` | thinking | 0.6 | 0.95 | 40 |
| `naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B` | not applicable | 0.5 | 0.60 | — |
| `naver-hyperclovax/HyperCLOVAX-SEED-Think-32B` | thinking | 0.7 | 0.90 | — |
| `internlm/Intern-S2-Mobius` | not applicable | 0.8 | 1.00 | 50 |
| `Kwai-Keye/Keye-VL-1_5-8B` | not applicable | 0.3 | 0.80 | 20 |
| `LanguageBind/Video-LLaVA-7B` | not applicable | — | — | — |
| `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` | not applicable | 0.7 | 1.00 | — |
| `allenai/Molmo2-8B` | not applicable | 0.8 | 0.95 | 50 |
| `allenai/MolmoWeb-4B` | not applicable | 0.8 | 0.95 | 50 |
| `allenai/MolmoWeb-8B` | not applicable | 0.8 | 0.95 | 50 |
| `allenai/Molmo2-4B` | not applicable | 0.8 | 0.95 | 50 |
| `meta-models/Muse-Glimmer-30B` | not applicable | 1.0 | 0.95 | 64 |

Table 4 does not specify top-k for the paper models, so their inference
commands deliberately omit `--top-k` and use each checkpoint's generation
configuration.

### Start Qwen3.5 on all four GPUs

```bash
model=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3.5-35B-A3B
server_pids=()
ports=(8000)

nohup env \
  PORT=8000 \
  MAX_MODEL_LEN=262144 \
  REASONING_PARSER=qwen3 \
  TENSOR_PARALLEL_SIZE=4 \
  bash scripts/serve_vllm.sh "$model" \
  --gdn-prefill-backend triton \
  > logs/qwen3.5-35b-a3b-vllm.log 2>&1 &
server_pids+=("$!")
```

`--gdn-prefill-backend triton` avoids a FlashInfer JIT build that requires
`nvcc`. The serving wrapper defaults to `--mm-processor-cache-gb 0`, which
avoids an observed vLLM 0.19.1 multimodal cache consistency assertion during
concurrent video requests.

### Start four independent one-GPU replicas

Intern-S1, Qwen3-VL-8B, and Qwen3-Omni each fit on one GH200. Four independent
replicas were more reliable here than vLLM's built-in data-parallel coordinator.
Define this helper once inside the allocated shell:

```bash
start_four_replicas() {
  local model=$1
  local log_prefix=$2
  local reasoning_parser=${3:-}
  local use_eager=${4:-false}

  server_pids=()
  ports=(8000 8001 8002 8003)
  for gpu in 0 1 2 3; do
    local port=$((8000 + gpu))
    nohup env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PORT="$port" \
      MAX_MODEL_LEN=65536 \
      REASONING_PARSER="$reasoning_parser" \
      ENFORCE_EAGER="$use_eager" \
      TENSOR_PARALLEL_SIZE=1 \
      bash scripts/serve_vllm.sh "$model" \
      > "logs/${log_prefix}-gpu${gpu}.log" 2>&1 &
    server_pids+=("$!")
  done
}
```

Call it for exactly one model:

```bash
# Intern-S1-9B in the paper
model=/scratch/u6sn/yangw.u6sn/huggingface_models/internlm/Intern-S1-mini
start_four_replicas "$model" intern-s1-mini-vllm ""

# Or Qwen3-VL-8B-Think
model=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-VL-8B-Thinking
start_four_replicas "$model" qwen3-vl-8b-thinking-vllm qwen3 true

# Or Qwen3-Omni-30B-A3B-Thinking
model=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-Omni-30B-A3B-Thinking
start_four_replicas "$model" qwen3-omni-30b-thinking-vllm qwen3 true
```

The documented Qwen3-VL and Qwen3-Omni runs use eager mode because those exact
recipes passed validation. An earlier TP=4 compiled Qwen3-VL run produced
repetitive output, but changing both tensor parallelism and execution mode does
not prove compilation alone caused it. Eager mode disables `torch.compile` and
CUDA graphs; it does not disable FlashAttention (the validated logs select
FlashAttention 3). To test a TP=1 compiled replica, omit the final `true`.

Wait for every selected server to be ready before inference:

```bash
for port in "${ports[@]}"; do
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; do
    sleep 10
  done
  echo "port ${port} ready"
done
```

### Run inference after vLLM is ready

Use this command for Qwen3.5's single four-GPU server:

```bash
python scripts/run_inference.py \
  --model /scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3.5-35B-A3B \
  --data-dir /scratch/u6sn/yangw.u6sn/huggingface_data/extraordinarylab/drivel-hub-plus/data \
  --metadata-csv metadata.csv \
  --output-jsonl outputs/Qwen/Qwen3.5-35B-A3B/predictions.jsonl \
  --mode full \
  --enable-thinking \
  --temperature 1.0 \
  --top-p 0.95 \
  --workers 4
```

For a four-replica model, repeat `--base-url` once per local endpoint. Intern-S1
uses no thinking parser:

```bash
python scripts/run_inference.py \
  --model /scratch/u6sn/yangw.u6sn/huggingface_models/internlm/Intern-S1-mini \
  --base-url http://127.0.0.1:8000/v1 \
  --base-url http://127.0.0.1:8001/v1 \
  --base-url http://127.0.0.1:8002/v1 \
  --base-url http://127.0.0.1:8003/v1 \
  --data-dir /scratch/u6sn/yangw.u6sn/huggingface_data/extraordinarylab/drivel-hub-plus/data \
  --metadata-csv metadata.csv \
  --output-jsonl outputs/internlm/Intern-S1-mini/predictions.jsonl \
  --mode full \
  --temperature 0.8 \
  --top-p 1.0 \
  --workers 4
```

Qwen3-VL uses four concurrent requests per replica:

```bash
python scripts/run_inference.py \
  --model /scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-VL-8B-Thinking \
  --base-url http://127.0.0.1:8000/v1 \
  --base-url http://127.0.0.1:8001/v1 \
  --base-url http://127.0.0.1:8002/v1 \
  --base-url http://127.0.0.1:8003/v1 \
  --data-dir /scratch/u6sn/yangw.u6sn/huggingface_data/extraordinarylab/drivel-hub-plus/data \
  --metadata-csv metadata.csv \
  --output-jsonl outputs/Qwen/Qwen3-VL-8B-Thinking/predictions.jsonl \
  --mode full \
  --enable-thinking \
  --temperature 1.0 \
  --top-p 0.95 \
  --workers 16
```

Qwen3-Omni's vLLM serve path requires the video and its audio track as separate
inputs. `--use-audio-in-video` performs that extraction with PyAV:

```bash
python scripts/run_inference.py \
  --model /scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-Omni-30B-A3B-Thinking \
  --base-url http://127.0.0.1:8000/v1 \
  --base-url http://127.0.0.1:8001/v1 \
  --base-url http://127.0.0.1:8002/v1 \
  --base-url http://127.0.0.1:8003/v1 \
  --data-dir /scratch/u6sn/yangw.u6sn/huggingface_data/extraordinarylab/drivel-hub-plus/data \
  --metadata-csv metadata.csv \
  --output-jsonl outputs/Qwen/Qwen3-Omni-30B-A3B-Thinking/predictions.jsonl \
  --mode full \
  --enable-thinking \
  --temperature 0.6 \
  --top-p 0.95 \
  --use-audio-in-video \
  --workers 16
```

Output is append-only and resumable by filename. Rerun the same command without
`--overwrite` after an interruption; completed files are skipped. To move to
the next model in the same allocation, stop only the server PIDs recorded by
the current shell:

```bash
kill "${server_pids[@]}"
wait "${server_pids[@]}" 2>/dev/null || true
```

## Evaluation

Start four one-GPU Qwen3.8 judge replicas after stopping the inference servers:

```bash
model=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3.8-27B
server_pids=()
ports=(8000 8001 8002 8003)

for gpu in 0 1 2 3; do
  port=$((8000 + gpu))
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PORT="$port" \
    MAX_MODEL_LEN=65536 \
    TENSOR_PARALLEL_SIZE=1 \
    bash scripts/serve_vllm.sh "$model" \
    --gdn-prefill-backend triton \
    > "logs/qwen3.8-27b-judge-vllm-gpu${gpu}.log" 2>&1 &
  server_pids+=("$!")
done
```

After all four ports pass the readiness check above, evaluate a prediction
file with the VideoLLM judge:

```bash
python scripts/video_llm_judge.py \
  --model /scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3.8-27B \
  --base-url http://127.0.0.1:8000/v1 \
  --base-url http://127.0.0.1:8001/v1 \
  --base-url http://127.0.0.1:8002/v1 \
  --base-url http://127.0.0.1:8003/v1 \
  --data-dir /scratch/u6sn/yangw.u6sn/huggingface_data/extraordinarylab/drivel-hub-plus/data \
  --input-jsonl outputs/Qwen/Qwen3-Omni-30B-A3B-No-Thinking/predictions.jsonl \
  --output-jsonl judgments/Qwen/Qwen3-Omni-30B-A3B-No-Thinking/judgments.jsonl \
  --temperature 0.7 \
  --top-p 0.80 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --repetition-penalty 1.0 \
  --workers 16
```

This judge command is no-thinking by default. Add `--enable-thinking` only for
an explicitly thinking judge run. Judge results live under `judgments/` so
each model directory under `outputs/` contains only `predictions.jsonl`.
The command is resumable and skips filenames already judged. If a completed
run contains explicit `judge_error` rows, rerun the same command with
`--retry-error-rows`; it removes only those failed records and regenerates
them without duplicating successful rows.

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
