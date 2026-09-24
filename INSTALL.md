# Environment installation

This project uses the Conda environment `drivelology`, under the Conda root at
`/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3` (reachable as
`/scratch/u6sn/yangw.u6sn/miniforge3`; there is no `miniforge3` in `$HOME`).
The validated serving stack targets the cluster's ARM64 NVIDIA GH200 nodes and
CUDA 12.x driver.

Building the environment is compute, so submit it rather than running it in a
login shell — see [CLAUDE.md](CLAUDE.md). The environment lives on the shared
filesystem, so it is immediately available inside a later Slurm allocation.

```bash
sbatch --partition=interactive --reservation=interactive --time=4:00:00 \
  --gres=gpu:1 --cpus-per-task=32 install_drivelology.sbatch
```

The job requests one GPU so that it can run the verification below in the same
allocation. The steps it performs are:

```bash
CONDA_ROOT=/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate base

# uv is not in the lock, so it must not live in drivelology (see below).
mamba install -y -c conda-forge uv

conda create -n drivelology python=3.12 -y

export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=600
export UV_CACHE_DIR="/scratch/u6sn/${USER}/uv-cache"
export TMPDIR=/scratch/u6sn/yangw.u6sn/tmp/drivel-hub-plus

"$CONDA_ROOT/bin/uv" pip sync requirements-vllm.lock.txt \
  --python "$CONDA_ROOT/envs/drivelology/bin/python" \
  --torch-backend=cu129
```

## What each non-obvious pin is for

Most entries in `requirements-vllm.txt` are the serving stack. These are the
ones whose absence produces a confusing failure rather than an obvious one:

| Package | Needed by | Symptom when missing |
| --- | --- | --- |
| `librosa`, `soundfile` | `vllm[audio]` | Every `audio_url` request returns HTTP 500, `Please install vllm[audio] for audio support`. Silently makes every omni run vision-only. |
| `av` | `run_inference.py` | `ModuleNotFoundError: av`; no audio extraction and no vision-only remux. |
| `peft` | `jina-embeddings-v5-omni` remote code | `AutoProcessor.from_pretrained` refuses to load: "requires the following packages that were not found: peft". |
| `qwen-omni-utils` | `generate_embeddings.py --backend qwen-hidden-state` | LCO-Embedding, e5-omni and Qwen2.5-Omni retrieval runs die on import. |
| `sentence-transformers` | `--backend st-omni` | Omni-Embed-Nemotron ships as a SentenceTransformers module, not an `AutoModel`. |

## Keep uv out of `drivelology`

`uv pip sync` uninstalls every package the lock does not list, and
`requirements-vllm.lock.txt` does not list `uv`. Installing uv into
`drivelology` therefore makes the sync delete the tool running it, leaving a
half-synced environment. Keep uv in the base env, call it by absolute path, and
point it at the target interpreter with `--python`.

`--python` also removes the need to activate `drivelology` before syncing, so
the job never depends on which env happens to be active.

## CUDA libraries from the HPC SDK

The compute image ships `/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64`
on `LD_LIBRARY_PATH`, and that directory contains `libnvJitLink.so.12` at
version 12.6.77. The CUDA 12.9 wheels pinned here need a newer one: their
`libcusparse.so.12` references `__nvJitLinkGetErrorLogSize_12_9`, and the 12.6
library only exports symbols up to `_12_6`. The dynamic loader consults
`LD_LIBRARY_PATH` before torch's `RUNPATH`, so the system copy wins and
`import torch` fails with:

```
ImportError: .../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol:
__nvJitLinkGetErrorLogSize_12_9, version libnvJitLink.so.12
```

`install_drivelology.sbatch` fixes this on the environment, not in each caller,
by writing `etc/conda/activate.d/10-cuda-wheel-libs.sh`. The hook prepends every
`site-packages/nvidia/*/lib` directory to `LD_LIBRARY_PATH`, and the matching
`deactivate.d` hook restores the previous value. `conda activate drivelology` is
therefore enough — `serve_vllm.sh`, `run_inference.py` and `video_llm_judge.py`
need no change. Confirm it with:

```bash
ldd "$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cusparse/lib/libcusparse.so.12" \
  | grep nvjitlink
```

It must resolve inside `$CONDA_PREFIX`, not under `/opt/nvidia/hpc_sdk`.

Use `requirements-vllm.lock.txt` to reproduce the tested environment. It pins
the complete transitive dependency graph, including hashes and the CUDA 12.9
PyTorch builds. `uv pip sync` also removes packages that are not in the lock,
so start with a new `drivelology` environment unless removing extra packages is
intentional.

`requirements-vllm.txt` is the short, human-edited source specification. It is
useful when updating dependencies, but `uv pip install -r
requirements-vllm.txt` resolves transitive dependencies again and is therefore
less reproducible than syncing the lock. Both files use the official ARM64
vLLM and FlashAttention wheels from GitHub releases; direct wheel URLs work
normally with uv and are recorded with hashes in the lock. Do not replace them
with x86_64 wheels.

This repository intentionally uses `uv pip sync`, not `uv sync`. The latter is
designed around a `pyproject.toml` project and normally creates/manages a
`.venv`; this cluster workflow instead uses the shared Conda environment named
`drivelology`.

When intentionally changing a top-level pin, edit `requirements-vllm.txt` and
regenerate the ARM64 lock. Resolution only reads package metadata and installs
nothing, so it is cheap enough for a login shell.

Compile against `manylinux_2_34`, not `2_31`. The nodes run glibc 2.38, and the
older tag silently drops wheels built for 2.34 — which is how a `tilelang`
aarch64 wheel that exists on PyPI came back as "no solution found":

```bash
uv pip compile requirements-vllm.txt \
  --python-platform aarch64-manylinux_2_34 \
  --python-version 3.12 \
  --torch-backend=cu129 \
  --only-binary=:all: \
  --generate-hashes \
  --output-file requirements-vllm.lock.txt
```

## Updating the env while jobs are running

`uv pip sync` removes whatever the lock does not list, so a sync can break a
running job mid-flight. Before syncing a shared env that something is using,
diff the two locks and confirm the change is purely additive — no removals and
no version changes. Adding packages is safe, because a running process has
already imported what it needs; changing or removing them is not.

`update_env.sbatch` performs the sync and then loads the interfaces that were
previously failing, rather than only checking that the packages import.

## Verify on a GPU node

`install_drivelology.sbatch` runs this check at the end of the build, so the
install log already contains the output. To re-check by hand, request a Slurm
GPU allocation, activate `drivelology` and run:

```bash
python - <<'PY'
import torch
import vllm
import transformers
from flash_attn import flash_attn_func

print("vLLM:", vllm.__version__)
print("Transformers:", transformers.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

q = torch.randn(1, 8, 2, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q)
print("FlashAttention CUDA test:", out.shape, out.device)
PY
```

Expected core versions are:

- vLLM `0.19.1`
- PyTorch `2.10.0+cu129`
- Transformers `5.5.3`
- FlashAttention `2.8.1+cu12torch2.10` for CPython 3.12 ARM64
- NumPy `2.2.6`
- PyAV `18.1.0`

## Why these exact packages are pinned

- vLLM 0.11.2 cannot recognize the Qwen3.5/Qwen3.6
  `qwen3_5_moe` architecture.
- The CUDA 13 vLLM/PyTorch wheels cannot initialize on this cluster's CUDA
  12.x-compatible NVIDIA driver. Use the default vLLM ARM64 wheel and the
  PyTorch `cu129` backend.
- Qwen3.5's tokenizer mapping is absent from Transformers 4.57.6. vLLM 0.19.1
  supports the tested Transformers 5.5.3 release.
- FlashAttention 2.8.3's available CUDA-12 ARM64 wheel targets PyTorch 2.9 and
  fails with an undefined `c10_cuda_check_implementation` symbol under PyTorch
  2.10. The official FlashAttention 2.8.1 wheel pinned here exactly matches
  CUDA 12, PyTorch 2.10, CPython 3.12, CXX11 ABI, and ARM64.
- NumPy newer than 2.2 is incompatible with the pinned Numba release used by
  vLLM.
- The compute image has no `ffmpeg`/`ffprobe` binary or environment module.
  PyAV 18.1.0 supplies ARM64 wheels with the media libraries needed by
  `run_inference.py` to extract audio and remux vision-only video without a
  separate system package.

## Troubleshooting

- Package downloads time out on a GPU compute node: keep uv's cache on scratch
  (`UV_CACHE_DIR`) and resubmit — completed wheels are reused, so the retry
  resumes rather than restarts. Only if the job keeps timing out, fall back to
  installing from a login shell into the same shared Conda environment.
- `uv: command not found` partway through, or a sync that ends with fewer
  packages than the lock: uv was installed into `drivelology` and the sync
  removed it. Reinstall uv into the base env and resync.
- `The NVIDIA driver on your system is too old`: a CUDA 13 PyTorch wheel was
  installed accidentally. Force the CUDA 12.9 build with
  `uv pip install --reinstall torch==2.10.0 torchvision==0.25.0
  torchaudio==2.10.0 --torch-backend=cu129`.
- `KeyError: 'Qwen3_5MoeConfig'`: verify vLLM is 0.19.1 and Transformers is
  5.5.3.
- `undefined symbol ... c10_cuda_check_implementation`: uninstall the stale
  FlashAttention build and reinstall the exact wheel pinned in
  `requirements-vllm.txt`.
- `Numba needs NumPy 2.2 or less`: restore `numpy==2.2.6`.
- `undefined symbol: __nvJitLinkGetErrorLogSize_12_9`: the HPC SDK's CUDA 12.6
  libraries are shadowing the wheels. See "CUDA libraries from the HPC SDK";
  the activation hook is missing or the env was not activated through conda.
- `ModuleNotFoundError: av`: sync the lock again. Qwen3-Omni full-video input
  and the audio/vision ablation modes require the pinned PyAV package.
- uv hardlink warning: keep `UV_LINK_MODE=copy`; home and scratch are different
  filesystems.

Slurm allocation, vLLM server, and inference commands belong in
[`README.md`](README.md), not this installation guide.
