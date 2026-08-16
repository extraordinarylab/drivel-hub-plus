# Environment installation

This project uses the Conda environment `drivelology`. The validated serving
stack targets the cluster's ARM64 NVIDIA GH200 nodes and CUDA 12.x driver.

Run installation on a login node, where GitHub and package downloads are much
faster than on interactive GPU nodes. The Conda environment is on the shared
filesystem, so it is immediately available inside a later Slurm allocation.

```bash
conda create -n drivelology python=3.12 -y
conda activate drivelology

export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=600
export UV_CACHE_DIR="/scratch/u6sn/${USER}/uv-cache"

uv pip sync requirements-vllm.lock.txt --torch-backend=cu129
```

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
regenerate the ARM64 lock on a login node:

```bash
uv pip compile requirements-vllm.txt \
  --python-platform aarch64-manylinux_2_31 \
  --python-version 3.12 \
  --torch-backend=cu129 \
  --only-binary=:all: \
  --generate-hashes \
  --output-file requirements-vllm.lock.txt
```

## Verify on a GPU node

After requesting a Slurm GPU allocation, activate `drivelology` and run:

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

- Package downloads time out on a GPU compute node: leave the Slurm shell
  running and install from a login shell into the same shared Conda
  environment. Keep uv's cache on scratch and rerun; completed wheels are
  reused.
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
- `ModuleNotFoundError: av`: sync the lock again. Qwen3-Omni full-video input
  and the audio/vision ablation modes require the pinned PyAV package.
- uv hardlink warning: keep `UV_LINK_MODE=copy`; home and scratch are different
  filesystems.

Slurm allocation, vLLM server, and inference commands belong in
[`README.md`](README.md), not this installation guide.
