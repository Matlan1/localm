# GPU Setup

## Choosing a backend (`setup-llama --backend`)

The installer detects your GPU and provisions the matching llama.cpp backend; you
can also do it (or change it) directly:

```bash
localm setup-llama                      # auto-detect and pick the right backend
localm setup-llama --backend vulkan     # any GPU (AMD/NVIDIA/Intel), no toolkit
localm setup-llama --backend cuda       # NVIDIA, peak perf (needs CUDA runtime)
localm setup-llama --backend amd-rocm   # AMD RX 6000 (gfx103X), self-contained
localm setup-llama --backend cpu        # no GPU
localm setup-llama --from <build dir>   # your own llama.cpp build (any backend)
```

| Backend | Runs on | Notes |
|---|---|---|
| `vulkan` | any AMD / NVIDIA / Intel GPU | universal default - only the normal display driver, no CUDA/ROCm/oneAPI toolkit |
| `cuda` | NVIDIA | peak performance; needs the CUDA runtime present |
| `amd-rocm` | AMD RX 6000 (gfx103X) | self-contained ROCm build (bundles its runtime) |
| `hip` | AMD (any gfx) | upstream ROCm build; needs the ROCm/HIP runtime |
| `sycl` | Intel Arc | needs the oneAPI runtime |
| `cpu` | no GPU | always works |

Binaries come from official [llama.cpp](https://github.com/ggml-org/llama.cpp)
releases (AMD uses a self-contained build); the loader auto-detects the GPU at
runtime. **macOS/Metal is experimental and unverified - treat it as best-effort.
If inference fails or hangs on macOS, fall back to CPU:
`localm setup-llama --backend cpu --force`.**

## AMD (ROCm / HIP)

localm uses the native `llama.dll` HIP build for AMD GPUs. No Python GPU packages are needed for GGUF inference: only the prebuilt DLLs.

### gfx1030 (RX 6000 series: Navi21)

Required DLLs (load order matters):

```
ggml.dll
ggml-base.dll
ggml-cpu.dll
ggml-hip.dll        ← GPU compute via ROCm HIP
llama.dll
```

The DLL directory is auto-detected. To use a custom location:

```bash
set LLAMA_CPP_LIB=C:\path\to\llama.dll
localm run mymodel --prompt "hello"
```

Or set it permanently in your shell profile.

### Building from source (optional)

If you want to build `llama.dll` yourself from the llama.cpp source:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j8
```

Output: `build\bin\Release\llama.dll` and sibling `ggml*.dll` files.

### Verify GPU is being used

Run with a 13B+ model and watch VRAM usage in Task Manager or `rocm-smi`.  
localm prints at startup:

```
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 16368 MiB):
  Device 0: AMD Radeon RX 6900 XT, gfx1030
```

To limit GPU layers (partial offload):
```bash
localm run mymodel --gpu-layers 20 --prompt "hello"
```

## NVIDIA (CUDA)

The simplest path is `localm setup-llama --backend vulkan` (runs on the NVIDIA
GPU through the normal driver, no CUDA toolkit). For peak performance use
`localm setup-llama --backend cuda`, which fetches the upstream CUDA build (it
needs the CUDA runtime present). Either way the loader auto-detects the GPU at
startup; nothing else differs from the AMD path.

## Intel (Arc)

`localm setup-llama --backend vulkan` is the easy path (no toolkit). For the
oneAPI-optimized build use `--backend sycl` (needs the oneAPI runtime present).

## CPU-only

Drop the HIP/CUDA DLL. The DLL load order becomes:

```
ggml.dll → ggml-base.dll → ggml-cpu.dll → llama.dll
```

Set `--gpu-layers 0` to disable GPU offloading:

```bash
localm run mymodel --gpu-layers 0 --prompt "hello"
```

## HuggingFace Transformers (PyTorch)

GGUF inference needs no PyTorch - this section is only for HF-format models. The
installer sets PyTorch up to match your GPU; to do it by hand:

```bash
# AMD (ROCm 7.13, Windows, Python 3.12) - the [gpu] extra:
uv pip install -p .venv -e ".[gpu]"

# NVIDIA (CUDA):
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU:
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The `[gpu]` extra is AMD-ROCm-specific (Windows, Python 3.12) - do **not** install
it on an NVIDIA box; use the CUDA wheels above. GPU is selected automatically;
override with `--device cuda` / `--device cpu`.
