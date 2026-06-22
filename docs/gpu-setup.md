# GPU Setup

## Choosing a backend (`setup-llama --backend`)

The installer detects your GPU and provisions the matching llama.cpp backend; you
can also do it (or change it) directly:

```bash
localm setup-llama                      # auto-detect and pick the right backend
localm setup-llama --backend vulkan     # any GPU (AMD/NVIDIA/Intel), no toolkit
localm setup-llama --backend cuda       # NVIDIA, peak perf (Windows: fetches the runtime for you)
localm setup-llama --backend amd-rocm   # AMD RX 6000 (gfx103X), self-contained
localm setup-llama --backend cpu        # no GPU
localm setup-llama --from <build dir>   # your own llama.cpp build (any backend)
```

| Backend | Runs on | Notes |
|---|---|---|
| `vulkan` | any AMD / NVIDIA / Intel GPU | universal default - only the normal display driver, no CUDA/ROCm/oneAPI toolkit |
| `cuda` | NVIDIA | peak performance; on Windows setup fetches the CUDA runtime for you (no Toolkit), then load-tests and falls back to vulkan/cpu if it cannot load |
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
GPU through the normal driver, no CUDA toolkit) - this is what the installer
recommends.

For peak performance pick `--backend cuda`. On **Windows** this is a guided,
self-contained path: setup checks your driver, then fetches BOTH the CUDA
`llama` build and the matching `cudart` runtime bundle from the same llama.cpp
release, so **you do not need to install the CUDA Toolkit**. The dialogue covers:

- **Driver new enough** (supports CUDA >= 12.4): it offers to download the build
  + runtime (a few hundred MB) and sets it up.
- **Driver too old**: the GPU driver is the one piece setup cannot fetch for you
  (it needs admin + a reboot). Setup points you at NVIDIA's driver download and
  uses Vulkan for now so you are not stuck; re-run
  `localm setup-llama --backend cuda --force` after updating.
- **No NVIDIA detected** but you chose CUDA anyway: it proceeds at your request
  (a one-time heads-up), or you can take Vulkan.

After provisioning, setup **load-tests** the library exactly as `localm run`
will. If the CUDA build cannot load on your machine it automatically falls back
to Vulkan, then CPU, so setup never leaves you with a broken runtime. (On Linux,
`--backend cuda` uses the upstream CUDA build and expects the CUDA runtime
present; the same load-test + fallback applies.)

> `--sha256 <hex>` pins the **CUDA build** archive only. The separate cudart
> runtime bundle has no per-release published hash, so it is validated by size +
> archive shape rather than the pin; both come from the same upstream release.

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
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CPU:
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The `[gpu]` extra is AMD-ROCm-specific (Windows, Python 3.12) - do **not** install
it on an NVIDIA box; use the CUDA wheels above. GPU is selected automatically;
override with `--device cuda` / `--device cpu`.

## When something goes wrong

localm tries to fail like a good citizen: it tells you *what* broke and *why*,
and (for an unexpected error) offers to save a prefilled, editable bug report you
can read before anything is sent. You can also create one any time:

```bash
localm bug-report -m "short description of the problem"
```

It collects a safe snapshot - OS, GPU vendor, driver / CUDA capability, the
chosen backend, and the names of the provisioned libraries - and never your API
key, config, or chat data. The report is saved under your data dir
(`bug-reports/`); you can edit it, then email it to the maintainer, open a GitHub
issue (once you have repo access), or send the file however you like.
