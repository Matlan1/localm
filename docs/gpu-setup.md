# GPU Setup

## Choosing a backend (`setup-llama --backend`)

The installer detects your GPU and provisions the matching llama.cpp backend; you
can also do it (or change it) directly:

```bash
localm setup-llama                      # auto-detect
localm setup-llama --backend vulkan     # any GPU
localm setup-llama --backend cuda       # NVIDIA
localm setup-llama --backend amd-rocm   # AMD RX 6000 (gfx103X)
localm setup-llama --backend cpu        # no GPU
localm setup-llama --from <build dir>   # your own llama.cpp build
```

See the table below for what each backend needs and when to pick it.

| Backend | Runs on | Notes |
|---|---|---|
| `vulkan` | any AMD / NVIDIA / Intel GPU | universal default - only the normal display driver, no CUDA/ROCm/oneAPI toolkit |
| `cuda` | NVIDIA | peak performance; on Windows setup fetches the CUDA runtime for you (no Toolkit), then load-tests and falls back to vulkan/cpu if it cannot load |
| `amd-rocm` | AMD RX 6000 (gfx103X / RDNA2) | self-contained ROCm build (bundles its runtime); gfx103X-only - other AMD GPUs use `vulkan` or `hip` |
| `hip` | AMD (any gfx) | upstream ROCm build; needs the ROCm/HIP runtime |
| `sycl` | Intel Arc | needs the oneAPI runtime |
| `metal` | Apple Silicon | auto-picked on macOS; experimental and unverified - see the note below |
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
set LLAMA_CPP_LIB=D:\path\to\llama.dll
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

> `-DAMDGPU_TARGETS=gfx1030` targets RX 6000 (RDNA2). For another AMD GPU set it to
> your card's gfx arch (e.g. `gfx1100` for RX 7000 / RDNA3). On Linux,
> `localm setup-llama --backend hip` builds against the installed ROCm/HIP runtime
> instead of bundling one.

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

On **Windows** the installer recommends **`cuda`** for NVIDIA (peak performance,
self-contained - see below). The universal alternative is
`localm setup-llama --backend vulkan` (runs on the NVIDIA GPU through the normal
driver, no CUDA toolkit); on **Linux**, Vulkan is still what the installer
recommends by default.

For peak performance pick `--backend cuda`. On **Windows** this is a guided,
self-contained path: setup checks your driver, then fetches BOTH the CUDA
`llama` build and the matching `cudart` runtime bundle from the same llama.cpp
release, so **you do not need to install the CUDA Toolkit** (needs CUDA >= 12.4
for most GPUs). Setup also detects your GPU's own architecture and picks the
CUDA asset line it actually needs: NVIDIA Blackwell (RTX 50-series and
datacenter B100/B200) automatically gets the newer 13.x line - which needs a
newer driver in turn (CUDA >= 13.3) - since the 12.x build has no kernels for
Blackwell; every earlier architecture stays on the 12.x line. Setup checks
your driver against whichever line your GPU needs and, if it is too old or no
NVIDIA is found, uses Vulkan for now and tells you what to do; after updating a
driver, re-run
`localm setup-llama --backend cuda --force`.

**On Linux**, `--backend cuda` is also self-contained, the same way: upstream
llama.cpp does not publish a Linux CUDA binary itself, so setup fetches one
from an actively-maintained third-party builder, plus the CUDA runtime
libraries (cudart, cuBLAS) as separate small downloads - again, no CUDA
Toolkit install needed. This is newer than the Windows path and has not yet
been confirmed across real NVIDIA Linux hardware, so Vulkan stays the default
recommendation there for now; try `--backend cuda` explicitly if you want to
test it, and report back what you find.

After provisioning, setup **load-tests** the library exactly as `localm run`
will, on every platform. If the CUDA build cannot load on your machine it
automatically falls back to Vulkan, then CPU, so setup never leaves you with a
broken runtime.

> `--sha256 <hex>` pins the **CUDA build** archive only; the cudart runtime
> bundle (Windows) or the CUDA runtime libraries (Linux) are verified by their
> own separately published checksums instead.

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

## Mixture-of-Experts: reducing VRAM footprint

A Mixture-of-Experts (MoE) model's expert layers can be kept in system RAM
instead of VRAM, so it fits in far less GPU memory. Off by default:

```bash
localm config n_cpu_moe 16        # keep 16 layers' worth of experts on system RAM
localm config n_cpu_moe 0         # off - the default
```

It is a footprint dial, not a speedup - the same VRAM budget runs at about the
same tokens/sec either way. Measured on a 7B MoE model: GPU footprint dropped
from 3961 MiB to 241 MiB with all 16 layers set. Has no effect on a normal
(dense) model, and says so instead of silently doing nothing. The Settings
page has the same field ("MoE expert layers on CPU").

## Vision projector (image understanding) placement

Loading an image runs the vision projector (mtmd) on the GPU by default,
falling back to CPU only if the GPU attempt genuinely fails (logging why).
Set `LOCALM_MTMD_CPU=1` to force the old CPU-only behaviour:

```bash
set LOCALM_MTMD_CPU=1
localm run myvisionmodel --image photo.jpg --prompt "Describe this image."
```

## HuggingFace Transformers (PyTorch)

GGUF inference needs no PyTorch - this section is only for HF-format models. The
installer auto-detects your GPU and installs the matching torch wheels. To do it
by hand, use the line for your hardware:

```bash
# NVIDIA (CUDA), any OS:
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Intel (Arc / Xe), any OS - the wheels carry the oneAPI runtime:
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/xpu

# AMD on Linux - upstream ROCm wheels (broad gfx support):
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# AMD on Windows, RX 6000 / RDNA2 (gfx103X) - localm's bundled self-contained build:
uv pip install -p .venv -e ".[gpu]"

# AMD on Windows, RX 7000 / 9000 (RDNA3 / RDNA4) - AMD's Windows ROCm wheels (public preview):
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

# CPU (any machine):
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The `[gpu]` extra is the gfx103X (RX 6000) self-contained build (Windows, Python
3.12) - on other AMD Windows cards use the ROCm 6.4 preview wheels above, and on
Linux use the upstream ROCm index. AMD ROCm on Windows is a recent **public
preview**, so expect rough edges there. GPU is selected automatically; override
with `--device cuda` / `--device xpu` / `--device cpu`.

## When something goes wrong

If setup or a run fails, localm tells you what broke and why, and (for an
unexpected error) offers to save a prefilled, editable bug report you can read
before anything is sent. You can also create one any time:

```bash
localm bug-report -m "short description of the problem"
```

It collects a safe hardware/backend snapshot and never your API key, config, or
chat data. The report is saved under your data dir (`bug-reports/`); you can edit
it, then send it with `localm bug-report --send` (no GitHub account needed),
email it to the maintainer, or send the file however you like.
