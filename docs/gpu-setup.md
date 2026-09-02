# GPU Setup

## Choosing a backend (`setup-llama --backend`)

The installer detects your GPU and provisions the matching llama.cpp backend; you
can also do it (or change it) directly:

```bash
localm setup-llama                      # auto-detect
localm setup-llama --backend vulkan     # any GPU
localm setup-llama --backend cuda       # NVIDIA
localm setup-llama --backend amd-rocm   # AMD, self-contained (Windows only)
localm setup-llama --backend cpu        # no GPU
localm setup-llama --from <build dir>   # your own llama.cpp build
```

See the table below for what each backend needs and when to pick it.

| Backend | Runs on | Notes |
|---|---|---|
| `vulkan` | any AMD / NVIDIA / Intel GPU | the universal fallback - only the normal display driver, no CUDA/ROCm/oneAPI toolkit; auto-picked for Intel and for AMD with no ROCm/HIP toolkit detected |
| `cuda` | NVIDIA | auto-picked on every OS (peak performance); setup fetches the CUDA runtime for you (no Toolkit) on both Windows and Linux, then load-tests and falls back to vulkan/cpu if it cannot load |
| `amd-rocm` | AMD RX 6000/7000/9000 (gfx103X/110X/120X), Windows only | self-contained ROCm build matched to your card's family (bundles its runtime, no toolkit needed); auto-picked for RX 6000 (gfx103X), the best-tested family - request it explicitly (`--backend amd-rocm`) on RX 7000/9000 too, since auto-detect still defaults those to `hip`/`vulkan`; on Linux use `hip` or `--from` instead |
| `hip` | AMD (any gfx) | upstream ROCm build; auto-picked when a system ROCm/HIP toolkit is detected present, else `vulkan` |
| `sycl` | Intel Arc | needs the oneAPI runtime; opt-in only (no toolkit-presence probe for it yet) |
| `metal` | Apple Silicon | auto-picked on macOS; experimental and unverified - see the note below |
| `cpu` | no GPU | always works |

Binaries come from official [llama.cpp](https://github.com/ggml-org/llama.cpp)
releases (AMD uses a self-contained build); the loader auto-detects the GPU at
runtime. **macOS/Metal is experimental and unverified - treat it as best-effort.
If inference fails or hangs on macOS, fall back to CPU:
`localm setup-llama --backend cpu --force`.**

Before anything is kept installed, setup **load-tests** the library - and that
load includes localm's own ABI check: does this exact build's struct layout and
enum values still match what localm's native bindings expect (an upstream release
has broken this before without changing the visible API). A build that fails
either way is never left installed; setup explains why and falls back to a build
that is confirmed to work.

## Managing the runtime after install: pin, switch, or roll back

Once a backend is provisioned, `setup-llama` can also change or revert it in
place:

```bash
localm setup-llama --tag b10355     # install exactly this llama.cpp release and pin it
localm setup-llama --tag latest     # track upstream's newest release (untested by localm)
localm setup-llama --tag default    # back to the build localm ships and confirmed
localm setup-llama --rollback       # back to the previous build recorded for this backend
localm setup-llama --backend cuda   # switch backend on a machine that already has one
```

By default you get the newest upstream release localm has confirmed works
(`default`); an upstream release that turns out broken on your hardware is exactly
what `--rollback` is for. Both `--tag` and `--rollback` stick, including across
`localm update`. `localm doctor` shows which build is installed and whether it is
pinned.

The same actions are in the GUI under **Settings -> Updates -> Inference
runtime**: a **Check for runtime update** button (compares what is installed
against what `setup-llama` would install right now, without downloading
anything), an **Update runtime** button once a change is available (pick a
different backend or an exact build first, or leave both blank to just refresh the
current one), and a **Roll back** button that appears whenever an earlier build is
on record for the installed backend. This is also how a GUI-only user installs a
runtime for the first time on a machine that has none, or switches backend,
without touching a terminal.

## AMD (ROCm / HIP)

localm uses the native `llama.dll` HIP build for AMD GPUs. No Python GPU packages are needed for GGUF inference: only the prebuilt DLLs.

`--backend amd-rocm` detects your card's family from its name and fetches the
matching self-contained build automatically: gfx103X for RX 6000, gfx110X for RX
7000, gfx120X for RX 9000. An unrecognised AMD card falls back to the gfx103X
build. This is Windows only; on Linux use `--backend hip` (needs a system
ROCm/HIP toolkit) instead.

### gfx1030 (RX 6000 series: Navi21)

The layout below is illustrative for the gfx103X build specifically - RX 7000
(gfx110X) and RX 9000 (gfx120X) follow the same shape with their own DLL set,
fetched the same way.

Required DLLs (load order matters):

```
ggml-base.dll
ggml-cpu.dll
ggml-hip.dll        ← GPU compute via ROCm HIP
ggml.dll
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

The installer recommends **`cuda`** for NVIDIA on **both Windows and Linux**
(peak performance, self-contained - see below). The universal alternative is
`localm setup-llama --backend vulkan` (runs on the NVIDIA GPU through the
normal driver, no CUDA toolkit) if you would rather skip the runtime download.

On **Windows** this is a guided, self-contained path: setup checks your
driver, then fetches BOTH the CUDA `llama` build and the matching `cudart`
runtime bundle from the same llama.cpp release, so **you do not need to
install the CUDA Toolkit** (needs CUDA >= 12.4 for most GPUs). Setup also
detects your GPU's own architecture and picks the CUDA asset line it actually
needs: NVIDIA Blackwell (RTX 50-series and datacenter B100/B200) automatically
gets the newer 13.x line - which needs a newer driver in turn (CUDA >= 13.3) -
since the 12.x build has no kernels for Blackwell; every earlier architecture
stays on the 12.x line. Setup checks your driver against whichever line your
GPU needs and, if it is too old or no NVIDIA is found, uses Vulkan for now and
tells you what to do; after updating a driver, re-run
`localm setup-llama --backend cuda --force`.

**On Linux**, `--backend cuda` is self-contained the same way: upstream
llama.cpp does not publish a Linux CUDA binary itself, so setup fetches one
from an actively-maintained third-party builder, plus the CUDA runtime
libraries (cudart, cuBLAS) as separate small downloads - again, no CUDA
Toolkit install needed, and the same Blackwell-aware asset-line pick applies.
Real-hardware field testing (NVIDIA RTX PRO 4000 Blackwell) confirmed CUDA
works and outperforms Vulkan on Linux too, so it is now the default
recommendation there, the same as Windows.

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
ggml-base.dll → ggml-cpu.dll → ggml.dll → llama.dll
```

Set `--gpu-layers 0` to disable GPU offloading:

```bash
localm run mymodel --gpu-layers 0 --prompt "hello"
```

## Multiple GPUs: splitting a model across cards

With more than one GPU, a model too large for one card can be split across
several. This applies to `n_gpu_layers`/GGUF loading, not to media generation
(ComfyUI drives its own GPU placement separately).

```bash
localm config main_gpu_index 0          # which device to load onto (blank = device 0)
localm config gpu_split_indices 0,1     # which devices to split across (2+ needed)
localm config gpu_split_ratios 3,1      # relative weight per device, same order/length
```

Left blank, `gpu_split_indices` still spreads a model over every detected GPU by
free VRAM; set it explicitly to choose which cards, and `gpu_split_ratios` to pin
exact relative weights instead of following free VRAM at load time (blank
distributes evenly when free VRAM cannot be measured). Device indices are numbered
in the order llama.cpp itself sees them, which is not always the ggml enumeration
order - a laptop with an integrated GPU alongside a discrete one, for example, may
have the iGPU dropped from that list.

In the GUI, the same controls live in Settings' Live Tuning card as a **Main
GPU** selector and a "Split across GPUs" checkbox row (populated from the
detected device list), with a ratio-weight input beside each checked device for
`gpu_split_ratios`.

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

## GPU load in the GUI

The live GPU stats localm shows (VRAM and load percent) work on AMD and Intel
now, not only NVIDIA: NVIDIA reads `nvidia-smi`; AMD reads the driver's own ADL
sensor for whole-card load (so it is not fooled by ROCm/HIP compute, which does
not show up on the generic Windows GPU-engine counters); everything else on
Windows falls back to those generic counters. This is Windows-first: on Linux
only the NVIDIA reading is currently available. VRAM readings are corrected the
same way, since AMD's own driver query under-reports free VRAM by only counting
what the current process itself has allocated.

## HuggingFace Transformers (PyTorch)

GGUF inference needs no PyTorch - this section is only for HF-format models. The
installer auto-detects your GPU and installs the matching torch wheels. To do it
by hand, use the line for your hardware - if you are not sure which NVIDIA line
applies, ask localm's own detector rather than guessing. The commands below use
`.venv` because that is what the self-contained installers create; if you
installed with `pip install localm` there is no `.venv` to point at - drop
`-p .venv` (or `uv pip install -p .venv`'s `-p .venv`) and run the plain `python
-m localm.hwdetect torch-args cuda` / `pip install torch torchvision --index-url
...` in whatever environment `localm` is already installed into.

`.venv/bin/python -m localm.hwdetect torch-args cuda` (Windows:
`.venv\Scripts\python -m localm.hwdetect torch-args cuda`) prints the exact
`uv pip install` arguments for the card actually in this machine.

```bash
# NVIDIA (CUDA), pre-Blackwell (most cards shipped before 2026), any OS:
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cu126

# NVIDIA (CUDA), Blackwell and newer (RTX 50-series, RTX PRO Blackwell,
# B100/B200) - the cu126 wheels above carry no kernels for these and PyTorch
# would silently run CPU-only:
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Intel (Arc / Xe), any OS - the wheels carry the oneAPI runtime:
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/xpu

# AMD on Linux - upstream ROCm wheels (broad gfx support):
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# AMD on Windows, RX 6000 / RDNA2 (gfx103X) - localm's bundled self-contained build.
# `-e ".[gpu]"` is an editable install and needs a source checkout to point at;
# for `pip install localm` use the non-editable equivalent instead:
#   pip install --upgrade "localm[gpu]"
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

It collects a hardware/backend snapshot plus an allowlisted, scrubbed subset of
config (context sizing, ports, media URLs, privacy mode) - never the whole
config, and never your API key, environment variables, secrets, or chat/transcript
content. The report is saved under your data dir (`bug-reports/`); you can edit
it, then send it with `localm bug-report --send` (no GitHub account needed),
email it to the maintainer, or send the file however you like.
