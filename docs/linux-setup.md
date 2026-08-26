# localm on Linux (and macOS)

localm runs natively on Linux. The app code is platform-agnostic; only the
native llama.cpp library and a few setup scripts differ from Windows.

> **macOS is experimental and unverified.** `setup.sh` auto-detects Apple
> Silicon and defaults to the Metal llama.cpp build (the interactive picker
> also offers it as its own numbered choice); Intel Macs default to CPU-only,
> since no Metal build targets them. Detection itself has not been tested on
> real hardware - treat it as best-effort and expect rough edges.

## One-click install (the lazy path)

```sh
curl -fsSL https://raw.githubusercontent.com/Matlan1/localm/master/install.sh | bash
```

This clones localm to `~/localm`, installs `uv` if needed, creates a private
`.venv`, auto-detects your GPU, and runs a non-interactive setup that also
provisions the matching llama.cpp backend (CUDA for NVIDIA, HIP for AMD when a
system ROCm/HIP toolkit is present, Metal on Apple Silicon, Vulkan as the
universal fallback otherwise, CPU with no GPU - see
[gpu-setup.md](gpu-setup.md) for the full policy).
Override the location with `LOCALM_DIR=...`.

## Manual install

```sh
git clone https://github.com/Matlan1/localm.git
cd localm
bash setup.sh
```

`setup.sh` is interactive: it installs `uv` for you if you do not have it,
detects your accelerator, creates the venv, installs localm, lets you pick a
llama.cpp backend (or copy in your own build), picks a data directory, and adds
an app-menu entry. Pass `--yes` for defaults.

The PyTorch stack is only needed to run HuggingFace (non-GGUF) Transformers
models; GGUF chat through the llama.cpp backend needs none of it. There is no
separate yes/no prompt for it - `setup.sh` installs a matching PyTorch build
automatically for whichever GPU llama.cpp backend you pick, and skips it only
if you pick the `cpu` llama.cpp backend, which also gives up
GPU-accelerated GGUF chat. If you want GPU-accelerated GGUF chat but not
PyTorch, uninstall it after setup finishes
(`uv pip uninstall -p .venv torch torchvision`) - or add it back later
yourself (see the README extras).

Prerequisites:
- `uv` - `setup.sh` installs it for you if missing; only needed by hand if you
  skip `setup.sh`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- For the graphical launcher: Tk - `sudo apt install python3-tk` (Debian/Ubuntu),
  `sudo dnf install python3-tkinter` (Fedora), etc. The web GUI needs only a browser.

## The native llama.cpp library (GGUF backend)

`setup.sh` provisions this for you from the official llama.cpp releases - pick a
backend (or let auto-detect choose) and it downloads the matching Linux build:

```sh
.venv/bin/localm setup-llama --backend vulkan   # any GPU, no vendor toolkit
.venv/bin/localm setup-llama --backend cuda      # NVIDIA (needs CUDA runtime)
.venv/bin/localm setup-llama --backend hip       # AMD ROCm (needs ROCm runtime)
.venv/bin/localm setup-llama --backend sycl      # Intel Arc (needs the oneAPI runtime)
.venv/bin/localm setup-llama --backend cpu       # no GPU
```

If you would rather build llama.cpp yourself (e.g. a specific gfx target), build
it once and point localm at the output with `--from`:

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# CPU:
cmake -B build && cmake --build build --config Release -j

# AMD ROCm (set your gfx target, e.g. gfx1030 for RDNA2):
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# NVIDIA CUDA:
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Then provision it into localm:

```sh
.venv/bin/localm setup-llama --from /path/to/llama.cpp/build/bin
```

This copies `libllama.so` and its `libggml*.so` deps into localm's runtime. For
a one-off without copying, set `LLAMA_CPP_LIB=/path/to/libllama.so`.

Runtime deps: a ROCm/CUDA build needs its vendor runtime resolvable at load time.
The cleanest options are an rpath in your build, a system install (`ldconfig`,
`/opt/rocm/lib`), or `LD_LIBRARY_PATH` in your shell before launching localm.

## GPU acceleration for the HuggingFace/transformers backend

`setup.sh` installs PyTorch for the detected vendor from the matching PyTorch
index (ROCm / CUDA / CPU). If the version it picks does not suit your setup,
install torch yourself, e.g.:

```sh
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
# or .../whl/cu126 (CUDA, most cards) or .../whl/cu130 (CUDA, Blackwell and
# newer - cu126 has no kernels for these) or .../whl/cpu (CPU)
```

Not sure which CUDA line your card needs? `.venv/bin/python -m localm.hwdetect
torch-args cuda` prints the exact one for the GPU actually in this machine (see
[gpu-setup.md](gpu-setup.md#huggingface-transformers-pytorch) for the full table).

A HuggingFace AWQ (4-bit) checkpoint needs nothing beyond this PyTorch stack:
localm dequantizes AWQ layers natively (no `gptqmodel`/`autoawq`/`torchao`
compiled dependency), so it loads and runs the same way on Linux ROCm, CUDA,
or CPU as it does elsewhere.

### `pip install "localm[gpu]"` is Windows-only

The `[gpu]` pip extra pins AMD ROCm torch wheels and can only resolve them on
Windows. A pip extra cannot carry a custom package index, so on Linux
`pip install "localm[gpu]"` will NOT install a GPU torch build. On Linux use
`setup.sh` (it adds the right `--index-url`) or the manual
`uv pip install ... --index-url` command shown above.

### WSL2 and virtual machines

- **WSL2**: NVIDIA CUDA works well (CUDA-on-WSL is supported); AMD ROCm under
  WSL2 is experimental.
- **Plain VMs** (VMware / VirtualBox / default Hyper-V): only a virtual display
  adapter is exposed, so localm runs CPU-only. Real GPU acceleration in a VM
  needs PCI passthrough (VFIO/IOMMU, or Hyper-V DDA - usually a second GPU).

Linux AMD ROCm GPU acceleration is not yet verified end-to-end on native Linux;
treat it as best-effort until confirmed on real hardware.

## Running

```sh
./localm.sh gui              # web GUI in your browser
./localm-launcher.sh         # the graphical launcher (needs Tk)
./localm.sh run <model>      # terminal chat
.venv/bin/localm --help      # everything else
```

Data lives in `./home` (portable, the default) or a custom path you pick during
setup (recorded in `localm-home.cfg`) - there is no silent per-user fallback to a
shared `~/.localm`. Override at any time with `LOCALM_HOME=/path`.
