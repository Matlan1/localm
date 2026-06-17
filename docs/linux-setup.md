# localm on Linux (and macOS)

localm runs natively on Linux. The app code is platform-agnostic; only the
native llama.cpp library and a few setup scripts differ from Windows.

> **macOS is experimental and unverified.** `setup.sh` runs and will try to fetch
> the Apple-Silicon (Metal) llama.cpp build, but this path has not been tested -
> treat it as best-effort and expect rough edges.

## One-click install (the lazy path)

```sh
curl -fsSL https://raw.githubusercontent.com/Matlan1/localm/master/install.sh | bash
```

This clones localm to `~/localm`, installs `uv` if needed, creates a private
`.venv`, auto-detects your GPU, and runs a non-interactive setup that also
provisions the matching llama.cpp backend (Vulkan for any GPU, CPU otherwise).
Override the location with `LOCALM_DIR=...`.

## Manual install

```sh
git clone https://github.com/Matlan1/localm.git
cd localm
bash setup.sh
```

`setup.sh` is interactive: it detects your accelerator, creates the venv,
installs localm, optionally installs the PyTorch stack for your GPU, lets you
copy in a llama.cpp build, picks a data directory, and adds an app-menu entry.
Pass `--yes` for defaults.

Prerequisites:
- `uv` - `curl -LsSf https://astral.sh/uv/install.sh | sh`
- For the graphical launcher: Tk - `sudo apt install python3-tk` (Debian/Ubuntu),
  `sudo dnf install python3-tkinter` (Fedora), etc. The web GUI needs only a browser.

## The native llama.cpp library (GGUF backend)

`setup.sh` provisions this for you from the official llama.cpp releases - pick a
backend (or let auto-detect choose) and it downloads the matching Linux build:

```sh
.venv/bin/localm setup-llama --backend vulkan   # any GPU, no vendor toolkit
.venv/bin/localm setup-llama --backend cuda      # NVIDIA (needs CUDA runtime)
.venv/bin/localm setup-llama --backend hip       # AMD ROCm (needs ROCm runtime)
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
# or .../whl/cu124  (CUDA)  or  .../whl/cpu  (CPU)
```

The GGUF (llama.cpp) backend does not need PyTorch at all.

## Running

```sh
./localm.sh gui              # web GUI in your browser
./localm-launcher.sh         # the graphical launcher (needs Tk)
./localm.sh run <model>      # terminal chat
.venv/bin/localm --help      # everything else
```

Data lives in `./home` (portable) or `~/.localm` (shared), per your setup choice;
override with `LOCALM_HOME=/path`.
