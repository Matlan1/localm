# localm on Linux (and macOS)

localm runs natively on Linux. The app code is platform-agnostic; only the
native llama.cpp library and a few setup scripts differ from Windows.

> **macOS is experimental.** On Apple Silicon `setup.sh` defaults to the Metal
> llama.cpp build (the interactive picker also lists it, as choice 8, on Apple
> Silicon only); Intel Macs default to the `cpu` backend, since no Metal build
> targets them. A CI job on an Apple Silicon runner (daily, and on changes to
> detection or provisioning code) installs with `setup.sh --yes`, checks that
> the backend pick is Metal, runs `localm doctor`, starts a server with no
> model, and loads a tiny model and generates with it on the CPU (0 GPU
> layers). Metal GPU inference itself is not exercised there - treat it as
> best-effort and expect rough edges. To run a model on the CPU anyway, set
> `localm config n_gpu_layers 0` (or pass `-g 0` to `localm run` / `localm
> serve`); `--backend cpu` does not do that on Apple Silicon, because it
> installs the same arm64 build as `metal`.

## One-click install (the lazy path)

```sh
curl -fsSL https://raw.githubusercontent.com/Matlan1/localm/master/install.sh | bash
```

Needs `git` and `curl`. This clones localm (shallow) to `~/localm` - an
existing clone there is updated with `git pull --ff-only` instead - and runs
`bash setup.sh --yes`, which takes the default at every question: the Portable
choice (uv, its Python and its download cache stay inside the clone), a private
`.venv`, data in `./home`, the GUI in a browser tab, an application-menu entry
(a `.desktop` file in `~/.local/share/applications`), and no global `localm`
command. It auto-detects your GPU and provisions the
matching llama.cpp backend (CUDA for NVIDIA, HIP for AMD when a system ROCm/HIP
toolkit is present, Metal on Apple Silicon, Vulkan as the universal fallback
otherwise, CPU with no GPU - see [gpu-setup.md](gpu-setup.md) for the full
policy). Piped from `curl`, it skips the plugin picker; run `localm plugin
setup` afterwards to choose features. `localm` is not on your PATH afterwards:
run `~/localm/localm.sh <args>`, or run `bash setup.sh` yourself and answer `y`
to the global-command question.

Override the location on the `bash` side of the pipe (`LOCALM_REPO` overrides
the repository the same way):

```sh
curl -fsSL https://raw.githubusercontent.com/Matlan1/localm/master/install.sh | LOCALM_DIR=~/apps/localm bash
```

## Manual install

```sh
git clone https://github.com/Matlan1/localm.git
cd localm
bash setup.sh
```

`setup.sh` is interactive. In order, it asks whether uv, its Python and its
download cache stay inside the clone (Portable, the default) or use uv's normal
per-user locations (Shared); installs `uv` if it is missing (into `./.uv` for
Portable); asks whether to use a detected GPU; creates `.venv` (Python 3.12);
asks where data should live (`./home`, or a custom path recorded in
`localm-home.cfg`); asks whether the GUI opens in a browser tab or an app window
(the app window adds the `desktop` extra); installs localm (extras `coder`,
`voice`, `monitor`); lets you pick a llama.cpp backend (or copy in your own
build); installs PyTorch to match; builds the launcher; offers an
application-menu entry and an optional global `localm` command (a symlink in
`~/.local/bin`); and runs `localm plugin setup`. Pass `--yes` to take the
default at each question (the plugin picker still asks when run from a
terminal). `bash setup.sh --uninstall` removes what setup installed.

The PyTorch stack is only needed to run HuggingFace (non-GGUF) Transformers
models; GGUF chat through the llama.cpp backend needs none of it. There is no
separate yes/no prompt for it - `setup.sh` installs a matching PyTorch build
(and the `hf` and `audio` extras) automatically when the llama.cpp backend you
pick maps to one: CUDA wheels for `cuda`, ROCm wheels for `hip`, XPU wheels for
`sycl`. For `vulkan`, `metal` or your own build it follows the GPU it finds
instead: CUDA wheels with an NVIDIA GPU, XPU wheels with an Intel GPU, never
ROCm. It skips PyTorch for the `cpu` backend (which also gives up
GPU-accelerated GGUF chat) and for those vendor-neutral picks on a machine with
neither GPU, such as an AMD or Apple Silicon box, and prints how to add PyTorch
yourself. If you want GPU-accelerated GGUF chat but not PyTorch, uninstall it
after setup finishes (`uv pip uninstall -p .venv torch torchvision`) - or add it
back later yourself (see the README extras).

Prerequisites:
- `git` to clone, and `curl` (setup.sh uses it to fetch uv's installer when uv
  is missing).
- `uv` - `setup.sh` installs it for you if missing; only needed by hand if you
  skip `setup.sh`: install it with your package manager or download it from
  https://github.com/astral-sh/uv/releases. With the Portable choice uv lives
  at `./.uv/uv`, not on your PATH: wherever this page shows `uv pip ...`, run
  `./.uv/uv pip ...` (or `export PATH="$PWD/.uv:$PATH"` first).
- For the graphical launcher: Tk. localm's bundled Python includes it, so it
  normally works out of the box; if the launcher reports Tk missing (a system
  Python without it), install `sudo apt install python3-tk` (Debian/Ubuntu),
  `sudo dnf install python3-tkinter` (Fedora), etc. The web GUI needs only a
  browser.

## pip install

```sh
pip install localm
```

No clone needed. Needs Python **exactly 3.12.x already on your machine** -
`pip install` does not provision its own interpreter the way `setup.sh` does.
There is no `./localm.sh`, `./localm-launcher.sh`, or `.venv/` - `localm` is a
normal console script on your `PATH` once installed, so every command below
that shows `.venv/bin/localm ...` or `./localm.sh ...` is just `localm ...`
for you.

Set `LOCALM_HOME` before you do anything else: `setup.sh` above always asks
where data should live, but a plain `pip install` never does, and with
nothing set it falls back to a directory inside your Python environment. An
upgrade does not delete it, but your models then live inside site-packages, and
removing or recreating that environment takes them with it.

```sh
export LOCALM_HOME=~/localm-data
```

Then provision the native GGUF backend the same way as any other install -
see the next section.

## The native llama.cpp library (GGUF backend)

`setup.sh` provisions this for you from the official llama.cpp releases (the
Linux CUDA build comes from a third-party builder, see
[gpu-setup.md](gpu-setup.md#nvidia-cuda)) - pick a backend (or let auto-detect
choose) and it downloads the matching build (a pip install: run the same
`setup-llama` command as plain `localm setup-llama --backend ...`, no
`.venv/bin/` prefix). The prebuilt Linux downloads are x86_64 builds; on another
architecture build llama.cpp yourself (below).

```sh
.venv/bin/localm setup-llama                     # auto-detect (the policy in gpu-setup.md)
.venv/bin/localm setup-llama --backend vulkan   # any GPU, no vendor toolkit
.venv/bin/localm setup-llama --backend cuda      # NVIDIA (needs the driver; fetches its own CUDA runtime)
.venv/bin/localm setup-llama --backend hip       # AMD ROCm (needs ROCm runtime)
.venv/bin/localm setup-llama --backend sycl      # Intel GPU, incl. integrated (needs the oneAPI runtime)
.venv/bin/localm setup-llama --backend cpu       # no GPU
.venv/bin/localm setup-llama --backend metal     # Apple Silicon (macOS only)
```

Upstream's `cpu`/`vulkan` Linux builds link against the system's OpenMP
runtime (`libgomp.so.1`) but do not ship it; setup bundles a copy into the
runtime dir automatically (the same way manylinux Python wheels vendor it),
so a bare/minimal distro image works without installing anything yourself. A
load failure naming a *different* missing `.so` is a genuine system
dependency (e.g. `libvulkan1` for the `vulkan` build) - install the named
package and retry with `--force`.

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

This copies the shared libraries in that directory (`libllama.so` and its
`libggml*.so` deps; `.dylib` on macOS) into localm's runtime. For a one-off
without copying, set `LLAMA_CPP_LIB=/path/to/libllama.so`.

Runtime deps: a ROCm/CUDA build needs its vendor runtime resolvable at load time.
The cleanest options are an rpath in your build, a system install (`ldconfig`,
`/opt/rocm/lib`), or `LD_LIBRARY_PATH` in your shell before launching localm.

## GPU acceleration for the HuggingFace/transformers backend

`setup.sh` installs PyTorch for the backend you picked (see above) from the
matching PyTorch index (CUDA, ROCm or XPU), and none for `cpu`. If the version
it picks does not suit your setup, install torch yourself, e.g.:

```sh
uv pip install -p .venv torch torchvision --torch-backend=rocm6.2
# or cu126 (CUDA, most cards) or cu130 (CUDA, Blackwell and newer - cu126 has
# no kernels for these) or xpu (Intel) or cpu (CPU). --torch-backend, not
# --index-url: the latter replaces PyPI outright for the whole install, and
# every PyTorch wheel index caps setuptools below what this project requires.
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
`setup.sh` (it picks the right `--torch-backend`) or the manual
`uv pip install ... --torch-backend` command shown above.

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

The `./localm.sh` / `./localm-launcher.sh` wrappers and the `.venv/` they call
into are created by `setup.sh` inside the clone. A `pip install localm` has
none of that - `localm` is already on `PATH`, so the same commands are just
`localm gui`, `localm run <model>`, `localm --help`.

Data lives in `./home` (portable, the default) or a custom path you pick during
setup (recorded in `localm-home.cfg`) - there is no silent per-user fallback to a
shared `~/.localm`. Override at any time with `LOCALM_HOME=/path`. That
portable `./home` default only applies to a source checkout (`setup.sh`); a
`pip install` has no portable mode - see [pip install](#pip-install) above,
set `LOCALM_HOME` explicitly.
