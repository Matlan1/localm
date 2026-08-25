# SPDX-License-Identifier: AGPL-3.0-or-later
"""Best-effort accelerator detection for install-time and run-time backend choice."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

VENDORS = ("nvidia", "amd", "intel")


@dataclass
class Detection:
    """What was found, and a LEGACY backend guess (see ``recommended``)."""
    vendors: list = field(default_factory=list)   # subset of VENDORS, in priority order
    # LEGACY. Holds only "vulkan" or "cpu" and cannot express "cuda",
    # "amd-rocm" or "metal". Call recommended_install_backend() instead.
    recommended: str = "cpu"
    source: str = ""                               # how we decided (for messaging)
    gpu_names: str = ""                            # raw adapter name(s), lowercased

    # Whether the adapter enumeration ran. False means the question was never
    # answered, which is not the same as "it answered and found no GPU".
    probe_ok: bool = True
    probe_error: str = ""                          # short reason when probe_ok is False

    @property
    def has_gpu(self) -> bool:
        return bool(self.vendors)

    @property
    def gpu_state(self) -> str:
        """``'found'`` | ``'none'`` | ``'unknown'`` - the three outcomes this module can genuinely distinguish, kept apart because they need different responses."""
        if self.vendors:
            return "found"
        return "none" if self.probe_ok else "unknown"


def _run_ok(cmd: list) -> "tuple[str, bool]":
    """Run *cmd*; return (combined stdout+stderr, ran_to_completion)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
    except Exception:
        return "", False
    return (r.stdout or "") + (r.stderr or ""), r.returncode == 0


def _run(cmd: list) -> str:
    """Run *cmd* and return combined stdout+stderr, or '' on any failure."""
    return _run_ok(cmd)[0]


def _win_gpu_names() -> "tuple[str, bool]":
    """Names of the Windows display adapters (lowercased), best-effort, plus whether either enumeration tool actually answered."""
    out, ok = _run_ok([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object -ExpandProperty Name",
    ])
    if not out.strip():
        out, wmic_ok = _run_ok(["wmic", "path", "win32_VideoController", "get", "name"])
        ok = ok or wmic_ok
    return out.lower(), ok


def _linux_gpu_names() -> "tuple[str, bool]":
    """Names of Linux PCI display controllers (lowercased), best-effort, plus whether ``lspci`` actually answered (it is absent on plenty of minimal installs and containers, where 'no display controllers' would be a fabricated answer rather than a measured one)."""
    out, ok = _run_ok(["lspci"])
    return "\n".join(
        ln for ln in out.lower().splitlines()
        if "vga" in ln or "3d controller" in ln or "display" in ln
    ), ok


def detect() -> Detection:
    """Detect GPU vendors present and recommend a default llama.cpp backend."""
    found: list = []

    if sys.platform == "win32":
        names, probe_ok = _win_gpu_names()
        if "nvidia" in names or shutil.which("nvidia-smi"):
            found.append("nvidia")
        if "radeon" in names or "amd " in names or " amd" in names or shutil.which("rocm-smi"):
            found.append("amd")
        if "intel(r) arc" in names or "intel arc" in names or "arc graphic" in names:
            found.append("intel")
        src = "win32 video controllers"
    elif sys.platform == "darwin":
        # Apple Silicon: the GPU is integrated and driven via Metal, which the
        # official llama.cpp macos-arm64 build targets directly.
        machine, uname_ok = _run_ok(["uname", "-m"])
        machine = machine.strip().lower()
        if "arm64" in machine:
            return Detection(vendors=["apple"], recommended="metal", source="apple silicon")
        if not uname_ok:
            # uname could not answer, so the architecture is unknown rather than
            # Intel. Same cpu default, honest label.
            return Detection(vendors=[], recommended="cpu", source="macos, arch unknown",
                             probe_ok=False, probe_error="uname did not run")
        return Detection(vendors=[], recommended="cpu", source="macos intel")
    else:
        names, probe_ok = _linux_gpu_names()
        if shutil.which("nvidia-smi") or "nvidia" in names:
            found.append("nvidia")
        rocm_dir = Path("/opt/rocm").is_dir()    # hygiene-ok: generic ROCm system path
        if (shutil.which("rocminfo") or shutil.which("rocm-smi")
                or rocm_dir or "amd/ati" in names or "radeon" in names):
            found.append("amd")
        if "intel" in names and ("arc" in names or "xe" in names or "dg2" in names):
            found.append("intel")
        src = "lspci / driver tools"

    # Priority order, de-duplicated.
    ordered = [v for v in VENDORS if v in found]
    recommended = "vulkan" if ordered else "cpu"
    # probe_ok/probe_error describe the enumeration, not the conclusion, and are
    # recorded even when shutil.which identified a vendor anyway. gpu_state
    # resolves the two into an answer.
    return Detection(vendors=ordered, recommended=recommended, source=src,
                     gpu_names=names, probe_ok=probe_ok,
                     probe_error=("" if probe_ok
                                  else "the display-adapter enumeration did not run"))


def _amd_known_non_gfx103x(names: str) -> bool:
    """True when the AMD adapter name CLEARLY indicates a family the self-contained gfx103X (RDNA2 / RX 6000) ROCm build does NOT cover: RDNA1 (RX 5000), RDNA3 (RX 7000), RDNA4 (RX 9000), Radeon VII, Instinct."""
    return any(tag in names for tag in
               ("rx 5", "rx 7", "rx 9", "radeon vii", "instinct"))


def _rocm_toolkit_present() -> bool:
    """Whether a working system ROCm/HIP toolkit is installed on THIS machine, independent of GPU-name matching - the SAME signal ``detect()`` already probes (``rocminfo``/``rocm-smi`` on both platforms, ``/opt/rocm`` on Linux) to help decide 'is amd present', factored out here so ``recommended_install_bac..."""
    try:
        if shutil.which("rocminfo") or shutil.which("rocm-smi"):
            return True
        if sys.platform != "win32" and Path("/opt/rocm").is_dir():  # hygiene-ok: generic ROCm system path
            return True
    except Exception:
        return False
    return False


def amd_gfx_family(names: str) -> str:
    """Best-effort AMD GPU family from the adapter name, used to pick a Windows PyTorch ROCm wheel source."""
    n = names or ""
    if "rx 9" in n:
        return "gfx120x"
    if "rx 7" in n:
        return "gfx110x"
    if "rx 6" in n:
        return "gfx103x"
    return ""


def recommended_install_backend(det: "Detection | None" = None) -> str:
    """The backend the INSTALLER should provision by default - the ONE policy both setup.bat and setup.sh call, so the two detectors can never drift: * Apple Silicon -> metal * no GPU -> cpu * NVIDIA, any OS -> cuda (peak performance; llama.cpp ships a self-contained cudart bundle on both Windows and Linux..."""
    d = det or detect()
    if "apple" in d.vendors:
        return "metal"
    if not d.has_gpu:
        return "cpu"
    if "nvidia" in d.vendors:
        return "cuda"
    if d.vendors == ["amd"]:
        if sys.platform == "win32" and not _amd_known_non_gfx103x(d.gpu_names):
            return "amd-rocm"
        if _rocm_toolkit_present():
            return "hip"
    return "vulkan"


def recommended_torch_variant(backend: str, det: "Detection | None" = None) -> str:
    """The PyTorch variant the INSTALLER should provision for the HuggingFace / transformers backend, given the user's chosen llama.cpp *backend* and the detected hardware."""
    b = (backend or "").strip().lower()
    if b == "cuda":
        return "cuda"
    if b in ("amd-rocm", "rocm", "hip"):
        return "rocm"
    if b == "sycl":
        return "xpu"          # explicit Intel GGUF pick -> Intel HF torch (xpu)
    if b == "cpu":
        return "none"
    # Vendor-neutral runtime choice: route the HF torch by the detected GPU.
    # NVIDIA -> cuda, Intel -> xpu, never rocm. No GPU signal -> none.
    d = det or detect()
    if "nvidia" in d.vendors:
        return "cuda"
    if "intel" in d.vendors:
        return "xpu"
    return "none"


# PyTorch wheel index URLs by variant, shared by setup.bat and setup.sh.
# cu126 is the broadly-compatible CUDA line; cuda-blackwell is needed for
# Blackwell-and-newer architectures; xpu is Intel; rocm-linux is upstream;
# rocm-win is AMD's Windows preview. AMD-on-Windows resolves per gfx family in
# torch_pip_args.
_TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu126",
    "cuda-blackwell": "https://download.pytorch.org/whl/cu130",
    "xpu": "https://download.pytorch.org/whl/xpu",
    "rocm-linux": "https://download.pytorch.org/whl/rocm6.2",
    "rocm-win": "https://download.pytorch.org/whl/rocm6.4",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

# Mirrors setup_llama._BLACKWELL_MIN_CAP. Data-center Blackwell is compute
# capability 10.x and consumer/workstation Blackwell is 12.x, so (10, 0) is the
# lower bound covering both.
_CUDA_BLACKWELL_MIN_CAP = (10, 0)


def _cuda_compute_capabilities() -> list:
    """Every NVIDIA GPU's compute capability on this machine, as comparable tuples (one per card; multi-GPU boxes report one line per device)."""
    exe = shutil.which("nvidia-smi") or "nvidia-smi"
    out = _run([exe, "--query-gpu=compute_cap", "--format=csv,noheader"])
    caps = []
    for line in out.strip().splitlines():
        line = line.strip()
        try:
            caps.append(tuple(int(p) for p in line.split(".")))
        except ValueError:
            continue
    return caps


def pytorch_index_url(variant: str) -> "str | None":
    """The PyTorch wheel index URL for a torch *variant* key ('cuda' | 'xpu' | 'rocm-linux' | 'rocm-win' | 'cpu'), or None if unknown."""
    if variant == "cuda":
        caps = _cuda_compute_capabilities()
        if any(cap >= _CUDA_BLACKWELL_MIN_CAP for cap in caps):
            return _TORCH_INDEX["cuda-blackwell"]
    return _TORCH_INDEX.get(variant)


def torch_pip_args(backend: str, det: "Detection | None" = None) -> str:
    """The exact ``uv pip install -p .venv <ARGS>`` arguments to provision the HF PyTorch stack for *backend* on THIS machine - or '' when no verified prebuilt exists and the installer should skip and guide the user."""
    variant = recommended_torch_variant(backend, det)
    if variant == "cuda":
        # Routes through pytorch_index_url so the Blackwell detection lives in
        # one place.
        return f"torch torchvision --index-url {pytorch_index_url('cuda')}"
    if variant == "xpu":
        return f"torch torchvision --index-url {_TORCH_INDEX['xpu']}"
    if variant == "rocm":
        if sys.platform != "win32":
            return f"torch torchvision --index-url {_TORCH_INDEX['rocm-linux']}"
        fam = amd_gfx_family((det or detect()).gpu_names)
        if fam == "gfx103x":
            return "-e .[gpu]"          # bundled gfx1030 self-contained build
        if fam in ("gfx110x", "gfx120x"):
            return f"torch torchvision --index-url {_TORCH_INDEX['rocm-win']}"
        return ""                        # unknown AMD on Windows: no verified prebuilt
    return ""


def main(argv=None) -> int:
    """``python -m localm.hwdetect`` -> prints '<primary-vendor-or-none> <install-backend>' on one line, so the shell installers (setup.bat / setup.sh) share this single tested detector + policy instead of each rolling their own."""
    args = [] if argv is None else list(argv)
    if args and args[0] == "torch-args":
        backend = args[1] if len(args) > 1 else ""
        print(torch_pip_args(backend))
        return 0
    if args and args[0] == "torch":
        backend = args[1] if len(args) > 1 else ""
        print(recommended_torch_variant(backend))
        return 0
    d = detect()
    vendor = d.vendors[0] if d.vendors else "none"
    print(f"{vendor} {recommended_install_backend(d)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
