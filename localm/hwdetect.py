# SPDX-License-Identifier: AGPL-3.0-or-later
"""Best-effort accelerator detection for install-time and run-time backend choice.

Pure stdlib, and NEVER raises: detection is advisory, so a probe that fails just
contributes nothing. ``detect()`` reports which GPU vendors are present and the
recommended llama.cpp backend.

Backend-selection policy (the "anybody, out of the box" rule): pick the backend
that is fastest AND works with no user-installed toolkit on THIS machine.
  * NVIDIA on Windows -> ``cuda``: the same llama.cpp release ships a
    self-contained cudart bundle, so CUDA is out-of-the-box here (no Toolkit) and
    the fastest path on NVIDIA. (NVIDIA on Linux stays ``vulkan`` - the Linux cuda
    build needs a system CUDA toolkit we cannot self-provide.)
  * ``vulkan`` is the universal default for every other GPU - it runs on AMD,
    NVIDIA (Linux), and Intel through the vendor's normal display driver, with no
    CUDA/ROCm/oneAPI toolkit to install.
  * the remaining vendor-specific backends (``hip`` for AMD, ``sycl`` for Intel)
    are offered as an opt-in for maximum performance, not the default.
  * ``cpu`` when no GPU is detected.

Detection is intentionally conservative: a missing tool or an unparseable name
means "unknown", never a crash and never a wrong-vendor claim.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

VENDORS = ("nvidia", "amd", "intel")


@dataclass
class Detection:
    """What was found, and the backend we'd pick by default."""
    vendors: list = field(default_factory=list)   # subset of VENDORS, in priority order
    recommended: str = "cpu"                       # "vulkan" | "cpu"
    source: str = ""                               # how we decided (for messaging)
    gpu_names: str = ""                            # raw adapter name(s), lowercased

    @property
    def has_gpu(self) -> bool:
        return bool(self.vendors)


def _run(cmd: list) -> str:
    """Run *cmd* and return combined stdout+stderr, or "" on any failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def _win_gpu_names() -> str:
    """Names of the Windows display adapters (lowercased), best-effort."""
    out = _run([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object -ExpandProperty Name",
    ])
    if not out.strip():
        out = _run(["wmic", "path", "win32_VideoController", "get", "name"])
    return out.lower()


def _linux_gpu_names() -> str:
    """Names of Linux PCI display controllers (lowercased), best-effort."""
    out = _run(["lspci"])
    return "\n".join(
        ln for ln in out.lower().splitlines()
        if "vga" in ln or "3d controller" in ln or "display" in ln
    )


def detect() -> Detection:
    """Detect GPU vendors present and recommend a default llama.cpp backend.

    Never raises. Vendor order in the result is priority order (nvidia, amd,
    intel) for picking a vendor-optimized backend if the user opts in."""
    found: list = []

    if sys.platform == "win32":
        names = _win_gpu_names()
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
        machine = _run(["uname", "-m"]).strip().lower()
        if "arm64" in machine:
            return Detection(vendors=["apple"], recommended="metal", source="apple silicon")
        return Detection(vendors=[], recommended="cpu", source="macos intel")
    else:
        names = _linux_gpu_names()
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
    return Detection(vendors=ordered, recommended=recommended, source=src,
                     gpu_names=names)


def recommended_backend(det: "Detection | None" = None) -> str:
    """Convenience: the default backend for the current machine."""
    return (det or detect()).recommended


def _amd_known_non_gfx103x(names: str) -> bool:
    """True when the AMD adapter name CLEARLY indicates a family the self-contained
    gfx103X (RDNA2 / RX 6000) ROCm build does NOT cover: RDNA1 (RX 5000), RDNA3
    (RX 7000), RDNA4 (RX 9000), Radeon VII, Instinct. Conservative - an unknown or
    RX 6000 name returns False so the self-contained build stays the default (the
    setup-llama load-test + vulkan fallback is the net if this guess is ever wrong)."""
    return any(tag in names for tag in
               ("rx 5", "rx 7", "rx 9", "radeon vii", "instinct"))


def amd_gfx_family(names: str) -> str:
    """Best-effort AMD GPU family from the adapter name, used to pick a Windows
    PyTorch ROCm wheel source. Coarse and conservative - an unrecognised AMD card
    returns "". Families (checked most-recent first so "rx 9"/"rx 7" win):
      * "gfx120x" - RDNA4 / RX 9000 (Navi 4x)
      * "gfx110x" - RDNA3 / RX 7000 (Navi 3x)
      * "gfx103x" - RDNA2 / RX 6000 (Navi 2x) -> localm's bundled self-contained build
    RDNA1 (RX 5000) and older return "" - no current prebuilt Windows ROCm wheel.
    """
    n = names or ""
    if "rx 9" in n:
        return "gfx120x"
    if "rx 7" in n:
        return "gfx110x"
    if "rx 6" in n:
        return "gfx103x"
    return ""


def recommended_install_backend(det: "Detection | None" = None) -> str:
    """The backend the INSTALLER should provision by default - the ONE policy both
    setup.bat and setup.sh call, so the two detectors can never drift:
      * Apple Silicon                         -> metal
      * no GPU                                -> cpu
      * NVIDIA on Windows                      -> cuda      (peak performance; the same
        llama.cpp release ships a self-contained cudart bundle, so it is out-of-the-box
        with no CUDA Toolkit, and setup-llama's driver preflight + load-test fall back
        to vulkan if the driver is too old - never strands a box on a runtime it cannot load)
      * NVIDIA on Linux                        -> vulkan    (the Linux cuda build needs a
        system CUDA toolkit we cannot self-provide; vulkan runs via the display driver)
      * AMD on Windows, RX 6000 / unknown     -> amd-rocm  (self-contained gfx103X build)
      * AMD on Windows, clearly not RX 6000    -> vulkan    (gfx103X build won't fit)
      * any other GPU (Intel, Linux AMD, mixed)-> vulkan    (no vendor toolkit needed)
    The self-contained ROCm bundle is gfx103X + Windows only, and self-contained CUDA is
    Windows only, hence both vendor-optimized cases are narrowed to Windows."""
    d = det or detect()
    if "apple" in d.vendors:
        return "metal"
    if not d.has_gpu:
        return "cpu"
    if "nvidia" in d.vendors and sys.platform == "win32":
        return "cuda"
    if (sys.platform == "win32" and d.vendors == ["amd"]
            and not _amd_known_non_gfx103x(d.gpu_names)):
        return "amd-rocm"
    return "vulkan"


def recommended_torch_variant(backend: str, det: "Detection | None" = None) -> str:
    """The PyTorch variant the INSTALLER should provision for the HuggingFace /
    transformers backend, given the user's chosen llama.cpp *backend* and the
    detected hardware. Returns "cuda" | "rocm" | "xpu" | "none". Both setup.bat and
    setup.sh call this (via ``python -m localm.hwdetect torch <backend>``) AFTER the
    backend pick, so the GGUF runtime choice and the HF torch choice stay coherent.

    Why the BACKEND drives this and not the detected vendor alone (the SETUP-1 bug):
    a user who explicitly picks the vendor-neutral ``vulkan`` runtime on an AMD box
    has stepped OFF the ROCm path - which on Windows is the gfx103X-pinned ``.[gpu]``
    stack - so forcing ROCm torch on them is exactly the reported surprise. Policy:

      * ``cuda`` backend           -> cuda   (the user asked for the NVIDIA path)
      * ``amd-rocm`` / ``hip``     -> rocm   (the user asked for the AMD path)
      * ``sycl`` backend           -> xpu    (explicit Intel GGUF pick -> Intel HF torch)
      * ``cpu`` backend            -> none   (no GPU; HF torch is opt-in, see installer)
      * ``vulkan`` / ``own`` / other -> cuda when an NVIDIA GPU is present, xpu when an
                                       Intel GPU is present (both clean pip wheels that
                                       self-provision their runtime; the xpu wheels carry
                                       the oneAPI runtime), else none. NEVER rocm on a
                                       vendor-neutral pick.

    "none" means the installer SKIPS the heavy torch stack and tells the user how to add
    CPU / CUDA / ROCm / Intel-XPU torch themselves: honest, no surprise download, and no
    wrong-vendor stack installed behind their back."""
    b = (backend or "").strip().lower()
    if b == "cuda":
        return "cuda"
    if b in ("amd-rocm", "rocm", "hip"):
        return "rocm"
    if b == "sycl":
        return "xpu"          # explicit Intel GGUF pick -> Intel HF torch (xpu)
    if b == "cpu":
        return "none"
    # vulkan / own / metal / unknown / empty: vendor-neutral runtime choice. Route the HF
    # torch by the DETECTED GPU: NVIDIA -> cuda, Intel -> xpu (both clean pip wheels that
    # self-provision; the xpu wheels carry the oneAPI runtime). Never ROCm on a neutral
    # pick. No GPU signal -> none.
    d = det or detect()
    if "nvidia" in d.vendors:
        return "cuda"
    if "intel" in d.vendors:
        return "xpu"
    return "none"


# PyTorch wheel index URLs by variant. Centralised so setup.bat / setup.sh never
# drift on the source. cu126 = current CUDA line; xpu = Intel (the wheels carry the
# oneAPI runtime); rocm-linux = upstream ROCm wheels (broad gfx); rocm-win = AMD's
# Windows ROCm wheels (public preview, RDNA3/RDNA4). AMD-on-Windows is resolved PER
# gfx family in torch_pip_args - gfx103X uses localm's bundled self-contained build.
_TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu126",
    "xpu": "https://download.pytorch.org/whl/xpu",
    "rocm-linux": "https://download.pytorch.org/whl/rocm6.2",
    "rocm-win": "https://download.pytorch.org/whl/rocm6.4",
    "cpu": "https://download.pytorch.org/whl/cpu",
}


def pytorch_index_url(variant: str) -> "str | None":
    """The PyTorch wheel index URL for a torch *variant* key ("cuda" | "xpu" |
    "rocm-linux" | "rocm-win" | "cpu"), or None if unknown. Public accessor so other
    callers (the managed-ComfyUI fresh install picks the ComfyUI torch here) share
    this ONE index table instead of duplicating the URLs and drifting from it."""
    return _TORCH_INDEX.get(variant)


def torch_pip_args(backend: str, det: "Detection | None" = None) -> str:
    """The exact ``uv pip install -p .venv <ARGS>`` arguments to provision the HF
    PyTorch stack for *backend* on THIS machine - or "" when no verified prebuilt
    exists and the installer should skip and guide the user. SINGLE source of truth
    for the torch wheel SOURCE (setup.bat and setup.sh both consult it via
    ``python -m localm.hwdetect torch-args <backend>``), so the gfx-specific
    AMD-on-Windows routing lives in one tested place and the two installers cannot
    drift.

      * cuda            -> CUDA wheels (cu126); any OS with NVIDIA
      * xpu             -> Intel wheels (self-provision the oneAPI runtime)
      * rocm, Linux     -> upstream ROCm wheels (broad gfx support)
      * rocm, Windows, gfx103X (RX 6000 / RDNA2) -> localm's bundled self-contained
                           build, the ``[gpu]`` extra (no upstream Windows wheel for it)
      * rocm, Windows, gfx110X / gfx120X (RX 7000 / 9000) -> AMD's Windows ROCm
                           wheels (public preview)
      * rocm, Windows, other / unknown AMD       -> "" (no verified prebuilt; the
                           installer skips and prints how to add torch by hand)
      * none            -> ""

    Routing is unit-tested. Actual EXECUTION of the RX 7000/9000 Windows wheels is
    pending real RDNA3/RDNA4 hardware (this dev box is gfx1030); the gfx103X path is
    the verified one."""
    variant = recommended_torch_variant(backend, det)
    if variant == "cuda":
        return f"torch torchvision --index-url {_TORCH_INDEX['cuda']}"
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
    """``python -m localm.hwdetect`` -> prints '<primary-vendor-or-none> <install-backend>'
    on one line, so the shell installers (setup.bat / setup.sh) share this single
    tested detector + policy instead of each rolling their own.

    ``python -m localm.hwdetect torch <backend>`` -> prints the HF torch variant
    ("cuda" | "rocm" | "xpu" | "none") for that backend on this machine.

    ``python -m localm.hwdetect torch-args <backend>`` -> prints the exact
    ``uv pip install`` arguments for the torch wheel SOURCE on this machine
    (resolving AMD-on-Windows per gfx family), or "" to skip. The installers use
    this so the torch stack matches the user's runtime choice + hardware (SETUP-1)."""
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
