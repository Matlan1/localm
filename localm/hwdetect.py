"""Best-effort accelerator detection for install-time and run-time backend choice.

Pure stdlib, and NEVER raises: detection is advisory, so a probe that fails just
contributes nothing. ``detect()`` reports which GPU vendors are present and the
recommended llama.cpp backend.

Backend-selection policy (the "anybody, out of the box" rule):
  * ``vulkan`` is the universal default whenever ANY GPU is present - it runs on
    AMD, NVIDIA, and Intel through the vendor's normal display driver, with no
    CUDA/ROCm/oneAPI toolkit to install.
  * vendor-specific backends (``cuda`` for NVIDIA, ``hip`` for AMD, ``sycl`` for
    Intel) are offered as an opt-in for maximum performance, not the default.
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


def recommended_install_backend(det: "Detection | None" = None) -> str:
    """The backend the INSTALLER should provision by default - the ONE policy both
    setup.bat and setup.sh call, so the two detectors can never drift:
      * Apple Silicon                         -> metal
      * no GPU                                -> cpu
      * AMD on Windows, RX 6000 / unknown     -> amd-rocm  (self-contained gfx103X build)
      * AMD on Windows, clearly not RX 6000    -> vulkan    (gfx103X build won't fit)
      * any other GPU (NVIDIA, Intel, Linux    -> vulkan    (no vendor toolkit needed)
        AMD, mixed)
    The self-contained ROCm bundle is gfx103X + Windows only, hence the narrow amd-rocm case."""
    d = det or detect()
    if "apple" in d.vendors:
        return "metal"
    if not d.has_gpu:
        return "cpu"
    if (sys.platform == "win32" and d.vendors == ["amd"]
            and not _amd_known_non_gfx103x(d.gpu_names)):
        return "amd-rocm"
    return "vulkan"


def main(argv=None) -> int:
    """``python -m localm.hwdetect`` -> prints '<primary-vendor-or-none> <install-backend>'
    on one line, so the shell installers (setup.bat / setup.sh) share this single
    tested detector + policy instead of each rolling their own."""
    d = detect()
    vendor = d.vendors[0] if d.vendors else "none"
    print(f"{vendor} {recommended_install_backend(d)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
