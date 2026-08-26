# SPDX-License-Identifier: AGPL-3.0-or-later
"""Best-effort accelerator detection for install-time and run-time backend choice.

Pure stdlib, and NEVER raises: detection is advisory, so a probe that fails just
contributes nothing. ``detect()`` reports which GPU vendors are present and the
recommended llama.cpp backend.

Backend-selection policy - the best-performing backend this hardware actually
has, checked against what can genuinely be provisioned or is already present (a
self-contained bundle, an already-installed vendor toolkit, or a real prebuilt
binary this machine can run), before falling back to vulkan:
  * NVIDIA, any OS -> ``cuda``: llama.cpp ships a self-contained cudart bundle
    for both Windows and Linux (setup-llama fetches the CUDA runtime libraries
    itself, no Toolkit needed - see ``_provision_backend``'s cuda branch and
    ``NvidiaInfo.cuda_line`` for the architecture-aware build-line pick), so
    CUDA is out-of-the-box and the fastest path on NVIDIA regardless of
    platform. setup-llama's load-test plus vulkan fallback
    (``_provision_with_fallback``) covers a bad guess on any platform.
  * AMD, any OS -> ``hip`` when a working system ROCm/HIP toolkit is DETECTED
    present (``_rocm_toolkit_present`` - the same ``rocminfo``/``rocm-smi``/
    ``/opt/rocm`` signal ``detect()`` probes for vendor identification), else
    ``vulkan``. ``hip`` is a real downloadable prebuilt binary on both platforms
    (``setup_llama._BACKEND_ASSETS``). gfx103X on Windows keeps the
    self-contained ``amd-rocm`` bundle regardless, since it needs no system
    toolkit at all and so beats ``hip`` even when both are viable.
  * ``vulkan`` is the ONE catch-all for a GPU with no better path detected as
    actually runnable here (Intel - no toolkit-presence probe exists yet for
    oneAPI; AMD with no ROCm/HIP toolkit found; any mixed/unrecognised box).
  * ``cpu`` when no GPU is detected.

Detection is conservative: a missing tool or an unparseable name means
"unknown", never a crash and never a wrong-vendor claim.
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
        """``"found"`` | ``"none"`` | ``"unknown"`` - the three outcomes this
        module can genuinely distinguish, kept apart because they need different
        responses.

        ``has_gpu`` is a BOOLEAN and cannot express the third: it reads False
        both for a box with no GPU and for a box that could not be asked, so a
        caller branching on it treats an unanswered probe as proof of absent
        hardware. Prefer this property over ``has_gpu`` anywhere the answer is
        shown to a user or recorded in a diagnostic.

        A vendor found by a tool that DID answer (nvidia-smi / rocm-smi /
        rocminfo on PATH) is positive proof and stays ``"found"`` even when the
        adapter-name enumeration failed.

        NOT the same vocabulary as ``discover.GPU_PROBE_OK`` / ``_TIMEOUT`` /
        ``_BUSY`` / ``_INCONCLUSIVE``, and not shared with it: that one describes
        the RUNTIME torch/nvidia-smi device enumeration (an outcome per probe
        attempt), this one the INSTALL-TIME adapter-name detection (a conclusion
        about the machine). They cannot be unified: this module is pure stdlib,
        because ``setup.sh`` and ``setup.bat`` invoke it via
        ``python -m localm.hwdetect``, while ``discover`` imports ``localm.vram``
        and ``model_manager``."""
        if self.vendors:
            return "found"
        return "none" if self.probe_ok else "unknown"


def _run_ok(cmd: list) -> "tuple[str, bool]":
    """Run *cmd*; return (combined stdout+stderr, ran_to_completion).

    The flag reports exactly one thing: the process STARTED, did not time out,
    and exited 0. It is not a claim that the OUTPUT is trustworthy - powershell
    in particular can exit 0 having written a cmdlet error to stderr, which this
    function still returns as text, and ``detect``'s substring matching
    tolerates that."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
    except Exception:
        return "", False
    return (r.stdout or "") + (r.stderr or ""), r.returncode == 0


def _run(cmd: list) -> str:
    """Run *cmd* and return combined stdout+stderr, or "" on any failure.

    A thin wrapper over ``_run_ok``, so both share one subprocess result."""
    return _run_ok(cmd)[0]


def _win_gpu_names() -> "tuple[str, bool]":
    """Names of the Windows display adapters (lowercased), best-effort, plus
    whether either enumeration tool actually answered.

    ``ok`` is True when EITHER tool ran to completion: powershell exiting 0 with
    no adapters is a real answer ("this box has none"), so a subsequent wmic
    failure must not downgrade it."""
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
    """Names of Linux PCI display controllers (lowercased), best-effort, plus
    whether ``lspci`` actually answered (it is absent on plenty of minimal
    installs and containers, where "no display controllers" would be a fabricated
    answer rather than a measured one)."""
    out, ok = _run_ok(["lspci"])
    return "\n".join(
        ln for ln in out.lower().splitlines()
        if "vga" in ln or "3d controller" in ln or "display" in ln
    ), ok


def detect() -> Detection:
    """Detect GPU vendors present and recommend a default llama.cpp backend.

    Never raises. Vendor order in the result is priority order (nvidia, amd,
    intel) for picking a vendor-optimized backend if the user opts in.

    An enumeration that could not RUN is reported as such (``probe_ok`` False,
    ``gpu_state`` "unknown") instead of being rendered as "no GPU found". The
    recommendation is unchanged in that case - ``cpu`` remains the only safe
    default when nothing is known - but the caller can now tell a measured
    absence from an unanswered question, which is the difference between "this
    box has no GPU" and "we never got to look"."""
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
    """True when the AMD adapter name CLEARLY indicates a family the self-contained
    gfx103X (RDNA2 / RX 6000) ROCm build does NOT cover: RDNA1 (RX 5000), RDNA3
    (RX 7000), RDNA4 (RX 9000), Radeon VII, Instinct. Conservative - an unknown or
    RX 6000 name returns False so the self-contained build stays the default (the
    setup-llama load-test + vulkan fallback is the net if this guess is ever wrong)."""
    return any(tag in names for tag in
               ("rx 5", "rx 7", "rx 9", "radeon vii", "instinct"))


def _rocm_toolkit_present() -> bool:
    """Whether a working system ROCm/HIP toolkit is installed on THIS machine,
    independent of GPU-name matching - the SAME signal ``detect()`` probes
    (``rocminfo``/``rocm-smi`` on both platforms, ``/opt/rocm`` on Linux) to
    decide "is amd present", factored out so ``recommended_install_backend()``
    can act on it directly. The ``hip`` llama.cpp backend is a real downloadable
    prebuilt binary on both platforms (see ``setup_llama._BACKEND_ASSETS``) that
    NEEDS this toolkit present to load. Best-effort like the rest of this module
    - False on any failure, never raises."""
    try:
        if shutil.which("rocminfo") or shutil.which("rocm-smi"):
            return True
        if sys.platform != "win32" and Path("/opt/rocm").is_dir():  # hygiene-ok: generic ROCm system path
            return True
    except Exception:
        return False
    return False


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
    setup.bat and setup.sh call:
      * Apple Silicon                         -> metal
      * no GPU                                -> cpu
      * NVIDIA, any OS                        -> cuda      (llama.cpp ships a
        self-contained cudart bundle on both Windows and Linux, so it is
        out-of-the-box with no CUDA Toolkit on either, and setup-llama's driver
        preflight + load-test fall back to vulkan if the driver is too old or the
        build fails to load)
      * AMD, Windows, RX 6000 / unknown       -> amd-rocm  (self-contained gfx103X
        build; needs no system toolkit at all, so it wins even when a system ROCm
        install is ALSO present)
      * AMD, any OS, elsewhere, WITH a working system ROCm/HIP toolkit detected
        (see ``_rocm_toolkit_present``)       -> hip       (a real downloadable
        prebuilt binary on both platforms - see ``setup_llama._BACKEND_ASSETS`` -
        that needs this toolkit to load; covers Linux AMD and Windows
        gfx110X/gfx120X, where no SELF-CONTAINED build exists)
      * AMD, any OS, elsewhere, WITHOUT a detected toolkit -> vulkan (the vendor
        path cannot run here)
      * Intel, any OS                         -> vulkan    (``sycl`` is a real
        downloadable binary too, but there is no toolkit-presence probe for
        oneAPI yet, so it stays opt-in)
    The self-contained ROCm bundle is gfx103X + Windows only; self-contained CUDA
    is both-OS, so only the AMD gfx103X case is narrowed to Windows."""
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
    """The PyTorch variant the INSTALLER should provision for the HuggingFace /
    transformers backend, given the user's chosen llama.cpp *backend* and the
    detected hardware. Returns "cuda" | "rocm" | "xpu" | "none". Both setup.bat and
    setup.sh call this (via ``python -m localm.hwdetect torch <backend>``) AFTER the
    backend pick, so the GGUF runtime choice and the HF torch choice stay coherent.

    The chosen BACKEND drives this, not the detected vendor alone: a user who
    picks the vendor-neutral ``vulkan`` runtime on an AMD box has stepped OFF the
    ROCm path, so ROCm torch is never forced on them. Policy:

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
    CPU / CUDA / ROCm / Intel-XPU torch themselves."""
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
    """Every NVIDIA GPU's compute capability on this machine, as comparable
    tuples (one per card; multi-GPU boxes report one line per device).
    Best-effort like the rest of this module - [] on any failure, never raises.
    A small local probe (mirroring setup_llama.nvidia_preflight's own nvidia-smi
    query) rather than an import from setup_llama: this module runs standalone
    via ``python -m localm.hwdetect torch-args <backend>``, a SEPARATE process
    from ``localm setup-llama``, and stays dependency-free."""
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
    """The PyTorch wheel index URL for a torch *variant* key ("cuda" | "xpu" |
    "rocm-linux" | "rocm-win" | "cpu"), or None if unknown. Public accessor, so
    other callers (the managed-ComfyUI fresh install picks the ComfyUI torch
    here) share this ONE index table.

    For "cuda" specifically, this ALSO detects whether any installed NVIDIA GPU
    needs the Blackwell-and-newer wheel line and returns that instead of the
    cu126 default. Self-contained here rather than requiring every caller to pass
    compute-capability info through, so ComfyUI's torch install and the HF
    backend's get the same line; a flat cu126 on Blackwell loads but has no
    matching kernels and runs CPU-only."""
    if variant == "cuda":
        caps = _cuda_compute_capabilities()
        if any(cap >= _CUDA_BLACKWELL_MIN_CAP for cap in caps):
            return _TORCH_INDEX["cuda-blackwell"]
    return _TORCH_INDEX.get(variant)


def torch_pip_args(backend: str, det: "Detection | None" = None) -> str:
    """The exact ``uv pip install -p .venv <ARGS>`` arguments to provision the HF
    PyTorch stack for *backend* on THIS machine - or "" when no verified prebuilt
    exists and the installer should skip and guide the user. SINGLE source of truth
    for the torch wheel SOURCE (setup.bat and setup.sh both consult it via
    ``python -m localm.hwdetect torch-args <backend>``), so the gfx-specific
    AMD-on-Windows routing lives in one place.

      * cuda            -> CUDA wheels (cu126, or the Blackwell-and-newer line
                           when this machine has one - see pytorch_index_url);
                           any OS with NVIDIA
      * xpu             -> Intel wheels (self-provision the oneAPI runtime)
      * rocm, Linux     -> upstream ROCm wheels (broad gfx support)
      * rocm, Windows, gfx103X (RX 6000 / RDNA2) -> localm's bundled self-contained
                           build, the ``[gpu]`` extra (no upstream Windows wheel for it)
      * rocm, Windows, gfx110X / gfx120X (RX 7000 / 9000) -> AMD's Windows ROCm
                           wheels (public preview)
      * rocm, Windows, other / unknown AMD       -> "" (no verified prebuilt; the
                           installer skips and prints how to add torch by hand)
      * none            -> ""
    """
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
    """``python -m localm.hwdetect`` -> prints '<primary-vendor-or-none> <install-backend>'
    on one line, for the shell installers (setup.bat / setup.sh).

    ``python -m localm.hwdetect torch <backend>`` -> prints the HF torch variant
    ("cuda" | "rocm" | "xpu" | "none") for that backend on this machine.

    ``python -m localm.hwdetect torch-args <backend>`` -> prints the exact
    ``uv pip install`` arguments for the torch wheel SOURCE on this machine
    (resolving AMD-on-Windows per gfx family), or "" to skip."""
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
