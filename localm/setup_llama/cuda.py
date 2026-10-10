# SPDX-License-Identifier: AGPL-3.0-or-later
"""NVIDIA detection, the CUDA asset line a GPU needs, the CUDA runtime libraries
fetched from PyPI on Linux, and the interactive CUDA dialogue.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import click

from localm.debuglog import logger
from localm.setup_llama._common import console
from localm.setup_llama.download import _extract_archive, _validate_archive, ArtifactError
import localm.setup_llama as _sl

# --------------------------------------------------------------------------- #
#  NVIDIA / CUDA preflight + self-assembly                                     #
#                                                                              #
#  CUDA is the visible "peak NVIDIA performance" option, so picking it has to  #
#  LAND. The CUDA llama build needs the CUDA *runtime* libraries (cudart /     #
#  cublas) at load time. Upstream ships a self-contained                       #
#  ``cudart-llama-bin-win-cuda-<ver>`` bundle in the SAME release, so we make  #
#  CUDA work WITHOUT the user installing the full CUDA Toolkit: fetch the      #
#  build + the matching cudart bundle into the same lib dir. The one thing we  #
#  cannot self-assemble is the GPU DRIVER (a system component needing admin +  #
#  a reboot); a too-old driver is the single "you must do this part" branch.   #
# --------------------------------------------------------------------------- #

# Which upstream CUDA asset line to fetch is a function of the GPU's
# ARCHITECTURE, not just the platform: upstream ships both a 12.x line (broad
# compatibility - runs on any driver new enough for CUDA 12.4) and a 13.x line
# (needed for Blackwell-class GPUs, but itself needing a newer driver and
# dropping some pre-Turing arch support). NVIDIA Blackwell (datacenter sm_100,
# consumer/workstation sm_120 - e.g. RTX 50-series) is not supported by our
# pinned 12.4 build's fatbin: upstream only added Blackwell kernels starting
# CUDA 12.8, and our two pinned lines are 12.4 and 13.4. So a Blackwell card
# needs the 13.x line; every older architecture stays on 12.x, the
# broad-compatibility default. _CUDA_LINE is the fallback used when no
# architecture information is available at all (see NvidiaInfo.cuda_line).
_CUDA_LINE = "cuda-12"


# Compute-capability floor for "needs the 13.x line" (nvidia-smi's
# ``compute_cap`` query, e.g. "8.9", "12.0" - the GPU's sm/arch level, NOT the
# driver's max CUDA version). Per NVIDIA's published architecture numbers,
# Blackwell datacenter parts (B100/B200/GB100) report compute capability 10.0
# (sm_100) and Blackwell consumer/workstation parts (RTX 5090/5080/5070
# Ti/5070/5060) report 12.0 (sm_120); CUDA 12.8 was the first toolkit release
# to add Blackwell kernels. This is NOT verified against real Blackwell
# hardware (none is available here) - only the offline selection logic below
# is. >= 10.0 catches both variants and any later architecture
# without a new special case each time.
_BLACKWELL_MIN_CAP = (10, 0)


# Minimum driver-reported CUDA version ("cuda_capability") to trust each
# line's build, keyed by the line itself since a newer line needs a newer
# driver. Both match the PINNED asset's own X.Y (not just its major version):
# the 12.4 entry is the original, long-verified threshold; the 13.4 entry
# mirrors that same convention rather than relying on CUDA's minor-version-
# compatibility guarantee across the (very recent) 13.x series, which there is
# no Blackwell hardware here to confirm against directly. Being exact-match
# here is the conservative side of that unknown: it can only route a
# borderline driver to the safe Vulkan fallback, never hand it a build that
# fails to load.
_MIN_DRIVER_CUDA = {
    "cuda-12": (12, 4),
    "cuda-13": (13, 4),
}


def _ver_tuple(v: str) -> Optional[tuple]:
    # Return None (not (0,0)) on an unparseable version so an unmeasurable
    # capability reads as "unknown", never as a too-old driver we falsely block.
    try:
        return tuple(int(x) for x in str(v).split(".")[:2])
    except Exception:
        return None


def _ver_at_least(parsed: tuple, minimum: tuple) -> bool:
    """*parsed* >= *minimum*, treating a bare-major version (no minor
    component, e.g. "10" -> (10,)) as ".0". Plain tuple comparison would
    otherwise get this wrong: Python considers a tuple that is a strict
    PREFIX of another to be the smaller one regardless of the missing
    component's value, so (10,) >= (10, 0) is False even though 10 == 10.
    _ver_tuple's own contract (a bare major parses to a 1-element tuple, not
    padded) is intentional and unchanged - this is where the padding belongs,
    at the comparison, not the parse."""
    padded = parsed + (0,) * (len(minimum) - len(parsed))
    return padded >= minimum


@dataclass
class NvidiaInfo:
    """What nvidia-smi told us. Advisory only; every field may be empty."""
    present: bool = False           # an NVIDIA GPU + usable driver was found
    gpu_name: str = ""
    driver_version: str = ""
    cuda_capability: str = ""       # max CUDA the driver supports, e.g. "12.4"
    compute_capability: str = ""    # the GPU's own sm/arch level, e.g. "12.0" (Blackwell/sm_120)
    # Every GPU nvidia-smi lists, in nvidia-smi's own order:
    # [{"index", "name", "total_mib", "free_mib"}, ...].
    gpus: list = field(default_factory=list)

    @property
    def cuda_line(self) -> str:
        """Which upstream CUDA asset line this GPU's ARCHITECTURE needs:
        'cuda-12' (broad-compatibility default) or 'cuda-13' (required for
        Blackwell and newer - see _BLACKWELL_MIN_CAP). Unknown or unparseable
        capability stays on cuda-12: an unmeasured architecture is not
        evidence it needs the newer, narrower-compatibility line (same
        "unknown != too old" reasoning as driver_ok below)."""
        cap = _ver_tuple(self.compute_capability)
        if cap is not None and _ver_at_least(cap, _BLACKWELL_MIN_CAP):
            return "cuda-13"
        return "cuda-12"

    @property
    def driver_ok(self) -> bool:
        """True when the driver is new enough for the CUDA line THIS GPU's
        architecture needs (see cuda_line) - the minimum is not a single fixed
        threshold, since Blackwell and older cards need different lines.
        Unknown driver capability is treated as OK (do not block on a parse
        miss)."""
        if not self.cuda_capability:
            return True
        parsed = _ver_tuple(self.cuda_capability)
        # An unparseable capability is unknown, not old: cannot judge, do not block.
        if parsed is None:
            return True
        return _ver_at_least(parsed, _MIN_DRIVER_CUDA[self.cuda_line])


def _nvidia_smi(*args: str) -> str:
    """Combined nvidia-smi output, or "" if it is not present/usable."""
    exe = shutil.which("nvidia-smi") or "nvidia-smi"
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=8)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def nvidia_preflight() -> NvidiaInfo:
    """Detect the NVIDIA GPU + driver, the max CUDA version the DRIVER
    supports, and the GPU's own compute capability (its architecture - e.g.
    "12.0" for Blackwell/sm_120). These are two different questions: the
    driver's version says what it CAN run; the compute capability says what
    the CARD IS, which is what decides whether the 12.x build's fatbin even
    has kernels for it (see NvidiaInfo.cuda_line).

    Never raises. Asks explicitly for the driver version, the (untruncated)
    GPU name, the compute capability and every GPU's memory. The max CUDA
    version has no query field, so it comes from the banner ("CUDA Version:
    Y"), or from ``nvidia-smi -q`` when the banner does not carry it; the
    driver version falls back to the banner the same way. A field that still
    cannot be parsed stays empty and the start of the raw output is logged at
    debug."""
    out = _sl._nvidia_smi()
    if not out.strip():
        return NvidiaInfo(present=False)
    info = NvidiaInfo(present=True)
    info.driver_version = _first_version_line(
        _sl._nvidia_smi("--query-gpu=driver_version", "--format=csv,noheader"))
    long_out = ""
    for text in (out, None):
        if text is None:
            if info.driver_version and info.cuda_capability:
                break
            long_out = _sl._nvidia_smi("-q")
            text = long_out
        if not info.driver_version:
            m = _DRIVER_VERSION_RE.search(text)
            if m:
                info.driver_version = m.group(1)
        if not info.cuda_capability:
            m = _CUDA_VERSION_RE.search(text)
            if m:
                info.cuda_capability = m.group(1)
    name = _sl._nvidia_smi("--query-gpu=name", "--format=csv,noheader").strip().splitlines()
    if name:
        info.gpu_name = name[0].strip()
    cap = _sl._nvidia_smi("--query-gpu=compute_cap", "--format=csv,noheader").strip().splitlines()
    if cap:
        info.compute_capability = cap[0].strip()
    info.gpus = _parse_gpu_memory(_sl._nvidia_smi(
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits"))
    missing = [label for label, value in (("driver version", info.driver_version),
                                          ("CUDA version", info.cuda_capability),
                                          ("GPU name", info.gpu_name))
               if not value]
    if missing:
        logger.debug("nvidia-smi: could not parse %s; start of its output: %r; "
                     "start of 'nvidia-smi -q': %r", ", ".join(missing),
                     out[:_RAW_LOG_CHARS], long_out[:_RAW_LOG_CHARS])
    return info


# How much raw nvidia-smi output nvidia_preflight logs when a field is missing.
_RAW_LOG_CHARS = 800

# "Driver Version: 552.22" in the banner, "Driver Version    : 552.22" in -q.
_DRIVER_VERSION_RE = re.compile(r"Driver Version[ \t]*:[ \t]*([0-9]+(?:\.[0-9]+)+)")
_CUDA_VERSION_RE = re.compile(r"CUDA Version[ \t]*:[ \t]*([0-9]+\.[0-9]+)")
_VERSION_LINE_RE = re.compile(r"[0-9]+(?:\.[0-9]+)+")


def _first_version_line(text: str) -> str:
    """The first line of *text* when it is a dotted version number, else ""
    (an error sentence from an nvidia-smi that rejects the query field)."""
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line if _VERSION_LINE_RE.fullmatch(line) else ""
    return ""


def _parse_gpu_memory(text: str) -> list:
    """``[{"index", "name", "total_mib", "free_mib"}, ...]`` from nvidia-smi's
    ``--query-gpu=index,name,memory.total,memory.free
    --format=csv,noheader,nounits`` output. A line that does not parse is
    skipped; an error sentence parses to ``[]``."""
    out = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            out.append({"index": int(parts[0]), "name": parts[1],
                        "total_mib": int(parts[2]), "free_mib": int(parts[3])})
        except ValueError:
            continue
    return out


# NVIDIA publishes its own CUDA runtime libraries (cudart, cuBLAS) as plain
# PyPI wheels - the SAME channel localm's own HF/torch backend already
# depends on for the identical libraries (recommended_torch_variant's cu126
# index, hwdetect.py). NVIDIA renamed the CUDA-13-line packages to be
# UNSUFFIXED (nvidia-cublas-cu13 etc. are now deprecated stubs pointing at
# bare nvidia-cublas) - both lines are listed explicitly so neither naming
# scheme is guessed at.
#
# NCCL is NOT included: the binary this fetches alongside
# (hybridgroup/llama-cpp-builder's Linux CUDA build, see _CUDA_LINUX_REPO) does
# not link against it - none of its shared libraries reference "libnccl". A
# different CUDA build may, so re-check the binary actually in use before
# adding nccl back.
_CUDA_RUNTIME_PYPI_PACKAGES = {
    "cuda-12": ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"),
    "cuda-13": ("nvidia-cuda-runtime", "nvidia-cublas"),
}

# package -> (version, sha256 of its Linux x86_64 wheel). The wheel fetched is
# exactly this version, verified against this digest.
_CUDA_RUNTIME_PIN = {
    "nvidia-cuda-runtime-cu12": ("12.9.79",
        "25bba2dfb01d48a9b59ca474a1ac43c6ebf7011f1b0b8cc44f54eb6ac48a96c3"),
    "nvidia-cublas-cu12": ("12.9.2.10",
        "e4f53a8ca8c5d6e8c492d0d0a3d565ecb59a751b19cfdaa4f6da0ab2104c1702"),
    "nvidia-cuda-runtime": ("13.4.92",
        "9641f797da20ce1dd8e779b6e96d08cf9ba564cec8e8225458811ee26423f3a5"),
    "nvidia-cublas": ("13.8.1.7",
        "c11a27fd4379510e5b1f84b367a2514d1e52fe5cc13442117a0e0a1addee3cf2"),
}


def _pypi_wheel_url_and_sha(package: str) -> tuple:
    """The (url, sha256) of *package*'s pinned Linux x86_64 wheel from PyPI's
    JSON API, or (None, None) when *package* has no entry in _CUDA_RUNTIME_PIN
    or the lookup fails. The digest returned is the pin's own, so the download
    is verified against the pin and not against whatever PyPI reports. Never
    raises (mirrors _release_assets' contract: a best-effort lookup whose
    caller always has a fallback path, so a network hiccup here must not crash
    setup)."""
    pinned = _sl._CUDA_RUNTIME_PIN.get(package)
    if pinned is None:
        return None, None
    version, sha = pinned
    api = f"https://pypi.org/pypi/{package}/{version}/json"
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/json",
                                                    "User-Agent": "localm-setup-llama"})
        with _sl.verified_urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        for f in data["urls"]:
            name = str(f.get("filename", ""))
            if name.endswith(".whl") and "x86_64" in name and "linux" in name.lower():
                url = f.get("url")
                if url:
                    return url, sha
    except Exception as e:
        logger.debug("PyPI wheel lookup failed for %s (%s)", package, e)
    return None, None


def _fetch_pypi_runtime_lib(package: str, target: Path) -> int:
    """Download *package*'s Linux wheel from PyPI, verify it, and copy every
    ``.so*`` file it contains into *target* (flat - matches how the llama.cpp
    runtime dir is already laid out). Returns the number of files copied.
    Raises :class:`ArtifactError` on a download or validation failure, same
    contract as :func:`_fetch_and_place`, so callers decide fatal-vs-fallback
    identically for either artifact source.

    NEVER reads from any OTHER environment already on the user's machine: this
    always fetches and places a PRIVATE copy into *target*, exactly like the
    Windows cudart bundle, which is never "detected" on the user's system, only
    ever fetched fresh."""
    url, sha = _sl._pypi_wheel_url_and_sha(package)
    if url is None:
        raise ArtifactError(f"could not resolve a PyPI Linux wheel for {package!r}")
    with tempfile.TemporaryDirectory() as tmp:
        wheel = Path(tmp) / f"{package}.whl"
        dl = _sl._download(url, wheel)
        _validate_archive(wheel, expected_sha256=sha, dl=dl)   # a wheel is a zip
        ex = Path(tmp) / "x"
        _extract_archive(wheel, ex)
        n = 0
        for f in sorted(ex.rglob("*.so*")):
            if f.is_file() and not f.is_symlink():
                shutil.copy2(f, target / f.name)
                n += 1
        return n


def _fetch_cuda_runtime_libs(cuda_line: str, target: Path) -> int:
    """Fetch every PyPI-hosted CUDA runtime library for *cuda_line* ('cuda-12'
    or 'cuda-13') into *target*. Returns the total files copied. Raises
    ArtifactError (from the first failing package) on any failure - a
    partially-assembled CUDA runtime is worse than none, so this does not
    swallow a single package's failure and continue with the rest."""
    packages = _CUDA_RUNTIME_PYPI_PACKAGES.get(cuda_line, ())
    total = 0
    for pkg in packages:
        console.print(f"[dim]Fetching CUDA runtime library:[/dim] {pkg}")
        total += _sl._fetch_pypi_runtime_lib(pkg, target)
    return total


def _cuda_setup_dialogue(info: NvidiaInfo, assume_yes: bool, det=None) -> tuple:
    """Given the preflight, walk the user through making CUDA land. Returns
    ``(backend_to_provision, fetch_cudart_bundle)``.

    Branches:
      * driver new enough  -> offer the self-contained build+runtime fetch
        (default yes); declining falls back to vulkan.
      * driver too old     -> a driver cannot be self-assembled; recommend
        vulkan now and tell them how to enable CUDA later.
      * no NVIDIA detected, hardware unknown -> the user forced cuda;
        confirm-continue, else vulkan (warn-once, do not block).
      * no NVIDIA detected, but *det* (the SAME hwdetect.Detection
        _warn_off_profile already computed) shows a DIFFERENT vendor is
        actually present -> name it, recommend the real policy-backed match
        for it (hwdetect.recommended_install_backend - not a hardcoded
        vulkan regardless of hardware), and offer a genuine three-way choice
        (continue / switch to the recommendation / quit) instead of a binary
        confirm whose "no" silently imposes vulkan either way.

    *det* is optional and changes nothing when it is None or shows no vendor
    other than nvidia (the vast majority of existing callers/tests) - the
    dialogue falls back to the original generic behaviour unchanged.
    """
    console.print("[bold]CUDA selected[/bold] (peak NVIDIA performance). "
                  "Checking your system...")
    other_vendors = [v for v in (det.vendors if det else []) if v != "nvidia"]
    if info.present:
        console.print(f"  [green]OK[/green] NVIDIA GPU: {info.gpu_name or 'detected'}")
        if info.compute_capability:
            line_note = " (Blackwell)" if info.cuda_line == "cuda-13" else ""
            console.print(f"  [dim]Compute capability {info.compute_capability}{line_note} "
                          f"-> {info.cuda_line} line[/dim]")
        if info.cuda_capability:
            colour = "green" if info.driver_ok else "red"
            mark = "OK " if info.driver_ok else "no "
            need = _MIN_DRIVER_CUDA[info.cuda_line]
            console.print(f"  [{colour}]{mark}[/{colour}] Driver {info.driver_version} "
                          f"supports CUDA {info.cuda_capability} "
                          f"(need >= {need[0]}.{need[1]} for the {info.cuda_line} line)")
    elif other_vendors:
        seen = "/".join(v.upper() for v in other_vendors)
        console.print(f"  [yellow]?[/yellow] Could not run nvidia-smi - this machine "
                      f"looks like [bold]{seen}[/bold], not NVIDIA.")
    else:
        console.print("  [yellow]?[/yellow] Could not run nvidia-smi - no NVIDIA driver "
                      "detected here (or it is not on PATH).")

    # Driver too old: the one thing we cannot fetch for the user.
    if info.present and info.cuda_capability and not info.driver_ok:
        console.print(f"  GPU driver update required for CUDA "
                      f"(the {info.cuda_line} line this GPU needs).")
        console.print("  [dim]To enable later: update driver, reboot, run setup-llama --backend cuda[/dim]")
        console.print("  [green]Using Vulkan now[/green].")
        return "vulkan", False

    # No NVIDIA detected, but the user explicitly asked for cuda: their call.
    if not info.present:
        if not other_vendors:
            # No specific alternative to recommend - the original generic
            # continue-or-vulkan choice (also what a fully headless machine,
            # or one where hwdetect itself failed, falls back to).
            if assume_yes:
                console.print("  [dim]--yes: using Vulkan (no NVIDIA GPU detected).[/dim]")
                return "vulkan", False
            _sl._flush_stdin()
            if click.confirm("  Continue with CUDA anyway? (No = use Vulkan)", default=False):
                return "cuda", True
            return "vulkan", False

        # We KNOW what IS actually here - recommend the real match for it
        # rather than a hardcoded vulkan, via the SAME policy setup.bat/sh use.
        from localm import hwdetect as _hwdetect
        recommended = _hwdetect.recommended_install_backend(det)
        if assume_yes:
            seen = "/".join(v.upper() for v in other_vendors)
            console.print(f"  [dim]--yes: using {recommended} "
                          f"({seen} detected, not NVIDIA).[/dim]")
            return recommended, False
        _sl._flush_stdin()
        console.print("    [1] Continue with CUDA anyway")
        console.print(f"    [2] Switch to {recommended} (recommended for your hardware)")
        console.print("    [3] Quit")
        pick = click.prompt("  Pick 1-3", type=click.Choice(["1", "2", "3"]), default="2")
        if pick == "1":
            return "cuda", True
        if pick == "3":
            sys.exit(1)
        return recommended, False

    # Driver OK (or capability unknown but a GPU is present): offer the fetch.
    console.print("  [yellow]i[/yellow] Fetching self-contained CUDA runtime bundle. [bold]No Toolkit needed[/bold].")
    _sl._flush_stdin()
    if assume_yes or click.confirm("  Download the CUDA build + runtime now?", default=True):
        return "cuda", True
    console.print("  [dim]Falling back to Vulkan (works on your driver).[/dim]")
    return "vulkan", False


# --------------------------------------------------------------------------- #
#  Staged CUDA runtimes (container images built on a host without a GPU)       #
# --------------------------------------------------------------------------- #

STAGED_NOTE = ".localm-cuda-staged"

# Container image tag that carries each CUDA asset line.
IMAGE_TAG_FOR_LINE = {"cuda-12": "cuda", "cuda-13": "cuda13"}

# Exit status of cuda_container_check when the staged runtime cannot be used.
CUDA_CHECK_FAILED = 3


def record_staged_cuda(target: Path, cuda_line: str) -> None:
    """Record in *target* that its CUDA runtime was fetched for *cuda_line*
    without a GPU, so its load has not been tested. Raises ``OSError`` when the
    note cannot be written: the container start check depends on it."""
    (target / STAGED_NOTE).write_text(cuda_line + "\n", encoding="utf-8")


def staged_cuda_line(target: Path) -> Optional[str]:
    """The CUDA line recorded by :func:`record_staged_cuda`, or None when
    *target* holds no staged runtime (no note, or one naming an unknown line)."""
    try:
        line = (target / STAGED_NOTE).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return line if line in _MIN_DRIVER_CUDA else None


def check_staged_cuda_runtime(target: Path, allow_no_gpu: bool = False) -> tuple:
    """Whether the staged CUDA runtime in *target* can be used on this machine.
    Returns ``(ok, lines)``: the lines to show the user either way. ``(True,
    [])`` when *target* holds no staged runtime.

    Fails when no NVIDIA GPU is visible (unless *allow_no_gpu*, which reports
    that the GPU is not in use and passes), when the GPU needs the other CUDA
    line, when the driver is too old for the staged line, and when the runtime
    does not load and register a GPU device. Never raises."""
    line = staged_cuda_line(target)
    if line is None:
        return True, []
    try:
        info = _sl.nvidia_preflight()
    except Exception as e:
        return False, [f"could not query the NVIDIA GPU: {e}"]
    if not info.present:
        if allow_no_gpu:
            return True, [f"no NVIDIA GPU is visible; the {line} runtime is not in use "
                          "(LOCALM_ALLOW_NO_GPU is set)"]
        return False, [
            "no NVIDIA GPU is visible to this container.",
            "Start it with the NVIDIA Container Toolkit installed and --gpus all "
            "(docker run --gpus all ...), or set LOCALM_ALLOW_NO_GPU=1 to start "
            "without GPU acceleration.",
        ]
    gpu = info.gpu_name or "NVIDIA GPU"
    if info.cuda_line != line:
        want = IMAGE_TAG_FOR_LINE[info.cuda_line]
        have = IMAGE_TAG_FOR_LINE[line]
        return False, [
            f"this image carries the {line} runtime ({have} tag), but {gpu} "
            f"(compute capability {info.compute_capability or 'unknown'}) needs the "
            f"{info.cuda_line} runtime.",
            f"Use the {want} image tag instead.",
        ]
    if not info.driver_ok:
        need = _MIN_DRIVER_CUDA[line]
        return False, [
            f"the host driver {info.driver_version or 'version unknown'} supports CUDA "
            f"{info.cuda_capability}, and the {line} runtime needs {need[0]}.{need[1]} "
            "or newer. Update the host NVIDIA driver."]
    ok, detail = _sl._native_gpu_loads_ok()
    if not ok:
        return False, [f"the {line} runtime did not load for {gpu}: {detail}"]
    return True, [f"CUDA runtime ({line}) loaded for {gpu}"]


def _check_image_record(target: Path) -> tuple:
    """When LOCALM_IMAGE_BACKEND names a CUDA image, require *target* to hold the
    staged-runtime record for that image's line. Returns ``(ok, lines)``; ``(True,
    [])`` outside a CUDA image."""
    tag = os.environ.get("LOCALM_IMAGE_BACKEND", "")
    expected = next((line for line, name in IMAGE_TAG_FOR_LINE.items() if name == tag), None)
    if expected is None:
        return True, []
    found = staged_cuda_line(target)
    if found == expected:
        return True, []
    return False, [
        f"the {tag} image's CUDA runtime record in {target} is "
        f"{'missing or unreadable' if found is None else 'for ' + found}, "
        f"expected {expected}. The image is damaged; pull it again."]


def cuda_container_check() -> int:
    """Entrypoint hook: run :func:`check_staged_cuda_runtime` for this install's
    runtime directory, print the outcome to stderr, and return 0 when the
    container may start or :data:`CUDA_CHECK_FAILED`. LOCALM_ALLOW_NO_GPU=1
    lets a container without a GPU start."""
    target = _sl._repo_runtime_lib()
    allow = os.environ.get("LOCALM_ALLOW_NO_GPU", "") == "1"
    ok, lines = _check_image_record(target)
    if ok:
        ok, lines = check_staged_cuda_runtime(target, allow_no_gpu=allow)
    for index, text in enumerate(lines):
        refusal = "refusing to start: " if index == 0 and not ok else ""
        print(f"localm: {refusal}{text}", file=sys.stderr)
    return 0 if ok else CUDA_CHECK_FAILED
