"""``localm setup-llama`` - provision the native llama.cpp binaries locally.

Makes localm self-contained: the native inference runtime (the llama shared
library + its ggml deps, plus a matched GPU runtime when the prebuilt ships one)
is placed inside the project's own ``localm-llama-runtime`` wheel rather than
depending on a folder elsewhere on disk.

Backends (``--backend``), so any machine has a working out-of-the-box path:
  * ``auto`` (default) - detect the GPU and pick the broadest WORKING backend:
    AMD -> the self-contained ROCm build; any other GPU -> ``vulkan`` (runs on
    NVIDIA/Intel/AMD through the normal display driver, no vendor toolkit);
    Apple Silicon -> ``metal``; no GPU -> ``cpu``.
  * ``vulkan`` - universal GPU build from upstream llama.cpp (recommended for
    NVIDIA/Intel, and a no-toolkit option for AMD).
  * ``cuda`` / ``sycl`` / ``cpu`` - upstream llama.cpp prebuilts. ``cuda`` and
    ``sycl`` deliver peak vendor performance but need that vendor's runtime
    (CUDA toolkit / oneAPI) present; ``vulkan`` and ``cpu`` are self-contained.
  * ``amd-rocm`` - the self-contained gfx103X (RDNA2) ROCm build (bundles its
    own ROCm runtime; the current default for AMD RX 6000).

Sources, in order of preference:
  * ``--from <dir>``  - copy from a local llama.cpp build output (any backend).
  * ``--url <url>``   - an explicit prebuilt archive URL.
  * ``--backend ...`` - resolve the matching upstream llama.cpp release asset
    (``ggml-org/llama.cpp``); the latest release is used, with a pinned
    fallback if the release lookup is unavailable.

After placing the files it installs the runtime wheel editable so the loader can
import it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

console = Console(highlight=False)

# Self-contained AMD build: lemonade-sdk llama.cpp ROCm build for gfx103X
# (RDNA2), Windows-only. Bundles its own ROCm runtime, so AMD RX 6000 users need
# no separate HIP SDK. See rocm-canary-forge/windows-native for the provenance.
DEFAULT_URL = (
    "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/"
    "b1288/llama-b1288-windows-rocm-gfx103X-x64.zip"
)

# Upstream llama.cpp prebuilts (ggml-org/llama.cpp). We resolve the latest
# release tag at runtime; this pin is the fallback if that lookup fails.
_UPSTREAM_REPO = "ggml-org/llama.cpp"
_FALLBACK_TAG = "b9682"

# Per-backend asset matcher: substrings that must appear in the release asset
# name for (platform, backend). Substring matching (not exact names) keeps this
# robust to upstream version suffixes drifting (e.g. cuda-12.4, rocm-7.2).
_ASSET_MATCH = {
    "win32": {
        "cpu":    ["bin-win-cpu-x64"],
        "vulkan": ["bin-win-vulkan-x64"],
        "cuda":   ["bin-win-cuda-12"],          # prefer the 12.x runtime line
        "sycl":   ["bin-win-sycl-x64"],
        "hip":    ["bin-win-hip-radeon-x64"],   # needs AMD HIP SDK present
    },
    "linux": {
        "cpu":    ["bin-ubuntu-x64"],
        "vulkan": ["bin-ubuntu-vulkan-x64"],
        "cuda":   ["bin-ubuntu-cuda"],
        "sycl":   ["bin-ubuntu-sycl-fp16", "bin-ubuntu-sycl"],
        "hip":    ["bin-ubuntu-rocm"],
    },
    "darwin": {
        "cpu":    ["bin-macos-arm64", "bin-macos-x64"],
        "metal":  ["bin-macos-arm64"],
    },
}

# Backends a user may request directly (in addition to the special "auto" and
# the self-contained "amd-rocm").
_UPSTREAM_BACKENDS = ("vulkan", "cuda", "sycl", "hip", "cpu", "metal")

# SEC-8: a prebuilt llama runtime archive is many megabytes. Anything below this
# floor is almost certainly an error page, a redirect stub, or a truncated
# transfer, never the real artifact. This is the always-on lower bound; the
# valid-archive structural check below is the second always-on guard. A pinned
# sha256 is the opt-in third guard (we do not hardcode a brittle hash for the
# live URLs, which move with every upstream release).
_MIN_ARTIFACT_BYTES = 256 * 1024   # 256 KiB


class ArtifactError(Exception):
    """A downloaded artifact failed integrity validation (size, archive shape,
    or a provided sha256 pin) and must NOT be extracted or installed."""


def _platform_key() -> str:
    if sys.platform == "win32":
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def _lib_name() -> str:
    """The loadable llama library filename for this platform."""
    if sys.platform == "win32":
        return "llama.dll"
    if sys.platform == "darwin":
        return "libllama.dylib"
    return "libllama.so"


def _is_wanted(f: Path) -> bool:
    """Whether to copy *f*: the loadable library, its ggml deps, and the runtime
    libraries - matched by platform-appropriate naming (incl. versioned .so.N)."""
    n = f.name.lower()
    if sys.platform == "win32":
        return n.endswith(".dll") or n.endswith(".exe")
    if sys.platform == "darwin":
        return n.endswith(".dylib")
    return ".so" in n          # libfoo.so and libfoo.so.1


def _repo_runtime_lib() -> Path:
    """The localm-llama-runtime wheel's lib/ dir."""
    try:
        import localm_llama_runtime
        return Path(localm_llama_runtime.LIB_DIR)
    except Exception:
        repo_root = Path(__file__).resolve().parent.parent
        return repo_root / "runtime" / "localm_llama_runtime" / "lib"


def _runtime_pkg_dir() -> Path:
    """The runtime wheel project dir (for `pip install -e`)."""
    return _repo_runtime_lib().parent.parent


# --------------------------------------------------------------------------- #
#  Backend resolution                                                          #
# --------------------------------------------------------------------------- #

def _auto_backend() -> str:
    """Pick the broadest WORKING backend for this machine (see module docstring).

    AMD keeps the self-contained ROCm build (no toolkit, current behaviour);
    every other GPU uses vulkan (no vendor toolkit needed); Apple Silicon uses
    metal; a machine with no GPU uses cpu."""
    try:
        from localm import hwdetect
        det = hwdetect.detect()
    except Exception:
        return "cpu"
    if not det.has_gpu:
        return "cpu" if sys.platform != "darwin" else det.recommended
    if det.vendors == ["amd"] and sys.platform == "win32":
        return "amd-rocm"                 # self-contained gfx103X build
    if "apple" in det.vendors:
        return "metal"
    return "vulkan"                       # NVIDIA / Intel / mixed: universal


def _latest_tag() -> str:
    """The latest ggml-org/llama.cpp release tag, or the pinned fallback if the
    lookup is unavailable (offline, rate-limited, etc.)."""
    api = f"https://api.github.com/repos/{_UPSTREAM_REPO}/releases/latest"
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "localm-setup-llama"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        tag = data.get("tag_name")
        if isinstance(tag, str) and tag:
            return tag
    except Exception:
        pass
    return _FALLBACK_TAG


def _resolve_backend_url(backend: str) -> str:
    """Resolve a backend name to a downloadable archive URL.

    ``amd-rocm`` is the self-contained lemonade build (special-cased). Every
    other backend maps to an upstream llama.cpp release asset for this platform.
    Raises ``click.ClickException`` if the backend is not available here."""
    if backend == "amd-rocm":
        if sys.platform != "win32":
            raise click.ClickException(
                "the self-contained 'amd-rocm' build is Windows-only; on Linux "
                "use --backend hip (needs ROCm) or build with --from.")
        return DEFAULT_URL

    plat = _platform_key()
    matchers = _ASSET_MATCH.get(plat, {}).get(backend)
    if not matchers:
        avail = ", ".join(sorted(_ASSET_MATCH.get(plat, {})))
        raise click.ClickException(
            f"backend {backend!r} is not available on this platform "
            f"({plat}). Available: {avail or 'none'}.")

    tag = _latest_tag()
    # Ask the release for its assets so we match the real (version-suffixed)
    # name; fall back to a templated guess if the asset list is unavailable.
    api = f"https://api.github.com/repos/{_UPSTREAM_REPO}/releases/tags/{tag}"
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "localm-setup-llama"})
        with urllib.request.urlopen(req, timeout=10) as r:
            assets = json.loads(r.read().decode("utf-8")).get("assets", [])
        for a in assets:
            name = str(a.get("name", "")).lower()
            if any(m in name for m in matchers) and a.get("browser_download_url"):
                return a["browser_download_url"]
    except Exception:
        pass
    # Fallback: construct the canonical URL from the first matcher token.
    stem = matchers[0]
    ext = "zip" if plat == "win32" else "tar.gz"
    fname = f"llama-{tag}-{stem.replace('bin-', 'bin-')}.{ext}"
    return (f"https://github.com/{_UPSTREAM_REPO}/releases/download/{tag}/{fname}")


# --------------------------------------------------------------------------- #
#  Download / validate / extract                                              #
# --------------------------------------------------------------------------- #

def _download(url: str, dest: Path) -> None:
    console.print(f"[dim]Downloading {url}[/dim]")
    last = [-1]

    def _hook(block: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, block * block_size * 100 // total)
        if pct != last[0] and pct % 5 == 0:
            last[0] = pct
            mb = total / 1024 ** 2
            console.print(f"[dim]  {pct:3d}%  ({mb:.0f} MB)[/dim]", end="\r")

    urllib.request.urlretrieve(url, dest, _hook)
    console.print()


def _sha256_file(path: Path) -> str:
    """Stream the file through sha256 so a multi-hundred-MB artifact is not
    read into memory at once."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_supported_archive(path: Path) -> bool:
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def _validate_archive(
    path: Path,
    expected_sha256: Optional[str] = None,
    min_size: int = _MIN_ARTIFACT_BYTES,
) -> None:
    """SEC-8: validate a downloaded artifact BEFORE it is extracted or installed.
    Raises :class:`ArtifactError` on any failure.

    Three checks, in cheapest-first order:
      1. size: a real prebuilt runtime archive is many MB; a tiny/empty body is
         an error page, a redirect stub, or a truncated transfer (always on).
      2. shape: it must be a structurally valid zip OR tar archive, so a
         200-with-HTML or a half-transferred file is rejected before we hand it
         to extraction (always on).
      3. provenance: when *expected_sha256* is given, the file's digest must
         match it (opt-in; refuses on mismatch). Comparison is whitespace- and
         case-insensitive so a pasted hash from any source works.
    """
    try:
        size = path.stat().st_size
    except OSError as e:
        raise ArtifactError(f"could not stat downloaded file: {e}") from e
    if size < min_size:
        raise ArtifactError(
            f"download is too small ({size} bytes < {min_size} minimum) - "
            "this is almost certainly an error page or a truncated transfer, "
            "not the prebuilt runtime."
        )
    if not _is_supported_archive(path):
        raise ArtifactError(
            "download is not a valid zip or tar archive - it may be a truncated "
            "transfer, an HTML error page served with a 200, or a tampered "
            "payload."
        )
    if expected_sha256:
        want = expected_sha256.strip().lower()
        got = _sha256_file(path)
        if got != want:
            raise ArtifactError(
                "download sha256 does not match the expected pin "
                f"(expected {want}, got {got}). Refusing to install a "
                "possibly tampered or wrong artifact."
            )


def _extract_archive(path: Path, dest: Path) -> None:
    """Extract a validated zip or tar.gz into *dest*. Tar extraction uses the
    'data' filter (Python 3.12+) so a malicious member cannot escape *dest*."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
        return
    with tarfile.open(path) as tf:
        try:
            tf.extractall(dest, filter="data")     # py3.12+: path-traversal safe
        except TypeError:
            tf.extractall(dest)                    # older python: best effort


def _copy_binaries(src_dir: Path, target: Path) -> int:
    """Copy the llama/ggml/runtime libraries from *src_dir* (recursively) into
    *target*. Returns the number of files copied."""
    n = 0
    for f in src_dir.rglob("*"):
        if f.is_file() and _is_wanted(f):
            shutil.copy2(f, target / f.name)
            n += 1
    return n


def _install_runtime_wheel(pkg_dir: Path) -> bool:
    """Install the runtime wheel editable into the active venv. Tries uv, then
    pip. Returns True on success."""
    for cmd in (["uv", "pip", "install", "-e", str(pkg_dir)],
                [sys.executable, "-m", "pip", "install", "-e", str(pkg_dir)]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return True
        except FileNotFoundError:
            continue
    return False


@click.command("setup-llama", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--from", "from_dir", default=None, type=click.Path(exists=True, file_okay=False),
              help="Copy binaries from a local llama.cpp build directory instead of downloading.")
@click.option("--backend", default="auto",
              type=click.Choice(["auto", "vulkan", "cuda", "sycl", "hip", "cpu",
                                 "metal", "amd-rocm"], case_sensitive=False),
              help="Which prebuilt to fetch. 'auto' detects your GPU and picks "
                   "the broadest working backend (vulkan for NVIDIA/Intel, the "
                   "self-contained ROCm build for AMD, cpu if no GPU).")
@click.option("--url", default=None, help="Override with an explicit prebuilt archive URL.")
@click.option("--sha256", "sha256", default=None,
              help="Expected sha256 of the downloaded archive. When given, the "
                   "download is refused unless its digest matches (opt-in "
                   "integrity pin).")
@click.option("--force", is_flag=True, help="Re-provision even if binaries are already present.")
def main(from_dir: Optional[str], backend: str, url: Optional[str],
         sha256: Optional[str], force: bool) -> None:
    """Download or copy the native llama.cpp binaries into localm's own venv.

    \b
      localm setup-llama                        # auto-detect GPU, fetch the right prebuilt
      localm setup-llama --backend vulkan       # universal GPU build (any vendor)
      localm setup-llama --backend cuda         # NVIDIA (needs CUDA runtime)
      localm setup-llama --backend cpu          # no GPU
      localm setup-llama --from /path/to/llama.cpp/build/bin
      localm setup-llama --url https://.../llama-...zip
      localm setup-llama --sha256 <hex>         # pin the expected archive digest
    """
    lib_name = _lib_name()
    target = _repo_runtime_lib()
    target.mkdir(parents=True, exist_ok=True)

    already = (target / lib_name).exists()
    if already and not force:
        console.print(f"[green]Already provisioned[/green] at {target}")
        console.print("[dim]Use --force to re-download/replace.[/dim]")
        _ensure_importable()
        return

    if from_dir:
        src = Path(from_dir)
        console.print(f"Copying binaries from [bold]{src}[/bold] …")
        n = _copy_binaries(src, target)
        if not (target / lib_name).exists():
            console.print(f"[red]No {lib_name} found in the source directory.[/red] "
                          f"Point --from at the build output containing {lib_name}.")
            sys.exit(1)
        console.print(f"[green]Copied {n} file(s)[/green] into {target}")
    else:
        # Resolve the download URL: explicit --url wins, else the chosen backend
        # (auto-detected when 'auto').
        if url:
            download_url = url
            chosen = "url"
        else:
            chosen = _auto_backend() if backend == "auto" else backend
            try:
                download_url = _resolve_backend_url(chosen)
            except click.ClickException as e:
                console.print(f"[red]{e.message}[/red]")
                console.print("Build llama.cpp for your hardware and run:  "
                              "[bold]localm setup-llama --from <build dir>[/bold], "
                              "or pass --url.")
                sys.exit(1)
            note = {
                "vulkan": "universal GPU build (AMD/NVIDIA/Intel via the display driver)",
                "amd-rocm": "self-contained AMD ROCm build (gfx103X / RX 6000)",
                "cuda": "NVIDIA CUDA build (needs the CUDA runtime present)",
                "sycl": "Intel oneAPI build (needs the oneAPI runtime present)",
                "hip": "AMD ROCm build (needs the ROCm/HIP runtime present)",
                "cpu": "CPU-only build (no GPU)",
                "metal": "Apple Silicon (Metal) build",
            }.get(chosen, chosen)
            console.print(f"[dim]Backend:[/dim] [bold]{chosen}[/bold]  ({note})")

        with tempfile.TemporaryDirectory() as tmp:
            suffix = ".zip" if download_url.lower().endswith(".zip") else ".tar.gz"
            arc_path = Path(tmp) / f"llama-prebuilt{suffix}"
            try:
                _download(download_url, arc_path)
            except Exception as e:
                console.print(f"[red]Download failed:[/red] {e}")
                console.print("Provide a local build with --from instead, or a "
                              "different --url.")
                sys.exit(1)
            # SEC-8: validate the artifact (size, valid-archive, optional sha256
            # pin) BEFORE extracting or installing anything from it.
            try:
                _validate_archive(arc_path, expected_sha256=sha256)
            except ArtifactError as e:
                console.print(f"[red]Refusing to install:[/red] {e}")
                console.print("Provide a local build with --from instead, or a "
                              "different --url (and --sha256 if you pin one).")
                sys.exit(1)
            extract_dir = Path(tmp) / "x"
            try:
                _extract_archive(arc_path, extract_dir)
            except (zipfile.BadZipFile, tarfile.TarError):
                console.print("[red]The downloaded file is not a valid archive.[/red]")
                sys.exit(1)
            n = _copy_binaries(extract_dir, target)
            if not (target / lib_name).exists():
                console.print(f"[red]The archive did not contain {lib_name}.[/red] "
                              "Try a different --backend/--url or use --from.")
                sys.exit(1)
            console.print(f"[green]Extracted {n} binary file(s)[/green] into {target}")

    console.print("[dim]Installing the runtime wheel into this venv …[/dim]")
    if _install_runtime_wheel(_runtime_pkg_dir()):
        console.print("[green]✓[/green] localm-llama-runtime installed.")
    else:
        console.print("[yellow]Could not auto-install the runtime wheel.[/yellow] "
                      f"Run:  uv pip install -e {_runtime_pkg_dir()}")

    _verify()


def _ensure_importable() -> None:
    try:
        import localm_llama_runtime  # noqa: F401
    except Exception:
        if _install_runtime_wheel(_runtime_pkg_dir()):
            console.print("[green]✓[/green] localm-llama-runtime installed.")


def _verify() -> None:
    try:
        from localm.inference.backends.llamacpp._loader import runtime_binary_dir
        d = runtime_binary_dir()
        if d:
            console.print(f"[bold green]Native runtime ready[/bold green] → {d}")
            console.print("Try it:  [bold]localm run <model>[/bold]")
        else:
            console.print("[yellow]Binaries placed but not yet resolvable - "
                          "restart your shell so the new package is importable.[/yellow]")
    except Exception:
        pass
