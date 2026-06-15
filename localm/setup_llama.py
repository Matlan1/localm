"""``localm setup-llama`` - provision the native llama.cpp binaries locally.

Makes localm self-contained: the native inference runtime (the llama shared
library + its ggml deps, plus a matched ROCm/CUDA runtime when the prebuilt
ships one) is placed inside the project's own ``localm-llama-runtime`` wheel
rather than depending on a folder elsewhere on disk.

Two sources:
  * ``--download`` (default, Windows only for now): fetch a prebuilt release zip
    and extract it. Defaults to the lemonade-sdk gfx1030 (RDNA2) Windows build.
    There is no hosted prebuilt for Linux/macOS - build llama.cpp for your GPU
    and use ``--from`` (see docs/linux-setup.md).
  * ``--from <dir>``: copy the binaries from a local llama.cpp build output
    (e.g. produced by rocm-canary-forge's build script, or a stock cmake build).

After placing the files it installs the runtime wheel editable so the loader can
import it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

console = Console(highlight=False)

# Default prebuilt: lemonade-sdk llama.cpp ROCm build for gfx103X (RDNA2),
# Windows-only. See rocm-canary-forge/windows-native for the provenance.
DEFAULT_URL = (
    "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/"
    "b1288/llama-b1288-windows-rocm-gfx103X-x64.zip"
)


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
@click.option("--url", default=None, help="Override the prebuilt download URL.")
@click.option("--force", is_flag=True, help="Re-provision even if binaries are already present.")
def main(from_dir: Optional[str], url: Optional[str], force: bool) -> None:
    """Download or copy the native llama.cpp binaries into localm's own venv.

    \b
      localm setup-llama                       # Windows: download the default prebuilt
      localm setup-llama --from /path/to/llama.cpp/build/bin
      localm setup-llama --url https://.../llama-rocm-gfxXXXX.zip
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
        download_url = url or (DEFAULT_URL if sys.platform == "win32" else None)
        if not download_url:
            console.print(f"[red]No default prebuilt for this platform "
                          f"({sys.platform}).[/red]")
            console.print("Build llama.cpp for your GPU and run:  "
                          "[bold]localm setup-llama --from <build dir>[/bold]  "
                          "(see docs/linux-setup.md), or pass --url.")
            sys.exit(1)
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "llama-prebuilt.zip"
            try:
                _download(download_url, zip_path)
            except Exception as e:
                console.print(f"[red]Download failed:[/red] {e}")
                console.print("Provide a local build with --from instead, or a "
                              "different --url.")
                sys.exit(1)
            extract_dir = Path(tmp) / "x"
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(extract_dir)
            except zipfile.BadZipFile:
                console.print("[red]The downloaded file is not a valid zip.[/red]")
                sys.exit(1)
            n = _copy_binaries(extract_dir, target)
            if not (target / lib_name).exists():
                console.print(f"[red]The archive did not contain {lib_name}.[/red] "
                              "Try a different --url or use --from.")
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
