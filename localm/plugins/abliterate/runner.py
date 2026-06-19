# SPDX-License-Identifier: AGPL-3.0-or-later
"""Locate and invoke Heretic as an external program.

Heretic (https://github.com/Matlan1/heretic-win-AMD) is licensed under
AGPL-3.0-or-later. localm treats it strictly as a **separate program**: this
module only ever *runs* it via subprocess and talks to it through command-line
arguments and the filesystem. Nothing here (or anywhere in localm) imports the
Heretic Python package, and Heretic's source is never committed into localm - it
is fetched into a gitignored folder on the user's machine on request.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# The Windows/AMD ROCm fork of Heretic.
HERETIC_FORK_URL = "https://github.com/Matlan1/heretic-win-AMD.git"


@dataclass
class HereticLocation:
    """How to invoke Heretic.

    Either a checkout directory (invoked via ``uv run heretic``) or a resolved
    ``heretic`` executable found on PATH.
    """

    repo_dir: Optional[Path]
    executable: Optional[str]

    @property
    def found(self) -> bool:
        return self.repo_dir is not None or self.executable is not None


def _is_heretic_repo(path: Path) -> bool:
    """Return True if ``path`` looks like a Heretic checkout."""
    return (path / "pyproject.toml").is_file() and (path / "src" / "heretic").is_dir()


def default_clone_dir(home_dir: Path) -> Path:
    """Default location for a fetched Heretic checkout (gitignored)."""
    return home_dir / ".heretic"


def find_heretic(config_path: Optional[str], home_dir: Path) -> HereticLocation:
    """Locate Heretic.

    Search order: the ``LOCALM_HERETIC_PATH`` environment variable, the
    ``heretic_path`` config value, the default ``<home>/.heretic`` checkout, then
    a ``heretic`` executable on PATH. Each path may point at a checkout directory
    or directly at the executable.
    """
    candidates: list[Path] = []
    env_path = os.environ.get("LOCALM_HERETIC_PATH")
    if env_path:
        candidates.append(Path(env_path))
    if config_path:
        candidates.append(Path(config_path))
    candidates.append(default_clone_dir(home_dir))

    for candidate in candidates:
        if candidate.is_dir() and _is_heretic_repo(candidate):
            return HereticLocation(repo_dir=candidate.resolve(), executable=None)
        if candidate.is_file() and candidate.name.lower().startswith("heretic"):
            return HereticLocation(repo_dir=None, executable=str(candidate.resolve()))

    on_path = shutil.which("heretic")
    if on_path:
        return HereticLocation(repo_dir=None, executable=on_path)

    return HereticLocation(repo_dir=None, executable=None)


def clone_heretic(dest: Path) -> bool:
    """Clone the Heretic fork into ``dest`` and run a base ``uv sync``.

    Returns True on success. Heretic remains a separate program; this only
    fetches it onto the user's machine. The ROCm-specific setup runs on
    Heretic's own first launch.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    clone = subprocess.run(
        ["git", "clone", "--depth", "1", HERETIC_FORK_URL, str(dest)]
    )
    if clone.returncode != 0:
        return False
    # Best-effort base environment sync; non-fatal if it fails.
    subprocess.run(["uv", "sync"], cwd=str(dest))
    return True


def build_command(
    location: HereticLocation,
    model: str,
    *,
    gguf_file: Optional[str] = None,
    export_strategy: str = "merge",
    export_gguf: Optional[str] = None,
) -> list[str]:
    """Build the argument list that invokes Heretic.

    The model is passed via ``--model`` (Heretic requires it as an option, not a
    bare positional argument, once other flags follow).
    """
    if location.repo_dir is not None:
        command = ["uv", "run", "heretic"]
    elif location.executable is not None:
        command = [location.executable]
    else:
        raise ValueError("Heretic location is empty")

    command += ["--model", model]
    if gguf_file:
        command += ["--gguf-file", gguf_file]
    if export_strategy:
        command += ["--export-strategy", export_strategy]
    if export_gguf:
        command += ["--export-gguf", export_gguf]
    return command


def derive_name(model: str) -> str:
    """Derive a localm registry name from a model id or path.

    Examples:
        ``Qwen/Qwen3-4B-Instruct-2507`` -> ``Qwen3-4B-Instruct-2507-heretic``
        ``D:\\models\\foo.gguf``         -> ``foo-heretic``
    """
    base = model.replace("\\", "/").rstrip("/").split("/")[-1]
    if base.lower().endswith(".gguf"):
        base = base[: -len(".gguf")]
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-")
    return f"{base or 'model'}-heretic"
