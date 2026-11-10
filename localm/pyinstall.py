# SPDX-License-Identifier: AGPL-3.0-or-later
"""Python package installs that localm starts (pip and uv): the network policy
check every install runs first, and the hashed requirements an install takes
from the ``uv.lock`` that ships with localm.

- :func:`check_command` refuses an install command whose package indexes the
  network policy refuses (``net_mode = off`` refuses unless
  ``net_allow_model_downloads`` is set).
- :func:`sync_command` is the dependency step of an update: ``uv sync`` from
  the shipped lock, which verifies every downloaded file against the lock's
  sha256.
- :func:`locked_requirements` reads, from the shipped lock, the exact versions
  and hashes of what a set of localm extras needs, for an install with
  ``--require-hashes``.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

#: The package index and its file host every pip or uv install reaches.
PYPI_URLS = ("https://pypi.org/simple/", "https://files.pythonhosted.org/")

#: The lock file shipped beside ``pyproject.toml``.
LOCK_NAME = "uv.lock"

_INDEX_OPTIONS = ("--index-url", "-i", "--extra-index-url", "--default-index", "--index",
                  "--find-links", "-f")
_EXPORT_TIMEOUT_S = 120


class InstallRefused(Exception):
    """An install was refused before anything ran (the network policy, or no
    lock where one is required). The message says why."""


def index_urls(cmd: Iterable) -> list:
    """The URLs an install command *cmd* downloads from: :data:`PYPI_URLS`,
    plus every index or find-links URL its options name and every ``http(s)``
    requirement it lists. Ignores options that name local paths."""
    args = [str(a) for a in cmd]
    urls = list(PYPI_URLS)
    for i, a in enumerate(args):
        value = None
        for opt in _INDEX_OPTIONS:
            if a == opt and i + 1 < len(args):
                value = args[i + 1]
            elif a.startswith(opt + "=") and opt.startswith("--"):
                value = a.split("=", 1)[1]
        if value is None and re.match(r"(?i)https?://", a):
            value = a
        if value is None:
            m = re.search(r"@\s*(https?://\S+)", a)
            value = m.group(1) if m else None
        if value and re.match(r"(?i)https?://", value) and value not in urls:
            urls.append(value)
    return urls


def check_network(urls: Iterable[str] = PYPI_URLS) -> None:
    """Raise :class:`InstallRefused` when the network policy refuses any of
    *urls* (``netpolicy.check_url`` with ``allow_when_off=True``, so
    ``net_allow_model_downloads`` exempts ``net_mode = off``)."""
    from localm.netpolicy import NetworkPolicyError, check_url
    for url in urls:
        try:
            check_url(url, allow_when_off=True)
        except NetworkPolicyError as e:
            raise InstallRefused(f"localm cannot download Python packages: {e}") from e


def check_command(cmd: Iterable) -> None:
    """Raise :class:`InstallRefused` when the network policy refuses an index
    the install command *cmd* uses (:func:`index_urls`)."""
    check_network(index_urls(cmd))


def is_install_command(cmd: Iterable) -> bool:
    """Whether *cmd* is a ``pip install`` or ``uv pip install`` / ``uv sync``
    command (by its arguments, not its program)."""
    args = [str(a) for a in cmd]
    for i, a in enumerate(args[:-1]):
        if a == "pip" and args[i + 1] == "install":
            return True
    return "sync" in args[1:2]


def lock_file(root) -> Optional[Path]:
    """The ``uv.lock`` beside the ``pyproject.toml`` at *root*, or None when
    either is missing."""
    root = Path(root)
    lock = root / LOCK_NAME
    return lock if lock.is_file() and (root / "pyproject.toml").is_file() else None


def sync_command(uv: str, extras: Iterable[str]) -> list:
    """The argv that installs the project at the current directory and its
    dependencies plus *extras* from its ``uv.lock`` into its ``.venv`` with
    *uv*: exact versions, each file checked against the lock's sha256, the lock
    left unchanged, packages the lock does not name left installed."""
    cmd = [uv, "sync", "--frozen", "--inexact", "--no-config"]
    for e in extras:
        cmd += ["--extra", str(e)]
    return cmd


@dataclass
class LockedRequirement:
    """One requirement of an exported lock: ``name==version`` with its
    environment marker (``""`` for none) and sha256 hashes."""
    name: str
    version: str
    marker: str = ""
    hashes: list = field(default_factory=list)

    def line(self) -> str:
        """The requirement as one requirements-file line."""
        marker = f" ; {self.marker}" if self.marker else ""
        return (f"{self.name}=={self.version}{marker} "
                + " ".join(f"--hash={h}" for h in self.hashes))

    def applies(self) -> bool:
        """Whether the marker holds on this machine (True when it has none)."""
        if not self.marker:
            return True
        from packaging.markers import Marker
        return bool(Marker(self.marker).evaluate())


def parse_export(text: str) -> list:
    """The :class:`LockedRequirement` entries of ``uv export --format
    requirements-txt`` output. Raises ValueError on an entry that is not an
    exact pin with at least one hash."""
    joined = re.sub(r"\\\s*\n", " ", text)
    out = []
    for raw in joined.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        hashes = re.findall(r"--hash=(sha256:[0-9a-fA-F]{64})", line)
        body = re.split(r"\s--hash=", " " + line, maxsplit=1)[0].strip()
        req, _, marker = body.partition(";")
        m = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)", req.strip())
        if not m or not hashes:
            raise ValueError(f"not a hashed exact pin: {line[:120]}")
        out.append(LockedRequirement(m.group(1), m.group(2), marker.strip(), hashes))
    return out


def locked_requirements(uv: str, root, extras: Iterable[str], *,
                        run: Callable = subprocess.run,
                        env: Optional[dict] = None) -> list:
    """What localm's dependencies plus *extras* resolve to in the ``uv.lock``
    at *root*, as :class:`LockedRequirement` entries (localm itself left out).
    Reads the lock with *uv* (``uv export --frozen --offline``), so nothing is
    downloaded and the lock is not changed.

    Raises :class:`InstallRefused` when *root* has no lock, and RuntimeError
    when uv fails or prints an entry without hashes."""
    root = Path(root)
    if lock_file(root) is None:
        raise InstallRefused(f"there is no {LOCK_NAME} at {root}")
    with tempfile.TemporaryDirectory(prefix="localm-lock-") as tmp:
        out = Path(tmp) / "requirements.txt"
        cmd = [uv, "export", "--frozen", "--offline", "--no-config", "--no-emit-project",
               "--format", "requirements-txt", "--no-header", "--no-annotate",
               "--output-file", str(out)]
        for e in extras:
            cmd += ["--extra", str(e)]
        kw: dict = {"cwd": str(root), "capture_output": True, "encoding": "utf-8",
                    "errors": "replace", "timeout": _EXPORT_TIMEOUT_S,
                    "stdin": subprocess.DEVNULL}
        if env is not None:
            kw["env"] = env
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            proc = run(cmd, **kw)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"uv could not read {LOCK_NAME}: {e}") from e
        if proc.returncode != 0:
            raise RuntimeError(f"uv could not read {LOCK_NAME}: "
                               + (proc.stderr or proc.stdout or "").strip()[-600:])
        try:
            text = out.read_text(encoding="utf-8")
        except OSError as e:
            raise RuntimeError(f"uv wrote no requirements from {LOCK_NAME}: {e}") from e
    try:
        return parse_export(text)
    except ValueError as e:
        raise RuntimeError(f"{LOCK_NAME} gave a requirement localm cannot check: {e}") from e
