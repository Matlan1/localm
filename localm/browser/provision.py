# SPDX-License-Identifier: AGPL-3.0-or-later
"""Provision the Chromium build the automated browser drives.

The ``browser`` pip extra (``pip install "localm[browser]"``) installs the
playwright DRIVER only. The Chromium build it drives is a separate download,
one exact build per playwright version (see pyproject.toml's ``browser``
extra), and previously only ``python -m playwright install chromium`` could
fetch it - this wraps that step as a localm-native command, the same shape as
``setup-llama`` and ``setup-embeddings``: it goes through the network policy
like every other explicit download, and it never reports success it has not
verified (AGENTS.md rule 5).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: How long a Chromium+ffmpeg+headless-shell download may run before it is
#: treated as hung and killed. Generous rather than tight: these assets
#: together are well under a GB (~300 MB measured), but a slow connection
#: must still finish rather than being cut off mid-transfer.
_INSTALL_TIMEOUT_S = 1200

PIP_INSTALL_HINT = 'pip install "localm[browser]"'

#: A caller-supplied progress sink for one output line at a time. Mirrors
#: managed_comfy_provision's on_progress shape.
ProgressCb = Optional[Callable[[str], None]]


@dataclass
class ProvisionResult:
    ok: bool
    message: str
    #: True when nothing was downloaded because Chromium was already present
    #: (only meaningful when ok is True).
    already_installed: bool = False


def chromium_executable_path() -> Optional[Path]:
    """The Chromium executable this playwright version drives, resolved
    WITHOUT installing or launching anything - or None when the playwright
    package is not importable, or its driver could not be started.

    Goes through playwright's OWN resolution rather than reconstructing the
    ``~/AppData/.../ms-playwright/chromium-<rev>/...`` layout by hand (a
    playwright implementation detail that differs by platform and by
    version), so this can never disagree with what playwright itself will
    try to launch."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    try:
        with sync_playwright() as p:
            return Path(p.chromium.executable_path)
    except Exception as e:                            # noqa: BLE001
        logger.debug("could not resolve the chromium executable path: %s", e)
        return None


def is_chromium_installed() -> bool:
    """Whether the Chromium build this playwright version drives is already
    on disk. False (never raises) when playwright itself is not installed or
    its driver could not answer."""
    path = chromium_executable_path()
    return path is not None and path.exists()


def _stream_install(cmd: list, *, on_progress: ProgressCb,
                    timeout: int) -> tuple[Optional[int], list]:
    """Run *cmd*, streaming its combined output to *on_progress* as each line
    arrives. Returns ``(returncode, lines)``; returncode is None only when the
    process was killed after *timeout* (the timeout itself is then the last
    entry in *lines*).

    Same shape as managed_comfy_provision._run: a background reader thread
    does the line I/O while this thread enforces the timeout via
    ``proc.wait(timeout=...)``, so a hung child is killed rather than left to
    block forever."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1)

    lines: list = []

    def _read_stdout() -> None:
        try:
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\r\n")
                lines.append(line)
                if on_progress is not None:
                    try:
                        on_progress(line)
                    except Exception:                  # noqa: BLE001
                        pass
        except ValueError:
            pass  # the pipe was closed under us (process killed on timeout)

    reader = threading.Thread(target=_read_stdout, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        reader.join(timeout=5)
        lines.append(f"(installer timed out after {timeout}s and was killed)")
        return None, lines
    reader.join(timeout=5)
    return proc.returncode, lines


def install_chromium(*, force: bool = False,
                     on_progress: ProgressCb = None) -> ProvisionResult:
    """Download the Chromium build via ``python -m playwright install
    chromium``, in THIS interpreter's venv, honestly reporting whether it
    actually worked.

    Nothing to do, and no network touched, when Chromium is already present
    and *force* is False - mirrors ``voice.stt_available``'s "already cached
    skips the policy gate" shape, since there is no request for the policy to
    govern.

    Network policy: this is an explicit user action (the CLI command below,
    and any future first-use provisioning of the coder's browser tool), so it
    is refused outright under ``net_mode=off`` unless
    ``net_allow_model_downloads`` exempts it - the same off-floor every other
    explicit download in this project uses (see ``localm.voice``,
    ``localm.inference.embedder``, ``localm.model_manager.pull``) - rather
    than letting playwright's own downloader attempt the fetch and surface a
    raw Node stack trace for what is, from here, a policy refusal.

    *force* is passed through as playwright's own ``--force``. NOTE this
    makes playwright remove the existing build BEFORE it redownloads, so a
    failed ``--force`` run can leave NO Chromium installed even though one
    was present before the call; the returned message says so when that
    happens.

    Never returns ``ok=True`` on the strength of the subprocess exit code
    alone: the Chromium executable is re-resolved from disk afterward and
    must actually be there (AGENTS.md rule 5 - a provisioning step that fails
    must never report success)."""
    try:
        import playwright  # noqa: F401
    except ImportError:
        return ProvisionResult(
            ok=False,
            message="The browser automation extra is not installed. Install "
                    f"it with:  {PIP_INSTALL_HINT}")

    was_installed = is_chromium_installed()
    if was_installed and not force:
        return ProvisionResult(
            ok=True, already_installed=True,
            message=f"Chromium is already installed at "
                    f"{chromium_executable_path()}.")

    from localm.netpolicy import downloads_allowed_when_off, network_mode
    if network_mode() == "off" and not downloads_allowed_when_off():
        return ProvisionResult(
            ok=False,
            message="Network access is disabled (net_mode=off). Enable it "
                    "with:  localm config net_mode ask - or allow just "
                    "downloads:  localm config net_allow_model_downloads true")

    cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
    if force:
        cmd.append("--force")
    try:
        returncode, lines = _stream_install(
            cmd, on_progress=on_progress, timeout=_INSTALL_TIMEOUT_S)
    except OSError as e:
        return ProvisionResult(
            ok=False, message=f"Could not start the playwright installer: {e}")

    now_installed = is_chromium_installed()
    if returncode != 0 or not now_installed:
        if returncode is None:
            reason = f"the installer did not finish within {_INSTALL_TIMEOUT_S}s"
        elif returncode != 0:
            reason = f"the installer exited with code {returncode}"
        else:
            reason = ("the installer reported success but the Chromium "
                      "executable is still not on disk")
        tail = "\n".join(lines[-15:])
        detail = f"\n{tail}" if tail else ""
        warning = ""
        if force and was_installed and not now_installed:
            warning = ("\nThe previous Chromium build was removed before this "
                       "attempt (--force) and the reinstall did not complete, "
                       "so no Chromium build is installed right now.")
        return ProvisionResult(
            ok=False, message=f"Could not install Chromium: {reason}.{warning}{detail}")

    return ProvisionResult(
        ok=True, message=f"Chromium installed at {chromium_executable_path()}.")
