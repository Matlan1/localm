# SPDX-License-Identifier: AGPL-3.0-or-later
"""Whether the new version starts: the probe of an update where no server of it will
be started.

``localm _update-probe`` serves the real app on a loopback port without a model,
without its background services and without anything it would write on its own, prints
``{"port": N}`` and serves until its input ends.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections.abc import Mapping
from http.client import HTTPConnection, HTTPException
from typing import Optional

from . import steps_update

LIMIT_S = 180.0
_REASON_CHARS = 600


def probe(python, root, *, version: str, home, env: Mapping, update_id: str = "",
          limit_s: float = LIMIT_S) -> Optional[str]:
    """Start the app of the install at *root* with *python* and ask it which version it
    is. None when it answers *version* within *limit_s* seconds, else why not, for the
    user."""
    flags: dict = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    child_env = steps_update.child_env(env, root, home, update_id)
    try:
        proc = subprocess.Popen([str(python), "-m", "localm", "_update-probe"], cwd=str(root),
                                env=child_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", **flags)
    except OSError as e:
        return f"localm {version} could not be started: {e}"
    lines: list = []
    ports: queue.SimpleQueue = queue.SimpleQueue()

    def read() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.startswith('{"port"'):
                ports.put(line)
            elif line.strip():
                lines.append(line.rstrip())
                del lines[:-40]
        ports.put(None)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    deadline = time.monotonic() + limit_s
    try:
        try:
            first = ports.get(timeout=limit_s)
        except queue.Empty:
            return f"localm {version} did not start within {limit_s:.0f} seconds{_tail(lines)}"
        if first is None:
            proc.wait(timeout=10)
            return f"localm {version} does not start{_tail(lines)}"
        answer = _ask(int(json.loads(first)["port"]), deadline)
        if answer is None:
            return f"localm {version} started and does not answer{_tail(lines)}"
        if answer != version:
            return f"localm {answer} answered instead of {version}"
        return None
    except (ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as e:
        return f"localm {version} could not be checked: {e}{_tail(lines)}"
    finally:
        _end(proc)
        reader.join(timeout=5)


def _ask(port: int, deadline: float) -> Optional[str]:
    """The version ``GET /whoami`` on the loopback *port* answers, None when it does not
    answer until *deadline*."""
    while True:
        conn = HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            conn.request("GET", "/whoami")
            response = conn.getresponse()
            body = json.loads(response.read(64 * 1024).decode("utf-8"))
            if response.status == 200 and isinstance(body, dict):
                return str(body.get("version"))
        except (OSError, HTTPException, ValueError):
            pass
        finally:
            conn.close()
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.2)


def _end(proc: subprocess.Popen) -> None:
    try:
        if proc.stdin is not None:
            proc.stdin.close()
        proc.wait(timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        proc.kill()
        proc.wait()


def _tail(lines: list) -> str:
    text = " ".join(line.strip() for line in lines[-3:])[-_REASON_CHARS:]
    return f": {text}" if text else ""
