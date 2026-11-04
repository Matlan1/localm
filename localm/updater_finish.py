# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the finishing step a build ships for its own update.

A build may contain ``localm/_finish_update.py`` (:data:`MODULE`). After the swap,
``updater.apply`` runs it from the tree the signature check verified::

    <the install's python> -I -X utf8 -u <install>/localm/_finish_update.py OLD NEW CLASS RESULT

with the install as the working directory and no console window. OLD and NEW are
the versions, CLASS the update class, RESULT the path of its result file. It uses
only the standard library.

- Exit status 0: the update continues. Any other status, or being stopped at a
  limit below: the update is rolled back, with the last line of output as the reason.
- RESULT holds ``{"took_over": [...]}``, naming the built-in steps it carried out
  among :data:`STEPS`; ``apply`` skips those and runs every other one as usual. No
  file, or a file that cannot be read, means nothing was taken over.
- Every line of output goes to the user the way the built-in steps' output does. A
  long step prints a line at least every :data:`SILENCE_LIMIT_S` seconds.
- It is stopped, with its child processes, after :data:`SILENCE_LIMIT_S` seconds
  without output or :data:`TOTAL_LIMIT_S` seconds in all.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

MODULE = "localm/_finish_update.py"
RESULT_FILE = "finish-result.json"
STEPS = frozenset({"dependencies", "runtime", "components"})
SILENCE_LIMIT_S = 600.0
TOTAL_LIMIT_S = 7200.0
_DRAIN_S = 2.0
_REASON_CHARS = 500


class FinishFailed(Exception):
    """The finishing step failed; ``str()`` is the reason to show the user."""


def present(root) -> bool:
    """Whether the tree at *root* ships a finishing step."""
    return (Path(root) / MODULE).is_file()


def interpreter(root) -> str:
    """The full path of the python of the install at *root*: its venv's, else the
    running one."""
    parts = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    own = Path(root).joinpath(".venv", *parts)
    return str(own) if own.is_file() else sys.executable


def _warn(msg, *args) -> None:
    from localm import updater
    updater._apply_warn(msg, *args)


def _show(line: str, say: Optional[Callable]) -> None:
    try:
        if say:
            say(line)
        else:
            print(line, flush=True)
    except Exception as e:
        _warn("could not show a line of the update's finishing step: %s", e)


def _stop_tree(proc: subprocess.Popen) -> None:
    """Stop *proc* and its child processes and wait for it."""
    if sys.platform == "win32":
        from localm import sysprog
        taskkill = sysprog.find("taskkill")
        if taskkill:
            try:
                subprocess.run([taskkill, "/F", "/T", "/PID", str(proc.pid)],
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=30,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            except (OSError, subprocess.SubprocessError) as e:
                _warn("could not stop the update's finishing step with taskkill: %s", e)
        else:
            _warn("taskkill was not found, so child processes of the update's finishing "
                  "step may keep running")
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError as e:
            _warn("could not stop the process group of the update's finishing step: %s", e)
    try:
        proc.kill()
    except OSError as e:
        _warn("could not stop the update's finishing step: %s", e)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _warn("the update's finishing step (pid %s) did not exit after being stopped", proc.pid)


def _read_result(path: Path, say: Optional[Callable]) -> frozenset:
    """The steps the result file at *path* names (see the module docstring)."""
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
        named = data["took_over"]
        if not isinstance(named, list) or not all(isinstance(n, str) for n in named):
            raise ValueError("took_over is not a list of names")
    except FileNotFoundError:
        return frozenset()
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as e:
        _warn("the result of the update's finishing step cannot be read: %s", e)
        _show(f"The result of the update's finishing step cannot be read ({e}), so the "
              "built-in steps run.", say)
        return frozenset()
    unknown = sorted(set(named) - STEPS)
    if unknown:
        _warn("the update's finishing step named unknown steps: %s", unknown)
    return frozenset(named) & STEPS


def run(root, *, old: str, new: str, klass: str, result_file, env: dict,
        say: Optional[Callable] = None) -> frozenset:
    """Run the finishing step of the install at *root* (see the module docstring)
    and return the built-in steps it took over. *say* receives each line of its
    output; without it the lines go to stdout. Raises :class:`FinishFailed` with the
    reason when it cannot start, exits non-zero, is stopped at a limit or is
    interrupted; its child processes are stopped when it is stopped."""
    root, result_file = Path(root), Path(result_file)
    try:
        result_file.unlink(missing_ok=True)
    except OSError as e:
        raise FinishFailed(f"an earlier result of the finishing step cannot be removed: {e}") from e
    cmd = [interpreter(root), "-I", "-X", "utf8", "-u", str(root / MODULE),
           str(old), str(new), str(klass), str(result_file)]
    flags: dict = ({"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32"
                   else {"start_new_session": True})
    try:
        proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", **flags)
    except OSError as e:
        raise FinishFailed(f"the finishing step could not start: {e}") from e

    lines: queue.SimpleQueue = queue.SimpleQueue()

    def read() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            lines.put(raw)

    threading.Thread(target=read, daemon=True, name="update-finish-output").start()

    start = last_output = time.monotonic()
    exited_at = None
    last_line = ""
    try:
        while True:
            rc = proc.poll()
            if rc is not None and exited_at is None:
                exited_at = time.monotonic()
            try:
                line = lines.get(timeout=0.2).rstrip("\r\n")
            except queue.Empty:
                if rc is not None:
                    break
            else:
                last_output = time.monotonic()
                if line.strip():
                    last_line = line.strip()
                    _show(line, say)
            now = time.monotonic()
            if exited_at is not None:
                if now - exited_at > _DRAIN_S:
                    break
                continue
            if now - last_output > SILENCE_LIMIT_S:
                _stop_tree(proc)
                raise FinishFailed(f"the finishing step printed nothing for "
                                   f"{SILENCE_LIMIT_S:.0f} s, so it was stopped")
            if now - start > TOTAL_LIMIT_S:
                _stop_tree(proc)
                raise FinishFailed(f"the finishing step was still running after "
                                   f"{TOTAL_LIMIT_S:.0f} s, so it was stopped")
    except KeyboardInterrupt:
        _stop_tree(proc)
        raise FinishFailed("the finishing step was interrupted") from None
    if rc != 0:
        raise FinishFailed(last_line[-_REASON_CHARS:] or f"the finishing step exited {rc}")
    took_over = _read_result(result_file, say)
    try:
        result_file.unlink(missing_ok=True)
    except OSError as e:
        _warn("could not remove the result of the update's finishing step: %s", e)
    if took_over:
        _show("The update's finishing step took over: " + ", ".join(sorted(took_over)) + ".", say)
    return took_over
