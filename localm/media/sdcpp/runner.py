# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parent-side handle to the stable-diffusion.cpp worker process.

``SdRunner`` spawns ``_child.worker_main`` in a ``multiprocessing`` "spawn"
child, keeps it (and its loaded model) between requests, relays progress, and
turns a crash, a hang or a cancel into a Python result. A native abort kills
only the child. Not thread-safe: callers serialise their own use.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import time
from typing import Any, Callable, Optional

_POLL_INTERVAL = 0.2

PROBE_TIMEOUT = 120.0
LOAD_TIMEOUT = 1800.0
GENERATE_TIMEOUT = 7200.0

# How long a cancelled generation may take to stop before the worker is killed.
CANCEL_GRACE = 15.0


class SdCancelled(Exception):
    """The generation was cancelled."""


class SdWorkerError(RuntimeError):
    """The worker reported a failure, crashed or timed out."""


EventCallback = Callable[[tuple], None]


class SdRunner:
    """One isolated sd.cpp worker. ``loaded_key`` names the model set it holds."""

    def __init__(self) -> None:
        self._proc = None
        self._req_q = None
        self._resp_q = None
        self._cancel = None
        self._crash_trace_path = None
        self.loaded_key: Optional[Any] = None
        self.load_info: Optional[dict] = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    @property
    def pid(self) -> Optional[int]:
        return None if self._proc is None else self._proc.pid

    def _spawn(self) -> None:
        from localm._mp_spawn import ensure_spawn_uses_venv_python
        from ._child import worker_main
        ensure_spawn_uses_venv_python()
        ctx = mp.get_context("spawn")
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._cancel = ctx.Event()
        self._crash_trace_path = None
        from localm.audit import diagnostics_allowed
        if diagnostics_allowed():
            from localm.debuglog import child_crash_trace_path, logger
            try:
                self._crash_trace_path = child_crash_trace_path("sdcpp-worker")
            except OSError as e:
                logger.warning("could not allocate a native-fault trace file (%s); a "
                               "native fault in the image worker will not be traced", e)
        self._proc = ctx.Process(
            target=worker_main,
            args=(self._req_q, self._resp_q, self._cancel, self._crash_trace_path),
            name="localm-sdcpp-worker", daemon=True)
        self._proc.start()

    def _exit_reason(self) -> str:
        from localm._mp_spawn import describe_exit_code
        return describe_exit_code(None if self._proc is None else self._proc.exitcode)

    def _crash_detail(self) -> str:
        path = self._crash_trace_path
        if path is None:
            return " No native stack trace was captured."
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            text = ""
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if not text:
            return " No native stack trace was captured."
        from localm.debuglog import logger
        logger.error("sd.cpp worker native fault trace:\n%s", text)
        return f" Native fault: {text.splitlines()[0].strip()}."

    def _wait(self, label: str, timeout: float, *,
              on_event: Optional[EventCallback] = None,
              cancel_check: Optional[Callable[[], bool]] = None):
        """The final reply to the command in flight, relaying events to
        *on_event*. Raises :class:`SdCancelled` or :class:`SdWorkerError`."""
        proc, resp_q, cancel = self._proc, self._resp_q, self._cancel
        if proc is None or resp_q is None or cancel is None:
            raise SdWorkerError("The image worker is not running.")
        deadline = time.monotonic() + timeout
        cancel_deadline = None
        while True:
            if cancel_check is not None and cancel_deadline is None:
                try:
                    wanted = bool(cancel_check())
                except Exception:
                    wanted = False
                if wanted:
                    cancel.set()
                    cancel_deadline = time.monotonic() + CANCEL_GRACE
            try:
                msg = resp_q.get(timeout=_POLL_INTERVAL)
            except _queue.Empty:
                if not proc.is_alive():
                    reason = self._exit_reason()
                    detail = self._crash_detail()
                    self._reset()
                    if cancel_deadline is not None:
                        raise SdCancelled("Generation cancelled.") from None
                    raise SdWorkerError(
                        f"The image worker process crashed (exit code {reason}) during "
                        f"'{label}'. The server stayed up.{detail}") from None
                now = time.monotonic()
                if cancel_deadline is not None and now > cancel_deadline:
                    self.shutdown(grace=0)
                    raise SdCancelled(
                        "Generation cancelled (the image worker was stopped).") from None
                if now > deadline:
                    self.shutdown(grace=0)
                    raise SdWorkerError(
                        f"The image worker '{label}' timed out after {timeout:.0f}s and "
                        "was stopped. The server stayed up.") from None
                continue
            kind = msg[0]
            if kind in ("progress", "log"):
                if on_event is not None:
                    try:
                        on_event(msg)
                    except Exception:
                        pass
                continue
            if cancel_deadline is not None:
                cancel.clear()
                if kind == "ok":
                    raise SdCancelled("Generation cancelled.")
            if kind == "ok":
                return msg[1]
            if kind == "cancelled":
                raise SdCancelled(msg[1] or "Generation cancelled.")
            if kind == "error":
                raise SdWorkerError(str(msg[1]))
            raise SdWorkerError(f"Unexpected reply from the image worker: {msg!r}")

    def _request(self, name: str, payload, timeout: float, **kw):
        if self._req_q is None or self._cancel is None or not self.is_alive():
            raise SdWorkerError("The image worker is not running.")
        self._cancel.clear()
        self._req_q.put((name, payload))
        return self._wait(name, timeout, **kw)

    def probe(self, runtime_dir, extra_dirs=None, timeout: float = PROBE_TIMEOUT) -> dict:
        """Load the runtime in a fresh worker, check its ABI and list its
        devices, then stop the worker. Raises :class:`SdWorkerError`."""
        self.shutdown(grace=0)
        self._spawn()
        try:
            return self._request("probe", {"runtime_dir": str(runtime_dir),
                                           "extra_dirs": [str(d) for d in extra_dirs or []]},
                                 timeout)
        finally:
            self.shutdown()

    def ensure_loaded(self, key, runtime_dir, ctx: dict, *, extra_dirs=None,
                      timeout: float = LOAD_TIMEOUT,
                      on_event: Optional[EventCallback] = None,
                      cancel_check: Optional[Callable[[], bool]] = None) -> dict:
        """Hold the model set *ctx* (identified by *key*) in a live worker,
        spawning or replacing the worker when it holds something else.
        Returns the load info (model version, capabilities, devices)."""
        if self.is_alive() and self.loaded_key == key and self.load_info is not None:
            return self.load_info
        self.shutdown()
        self._spawn()
        try:
            info = self._request("load", {"runtime_dir": str(runtime_dir),
                                          "extra_dirs": [str(d) for d in extra_dirs or []],
                                          "ctx": ctx},
                                 timeout, on_event=on_event, cancel_check=cancel_check)
        except BaseException:
            self.shutdown(grace=0)
            raise
        self.loaded_key = key
        self.load_info = info
        return info

    def generate_image(self, params: dict, *, timeout: float = GENERATE_TIMEOUT,
                       on_event: Optional[EventCallback] = None,
                       cancel_check: Optional[Callable[[], bool]] = None) -> dict:
        """Generate one image with the loaded model. Returns ``{"width",
        "height", "channel", "data", "seed"}``; raises :class:`SdCancelled` or
        :class:`SdWorkerError`."""
        return self._request("generate_image", params, timeout,
                             on_event=on_event, cancel_check=cancel_check)

    def _reset(self) -> None:
        for q in (self._req_q, self._resp_q):
            if q is not None:
                try:
                    q.close()
                    q.cancel_join_thread()
                except Exception:
                    pass
        self._proc = None
        self._req_q = None
        self._resp_q = None
        self._cancel = None
        self.loaded_key = None
        self.load_info = None

    def shutdown(self, grace: float = 5.0) -> None:
        """Stop the worker (asking first, then killing). Process exit releases
        all of its VRAM. Safe to call repeatedly."""
        proc = self._proc
        if proc is None:
            return
        if proc.is_alive() and grace > 0 and self._req_q is not None:
            try:
                self._req_q.put(("shutdown", None))
            except Exception:
                pass
            proc.join(timeout=grace)
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:
                pass
            proc.join(timeout=5)
        if self._crash_trace_path is not None:
            try:
                self._crash_trace_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._crash_trace_path = None
        self._reset()
