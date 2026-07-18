# SPDX-License-Identifier: AGPL-3.0-or-later
"""Subprocess isolation for the embedding-model lifecycle (load, embed, unload) -
the same fix PR #606 gave the chat GGUF backend (see
``backends/llamacpp/_runner.py``), applied to ``embedder.py``'s separate,
previously-unisolated GGUF/llama.cpp load path (honesty-audit follow-up,
2026-07-14): ``llama_load_model_from_file`` and ``llama_decode`` (called by
every ``embed()``) can hard-``abort()`` the whole process on a native CUDA/HIP
driver failure - no Python ``try/except`` can catch that.

Lighter-weight than the chat runner on purpose: the embedder's usage pattern
is load-once-serve-many, with no streaming and no mid-call cancellation, so
this needs only a plain request/response protocol - closer in shape to
``localm/voice.py``'s worker than to ``llamacpp/_runner.py``'s. Still a
long-lived child (not one process per call, unlike ``voice.py``'s
respawn-per-death-only model) so a small embedding model is not reloaded on
every ``embed()``.

Protocol (two ``multiprocessing.Queue``s, tagged tuples):

``req_q`` (parent -> child), one command processed at a time:
    ("load", {model_path, n_gpu_layers, n_ctx, pooling_type})
    ("embed", texts)
    ("shutdown", None)

``resp_q`` (child -> parent):
    ("ok", value)      - success (a {"dim": N} dict for load, a list of
                          vectors for embed)
    ("error", message) - a clean, expected failure (e.g. a bad model path)

A native abort, or any other uncaught fault in the child's dispatch loop,
produces NO envelope - the parent detects the dead child via
``proc.is_alive()``/``exitcode``, exactly like ``ModelRunner``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue
import time
from typing import List

# Fault-injection hook, honoured by the child ONLY when this environment
# variable is set. Exists exclusively so the test suite can prove the
# crash-containment property with a REAL uncatchable fault (the same code
# path a genuine native abort would take); never set in production. Values:
# "abort" (a genuine uncatchable native abort), "exit" (a hard process exit,
# no Python traceback), "hang" (a wedged native call). Mirrors
# llamacpp/_runner.py's _FAULT_ENV / voice.py's _FAULT_ENV.
_FAULT_ENV = "LOCALM_EMBEDDER_FAULT_FOR_TEST"


def _simulate_fault(mode: str) -> None:
    if mode == "hang":
        while True:                              # a wedged native call
            time.sleep(3600)
    if mode == "exit":
        os._exit(134)                            # vanish with no Python traceback
    os.abort()                                    # genuine uncatchable native abort


# --------------------------------------------------------------------------- #
# Child side - runs ONLY inside the isolated worker process.
# --------------------------------------------------------------------------- #

def _runner_main(req_q, resp_q) -> None:
    """Long-lived child: owns one GGUFEmbedder (one loaded model) for its
    whole process lifetime, dispatching one request at a time."""
    from localm.debuglog import attach_child_logging
    attach_child_logging()   # native load-failure diagnostics land in the
                              # shared debug log from this process too.

    from localm._mp_spawn import (install_parent_death_watchdog,
                                   suppress_native_error_dialogs)
    install_parent_death_watchdog()   # die with the parent even on a hard kill
                                       # (End Task / force-close); daemon=True is
                                       # atexit-gated and does not cover that, so
                                       # else this worker outlives the server
                                       # holding its embedding model in VRAM.
    suppress_native_error_dialogs()   # a native DLL failure here (loading llama.dll
                                       # itself, or the torch/ROCm conflict this
                                       # worker's own VRAM checks are known to hit -
                                       # see _sizing.py) must degrade to a catchable
                                       # exception, never a blocking modal dialog.

    embedder = None

    while True:
        cmd = req_q.get()
        if cmd is None:
            return
        name = cmd[0]
        payload = cmd[1] if len(cmd) > 1 else None

        fault = os.environ.get(_FAULT_ENV)
        if fault:
            _simulate_fault(fault)   # test-only; never returns cleanly

        if name == "shutdown":
            if embedder is not None:
                embedder.close()
            return

        if name == "load":
            payload = dict(payload)
            # cpu_only: n_gpu_layers=0 alone only controls WEIGHT placement -
            # ggml's scheduler still considers a REGISTERED GPU backend for
            # individual ops (a large enough matmul dispatches to vendor BLAS
            # regardless of how many layers were offloaded; confirmed live:
            # a small model like bge-small never crosses that threshold and
            # is unaffected either way, but a multi-billion-parameter model
            # like Qwen3-Embedding-4B does, and hits the identical rocBLAS/
            # Tensile crash on a "CPU-only" n_gpu_layers=0 load - issue #749).
            # Clearing the vendor-visible-device env vars BEFORE importing
            # anything native makes the HIP/ROCm/CUDA runtime itself report
            # ZERO devices, so ggml's backend registration finds none and the
            # scheduler has only CPU to dispatch to - a guarantee, not a
            # heuristic, independent of model size. Set in THIS child process
            # only (freshly spawned, isolated) - never touches the parent's
            # environment or a concurrently-loading chat model.
            if payload.pop("cpu_only", False):
                # "-1" (not "") - Windows' CRT putenv("VAR=") with nothing
                # after '=' REMOVES the variable rather than setting it empty
                # (confirmed live: an empty string left the device visible,
                # "llama_prepare_model_devices: using device ROCm0" still
                # fired). "-1" is the standard CUDA/HIP convention for "no
                # valid device index", which the runtime cannot silently
                # treat as "unset".
                os.environ["HIP_VISIBLE_DEVICES"] = "-1"
                os.environ["ROCR_VISIBLE_DEVICES"] = "-1"
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            from localm.inference.embedder import GGUFEmbedder
            try:
                embedder = GGUFEmbedder(**payload)
                # The pooling facts travel back with the load so the PARENT can
                # warn about a mis-pooled model: only the child ever holds the
                # model handle the declared type is read from.
                resp_q.put(("ok", {
                    "dim": embedder.dim,
                    "declared_pooling": embedder.declared_pooling,
                    "effective_pooling": embedder.pooling_type,
                }))
            except Exception as e:
                resp_q.put(("error", str(e)))
            # A hard native abort during GGUFEmbedder(...) is NOT caught here -
            # the process dies, and the parent detects that via is_alive().
            continue

        if name == "embed":
            try:
                resp_q.put(("ok", embedder.embed(payload)))
            except Exception as e:
                resp_q.put(("error", str(e)))
            # Same as above: a native abort during llama_decode propagates
            # out of this whole function, uncaught, on purpose.
            continue

        # An unrecognized command is a bug in this module's own parent-side
        # caller, not untrusted input - but rule 5 says never silently drop
        # it either way.
        resp_q.put(("error", f"unknown embedder-runner command: {name!r}"))


# --------------------------------------------------------------------------- #
# Parent side - worker lifecycle + dispatch.
# --------------------------------------------------------------------------- #

# How often spawn_and_load/embed re-check proc.is_alive() while polling for a
# response - short, since it is purely a local poll interval, not the
# command's own deadline (see the timeouts below).
_POLL_INTERVAL = 0.2

# Default model-load timeout. Embedding models are far smaller than chat
# models (24-90 MB for the built-in bge/nomic choices - see embedder.py's
# module docstring), but a user-configured `embedding_model` can still be
# multi-GB (confirmed live: Qwen3-Embedding-8B-Q8_0.gguf, 7.49 GB, from the
# incident that prompted this module). Generous but bounded, mirroring
# llamacpp/_runner.py's LOAD_TIMEOUT_DEFAULT reasoning - a stalled load has no
# safe "unmeasurable" fallback, it must raise rather than hang forever.
LOAD_TIMEOUT_DEFAULT = 300.0

# Bounded wait for one embed() RPC. A large RAG re-index batch can embed
# hundreds of chunks in one call, so this is generous rather than per-token
# tight like the chat runner's stream-chunk timeout; a genuinely wedged child
# is still detected via the is_alive() poll below well before this elapses.
_EMBED_TIMEOUT_DEFAULT = 300.0


class EmbedderRunner:
    """Parent-side handle to one isolated embedder worker process."""

    def __init__(self) -> None:
        self._proc = None
        self._req_q = None
        self._resp_q = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _spawn(self) -> None:
        from localm._mp_spawn import ensure_spawn_uses_venv_python
        ensure_spawn_uses_venv_python()   # #617: avoid a renamed-launcher WinError 2
        ctx = mp.get_context("spawn")   # explicit: identical on every OS
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._proc = ctx.Process(
            target=_runner_main, args=(self._req_q, self._resp_q),
            name="localm-embedder-worker", daemon=True)
        self._proc.start()

    def spawn_and_load(self, params: dict, timeout: float = LOAD_TIMEOUT_DEFAULT) -> dict:
        """Spawn the child and load the model. Returns ``{"dim": N}`` on
        success. Raises RuntimeError on a genuine load failure, a child crash
        (native abort - detected via is_alive(), never an exception this
        process had to catch), or a timeout (the child is killed)."""
        self._spawn()
        self._req_q.put(("load", params))
        return self._wait(timeout, "load")

    def embed(self, texts: List[str], timeout: float = _EMBED_TIMEOUT_DEFAULT) -> List[List[float]]:
        """Embed *texts* via the isolated worker. Raises RuntimeError on a
        clean failure, a child crash, or a timeout - the caller (embedder.py's
        IsolatedEmbedder) decides whether/how to recover.

        NOT safe to call concurrently on one runner, by design: the protocol
        above carries no request id, so two overlapping RPCs would be two
        threads blocked in the same resp_q.get(), each free to receive the
        OTHER's response. The sole caller, IsolatedEmbedder.embed(), serializes
        on its _rpc_lock to guarantee that (REG-643)."""
        self._req_q.put(("embed", texts))
        return self._wait(timeout, "embed")

    def _wait(self, timeout: float, label: str):
        deadline = time.monotonic() + timeout
        result = None
        while result is None:
            try:
                result = self._resp_q.get(timeout=_POLL_INTERVAL)
            except _queue.Empty:
                if not self._proc.is_alive():
                    code = self._proc.exitcode
                    raise RuntimeError(
                        f"The embedding worker process crashed (exit code "
                        f"{code}) during '{label}'. The server stayed up; see "
                        "the debug log for the native stack trace.")
                if time.monotonic() > deadline:
                    self.shutdown(grace=0)
                    raise RuntimeError(
                        f"Embedding worker '{label}' timed out after "
                        f"{timeout:.0f}s - the worker process may be hung "
                        "(see the debug log). The server stayed up and the "
                        "worker was stopped; retry the request.")
        kind = result[0]
        if kind == "ok":
            return result[1]
        if kind == "error":
            raise RuntimeError(result[1])
        raise RuntimeError(f"Unexpected response from the embedding worker: {result!r}")

    def shutdown(self, grace: float = 5.0) -> None:
        """Best-effort teardown: ask the worker to close cleanly, then kill it
        if it does not exit within *grace* seconds. Safe to call more than
        once, or when nothing is running."""
        proc = self._proc
        if proc is None:
            return
        if proc.is_alive():
            try:
                self._req_q.put(("shutdown", None))
            except Exception:
                pass
            if grace > 0:
                proc.join(timeout=grace)
        if proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass
            proc.join(timeout=5)
        for q in (self._req_q, self._resp_q):
            if q is not None:
                try:
                    q.close()
                    q.cancel_join_thread()   # do not let a feeder thread block exit
                except Exception:
                    pass
        self._proc = None
        self._req_q = None
        self._resp_q = None
