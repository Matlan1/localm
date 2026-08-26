# SPDX-License-Identifier: AGPL-3.0-or-later
"""A mid-stream client disconnect must not orphan the producer thread.

`_stream_sse` / `_stream_sse_completion` run `engine.chat_stream(...)` on a
daemon thread, and llama.py's `_generate` holds the per-model `_inference_lock`
across its whole generator body. Releasing only the request SEMAPHORE on
disconnect would leave the producer thread running to end-of-generation, still
holding `_inference_lock`, blocking the NEXT request to the same model.

These tests drive the REAL streaming coroutines. The engine stand-in's
`chat_stream` holds a REAL `threading.Lock` for the whole generator body and
releases it on exhaustion OR on close() - the same lock discipline as
`_generate`'s `with self._inference_lock:`. The only faked piece is native token
production; the cancel path in the HTTP layer and the generator-close cascade
both run for real.
"""

from __future__ import annotations

import asyncio
import threading
import time

from localm.inference.http_server import (
    _pin_engine,
    _stream_sse,
    _stream_sse_completion,
)


class _LockingEngine:
    """Engine whose chat_stream holds a real lock for its whole body, like
    llama.py `_generate` holds `_inference_lock`. With `ntokens=None` it never
    ends on its own, so the lock is released ONLY via the cancel path - a broken
    fix leaves it held forever and the test times out."""

    def __init__(self, ntokens: int | None = None, per_token_delay: float = 0.005):
        self.display_name = "lock-model"
        self.last_finish_reason = "stop"
        # Stands in for LlamaModel._inference_lock (one lock, shared across calls
        # to the same model - that is exactly what serialises requests).
        self.inference_lock = threading.Lock()
        self.entered = threading.Event()   # set once a generation actually starts
        self._ntokens = ntokens
        self._delay = per_token_delay

    # --- methods the streaming coroutines call ---
    def count_messages_tokens(self, messages):
        return 3

    def count_tokens(self, text):
        return len(str(text).split())

    def context_capacity(self):
        return None   # skip the compaction branch in _stream_sse

    def chat_stream(self, messages, **kwargs):
        return self._stream()

    def _stream(self):
        with self.inference_lock:          # mirrors `with self._inference_lock:`
            self.entered.set()
            i = 0
            while self._ntokens is None or i < self._ntokens:
                yield f"t{i} "
                i += 1
                time.sleep(self._delay)


async def _wait(cond, want=True, timeout: float = 3.0) -> bool:
    """Poll *cond* on the event loop until it equals *want* or *timeout* elapses.
    The condition flips from another (worker) thread, so we cannot just read it
    once."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if bool(cond()) == want:
            return True
        await asyncio.sleep(0.01)
    return bool(cond()) == want


_MSG = [{"role": "user", "content": "hi"}]


def test_chat_stream_disconnect_releases_inference_lock():
    async def scenario():
        eng = _LockingEngine()               # never-ending generation
        sem = asyncio.Semaphore(1)

        agen = _stream_sse(eng, _MSG, "lock-model", sem)
        role = await agen.__anext__()        # role announcement chunk
        assert "assistant" in role
        first = await agen.__anext__()        # first real token -> worker running
        assert "t0" in first
        assert await _wait(lambda: eng.inference_lock.locked(), True, 2.0), \
            "producer should hold the inference lock while generating"

        # Simulate the client disconnecting mid-stream.
        await agen.aclose()

        # The producer must observe the cancel, close the generator chain, and
        # release _inference_lock - not run to end-of-generation.
        assert await _wait(lambda: eng.inference_lock.locked(), False, 3.0), \
            "inference lock still held after disconnect: producer thread orphaned"

        # And a brand-new request over the SAME engine must acquire the lock and
        # stream promptly, instead of blocking behind the orphan.
        agen2 = _stream_sse(eng, _MSG, "lock-model", sem)
        await agen2.__anext__()               # role
        tok = await asyncio.wait_for(agen2.__anext__(), timeout=3.0)
        assert "t0" in tok
        await agen2.aclose()
        assert await _wait(lambda: eng.inference_lock.locked(), False, 3.0)

    asyncio.run(scenario())


def test_completions_stream_disconnect_releases_inference_lock():
    async def scenario():
        eng = _LockingEngine()
        sem = asyncio.Semaphore(1)

        agen = _stream_sse_completion(eng, _MSG, "lock-model", sem)
        first = await agen.__anext__()        # first token -> worker running
        assert "t0" in first
        assert await _wait(lambda: eng.inference_lock.locked(), True, 2.0)

        await agen.aclose()                   # disconnect

        assert await _wait(lambda: eng.inference_lock.locked(), False, 3.0), \
            "inference lock still held after disconnect (completions path)"

        agen2 = _stream_sse_completion(eng, _MSG, "lock-model", sem)
        tok = await asyncio.wait_for(agen2.__anext__(), timeout=3.0)
        assert "t0" in tok
        await agen2.aclose()

    asyncio.run(scenario())


def test_disconnect_through_pin_engine_releases_lock():
    """Production path: Starlette acloses the OUTER `_pin_engine` wrapper, not the
    inner `_stream_sse` directly. Closing the wrapper must propagate the cancel
    into the inner stream so the lock is released here and now."""
    async def scenario():
        eng = _LockingEngine()
        sem = asyncio.Semaphore(1)

        wrapped = _pin_engine(eng, _stream_sse(eng, _MSG, "lock-model", sem))
        await wrapped.__anext__()             # role
        await wrapped.__anext__()             # first token -> worker running
        assert await _wait(lambda: eng.inference_lock.locked(), True, 2.0)

        await wrapped.aclose()                # disconnect at the wrapper level

        assert await _wait(lambda: eng.inference_lock.locked(), False, 3.0), \
            "inference lock still held after wrapper-level disconnect"

    asyncio.run(scenario())


def test_normal_chat_stream_completes_and_releases_lock():
    """Happy path is unbroken: a finite generation streams all tokens, ends with
    [DONE]/finish_reason=stop, and leaves the lock free (the new try/finally +
    cancel_event must not corrupt a clean completion)."""
    async def scenario():
        eng = _LockingEngine(ntokens=3, per_token_delay=0.0)
        sem = asyncio.Semaphore(1)
        chunks = []
        async for chunk in _stream_sse(eng, _MSG, "lock-model", sem):
            chunks.append(chunk)
        body = "".join(chunks)
        assert "t0" in body and "t2" in body
        assert "[DONE]" in body
        assert '"finish_reason":"stop"' in body.replace(" ", "")
        assert "[inference error" not in body
        assert not eng.inference_lock.locked()

    asyncio.run(scenario())


def test_close_cascades_through_scrub_stream_releases_lock():
    """The cancel path relies on `.close()` cascading through the real backend
    wrapper generators. `engine.chat_stream` wraps the token stream in
    `textnorm.scrub_stream` (the outermost production wrapper); closing it must
    propagate GeneratorExit into the inner generator and release its lock."""
    from localm.textnorm import scrub_stream

    lock = threading.Lock()
    started = threading.Event()

    def inner():
        with lock:                            # like _generate's _inference_lock
            started.set()
            i = 0
            while True:
                yield f"t{i} "
                i += 1

    gen = scrub_stream(inner())
    # Drive it far enough that the inner generator has started (acquired the lock).
    for _ in range(8):
        next(gen)
    assert started.is_set()
    assert lock.locked()

    gen.close()                               # cascades into inner -> releases lock
    assert not lock.locked(), "close() did not cascade through scrub_stream"


# --------------------------------------------------------------------------- #
# NON-streaming disconnect (the _complete / _generate_full twin of the above).
#
# A non-streaming handler is a plain coroutine, so Starlette does NOT aclose it on
# a client disconnect the way it acloses a StreamingResponse's generator. The
# handler instead polls request.is_disconnected() and, on disconnect, signals the
# executor worker to stop and gen.close() the chain - releasing _inference_lock
# now rather than at end-of-generation. These tests drive the REAL
# _generate_full / _complete coroutines; the only faked pieces are token
# production (the lock-holding engine) and the disconnect signal (a controllable
# request).
# --------------------------------------------------------------------------- #


class _FakeRequest:
    """Stands in for a Starlette Request purely for is_disconnected() polling.
    Flip .disconnected from the test to simulate the client aborting."""

    def __init__(self, disconnected: bool = False):
        self.disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self.disconnected


def test_generate_full_disconnect_releases_inference_lock():
    from localm.inference.http_server import _generate_full

    async def scenario():
        eng = _LockingEngine()                # never-ending generation
        req = _FakeRequest()

        task = asyncio.ensure_future(_generate_full(eng, _MSG, req, max_tokens=0))
        # Worker starts and grabs the lock (mirrors _generate's _inference_lock).
        assert await _wait(lambda: eng.inference_lock.locked(), True, 2.0), \
            "worker should hold the inference lock while generating"
        assert not task.done()

        req.disconnected = True               # client aborts mid-generation

        # The poller must observe the disconnect, stop the worker, close the
        # generator chain, and release the lock - not run to end-of-generation.
        text = await asyncio.wait_for(task, timeout=3.0)
        assert isinstance(text, str)          # partial text, not an error
        assert await _wait(lambda: eng.inference_lock.locked(), False, 3.0), \
            "inference lock still held after non-stream disconnect: worker orphaned"

        # A brand-new generation over the SAME engine must acquire the lock
        # promptly instead of blocking behind the orphan.
        req2 = _FakeRequest()
        t2 = asyncio.ensure_future(_generate_full(eng, _MSG, req2, max_tokens=0))
        assert await _wait(lambda: eng.inference_lock.locked(), True, 3.0), \
            "follow-up generation blocked - lock was not released"
        req2.disconnected = True
        await asyncio.wait_for(t2, timeout=3.0)

    asyncio.run(scenario())


def test_generate_full_completes_and_releases_lock():
    """Happy path: a finite generation returns the full joined text and leaves the
    lock free (the cancel_event + watcher must not corrupt a clean completion)."""

    from localm.inference.http_server import _generate_full

    async def scenario():
        eng = _LockingEngine(ntokens=3, per_token_delay=0.0)
        req = _FakeRequest()                  # stays connected the whole time
        text = await asyncio.wait_for(
            _generate_full(eng, _MSG, req, max_tokens=64), timeout=5.0)
        assert text == "t0 t1 t2 "
        assert not eng.inference_lock.locked()

    asyncio.run(scenario())


def test_generate_full_none_request_runs_to_completion():
    """request=None (a caller with no disconnect signal) is inert: the generation
    runs to completion and the lock is released, never a spurious cancel."""

    from localm.inference.http_server import _generate_full

    async def scenario():
        eng = _LockingEngine(ntokens=2, per_token_delay=0.0)
        text = await asyncio.wait_for(
            _generate_full(eng, _MSG, None, max_tokens=64), timeout=5.0)
        assert text == "t0 t1 "
        assert not eng.inference_lock.locked()

    asyncio.run(scenario())


def test_complete_disconnect_releases_inference_lock():
    """End-to-end through the real chat non-streaming handler _complete: a mid-
    generation disconnect must release _inference_lock (and the semaphore) so the
    next request to the same model is not blocked."""

    from localm.inference.http_server import _complete

    async def scenario():
        eng = _LockingEngine()                # never-ending
        sem = asyncio.Semaphore(1)
        req = _FakeRequest()

        task = asyncio.ensure_future(
            _complete(eng, _MSG, "lock-model", sem, request=req, max_tokens=0))
        assert await _wait(lambda: eng.inference_lock.locked(), True, 2.0)
        assert not task.done()

        req.disconnected = True               # abort
        # _complete still returns (with partial content); the point is the lock is
        # freed, not the discarded response.
        await asyncio.wait_for(task, timeout=3.0)
        assert await _wait(lambda: eng.inference_lock.locked(), False, 3.0), \
            "inference lock still held after _complete disconnect"

        # Semaphore was released too: a fresh _complete acquires it and runs.
        req2 = _FakeRequest()
        t2 = asyncio.ensure_future(
            _complete(eng, _MSG, "lock-model", sem, request=req2, max_tokens=0))
        assert await _wait(lambda: eng.inference_lock.locked(), True, 3.0), \
            "follow-up _complete blocked - lock/semaphore not released"
        req2.disconnected = True
        await asyncio.wait_for(t2, timeout=3.0)

    asyncio.run(scenario())


def test_complete_happy_path_returns_response_and_releases_lock():
    """A non-streaming request that runs to completion still returns a valid
    ChatResponse JSON and leaves the lock free."""

    import json as _json

    from localm.inference.http_server import _complete

    async def scenario():
        eng = _LockingEngine(ntokens=3, per_token_delay=0.0)
        sem = asyncio.Semaphore(1)
        req = _FakeRequest()

        resp = await asyncio.wait_for(
            _complete(eng, _MSG, "lock-model", sem, request=req, max_tokens=64),
            timeout=5.0)
        body = _json.loads(bytes(resp.body))
        content = body["choices"][0]["message"]["content"]
        assert "t0" in content and "t2" in content
        assert body["choices"][0]["finish_reason"] == "stop"
        assert not eng.inference_lock.locked()

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# REAL native inference: the unit tests above stub token production with a
# lock-holding generator. This one drives the ACTUAL llama.py _generate and
# asserts it releases the ACTUAL _inference_lock when the real backend generator
# chain (scrub_stream -> gguf -> create_chat_completion -> _stream_chunks ->
# _decode_stream -> _generate) is closed mid-stream.
# @integration + @real_gguf: needs the native runtime + a small real model.
# --------------------------------------------------------------------------- #

import pytest   # noqa: E402

_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"


@pytest.fixture(scope="module")
def gguf_backend():
    try:
        from localm.inference.backends.llamacpp._loader import load_lib
        load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned (run 'localm setup-llama'): {e}")

    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=_REPO, filename=_FILE)
    except Exception as e:
        pytest.skip(f"could not fetch {_REPO}/{_FILE}: {e}")

    from localm.inference.backends.gguf import GgufBackend
    be = GgufBackend(path, n_ctx=2048)
    try:
        be.load()
    except Exception as e:
        pytest.skip(f"GGUF model failed to load on this machine: {e}")
    yield be
    be.unload()


@pytest.mark.integration
@pytest.mark.real_gguf
def test_real_gguf_midstream_close_releases_inference_lock(gguf_backend):
    """Close the REAL generator chain mid-generation and prove the model is
    NOT left wedged: a follow-up generation on the same loaded model must
    start and produce output promptly. This is the production mechanism the
    disconnect fix relies on.

    The real `_inference_lock` object cannot be inspected directly: the real
    LlamaCpp instance (and its lock) live inside an isolated worker PROCESS
    (see llamacpp/_runner.py), so no Python object in THIS process can observe
    its state. The property is asserted by timing instead: a bounded wall-clock
    assertion that the next generation starts promptly rather than hanging
    behind a still-held lock in the child."""
    import time

    be = gguf_backend

    gen = be.chat_stream(
        [{"role": "user", "content": "Count from 1 to 300, one number per line."}],
        max_tokens=300, temperature=0.0, seed=1,
    )
    first = next(gen)                        # prefill + first token(s): generation in flight
    assert isinstance(first, str)

    gen.close()                             # what the worker does on cancel/disconnect

    t0 = time.monotonic()
    out = "".join(be.chat_stream(
        [{"role": "user", "content": "Say only the word: hi"}],
        max_tokens=10, temperature=0.0, seed=1,
    ))
    elapsed = time.monotonic() - t0
    assert elapsed < 30.0, (
        f"follow-up generation took {elapsed:.1f}s after mid-stream close() - "
        "the worker's inference lock may still be held (would block real requests)"
    )
    assert out.strip()

    # A fresh generation on the SAME loaded model must run, not deadlock.
    out = "".join(be.chat_stream(
        [{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10, temperature=0.0, seed=1,
    )).strip()
    assert any(c.isalpha() for c in out), f"follow-up generation broke: {out!r}"
    assert be.loaded


# --------------------------------------------------------------------------- #
# REAL uvicorn server: a REAL client abort of a REAL non-streaming request over a
# REAL uvicorn server makes request.is_disconnected() fire, so the lock is freed.
# No model or network needed - a lock-holding fake engine stands in for the
# native backend, exactly as in the unit tests, so this runs in the default gate.
# --------------------------------------------------------------------------- #


class _ServerLockingEngine(_LockingEngine):
    """_LockingEngine wired to satisfy the real chat handler + get_engine: it must
    look loaded (so get_engine returns it directly) and text-only."""

    def __init__(self):
        super().__init__()               # ntokens=None -> never ends on its own
        self.loaded = True
        self.supports_images = False
        self.can_be_multimodal = False


def _wait_sync(cond, want=True, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bool(cond()) == want:
            return True
        time.sleep(0.02)
    return bool(cond()) == want


def _raw_chat_request(port: int) -> bytes:
    body = (
        b'{"model":"lock-model","messages":[{"role":"user","content":"hi"}],'
        b'"stream":false}'
    )
    return (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: close\r\n"
        b"\r\n" + body
    )


def test_real_uvicorn_nonstream_disconnect_releases_inference_lock():
    """Production path: a client that opens a non-streaming /v1/chat/completions
    request and then drops the socket mid-generation must make the server release
    _inference_lock (via request.is_disconnected() polling), so the model is not
    wedged for the next caller. Proves the disconnect DETECTION, not just the
    cancel mechanism."""
    import socket as _socket

    import uvicorn

    from localm.inference.http_server import create_app

    eng = _ServerLockingEngine()
    app = create_app(eng, api_landing=True)

    # Pre-bind an ephemeral port so we know it before the server starts.
    lsock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    lsock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    port = lsock.getsockname()[1]

    config = uvicorn.Config(app, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)

    def _serve():
        # Fresh event loop in this thread; uvicorn drives the ASGI app here.
        asyncio.run(server.serve(sockets=[lsock]))

    th = threading.Thread(target=_serve, daemon=True)
    th.start()
    try:
        assert _wait_sync(lambda: server.started, True, 10.0), "uvicorn did not start"

        # 1) Open a non-streaming request and abort it mid-generation.
        client = _socket.create_connection(("127.0.0.1", port), timeout=5.0)
        client.sendall(_raw_chat_request(port))
        assert _wait_sync(lambda: eng.inference_lock.locked(), True, 6.0), \
            "server never started generating (lock not held)"
        client.close()                          # <-- the disconnect under test

        # The disconnect poller must fire and free the lock - not run forever
        # (the fake engine never ends on its own, so a broken fix hangs here).
        assert _wait_sync(lambda: eng.inference_lock.locked(), False, 8.0), \
            "inference lock still held after real client disconnect: is_disconnected() never fired"

        # 2) The model is usable again: a second request acquires the lock promptly
        #    instead of blocking behind the orphaned first one.
        client2 = _socket.create_connection(("127.0.0.1", port), timeout=5.0)
        client2.sendall(_raw_chat_request(port))
        assert _wait_sync(lambda: eng.inference_lock.locked(), True, 6.0), \
            "follow-up request blocked - lock was not truly released"
        client2.close()
        assert _wait_sync(lambda: eng.inference_lock.locked(), False, 8.0)
    finally:
        server.should_exit = True
        th.join(timeout=10.0)


# --------------------------------------------------------------------------- #
# Eager engine error in the producer thread must not orphan the consumer.
#
# Engine.chat_stream is NOT a generator: it eagerly runs the auto-reload
# (self._backend.load()) and load_config() BEFORE returning the token generator,
# so it can RAISE at call time (a reload that OOMs, a since-removed GGUF). If the
# producer raises before the sentinel is enqueued, the consumer blocks forever at
# `await token_queue.get()` inside `async with sem`, deadlocking every later
# request to that model. These tests drive the REAL _stream_sse /
# _stream_sse_completion coroutines and require the failure to be surfaced and
# the semaphore released.
# --------------------------------------------------------------------------- #


class _EagerRaiseEngine:
    """Engine whose chat_stream raises EAGERLY, before returning a generator -
    exactly what Engine.chat_stream does when its auto-reload load() fails."""

    def __init__(self):
        self.display_name = "raise-model"
        self.last_finish_reason = "stop"
        self.calls = 0

    def count_messages_tokens(self, messages):
        return 3

    def count_tokens(self, text):
        return 1

    def context_capacity(self):
        return None   # skip the compaction branch

    def chat_stream(self, messages, **kwargs):
        self.calls += 1
        raise RuntimeError("reload failed (VRAM OOM)")


async def _drain(agen):
    chunks = []
    async for c in agen:
        chunks.append(c)
    return chunks


def test_chat_stream_eager_engine_error_releases_sem_and_surfaces_error():
    async def scenario():
        eng = _EagerRaiseEngine()
        sem = asyncio.Semaphore(1)

        # Must COMPLETE, not hang: a broken producer never enqueues the sentinel,
        # so the consumer awaits the queue forever and this wait_for times out.
        chunks = await asyncio.wait_for(
            _drain(_stream_sse(eng, _MSG, "raise-model", sem)), timeout=5.0)

        assert any("inference error" in c for c in chunks), \
            "the eager failure must be surfaced to the client, not swallowed"
        assert not sem.locked(), "semaphore leaked after an eager engine error"
        assert eng.calls == 1

        # And a brand-new request to the SAME model is not deadlocked behind a
        # held semaphore - it runs its handler (which also raises, also completes).
        chunks2 = await asyncio.wait_for(
            _drain(_stream_sse(eng, _MSG, "raise-model", sem)), timeout=5.0)
        assert any("inference error" in c for c in chunks2)
        assert not sem.locked()

    asyncio.run(scenario())


def test_completions_stream_eager_engine_error_releases_sem_and_surfaces_error():
    async def scenario():
        eng = _EagerRaiseEngine()
        sem = asyncio.Semaphore(1)

        chunks = await asyncio.wait_for(
            _drain(_stream_sse_completion(eng, _MSG, "raise-model", sem)), timeout=5.0)

        assert any("inference error" in c for c in chunks), \
            "the eager failure must be surfaced to the client (completions path)"
        assert not sem.locked(), "semaphore leaked after an eager engine error (completions)"

        chunks2 = await asyncio.wait_for(
            _drain(_stream_sse_completion(eng, _MSG, "raise-model", sem)), timeout=5.0)
        assert any("inference error" in c for c in chunks2)
        assert not sem.locked()

    asyncio.run(scenario())
