# SPDX-License-Identifier: AGPL-3.0-or-later
"""A mid-stream client disconnect must not orphan the producer thread.

Regression guard: `_stream_sse` / `_stream_sse_completion` run
`engine.chat_stream(...)` on a daemon thread. llama.py's `_generate` holds the
per-model `_inference_lock` across its whole generator body. Before the fix, a
disconnect released the request SEMAPHORE but left the producer thread running to
end-of-generation, so it kept holding `_inference_lock` and the NEXT request to
the same model blocked on it.

These tests drive the REAL streaming coroutines. The engine stand-in's
`chat_stream` holds a REAL `threading.Lock` for the whole generator body and
releases it on exhaustion OR on close() - exactly the lock discipline of
`_generate`'s `with self._inference_lock:`. The only faked piece is native token
production, which is not what the fix changed: the fix is the cancel path in the
HTTP layer plus the generator-close cascade, and both run for real here.
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
    from localm.inference.textnorm import scrub_stream

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
# REAL native inference: the unit tests above stub token production with a
# lock-holding generator. This one proves the ACTUAL llama.py _generate releases
# the ACTUAL _inference_lock when the real backend generator chain
# (scrub_stream -> gguf -> create_chat_completion -> _stream_chunks ->
# _decode_stream -> _generate) is closed mid-stream - closing the gap that would
# otherwise let the fix pass on a mock of exactly the thing that was broken.
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
    """Close the REAL generator chain mid-generation and prove the REAL
    _inference_lock is freed, so a follow-up generation on the same model is not
    blocked. This is the production mechanism the disconnect fix relies on."""
    be = gguf_backend
    lock = be._llm._inference_lock          # the real per-model serialisation lock

    gen = be.chat_stream(
        [{"role": "user", "content": "Count from 1 to 300, one number per line."}],
        max_tokens=300, temperature=0.0, seed=1,
    )
    first = next(gen)                        # prefill + first token(s): lock now held
    assert isinstance(first, str)
    # Non-reentrant Lock: a failed non-blocking acquire proves it is held (by the
    # suspended generator), i.e. generation is genuinely in flight.
    assert not lock.acquire(blocking=False), "lock should be held mid-generation"

    gen.close()                             # what the worker does on cancel/disconnect

    acquired = lock.acquire(timeout=5.0)
    assert acquired, "inference lock still held after mid-stream close(): would block next request"
    lock.release()

    # A fresh generation on the SAME loaded model must run, not deadlock.
    out = "".join(be.chat_stream(
        [{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10, temperature=0.0, seed=1,
    )).strip()
    assert any(c.isalpha() for c in out), f"follow-up generation broke: {out!r}"
    assert be.loaded
