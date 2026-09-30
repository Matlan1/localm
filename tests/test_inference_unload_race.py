# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Crash-safety contract for unload-during-generation.

The native llama.cpp context must never be freed while a generation step is
making a native call against it (a use-after-free that crashes the GPU
driver). The wrapper guarantees this with a per-instance lock + stop event:
close() signals stop, then frees *under* the lock, so it blocks until any
in-flight native region releases - and the generation loop bails at its next
step because stop is set / the context is gone.

These tests exercise that contract directly without loading the DLL.
"""

import os
import threading
import time
from unittest.mock import MagicMock, patch

import localm.inference.backends.llamacpp.llama as llama_mod
from localm import debuglog
from localm.inference.backends.llamacpp.llama import LlamaCpp
from tests._bare_llama import make_bare_llama
from tests._fake_batch import fake_batch_init


def _lockable_llama() -> LlamaCpp:
    return make_bare_llama(
        _cached_tokens=[1, 2, 3],
        _ctx_ptr=None,      # nothing to actually free in this unit test
        _model_ptr=None,
        _verbose=True,
    )


def test_close_signals_stop_immediately_and_waits_for_the_lock():
    """While a 'native region' holds _gen_lock, close() must not free - but it
    must set _stop right away so the generation loop bails at its next step."""
    llm = _lockable_llama()

    llm._gen_lock.acquire()              # simulate a decode in its locked region
    close_returned = threading.Event()

    def _do_close():
        llm.close()
        close_returned.set()

    t = threading.Thread(target=_do_close, daemon=True)
    t.start()
    time.sleep(0.05)

    # stop is signalled immediately (the generator will see it and abort)...
    assert llm._stop.is_set()
    # ...but close() is still blocked on the lock - it cannot free yet.
    assert not close_returned.is_set()

    llm._gen_lock.release()              # the native region finishes
    t.join(timeout=2)
    assert close_returned.is_set()       # now close() proceeds and frees
    assert llm._cached_tokens == []


def test_stop_set_before_lock_is_observable_by_a_generator_step():
    """A generator step checks `_stop.is_set() or _ctx_ptr is None` inside the
    lock; once close() has run, both are true, so the step aborts instead of
    touching a freed context."""
    llm = _lockable_llama()
    llm._ctx_ptr = 1234      # pretend a live context
    llm._model_ptr = 5678

    with patch("localm.inference.backends.llamacpp.llama.api", MagicMock()):
        llm.close()          # frees (mocked native) and sets stop

    assert llm._stop.is_set()
    # the conditions the locked native regions guard on are both satisfied
    with llm._gen_lock:
        aborts = llm._stop.is_set() or llm._ctx_ptr is None
    assert aborts


def _mock_native_api() -> MagicMock:
    """A mock api module for the KV-reuse prefill, the decode loop and free."""
    mock_api = MagicMock()
    mock_api.has_memory_api.return_value = True
    mock_api.llama_get_memory.return_value = 333
    mock_api.llama_memory_seq_rm.return_value = True
    mock_api.llama_decode.return_value = 0
    mock_api.llama_batch_init.side_effect = fake_batch_init
    mock_api.llama_sampler_sample.return_value = 42
    mock_api.llama_model_has_mrope.return_value = False
    return mock_api


def test_close_during_a_suspended_grammar_generation_does_not_deadlock(monkeypatch):
    """A non-verbose grammar generation is suspended at its yield while close()
    runs on another thread and the generator is resumed on a third: close()
    frees the native state, and both threads finish.

    On a deadlock the test releases the module's _stderr_lock so the stuck
    threads can finish, then fails."""
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222)
    llm._tokenizer.is_eog.return_value = False
    outcome = {}

    def _close():
        llm.close()
        outcome["closed"] = True

    with patch.object(llama_mod, "api", _mock_native_api()), \
         patch.object(llama_mod, "_build_sampler", return_value=999):
        gen = llm._generate(
            prompt_tokens=[1, 2, 3], max_new_tokens=50,
            temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1,
            grammar='root ::= "x"', grammar_lazy=True,
            grammar_triggers=[r"(<tool_call>[\s\S]*)"])
        assert next(gen) == 42

        def _resume():
            try:
                next(gen)
            except StopIteration:
                outcome["finished"] = True
            except BaseException as exc:
                outcome["error"] = exc

        closer = threading.Thread(target=_close, daemon=True)
        closer.start()
        deadline = time.monotonic() + 10
        while closer.is_alive() and time.monotonic() < deadline:
            if not llm._gen_lock.acquire(blocking=False):
                break
            llm._gen_lock.release()
            time.sleep(0.005)
        resumer = threading.Thread(target=_resume, daemon=True)
        resumer.start()
        deadline = time.monotonic() + 15
        for thread in (closer, resumer):
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

        freed = llm._ctx_ptr is None and llm._model_ptr is None
        stuck = closer.is_alive() or resumer.is_alive()
        if stuck:
            if llama_mod._stderr_lock.locked():
                llama_mod._stderr_lock.release()
            for thread in (closer, resumer):
                thread.join(timeout=15)

    assert freed, "close() never freed the native state while the generation was suspended"
    assert not stuck, "close() and the resumed generation deadlocked"
    assert outcome.get("closed"), outcome
    assert outcome.get("finished"), outcome
    assert llm.last_finish_reason == "error"


class _TracingLock:
    """A threading.Lock that records the name of every thread that acquires it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.acquirers: list = []

    def acquire(self, blocking=True, timeout=-1):
        self.acquirers.append(threading.current_thread().name)
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


def test_abandoning_a_generation_during_close_restores_fd2_in_order(monkeypatch, tmp_path):
    """A non-verbose grammar generation is abandoned at its yield on one thread
    while close() on another is inside its _quiet_stderr block: afterwards fd 2
    points where it pointed before the generation started."""
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    monkeypatch.setattr(debuglog, "_READER_JOIN_TIMEOUT", 2.0)
    swap_lock = _TracingLock()
    monkeypatch.setattr(llama_mod, "_stderr_lock", swap_lock)
    fd2_writers: list = []
    real_dup2 = os.dup2

    def _tracing_dup2(fd, fd2, *args, **kwargs):
        if fd2 == 2:
            fd2_writers.append(threading.current_thread().name)
        return real_dup2(fd, fd2, *args, **kwargs)

    in_free = threading.Event()
    release_free = threading.Event()

    def _blocking_free(_ptr):
        in_free.set()
        release_free.wait(timeout=15)

    mock_api = _mock_native_api()
    mock_api.llama_free.side_effect = _blocking_free
    llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222)
    llm._tokenizer.is_eog.return_value = False

    stream = tmp_path / "stderr.bin"
    original_fd2 = os.dup(2)
    target = os.open(str(stream), os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                     | getattr(os, "O_BINARY", 0))
    real_dup2(target, 2)
    os.close(target)
    monkeypatch.setattr(os, "dup2", _tracing_dup2)
    try:
        with patch.object(llama_mod, "api", mock_api), \
             patch.object(llama_mod, "_build_sampler", return_value=999):
            gen = llm._generate(
                prompt_tokens=[1, 2, 3], max_new_tokens=50,
                temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1,
                grammar='root ::= "x"', grammar_lazy=True,
                grammar_triggers=[r"(<tool_call>[\s\S]*)"])
            assert next(gen) == 42

            closer = threading.Thread(target=llm.close, name="closer", daemon=True)
            closer.start()
            assert in_free.wait(timeout=10), "close() never reached the native free"
            abandoner = threading.Thread(target=gen.close, name="abandoner", daemon=True)
            abandoner.start()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not (
                    "abandoner" in swap_lock.acquirers or "abandoner" in fd2_writers):
                time.sleep(0.005)
            release_free.set()
            for thread in (closer, abandoner):
                thread.join(timeout=15)
            finished = not (closer.is_alive() or abandoner.is_alive())

        os.write(2, b"MARKER-AFTER-THE-GENERATION\n")
        landed = b"MARKER-AFTER-THE-GENERATION" in stream.read_bytes()
    finally:
        release_free.set()
        real_dup2(original_fd2, 2)
        os.close(original_fd2)

    assert landed, "fd 2 no longer points where it pointed before the generation"
    assert finished, "close() or the abandoned generation did not finish"
