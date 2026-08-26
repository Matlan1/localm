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

import threading
import time
from unittest.mock import MagicMock, patch

from localm.inference.backends.llamacpp.llama import LlamaCpp
from tests._bare_llama import make_bare_llama


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
