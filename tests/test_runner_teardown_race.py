# SPDX-License-Identifier: AGPL-3.0-or-later
"""shutdown() racing an in-flight command must not look like a broken runtime.

``ModelRunner.shutdown()`` takes no lock (so teardown works while a command holds
``_q_lock``) and CLOSES the three queues BEFORE it nulls them. So a command
polling the response queue on another thread - a background preload, a live
stream, a token count - can land on either side of that window:

* closed queue: ``multiprocessing.Queue.get()`` after ``close()`` raises
  ``ValueError``, NOT ``Empty``, so it slips straight past
  ``except _queue.Empty``;
* ``_proc`` already None: ``AttributeError: 'NoneType' object has no attribute
  'is_alive'``.

Either one surfaces as "Native llama runtime failed to load: 'NoneType' object
has no attribute 'is_alive'. Provision or repair it with localm setup-llama" -
telling the user to repair a healthy install.
"""

import multiprocessing as mp
import queue as _queue

import pytest

from localm.inference.backends.base import ModelLoadCancelled
from localm.inference.backends.llamacpp import _runner as R


class _FakeProc:
    exitcode = None

    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


class _AlwaysEmpty:
    """A response queue that never yields, so the loop always takes the Empty
    branch - which is where the teardown check has to live."""

    def __init__(self, on_get=None):
        self._on_get = on_get

    def get(self, timeout=None):
        if self._on_get is not None:
            self._on_get()
        raise _queue.Empty

    def put(self, *a, **k):
        pass


def _runner_with(resp_q, proc=None):
    r = R.ModelRunner()
    r._resp_q = resp_q
    r._req_q = _AlwaysEmpty()
    r._ctrl_q = _AlwaysEmpty()
    r._proc = proc if proc is not None else _FakeProc()
    return r


# --------------------------------------------------------------------------- #
#  _poll: the single place the two teardown shapes are normalised              #
# --------------------------------------------------------------------------- #

def test_poll_lets_empty_through_untouched():
    """Empty is the normal keep-waiting signal every loop is built around; if
    _poll swallowed it, a load would spin instead of honouring its deadline."""
    r = _runner_with(_AlwaysEmpty())
    with pytest.raises(_queue.Empty):
        r._poll(0.01)


def test_poll_reports_a_nulled_queue_as_torn_down():
    r = _runner_with(None)
    with pytest.raises(R._RunnerTornDown):
        r._poll(0.01)


def test_poll_reports_a_really_closed_queue_as_torn_down():
    """Against a REAL multiprocessing.Queue, not a stub asserting the behaviour
    we assumed: the ValueError-not-Empty detail is what slips past the existing
    handler."""
    q = mp.get_context("spawn").Queue()
    q.close()
    q.cancel_join_thread()
    r = _runner_with(q)
    with pytest.raises(R._RunnerTornDown):
        r._poll(0.01)


def test_exitcode_is_none_rather_than_raising_once_the_proc_is_released():
    r = _runner_with(_AlwaysEmpty(), proc=_FakeProc())
    r._proc = None
    assert r._exitcode() is None


# --------------------------------------------------------------------------- #
#  spawn_and_load                                                              #
# --------------------------------------------------------------------------- #

def _load_runner(resp_q, on_get=None, monkeypatch=None):
    r = R.ModelRunner()

    def _fake_spawn():
        r._req_q = _AlwaysEmpty()
        r._ctrl_q = _AlwaysEmpty()
        r._resp_q = resp_q if resp_q is not None else _AlwaysEmpty(on_get)
        r._proc = _FakeProc()

    r._spawn = _fake_spawn
    return r


def test_load_racing_a_shutdown_reports_cancelled_not_a_broken_runtime():
    """_proc goes None between the get() and the liveness check. Unhandled, that
    raises AttributeError, which GgufBackend.load turns into "Native llama
    runtime failed to load ... repair it with localm setup-llama"."""
    r = None

    def _null_the_proc():
        r._proc = None            # shutdown() landing mid-poll

    r = _load_runner(None, on_get=_null_the_proc)
    with pytest.raises(ModelLoadCancelled):
        r.spawn_and_load({}, timeout=1.0)


def test_load_racing_a_queue_close_reports_cancelled_not_a_broken_runtime():
    """The other side of the same window: shutdown() closes the queues BEFORE it
    nulls them, so this ordering is the one hit FIRST."""
    q = mp.get_context("spawn").Queue()
    q.close()
    q.cancel_join_thread()
    r = _load_runner(q)
    with pytest.raises(ModelLoadCancelled):
        r.spawn_and_load({}, timeout=1.0)


def test_a_genuinely_dead_child_still_reports_a_crash():
    """The fix must not swallow the case the branch exists for. A child that
    really died - proc present, not alive - still gets the crash message with
    its exit code, not a cancellation."""
    r = R.ModelRunner()

    def _fake_spawn():
        r._req_q = _AlwaysEmpty()
        r._ctrl_q = _AlwaysEmpty()
        r._resp_q = _AlwaysEmpty()
        dead = _FakeProc(alive=False)
        dead.exitcode = 3
        r._proc = dead

    r._spawn = _fake_spawn
    with pytest.raises(RuntimeError, match="exit code 3"):
        r.spawn_and_load({}, timeout=1.0)


# --------------------------------------------------------------------------- #
#  the same window on the streaming and simple-command paths                   #
# --------------------------------------------------------------------------- #

def test_chat_stream_racing_a_shutdown_says_unloaded_not_attribute_error():
    r = None

    def _null_the_proc():
        r._proc = None

    r = _runner_with(_AlwaysEmpty(_null_the_proc))
    with pytest.raises(RuntimeError, match="unloaded"):
        list(r.chat_stream(messages=[{"role": "user", "content": "hi"}],
                           first_chunk_timeout=1.0))


def test_simple_request_racing_a_shutdown_says_unloaded_not_attribute_error():
    r = None

    def _null_the_proc():
        r._proc = None

    r = _runner_with(_AlwaysEmpty(_null_the_proc))
    with pytest.raises(RuntimeError, match="unloaded"):
        r.count_tokens("some text")


@pytest.mark.parametrize("exitcode,expect_native", [
    (-11, True),    # SIGSEGV: a genuine native fault
    (1, False),     # multiprocessing's signature for an uncaught Python exception
])
def test_a_genuinely_dead_child_is_reported_as_dead_not_unloaded(
        exitcode, expect_native, monkeypatch):
    """Fires-control for the stream path: a real death must be reported AS A
    DEATH, carrying its exit code - never as "unloaded", which is what the
    teardown-race window produces.

    Parametrized over both death classes rather than pinning one message while
    setting ``exitcode = 1``, which is multiprocessing's signature for an
    uncaught PYTHON exception and not a native fault at all. The load-bearing
    property is "reported as a death, with the code", not the particular
    wording, so the distinction is guarded on the teardown path here and not
    only in tests/test_worker_exit_code_decoding.py."""
    monkeypatch.setattr("localm._mp_spawn.os.name", "posix")
    r = _runner_with(_AlwaysEmpty(), proc=_FakeProc(alive=False))
    r._proc.exitcode = exitcode
    with pytest.raises(RuntimeError) as ei:
        list(r.chat_stream(messages=[{"role": "user", "content": "hi"}],
                           first_chunk_timeout=1.0))
    msg = str(ei.value)
    # A real death is never mistaken for an unload.
    assert "unloaded while" not in msg, msg
    assert str(exitcode) in msg, msg
    assert ("Native inference fault" in msg) is expect_native, msg
