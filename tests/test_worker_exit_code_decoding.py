# SPDX-License-Identifier: AGPL-3.0-or-later
"""A dead worker's exit code must be reported DECODED, not raw.

The raw number is the most discriminating fact available about a worker death:
on POSIX ``multiprocessing`` reports ``-N`` for death by signal N, so ``-4`` is
SIGILL - an illegal instruction, a different FAMILY of cause from a segfault
(-11) or an abort (-6), pointing the investigation somewhere different.

BOTH OS CONVENTIONS ARE TESTED ON WHATEVER PLATFORM THIS RUNS ON, via
``describe_exit_code``'s explicit *posix* parameter. That is the point of the
parameter: the conventions are mutually exclusive, so a Windows runner can never
produce ``-4``-means-SIGILL and a Linux runner can never produce an NTSTATUS.
Gating these on ``sys.platform`` instead would leave one branch permanently
unexercised while the file still read as covered.
"""

import os

import pytest

from localm._mp_spawn import death_was_a_native_fault, describe_exit_code
from localm.inference.backends.llamacpp import _runner as runner_mod
from localm.inference.backends.llamacpp._runner import ModelRunner


@pytest.fixture(autouse=True)
def _clean_fault_env():
    os.environ.pop(runner_mod._FAULT_ENV, None)
    yield
    os.environ.pop(runner_mod._FAULT_ENV, None)


class TestPosixSignalDecoding:
    def test_the_reported_code_names_sigill(self):
        """The exact code from issues 1222/1223."""
        out = describe_exit_code(-4, posix=True)
        assert "SIGILL" in out, out
        assert "-4" in out, out

    @pytest.mark.parametrize("code,name", [
        (-4, "SIGILL"),     # illegal instruction
        (-11, "SIGSEGV"),   # segfault: a DIFFERENT family of cause
        (-6, "SIGABRT"),    # a deliberate native abort, e.g. GGML_ABORT
        (-9, "SIGKILL"),    # killed from outside, not a fault at all
        (-8, "SIGFPE"),
    ])
    def test_each_fault_family_is_named_distinctly(self, code, name):
        """Naming them is only useful if they are told APART - a decoder that
        collapsed these would be no better than the raw number, since choosing
        between these families is the entire diagnostic value."""
        assert name in describe_exit_code(code, posix=True)

    def test_a_clean_nonzero_exit_is_not_read_as_a_signal(self):
        """A non-negative POSIX code is an ordinary exit status and says nothing
        about signals. Reporting exit 4 as "SIGILL" would invent a native crash
        that never happened - the failure mode this direction has to avoid."""
        out = describe_exit_code(4, posix=True)
        assert "SIGILL" not in out
        assert "signal" not in out.lower()
        assert out == "4"

    def test_a_negative_that_is_not_a_signal_says_what_is_known(self):
        """-99 is not a signal on any supported build. It must not crash and must
        not invent a name."""
        out = describe_exit_code(-99, posix=True)
        assert "-99" in out
        assert "None" not in out

    def test_posix_codes_are_not_decoded_through_the_host_signal_enum(self):
        """REGRESSION GUARD, so the decoder is not "simplified" back to a bare
        signal.Signals() lookup.

        The two enums disagree on the numbers that matter.

            Windows  SIGABRT == 22        Linux  SIGABRT ==  6
            Windows  6 is absent          Linux  22 == SIGTTOU

        So decoding a POSIX code through the HOST enum does not merely fail to
        name SIGABRT on a Windows box - for 22 it returns a CONFIDENTLY WRONG
        name. A POSIX code must be read with POSIX numbering regardless of which
        interpreter is doing the reading, which is what makes a Linux bug report
        legible to a maintainer on Windows."""
        assert "SIGABRT" in describe_exit_code(-6, posix=True)
        # 22 is SIGTTOU on Linux; it must NOT come back as Windows' SIGABRT.
        assert "SIGABRT" not in describe_exit_code(-22, posix=True)


class TestWindowsNtstatusDecoding:
    def test_negative_is_not_read_as_a_signal_on_windows(self):
        """THE REASON THE TWO BRANCHES ARE SPLIT. Python's own
        Process.terminate() calls TerminateProcess(handle, -1) on Windows, and
        _runner.shutdown() calls terminate() - so -1 is a code this project
        really produces there. Reading it as SIGHUP would label a deliberate
        teardown as a fatal signal."""
        out = describe_exit_code(-1, posix=False)
        assert "SIGHUP" not in out
        assert "signal" not in out.lower()

    def test_the_measured_native_abort_code_is_named(self):
        """3221226505 == 0xC0000409, the exit code a native os.abort()
        produces on Windows."""
        out = describe_exit_code(3221226505, posix=False)
        assert "0xC0000409" in out
        assert "abort" in out.lower()

    def test_illegal_instruction_is_named_on_windows_too(self):
        """The Windows equivalent of the reported SIGILL, so the same class of
        report from a Windows tester is equally readable."""
        assert "illegal instruction" in describe_exit_code(0xC000001D, posix=False)

    def test_the_dll_conflict_code_self_identifies(self):
        """0xC0000139 is the documented signature of the torch/HIP DLL-identity
        conflict this codebase already guards against. Decoding it means a future
        report says what it is instead of needing that root-cause session again."""
        out = describe_exit_code(0xC0000139, posix=False)
        assert "entry point not found" in out.lower()

    def test_an_ordinary_windows_code_is_left_alone(self):
        assert describe_exit_code(1, posix=False) == "1"


class TestNativeFaultClassification:
    """A worker death must not be CALLED a native fault unless it was one.

    Exit 1 is an uncaught Python exception rather than a native abort: a missing
    Pillow surfaces as "Native inference fault (worker exit 1)" with a plain
    ModuleNotFoundError in the log unless the classification is made at the site
    that WORDS the message, which a per-cause fix cannot reach.
    """

    @pytest.mark.parametrize("code", [-4, -11, -6, -9])
    def test_a_posix_signal_death_is_a_native_fault(self, code):
        assert death_was_a_native_fault(code, posix=True)

    @pytest.mark.parametrize("code", [0, 1, 3, 134])
    def test_an_ordinary_posix_exit_is_not(self, code):
        """1 is the one that mattered: multiprocessing's signature for an
        uncaught Python exception, and the code in the false message."""
        assert not death_was_a_native_fault(code, posix=True)

    def test_a_windows_ntstatus_is_a_native_fault(self):
        assert death_was_a_native_fault(0xC0000005, posix=False)
        assert death_was_a_native_fault(0xC000001D, posix=False)

    @pytest.mark.parametrize("code", [0, 1, 3, -1])
    def test_an_ordinary_windows_exit_is_not(self, code):
        """-1 is included: Process.terminate() produces it on Windows, and 3 is
        what an ARMED-faulthandler abort exits with. Neither is classifiable from
        the code alone, so neither may be asserted as native."""
        assert not death_was_a_native_fault(code, posix=False)

    def test_a_captured_trace_settles_it_on_either_platform(self):
        """The strongest evidence, and it leads rather than tiebreaks:
        faulthandler only fires on SIGSEGV/SIGFPE/SIGABRT/SIGBUS/SIGILL, so a
        trace means a native signal even when the exit code cannot say so - the
        Windows armed-abort-exits-3 case."""
        assert death_was_a_native_fault(3, trace_captured=True, posix=False)
        assert death_was_a_native_fault(1, trace_captured=True, posix=True)

    @pytest.mark.parametrize("bad", [None, "x", object()])
    def test_junk_is_not_a_native_fault_and_never_raises(self, bad):
        assert death_was_a_native_fault(bad) is False

    def test_a_real_python_exception_death_is_not_called_a_native_fault(self):
        """END TO END, through a REAL child: chat_stream before any load ->
        worker is None -> AttributeError -> uncaught -> exit 1.

        Sited at the runner rather than on the predicate because the property is
        how the message is WORDED, which a predicate test cannot see."""
        r = ModelRunner()
        r._spawn()
        try:
            with pytest.raises(RuntimeError) as ei:
                list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))
            msg = str(ei.value)
        finally:
            r.shutdown(grace=0)

        assert "Native inference fault" not in msg, (
            "an uncaught Python exception in the worker is still reported as a "
            f"native fault\n--- message ---\n{msg}")
        assert "exited unexpectedly" in msg, msg
        # The containment contract itself must survive the rewording.
        assert "reload on the next request" in msg, msg


class TestNeverRaises:
    """This decorates a message on a path that is ALREADY failing. Raising here
    would replace a real crash report with an unrelated traceback."""

    @pytest.mark.parametrize("bad", [None, "not-a-number", object(), 1.5])
    def test_junk_never_raises(self, bad):
        assert isinstance(describe_exit_code(bad), str)

    def test_none_is_explicit_not_the_string_none(self):
        """_exitcode() returns None once the child is released. "exit code None"
        reads like a bug in the reporter."""
        assert describe_exit_code(None) == "unknown"


class TestRunnerReportsTheDecodedCode:
    """The decoder existing is not the property; the RUNNER using it is. These
    are sited at the runner rather than only on the helper because every
    user-facing message must not interpolate the RAW code, which a helper-level
    test cannot see."""

    def test_exit_reason_is_decoded_for_a_signal_death(self, monkeypatch):
        r = ModelRunner()

        class _FakeProc:
            exitcode = -4

        r._proc = _FakeProc()
        monkeypatch.setattr("localm._mp_spawn.os.name", "posix")
        assert "SIGILL" in r._exit_reason()

    def test_a_real_native_death_reports_a_decoded_code_end_to_end(
            self, monkeypatch):
        """The behavioural half, driven through a REAL child process killed by a
        REAL native abort - not a fake exitcode.

        Needed alongside the source scan below because the two catch different
        things: the scan proves no call site interpolates the raw code, this
        proves the decoding survives into the message a user actually reads.

        THE WINDOWS ARM DOES NOT ASSERT 0xC0000409, AND MUST NOT BE "FIXED" TO.
        An unarmed Windows abort exits 0xC0000409, but the worker arms
        faulthandler before anything else and faulthandler installs a SIGABRT
        handler, so os.abort() takes the ordinary CRT path and exits 3 instead of
        __fastfail's NTSTATUS. On Windows this fault mode therefore yields no
        decodable code at all, and what characterises the fault is the captured
        trace instead."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "abort")
        r = ModelRunner()
        r._spawn()
        try:
            with pytest.raises(RuntimeError) as ei:
                list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))
            message = str(ei.value)
        finally:
            r.shutdown(grace=0)

        if os.name == "nt":
            # 3 is an ordinary exit status, not an NTSTATUS: the decoder must
            # leave it alone rather than invent a meaning, and above all must not
            # apply the POSIX negative-means-signal rule on this platform.
            assert "worker exit 3)" in message, message
            assert "signal" not in message.lower(), message
            # The fault is still characterised - by the captured trace.
            assert "Aborted" in message, message
        else:
            assert "SIGABRT" in message, message

    def test_no_user_facing_message_interpolates_the_raw_exitcode(self):
        """A grep-style guard on the source itself. Every crash message must go
        through _exit_reason(); a new one that reaches for _exitcode() directly
        would silently reintroduce the bare number, and no behavioural test would
        catch it because the message would still be produced.

        With the call sites reverted to the raw code, this is the only test in
        this file that goes red on Windows: the fault-injection abort exits 3
        there, and "3" is what both the raw and the decoded form produce, so
        nothing distinguishes them. On POSIX the behavioural test goes red too
        (SIGABRT disappears from the message)."""
        import inspect

        from localm.inference.backends.llamacpp import _runner

        src = inspect.getsource(_runner)
        # The raw exit code has exactly TWO legitimate consumers: the decoder
        # that RENDERS it (describe_exit_code) and the classifier that INTERPRETS
        # it (death_was_a_native_fault), both reached through one-line accessors.
        # Anything else on its way into a message uses _exit_reason().
        offenders = [
            line.strip() for line in src.splitlines()
            if "_exitcode()" in line
            and "describe_exit_code" not in line
            and "death_was_a_native_fault" not in line
            and "def _exitcode" not in line
        ]
        assert not offenders, (
            "these lines use the raw exit code where a decoded one belongs:\n"
            + "\n".join(offenders))

    @pytest.mark.parametrize("modname", [
        "localm.inference.backends._hf_runner",
        "localm.inference._embedder_runner",
    ])
    def test_sibling_runners_also_decode_their_exit_codes(self, modname):
        """The same guard for the OTHER two isolated workers.

        ``describe_exit_code`` lives in ``_mp_spawn`` shared rather than beside
        ``ModelRunner`` so all three runners reuse it. One claim, three workers,
        so one guard covering all three.

        Matches ``.exitcode`` (with the dot) rather than the bare word, so the
        module docstrings that mention ``is_alive()``/``exitcode`` in prose are
        not false positives; both modules scan clean today and both go red when
        a call site is reverted to the raw attribute."""
        import importlib
        import inspect

        src = inspect.getsource(importlib.import_module(modname))
        offenders = [
            line.strip() for line in src.splitlines()
            if ".exitcode" in line and "describe_exit_code" not in line
            and "death_was_a_native_fault" not in line
        ]
        assert not offenders, (
            f"{modname}: these lines use the raw exit code where a decoded one "
            "belongs:\n" + "\n".join(offenders))
