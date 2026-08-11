# SPDX-License-Identifier: AGPL-3.0-or-later
"""A dead worker's exit code must be reported DECODED, not raw.

Issues 1222/1223 reported ``Native inference fault (worker exit -4)`` and that
number was the only fact the product gave about the death. It is also the most
discriminating one available: on POSIX ``multiprocessing`` reports ``-N`` for
death by signal N, so ``-4`` is SIGILL - an illegal instruction, which is a
different FAMILY of cause from a segfault (-11) or an abort (-6) and points the
investigation somewhere different. Decoding it had to be done by hand before the
diagnosis could even pick a direction.

BOTH OS CONVENTIONS ARE TESTED ON WHATEVER PLATFORM THIS RUNS ON, via
``describe_exit_code``'s explicit *posix* parameter. That is the point of the
parameter: the conventions are mutually exclusive, so a Windows runner can never
produce ``-4``-means-SIGILL and a Linux runner can never produce an NTSTATUS -
and the field crash is on the branch this project's own box cannot reach. Gating
these on ``sys.platform`` instead would leave the branch that actually matters
permanently unexercised while the file still read as covered.
"""

import pytest

from localm._mp_spawn import describe_exit_code
from localm.inference.backends.llamacpp._runner import ModelRunner


class TestPosixSignalDecoding:
    def test_the_reported_code_names_sigill(self):
        """The exact code from issues 1222/1223."""
        out = describe_exit_code(-4, posix=True)
        assert "SIGILL" in out, out
        assert "-4" in out, out

    @pytest.mark.parametrize("code,name", [
        (-4, "SIGILL"),     # illegal instruction - the reported one
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
        """REGRESSION GUARD for a real defect this file caught, so nobody
        "simplifies" the decoder back to a bare signal.Signals() lookup.

        MEASURED: the two enums disagree on the numbers that matter.

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
        """3221226505 == 0xC0000409, measured from a real os.abort() on this
        project's own CRT while building the worker crash-trace capture."""
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
    are sited at the runner rather than only on the helper because the defect was
    that every user-facing message interpolated the RAW code - a helper-level
    test cannot see that (diff-review item 23: extracting a helper moves the
    tests away from where the bug lives)."""

    def test_exit_reason_is_decoded_for_a_signal_death(self, monkeypatch):
        r = ModelRunner()

        class _FakeProc:
            exitcode = -4

        r._proc = _FakeProc()
        monkeypatch.setattr("localm._mp_spawn.os.name", "posix")
        assert "SIGILL" in r._exit_reason()

    def test_no_user_facing_message_interpolates_the_raw_exitcode(self):
        """A grep-style guard on the source itself. Every crash message must go
        through _exit_reason(); a new one that reaches for _exitcode() directly
        would silently reintroduce the bare number, and no behavioural test would
        catch it because the message would still be produced."""
        import inspect

        from localm.inference.backends.llamacpp import _runner

        src = inspect.getsource(_runner)
        offenders = [
            line.strip() for line in src.splitlines()
            if "_exitcode()" in line
            and "describe_exit_code" not in line
            and "def _exitcode" not in line
        ]
        assert not offenders, (
            "these lines use the raw exit code where a decoded one belongs:\n"
            + "\n".join(offenders))
