# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ctrl+C in the server's console performs the same clean stop as the GUI's
Stop button.

Two halves, and both are needed for the console output to be honest:

* a spawned worker IGNORES the console interrupt, so an intentional stop is no
  longer reported as "gguf worker process crashed" with a KeyboardInterrupt
  traceback - a console interrupt reaches every process on the console, not
  only the server;
* the server, once its serving loop has ended for any reason, runs the same
  teardown the Stop button runs (job children, engines, embedder, crash
  marker) instead of leaving all of it to interpreter teardown.
"""

import importlib
import multiprocessing
import pathlib
import signal

import pytest

from localm import _mp_spawn
from localm.inference import http_server


def _child_report(q):
    """Run inside a REAL spawned child: apply the guard, report what it did."""
    from localm import _mp_spawn as mp

    applied = mp.ignore_interrupt_signals()
    again = mp.ignore_interrupt_signals()
    q.put({
        "applied": applied,
        "idempotent": again,
        "sigint_ignored": signal.getsignal(signal.SIGINT) is signal.SIG_IGN,
        "has_parent": multiprocessing.parent_process() is not None,
    })


class TestWorkerIgnoresConsoleInterrupt:
    def test_a_real_spawned_worker_ignores_sigint(self):
        """Driven through a REAL spawned child rather than a stand-in: a mocked
        parent_process() would say nothing about the signal disposition an
        actual worker ends up with, which is the whole property."""
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(target=_child_report, args=(q,), daemon=True)
        proc.start()
        try:
            report = q.get(timeout=180)
        finally:
            proc.join(timeout=30)
            if proc.is_alive():   # pragma: no cover - only on a wedged child
                proc.terminate()

        assert report["has_parent"], "fixture bug: this was not a spawned child"
        assert report["applied"] is True
        assert report["idempotent"] is True
        assert report["sigint_ignored"], "a spawned worker must ignore SIGINT"

    def test_the_main_process_stays_interruptible(self):
        """Ctrl+C is how a console user stops the server, so the guard is inert
        here. Asserted on the live disposition, not only on the return value: a
        guard that returned False having ALREADY set SIG_IGN passes a
        return-value-only check and breaks every Ctrl+C."""
        before = signal.getsignal(signal.SIGINT)
        assert _mp_spawn.ignore_interrupt_signals() is False
        assert signal.getsignal(signal.SIGINT) is before
        assert signal.getsignal(signal.SIGINT) is not signal.SIG_IGN


class TestEveryWorkerAppliesTheGuard:
    """The guard belongs wherever install_parent_death_watchdog is taken: both
    say "the parent owns this process's lifetime"."""

    ENTRY_POINTS = {
        "localm.inference.backends.llamacpp._runner": "_runner_main",
        "localm.inference.backends._hf_runner": "_runner_main",
        "localm.inference._embedder_runner": "_runner_main",
        "localm.voice": "_worker_main",
    }

    @pytest.mark.parametrize("module_name", sorted(ENTRY_POINTS))
    def test_worker_entry_point_ignores_interrupts(self, module_name):
        mod = importlib.import_module(module_name)
        func = getattr(mod, self.ENTRY_POINTS[module_name])
        names = set(func.__code__.co_names)
        assert "install_parent_death_watchdog" in names, (
            f"{module_name} changed shape; this test is pinned to the wrong "
            "function and is no longer checking anything")
        assert "ignore_interrupt_signals" in names, (
            f"{module_name}.{self.ENTRY_POINTS[module_name]} must ignore console "
            "interrupts, or a Ctrl+C aimed at the server tears this worker down "
            "and the intentional stop is reported as a crash")

    def test_no_other_worker_takes_the_watchdog_without_the_guard(self):
        """Wider than the four modules named above, which are a snapshot: any
        module registering the parent-death watchdog IS a worker, so a fifth one
        added later fails here instead of silently reintroducing the false crash
        report for its own model type."""
        root = pathlib.Path(_mp_spawn.__file__).resolve().parent
        offenders = []
        for path in sorted(root.rglob("*.py")):
            if path.name == "_mp_spawn.py":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "install_parent_death_watchdog()" not in text:
                continue
            if "ignore_interrupt_signals()" not in text:
                offenders.append(path.relative_to(root).as_posix())
        assert not offenders, (
            "these workers take the parent-death watchdog but not the interrupt "
            f"guard: {offenders}")


class TestServingStopRunsTheGuiButtonTeardown:
    def test_teardown_does_not_exit_the_process(self, monkeypatch):
        """_do_shutdown's os._exit is exactly what makes it unusable from a
        normal unwind, so the extracted teardown must not carry it."""
        import os

        def _boom(code):   # pragma: no cover - fires only on a regression
            raise AssertionError(f"_shutdown_teardown must not exit ({code})")

        monkeypatch.setattr(os, "_exit", _boom)
        monkeypatch.setattr(http_server, "_engine", None)
        monkeypatch.setattr(http_server, "_engines", {})
        http_server._shutdown_teardown()

    def test_do_shutdown_still_tears_down_then_exits(self, monkeypatch):
        """The Stop button's contract is unchanged by the split."""
        import os

        order = []

        def _fake_exit(code):
            order.append(("exit", code))
            raise SystemExit(code)

        monkeypatch.setattr(http_server, "_shutdown_teardown",
                            lambda **kw: order.append(("teardown", kw)))
        monkeypatch.setattr(os, "_exit", _fake_exit)
        with pytest.raises(SystemExit):
            http_server._do_shutdown(instance_id="inst-1")
        assert order == [("teardown", {"instance_id": "inst-1"}), ("exit", 0)]

    @staticmethod
    def _stub_serving(monkeypatch, make_run_server):
        """Drive run_advertised with the transport and the registry stubbed, so
        the assertion is about the stop sequence and not about binding a port.

        *make_run_server* is handed the shared call log and returns the
        portmux.run_server stand-in, so the transport's own entry lands in the
        same ordered list as the teardown's."""
        import contextlib

        calls = []
        run_server = make_run_server(calls)

        class _App:
            class state:
                instance_id = "inst-42"

        @contextlib.contextmanager
        def _advertise(*a, **kw):
            calls.append("advertise-enter")
            try:
                yield {}
            finally:
                calls.append("advertise-exit")

        monkeypatch.setattr("localm.instances.advertise", _advertise)
        monkeypatch.setattr("localm.portmux.run_server", run_server)
        monkeypatch.setattr(http_server, "_announce_stopping",
                            lambda: calls.append("announce"))
        monkeypatch.setattr(http_server, "_shutdown_teardown",
                            lambda **kw: calls.append(("teardown",
                                                       kw.get("instance_id"))))
        return calls, _App()

    def test_teardown_runs_when_serving_ends(self, monkeypatch):
        def _serves(calls):
            def _run_server(*a, **kw):
                calls.append("served")
            return _run_server

        calls, app = self._stub_serving(monkeypatch, _serves)
        http_server.run_advertised(app, "127.0.0.1", 1, mode="full")
        assert calls == ["advertise-enter", "served", "announce",
                         ("teardown", "inst-42"), "advertise-exit"]

    def test_teardown_runs_when_serving_raises(self, monkeypatch):
        """A bind failure still leaves a loaded model and live job children
        behind, so the teardown is in a finally rather than after the call."""
        def _explodes(calls):
            def _run_server(*a, **kw):
                raise RuntimeError("bind failed")
            return _run_server

        calls, app = self._stub_serving(monkeypatch, _explodes)
        with pytest.raises(RuntimeError):
            http_server.run_advertised(app, "127.0.0.1", 1, mode="full")
        assert ("teardown", "inst-42") in calls
        assert calls.index(("teardown", "inst-42")) < calls.index("advertise-exit")
