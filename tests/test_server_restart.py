# SPDX-License-Identifier: AGPL-3.0-or-later
"""An in-app RESTART endpoint so the user can restart the server from Settings
(it comes back on the same port) instead of only being able to shut down. The
restart sequence unloads the model BEFORE relaunching, like the shutdown
sequence."""

import os
import sys

from localm.inference import http_server


def test_restart_route_registered_and_gated():
    app = http_server.create_app(None)
    routes = {getattr(r, "path", None): r for r in app.routes}
    assert "/v1/server/restart" in routes
    route = routes["/v1/server/restart"]
    assert "POST" in route.methods
    # Server control must be auth-gated, not open to anyone on the network.
    assert route.dependencies, "restart endpoint must carry an auth dependency"


def test_restart_argv_is_canonical_python_m_localm():
    argv = http_server._restart_argv()
    assert argv[0] == sys.executable
    assert argv[1:3] == ["-m", "localm"]
    assert argv[3:] == sys.argv[1:]      # original subcommand + args preserved


def test_do_restart_unloads_before_relaunch(monkeypatch):
    order = []

    class _FakeEngine:
        def unload(self):
            order.append("unload")

    def _fake_relaunch(exe, argv):
        order.append(("relaunch", exe, tuple(argv)))
        raise SystemExit(0)   # stop _do_restart here instead of replacing pytest

    monkeypatch.setattr(http_server, "_engine", _FakeEngine())
    monkeypatch.setattr(os, "execv", _fake_relaunch)

    try:
        http_server._do_restart()
    except SystemExit:
        pass

    # Model unloaded BEFORE the relaunch (clean native teardown), then the canonical
    # re-launch command line.
    assert order and order[0] == "unload"
    assert order[-1][0] == "relaunch"
    assert order[-1][1] == sys.executable
    assert list(order[-1][2]) == http_server._restart_argv()


def test_do_restart_sets_restart_in_progress_flag_before_relaunch(monkeypatch):
    """A restart's re-exec'd process must not auto-open a NEW browser tab: the
    tab the user is already looking at shows a reconnect overlay that resumes in
    place (models.js's onServerUnreachable). _do_restart signals this to the
    re-exec'd process by setting LOCALM_RESTART_IN_PROGRESS right before
    os.execv, so it is present in the environment the new process image
    inherits; plugins/gui/cli.py's _should_auto_open_browser consumes it on the
    other end."""
    monkeypatch.setattr(http_server, "_engine", None)
    seen = {}

    def _fake_relaunch(exe, argv):
        seen["flag"] = os.environ.get("LOCALM_RESTART_IN_PROGRESS")
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _fake_relaunch)
    os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)
    try:
        try:
            http_server._do_restart()
        except SystemExit:
            pass
        assert seen.get("flag") == "1"
    finally:
        os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)


def test_do_restart_releases_embedder(monkeypatch):
    """The shared embedder (localm.inference.embedder) is a separate lifecycle
    from _engines and must be released before a restart's re-exec, or its native
    VRAM/RAM allocation leaks across the restart.

    Released via release_for_exit(), NOT reset_embedder(): the latter takes the
    embedder's load lock, which get_embedder() holds for a whole model load, so a
    restart issued mid-load blocks there and never reaches the teardown."""
    from localm.inference import embedder as emb

    def _fake_relaunch(exe, argv):
        raise SystemExit(0)

    calls = []
    monkeypatch.setattr(emb, "release_for_exit", lambda: (calls.append(1), True)[1])
    monkeypatch.setattr(http_server, "_engine", None)
    monkeypatch.setattr(os, "execv", _fake_relaunch)
    # release_for_exit() returning True makes _do_restart's own VRAM-release wait
    # fire - unmeasurable here so it skips immediately instead of hitting the
    # real GPU probe.
    import localm.discover as discover
    monkeypatch.setattr(discover, "vram_capacity", lambda *a, **kw: {})

    try:
        http_server._do_restart()
    except SystemExit:
        pass

    assert calls == [1]


def test_do_restart_disarms_crash_guard_before_relaunch(monkeypatch):
    # An intentional restart must not be reported as a crash.
    disarmed = []

    def _boom(*_a):
        raise SystemExit(0)

    monkeypatch.setattr(http_server, "_engine", None)
    import localm.bugreport as bugreport
    monkeypatch.setattr(bugreport, "disarm_crash_guard",
                        lambda instance_id=None: disarmed.append(True))
    monkeypatch.setattr(os, "execv", _boom)
    try:
        http_server._do_restart()
    except SystemExit:
        pass
    assert disarmed == [True]


# ------------------------ post-update watchdog ------------------------------
#
# _do_restart's update_watchdog param is the ONLY thing that distinguishes the
# post-update auto-restart from a plain /v1/server/restart click; these tests
# cover the plain path and the update path's wiring.

def test_do_restart_spawns_watchdog_when_given(monkeypatch):
    from localm import updater
    calls = []
    monkeypatch.setattr(updater, "spawn_health_watchdog",
                        lambda **kw: calls.append(kw) or True)
    monkeypatch.setattr(http_server, "_engine", None)

    def _stop_here(*_a):
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _stop_here)

    watchdog = {"host": "127.0.0.1", "port": 8642, "scheme": "http",
               "expect_version": "0.2.0"}
    try:
        http_server._do_restart(update_watchdog=watchdog)
    except SystemExit:
        pass
    assert calls == [{"host": "127.0.0.1", "port": 8642, "scheme": "http",
                      "expect_version": "0.2.0"}]


def test_do_restart_no_watchdog_by_default(monkeypatch):
    """The plain restart path (/v1/server/restart) must NEVER spawn a watchdog -
    it has no update to verify and no version to roll back to."""
    from localm import updater

    def _must_not_be_called(**_kw):
        raise AssertionError("spawn_health_watchdog must not fire for a plain restart")

    monkeypatch.setattr(updater, "spawn_health_watchdog", _must_not_be_called)
    monkeypatch.setattr(http_server, "_engine", None)

    def _stop_here(*_a):
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _stop_here)
    try:
        http_server._do_restart()
    except SystemExit:
        pass   # no AssertionError means the watchdog was correctly never spawned


def test_do_restart_watchdog_spawn_exception_does_not_block_execv(monkeypatch):
    """A watchdog spawn failure must never prevent the restart itself - a broken
    watchdog must not make updates worse than having none at all."""
    from localm import updater

    def _boom(**_kw):
        raise RuntimeError("spawn blew up")

    monkeypatch.setattr(updater, "spawn_health_watchdog", _boom)
    monkeypatch.setattr(http_server, "_engine", None)
    reached = []

    def _fake_execv(exe, argv):
        reached.append(True)
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _fake_execv)
    try:
        http_server._do_restart(update_watchdog={
            "host": "127.0.0.1", "port": 1, "scheme": "http", "expect_version": "x"})
    except SystemExit:
        pass
    assert reached == [True]   # execv still ran despite the watchdog spawn raising


def test_request_restart_threads_update_watchdog_through(monkeypatch):
    # Run the inner thread body synchronously so the test is not timing-dependent.
    class _SyncThread:
        def __init__(self, target, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr("threading.Thread", _SyncThread)
    captured = []
    monkeypatch.setattr(http_server, "_do_restart",
                        lambda **kw: captured.append(kw.get("update_watchdog")))

    watchdog = {"host": "127.0.0.1", "port": 9, "scheme": "https",
               "expect_version": "1.0.0"}
    http_server._request_restart(delay=0, update_watchdog=watchdog)
    assert captured == [watchdog]


# ------------------------------ VRAM-release wait -------------------------
#
# switch_engine's eviction loop waits for a native free to land
# (wait_for_vram_release) before constructing the replacement engine. The restart
# path waits too, only when there is something ACTUALLY loaded to wait for (not
# merely present in _engines), and never blocks the restart on failure.

def test_do_restart_waits_for_vram_release_when_engines_present(monkeypatch):
    order = []

    class _FakeEngine:
        loaded = True

        def unload(self):
            order.append("unload")

    def _fake_relaunch(exe, argv):
        order.append("relaunch")
        raise SystemExit(0)

    monkeypatch.setattr(http_server, "_engines", {"model-a": _FakeEngine()})
    monkeypatch.setattr(http_server, "_engine", None)
    monkeypatch.setattr(os, "execv", _fake_relaunch)

    import localm.discover as discover
    import localm.vram as vram

    monkeypatch.setattr(discover, "vram_capacity",
                        lambda *a, **kw: {"free": 10_000_000_000})

    calls = []

    def fake_wait(read_free, before_bytes=None, **kw):
        calls.append(before_bytes)
        order.append("wait")
        return (True, read_free())

    monkeypatch.setattr(vram, "wait_for_vram_release", fake_wait)

    try:
        http_server._do_restart()
    except SystemExit:
        pass

    # Waits AFTER the native unload, BEFORE the re-exec, with the free-VRAM
    # reading taken before teardown - the same before/after shape switch_engine
    # already uses at its own wait_for_vram_release call sites.
    assert order == ["unload", "wait", "relaunch"]
    assert calls == [10_000_000_000]


def test_do_restart_skips_vram_wait_when_nothing_was_loaded(monkeypatch):
    """A model-less restart (no chat engine, no embedder loaded) must not pay
    the wait's latency - there is nothing whose release needs confirming."""
    monkeypatch.setattr(http_server, "_engines", {})
    monkeypatch.setattr(http_server, "_engine", None)

    def _fake_relaunch(exe, argv):
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _fake_relaunch)

    import localm.vram as vram

    # Recorded rather than raised: wait_for_vram_release's own call site is
    # wrapped in a broad `except Exception`, which would silently swallow an
    # AssertionError raised from inside the mock and let the test pass either
    # way. Asserting on the call list AFTER _do_restart returns is outside
    # that try/except, so it actually observes what happened.
    calls = []
    monkeypatch.setattr(vram, "wait_for_vram_release", lambda *a, **kw: calls.append(1))

    try:
        http_server._do_restart()
    except SystemExit:
        pass
    assert calls == [], "wait_for_vram_release must not fire when nothing was unloaded"


def test_do_restart_skips_vram_wait_for_a_stale_unloaded_engine_entry(monkeypatch):
    """unload_all_models/idle-unload KEEP a now-unloaded engine's entry in
    _engines so a later request reloads it lazily, so _engines can be non-empty
    with nothing actually loaded.
    A dict-non-emptiness check would make every restart on a server that ever
    idle-unloaded a model pay the wait's full timeout for nothing freed."""
    class _StaleEngine:
        loaded = False   # present in _engines, but NOT resident - nothing to wait for

        def unload(self):
            pass   # already unloaded; a real GgufBackend.unload() is a no-op here too

    monkeypatch.setattr(http_server, "_engines", {"model-a": _StaleEngine()})
    monkeypatch.setattr(http_server, "_engine", None)

    def _fake_relaunch(exe, argv):
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _fake_relaunch)

    import localm.vram as vram

    # See test_do_restart_skips_vram_wait_when_nothing_was_loaded: recorded
    # rather than raised, since the call site is wrapped in `except Exception`.
    calls = []
    monkeypatch.setattr(vram, "wait_for_vram_release", lambda *a, **kw: calls.append(1))

    try:
        http_server._do_restart()
    except SystemExit:
        pass
    assert calls == [], ("wait_for_vram_release must not fire for a stale, "
                         "already-unloaded _engines entry")


def test_do_restart_skips_vram_wait_when_unmeasurable(monkeypatch):
    """Mirrors switch_engine's own 'measurable and free_before is not None'
    guard: an unmeasurable box (no 'free' key) must not hang the restart
    waiting for something it can never observe."""
    class _FakeEngine:
        loaded = True

        def unload(self):
            pass

    monkeypatch.setattr(http_server, "_engines", {"model-a": _FakeEngine()})
    monkeypatch.setattr(http_server, "_engine", None)

    def _fake_relaunch(exe, argv):
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _fake_relaunch)

    import localm.discover as discover
    import localm.vram as vram

    monkeypatch.setattr(discover, "vram_capacity", lambda *a, **kw: {})

    # See test_do_restart_skips_vram_wait_when_nothing_was_loaded: recorded
    # rather than raised, since the call site is wrapped in `except Exception`.
    calls = []
    monkeypatch.setattr(vram, "wait_for_vram_release", lambda *a, **kw: calls.append(1))

    try:
        http_server._do_restart()
    except SystemExit:
        pass
    assert calls == [], "wait_for_vram_release must not fire when VRAM is unmeasurable"


def test_do_restart_vram_wait_failure_does_not_block_restart(monkeypatch):
    """A wedged/erroring VRAM probe during the wait must not prevent the
    restart itself - best-effort, like every other teardown step here."""
    class _FakeEngine:
        loaded = True

        def unload(self):
            pass

    monkeypatch.setattr(http_server, "_engines", {"model-a": _FakeEngine()})
    monkeypatch.setattr(http_server, "_engine", None)

    reached = []

    def _fake_relaunch(exe, argv):
        reached.append(True)
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _fake_relaunch)

    import localm.discover as discover
    import localm.vram as vram

    monkeypatch.setattr(discover, "vram_capacity", lambda *a, **kw: {"free": 123})

    wait_calls = []

    def _boom(*_a, **_kw):
        wait_calls.append(1)
        raise RuntimeError("driver wedged")

    monkeypatch.setattr(vram, "wait_for_vram_release", _boom)

    try:
        http_server._do_restart()
    except SystemExit:
        pass
    # The wait was actually INVOKED (not silently skipped) and its failure
    # still let the restart proceed to execv.
    assert wait_calls == [1]
    assert reached == [True]   # execv still ran despite the wait raising


def test_do_restart_waits_for_vram_release_when_only_embedder_was_loaded(monkeypatch):
    """No chat engine, but the shared embedder WAS resident - its release also
    needs the wait, not just a chat-engine eviction."""
    monkeypatch.setattr(http_server, "_engines", {})
    monkeypatch.setattr(http_server, "_engine", None)

    def _fake_relaunch(exe, argv):
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _fake_relaunch)

    from localm.inference import embedder as emb
    # loaded_path() is the pre-release "was anything loaded" probe free_before's
    # skip gate now reads; without it the scenario is indistinguishable from
    # "nothing was loaded" even though release_for_exit() reports True.
    monkeypatch.setattr(emb, "loaded_path", lambda: "/fake/embedder/model.gguf")
    monkeypatch.setattr(emb, "release_for_exit", lambda: True)

    import localm.discover as discover
    import localm.vram as vram

    monkeypatch.setattr(discover, "vram_capacity", lambda *a, **kw: {"free": 5})
    calls = []
    monkeypatch.setattr(vram, "wait_for_vram_release",
                        lambda read_free, before_bytes=None, **kw: calls.append(1))

    try:
        http_server._do_restart()
    except SystemExit:
        pass
    assert calls == [1]


# ---------------------- free-VRAM probe skip (restart speed) ---------------
#
# vram_capacity() -> list_gpus() re-probes on every call with no TTL cache (see
# discover.list_gpus's own docstring), so a cold reading - an isolated
# subprocess cold-importing torch - can itself cost several real seconds.
# free_before's ONLY consumer is the had_engines/had_embedder-gated wait
# below, so reading it when neither is true wastes that time on every restart
# that has nothing loaded (--no-model, or a server that idle-unloaded
# everything) for a number that is never used.

def test_do_restart_skips_free_vram_probe_when_nothing_was_loaded(monkeypatch):
    monkeypatch.setattr(http_server, "_engines", {})
    monkeypatch.setattr(http_server, "_engine", None)

    from localm.inference import embedder as emb
    monkeypatch.setattr(emb, "loaded_path", lambda: None)

    def _fake_relaunch(exe, argv):
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _fake_relaunch)

    import localm.discover as discover

    # Recorded rather than raised: vram_capacity's call site is wrapped in
    # `except Exception`, which would silently swallow an AssertionError
    # raised from inside the mock and let the test pass either way (see
    # test_do_restart_skips_vram_wait_when_nothing_was_loaded above for the
    # same shape on the wait call). Asserting on the call list AFTER
    # _do_restart returns is outside that try/except, so it actually
    # observes what happened.
    calls = []
    monkeypatch.setattr(discover, "vram_capacity", lambda *a, **kw: calls.append(1))

    try:
        http_server._do_restart()
    except SystemExit:
        pass
    assert calls == [], (
        "vram_capacity must not be probed for free_before when nothing is "
        "loaded - its result would never be used")


def test_do_restart_probes_free_vram_when_engines_present(monkeypatch):
    """The mirror of the skip test above: when something IS loaded, free_before
    still has to be read (its result feeds the release-confirmation wait)."""
    class _FakeEngine:
        loaded = True

        def unload(self):
            pass

    monkeypatch.setattr(http_server, "_engines", {"model-a": _FakeEngine()})
    monkeypatch.setattr(http_server, "_engine", None)

    from localm.inference import embedder as emb
    monkeypatch.setattr(emb, "loaded_path", lambda: None)

    def _fake_relaunch(exe, argv):
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _fake_relaunch)

    import localm.discover as discover
    import localm.vram as vram

    calls = []

    def _probe(*_a, **_kw):
        calls.append(1)
        return {"free": 42}

    monkeypatch.setattr(discover, "vram_capacity", _probe)
    monkeypatch.setattr(vram, "wait_for_vram_release",
                        lambda read_free, before_bytes=None, **kw: (True, before_bytes))

    try:
        http_server._do_restart()
    except SystemExit:
        pass
    assert calls, "vram_capacity must still be probed when an engine is loaded"


def test_do_restart_probes_free_vram_when_only_embedder_present(monkeypatch):
    """Same mirror, for the embedder-only case: had_embedder alone must still
    trigger the probe, not just had_engines."""
    monkeypatch.setattr(http_server, "_engines", {})
    monkeypatch.setattr(http_server, "_engine", None)

    from localm.inference import embedder as emb
    monkeypatch.setattr(emb, "loaded_path", lambda: "/fake/embedder/model.gguf")
    monkeypatch.setattr(emb, "release_for_exit", lambda: True)

    def _fake_relaunch(exe, argv):
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _fake_relaunch)

    import localm.discover as discover
    import localm.vram as vram

    calls = []

    def _probe(*_a, **_kw):
        calls.append(1)
        return {"free": 42}

    monkeypatch.setattr(discover, "vram_capacity", _probe)
    monkeypatch.setattr(vram, "wait_for_vram_release",
                        lambda read_free, before_bytes=None, **kw: (True, before_bytes))

    try:
        http_server._do_restart()
    except SystemExit:
        pass
    assert calls, "vram_capacity must still be probed when the embedder is loaded"
