# SPDX-License-Identifier: AGPL-3.0-or-later
"""Failure-surfacing behaviour in the plugin builtins.

Each test covers a site that could hide a real failure - a swallowed exception,
a silenced warning, or a false cause reported in place of the true one - and
asserts that it SURFACES (a debug/WARNING log, or the honest error), keeps its
guard, and does not crash.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
#  jobs/scheduler.py - a failing tick must be logged, not silently swallowed
# --------------------------------------------------------------------------- #

def test_scheduler_loop_logs_tick_failure(monkeypatch):
    """An unreadable jobs.json makes store.list() raise every tick, which halts
    ALL scheduled jobs; the loop WARNs rather than swallowing it."""
    from localm.plugins.builtin.jobs.scheduler import JobScheduler

    class BoomStore:
        def list(self):
            raise RuntimeError("jobs store unreadable (simulated)")

    sched = JobScheduler(store=BoomStore(), run_job=lambda *a, **k: {}, poll_seconds=1)

    warnings: list = []
    from localm.debuglog import logger as dbg
    monkeypatch.setattr(dbg, "warning", lambda *a, **k: warnings.append(a))

    async def run_briefly():
        assert sched.start() is True
        for _ in range(100):            # bounded wait for the first failed tick
            if warnings:
                break
            await asyncio.sleep(0.02)
        sched.stop()
        await asyncio.sleep(0.01)

    asyncio.run(run_briefly())

    assert warnings, "a failing scheduler tick must be logged, not silently swallowed"
    joined = " ".join(str(a) for a in warnings).lower()
    assert "tick failed" in joined
    assert "unreadable" in joined


def test_scheduler_tick_error_logged_once_then_recovers(monkeypatch):
    """Log-once/on-change: the same failure is not re-logged every 30s poll, a
    recovery is noted once, and a later recurrence is logged again."""
    from localm.plugins.builtin.jobs.scheduler import JobScheduler

    sched = JobScheduler(store=object(), run_job=lambda *a, **k: {})
    warnings: list = []
    infos: list = []
    from localm.debuglog import logger as dbg
    monkeypatch.setattr(dbg, "warning", lambda *a, **k: warnings.append(a))
    monkeypatch.setattr(dbg, "info", lambda *a, **k: infos.append(a))

    sched._note_tick_error(RuntimeError("boom"))
    sched._note_tick_error(RuntimeError("boom"))     # same signature -> no re-log
    assert len(warnings) == 1

    sched._note_tick_ok()                            # recovery -> logged once
    assert len(infos) == 1
    sched._note_tick_ok()                            # already clear -> no repeat
    assert len(infos) == 1

    sched._note_tick_error(RuntimeError("boom"))     # recurs -> logged again
    assert len(warnings) == 2


# --------------------------------------------------------------------------- #
#  rag/plug.py - image-describe failure surfaces its true cause, not a false
#     empty description
# --------------------------------------------------------------------------- #

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def test_self_describe_image_propagates_transport_error(monkeypatch):
    import localm.plugins.builtin.rag.plug as ragplug
    import localm.selfclient as selfclient

    def boom(*a, **k):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(selfclient, "self_request", boom)
    fn = ragplug._make_self_describe_image("http://127.0.0.1:0", lambda: "m")
    with pytest.raises(RuntimeError, match="connection reset by peer"):
        fn(_PNG, "image/png")


def test_image_extract_reports_honest_cause_not_empty(monkeypatch):
    """End to end: extract_bytes wraps the REAL error ("Image description
    failed: ...") rather than reporting "returned empty description"."""
    import localm.plugins.builtin.rag.plug as ragplug
    import localm.selfclient as selfclient
    from localm.rag.extract import ExtractError, extract_bytes

    def boom(*a, **k):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(selfclient, "self_request", boom)
    describe = ragplug._make_self_describe_image("http://127.0.0.1:0", lambda: "m")

    with pytest.raises(ExtractError) as ei:
        extract_bytes(_PNG, "photo.png", describe_image_fn=describe)
    msg = str(ei.value)
    assert "Image description failed" in msg          # honest cause
    assert "empty description" not in msg             # NOT the false cause
    assert "connection reset by peer" in msg


class _FakeResp:
    def __init__(self, ok, status_code=500, detail=""):
        self.ok = ok
        self.status_code = status_code
        self._detail = detail

    def json(self):
        return {"detail": self._detail}


def test_self_describe_image_empty_success_returns_empty_string(monkeypatch):
    """A genuine empty-but-successful description stays a "" return (extract then
    honestly reports it as empty) - only FAILURES propagate."""
    import localm.plugins.builtin.rag.plug as ragplug
    import localm.selfclient as selfclient

    class OkEmpty:
        ok = True

        def json(self):
            return {"choices": [{"message": {"content": "   "}}]}

    monkeypatch.setattr(selfclient, "self_request", lambda *a, **k: OkEmpty())
    fn = ragplug._make_self_describe_image("http://x", lambda: "m")
    assert fn(_PNG, "image/png") == ""


def test_self_describe_image_no_vision_message(monkeypatch):
    import localm.plugins.builtin.rag.plug as ragplug
    import localm.selfclient as selfclient

    resp = _FakeResp(ok=False, status_code=400, detail="model cannot accept image input")
    monkeypatch.setattr(selfclient, "self_request", lambda *a, **k: resp)
    fn = ragplug._make_self_describe_image("http://x", lambda: "m")
    with pytest.raises(RuntimeError, match="does not support vision"):
        fn(_PNG, "image/png")


# --------------------------------------------------------------------------- #
#  memory/plug.py - an unreadable legacy file is not treated as empty, so the
#     migration is not falsely marked done
# --------------------------------------------------------------------------- #

def test_read_memory_distinguishes_absent_from_unreadable(tmp_path, monkeypatch):
    import localm.plugins.builtin.memory.plug as plug

    legacy = plug._legacy_memory_file()
    # Absent -> "" (no raise)
    if legacy.exists():
        legacy.unlink()
    assert plug._read_memory() == ""

    # Exists but unreadable -> raises OSError (NOT "")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("- a legacy fact\n", encoding="utf-8")
    real_read = Path.read_text

    def boom(self, *a, **k):
        if Path(self) == legacy:
            raise OSError("simulated unreadable legacy file")
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", boom)
    with pytest.raises(OSError):
        plug._read_memory()


def test_unreadable_legacy_memory_not_marked_migrated(tmp_path, monkeypatch):
    """An exists-but-unreadable chat-memory.md does not cause _migrate_legacy to
    write the permanent .legacy-imported marker."""
    monkeypatch.setenv("LOCALM_MODE", "log")             # persist enabled
    import localm.plugins.builtin.memory.plug as plug
    from localm.memory import MemoryStore

    legacy = plug._legacy_memory_file()
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("- a legacy fact\n- another\n", encoding="utf-8")

    real_read = Path.read_text

    def boom(self, *a, **k):
        if Path(self) == legacy:
            raise OSError("simulated unreadable legacy file")
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", boom)

    store = MemoryStore("owner", "chat", root=tmp_path / "mem")
    marker = store.path.with_suffix(".legacy-imported")
    assert not marker.exists()

    plug._migrate_legacy(store)

    assert not marker.exists(), (
        "an unreadable legacy file was marked as migrated - the migration never "
        "read it, so this is a false success (RULE 5)")


def test_rendered_text_tolerates_unreadable_legacy(tmp_path, monkeypatch):
    """The memory modal must show an empty textarea, not 500, when the legacy file
    is unreadable and the structured store is empty."""
    import localm.plugins.builtin.memory.plug as plug
    from localm.memory import MemoryStore

    legacy = plug._legacy_memory_file()
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("- x\n", encoding="utf-8")
    real_read = Path.read_text

    def boom(self, *a, **k):
        if Path(self) == legacy:
            raise OSError("simulated unreadable legacy file")
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", boom)
    store = MemoryStore("owner", "chat", root=tmp_path / "mem")
    assert plug._rendered_text(store) == ""            # graceful, no OSError raised


# --------------------------------------------------------------------------- #
#  mcpserver/server.py - a coder-availability probe failure is logged and
#     still fails closed (the coder tool hides, but not silently)
# --------------------------------------------------------------------------- #

def test_coder_availability_probe_logs_on_failure(monkeypatch):
    import localm.plugins.engine as pe
    import localm.plugins.mcpserver.server as srv

    class BoomPM:
        def __init__(self, *a, **k):
            pass

        def is_active(self, name):
            raise RuntimeError("plugin config unreadable")

    monkeypatch.setattr(pe, "PluginManager", BoomPM)

    calls: list = []
    from localm.debuglog import logger as dbg
    monkeypatch.setattr(dbg, "debug", lambda *a, **k: calls.append(a))

    assert srv._coder_available() is False             # fails CLOSED (safe)
    assert calls, "a coder-availability probe failure must be logged, not silent"
    assert "coder" in " ".join(str(a) for a in calls).lower()


# --------------------------------------------------------------------------- #
#  rag/plug.py - the classify tie-break failure is logged before falling back
# --------------------------------------------------------------------------- #

def test_self_classify_logs_on_failure(monkeypatch):
    import localm.plugins.builtin.rag.plug as ragplug
    import localm.selfclient as selfclient

    def boom(*a, **k):
        raise RuntimeError("classify endpoint down")

    monkeypatch.setattr(selfclient, "self_request", boom)

    calls: list = []
    from localm.debuglog import logger as dbg
    monkeypatch.setattr(dbg, "debug", lambda *a, **k: calls.append(a))

    fn = ragplug._make_self_classify("http://x", lambda: "m")
    assert fn("some snippet") is None                  # falls back to caller default
    assert calls, "a classify tie-break failure must be logged, not silent"


def test_self_classify_logs_on_non_ok_response(monkeypatch):
    """self_request never raises on a non-2xx, so a non-ok HTTP response is
    logged on its way to the fall-through as well."""
    import localm.plugins.builtin.rag.plug as ragplug
    import localm.selfclient as selfclient

    monkeypatch.setattr(selfclient, "self_request",
                        lambda *a, **k: _FakeResp(ok=False, status_code=503))

    calls: list = []
    from localm.debuglog import logger as dbg
    monkeypatch.setattr(dbg, "debug", lambda *a, **k: calls.append(a))

    fn = ragplug._make_self_classify("http://x", lambda: "m")
    assert fn("some snippet") is None
    joined = " ".join(str(a) for a in calls).lower()
    assert calls and "503" in joined, "a non-ok classify HTTP response must be logged"
