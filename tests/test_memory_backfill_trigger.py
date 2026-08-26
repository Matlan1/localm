# SPDX-License-Identifier: AGPL-3.0-or-later
"""The debounced full-root memory-vector backfill sweep on the chat outlet.

`memory.backfill.backfill_all` walks EVERY namespace to completion. The
auto-consolidate outlet only backfills the ONE store for whichever principal
just chatted, bounded to 64 records per pass
(`MemoryStore.backfill_vectors`), so an unrelated namespace, or a backlog
bigger than that bound, does not converge through it.

This exercises the sweep trigger wired into the same chat outlet, and
specifically that it is independent of memory_auto_consolidate and of whether
a chat model is loaded: a vector backfill needs only the embedder.
"""

from __future__ import annotations

import time

import pytest

from localm.memory.backfill import vectorless_total
from localm.memory.record import MemoryRecord
from localm.memory.store import MemoryStore


def _fake_embed(texts):
    one = not isinstance(texts, list)
    out = [[0.1, 0.2, 0.3, 0.4] for _ in ([texts] if one else texts)]
    return out[0] if one else out


@pytest.fixture
def mem_home(tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    import localm.plugins.builtin.memory.plug as plug
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_memory_root", lambda: tmp_path / "memory")
    monkeypatch.setenv("LOCALM_MODE", "log")            # writes/persist allowed
    # Reset module state between tests (module-level, not per-test).
    plug._sweep_running = False
    return tmp_path, plug


class _SyncThread:
    """Drop-in for threading.Thread that runs the target synchronously on
    .start()."""

    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def _record_thread(sink):
    """A Thread stub that appends the spawned target to *sink* instead of
    running it."""
    class _Rec:
        def __init__(self, target=None, daemon=None):
            sink.append(target)

        def start(self):
            pass
    return _Rec


def _seed_vectorless(root, principal="owner", n=3):
    st = MemoryStore(principal, "chat", "", root=root)
    for i in range(n):
        st.add(MemoryRecord(text=f"fact {i} for {principal}", source="user"),
               embed_fn=None)                      # no embedder: no vectors
    return st


def test_sweep_embeds_vectorless_records_across_the_whole_root(mem_home, monkeypatch):
    """The sweep reaches every namespace, not just whichever principal just
    chatted."""
    home, plug = mem_home
    root = home / "memory"
    _seed_vectorless(root, "owner", 3)
    _seed_vectorless(root, "other-principal", 2)
    monkeypatch.setattr(plug, "_embed_fn", lambda: _fake_embed)
    monkeypatch.setattr(plug, "_live_engine", lambda: None)   # no chat model needed
    monkeypatch.setattr(plug._threading, "Thread", _SyncThread)

    plug._memory_outlet("reply", [], {})

    assert vectorless_total(root) == 0, "the sweep must reach every namespace"


def test_sweep_is_debounced_within_the_interval(mem_home, monkeypatch):
    home, plug = mem_home
    root = home / "memory"
    _seed_vectorless(root, "owner", 3)
    monkeypatch.setattr(plug, "_embed_fn", lambda: _fake_embed)
    spawned = []
    monkeypatch.setattr(plug._threading, "Thread", _record_thread(spawned))
    plug._sweep_stamp(time.time())              # a sweep "just ran"

    plug._memory_outlet("reply", [], {})

    assert spawned == [], "ran again within the debounce interval"
    assert vectorless_total(root) == 3, "debounced: nothing should have been embedded"


def test_sweep_skips_and_still_stamps_when_nothing_pending(mem_home, monkeypatch):
    home, plug = mem_home
    root = home / "memory"
    st = _seed_vectorless(root, "owner", 2)
    st.backfill_vectors(_fake_embed)             # everything already has a vector
    assert vectorless_total(root) == 0
    monkeypatch.setattr(plug, "_embed_fn", lambda: _fake_embed)
    spawned = []
    monkeypatch.setattr(plug._threading, "Thread", _record_thread(spawned))

    plug._memory_outlet("reply", [], {})

    assert spawned == [], "nothing pending: no background pass needed"
    assert plug._sweep_last_run() > 0, "a clean scan must still stamp the marker"


def test_sweep_never_fires_in_privacy_mode(mem_home, monkeypatch):
    home, plug = mem_home
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    root = home / "memory"
    _seed_vectorless(root, "owner", 3)
    monkeypatch.setattr(plug, "_embed_fn", lambda: _fake_embed)
    spawned = []
    monkeypatch.setattr(plug._threading, "Thread", _record_thread(spawned))

    plug._memory_outlet("reply", [], {})

    assert spawned == [], "a backfill thread was spawned in privacy mode"
    assert vectorless_total(root) == 3


def test_sweep_runs_independent_of_memory_auto_consolidate_flag(mem_home, monkeypatch):
    """Backfill needs only the embedder, not fact extraction, so disabling
    auto-consolidate leaves the sweep running."""
    home, plug = mem_home
    (home / "config.json").write_text(
        '{"memory_auto_consolidate": false}', encoding="utf-8")
    root = home / "memory"
    _seed_vectorless(root, "owner", 3)
    monkeypatch.setattr(plug, "_embed_fn", lambda: _fake_embed)
    monkeypatch.setattr(plug, "_live_engine", lambda: None)
    monkeypatch.setattr(plug._threading, "Thread", _SyncThread)

    plug._memory_outlet("reply", [], {})

    assert vectorless_total(root) == 0, (
        "memory_auto_consolidate=False must not block the backfill sweep")


def test_sweep_runs_with_no_chat_model_loaded(mem_home, monkeypatch):
    """Backfill needs only the embedder, not a loaded chat model."""
    home, plug = mem_home
    root = home / "memory"
    _seed_vectorless(root, "owner", 2)
    monkeypatch.setattr(plug, "_embed_fn", lambda: _fake_embed)
    monkeypatch.setattr(plug, "_live_engine", lambda: None)   # no engine at all
    monkeypatch.setattr(plug._threading, "Thread", _SyncThread)

    plug._memory_outlet("reply", [], {})

    assert vectorless_total(root) == 0


def test_sweep_disabled_config_skips(mem_home, monkeypatch):
    home, plug = mem_home
    (home / "config.json").write_text(
        '{"memory_enabled": false}', encoding="utf-8")
    root = home / "memory"
    _seed_vectorless(root, "owner", 2)
    monkeypatch.setattr(plug, "_embed_fn", lambda: _fake_embed)
    spawned = []
    monkeypatch.setattr(plug._threading, "Thread", _record_thread(spawned))

    plug._memory_outlet("reply", [], {})

    assert spawned == [], "memory_enabled=False must skip the backfill sweep too"
    assert vectorless_total(root) == 2


def test_sweep_marker_persists_last_run_across_reads(mem_home):
    home, plug = mem_home
    assert plug._sweep_last_run() == 0.0
    plug._sweep_stamp(1234567.0)
    assert plug._sweep_last_run() == 1234567.0


def test_outlet_never_raises_when_sweep_trigger_errors(mem_home, monkeypatch):
    home, plug = mem_home
    monkeypatch.setattr(plug, "_maybe_sweep_backfill",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(plug, "_maybe_auto_consolidate", lambda *a, **k: None)
    assert plug._memory_outlet("the reply", [], {}) == "the reply"


def test_outlet_still_runs_sweep_when_consolidate_trigger_errors(mem_home, monkeypatch):
    """The two triggers are independent: a failure in auto-consolidate still
    leaves the backfill sweep to run its own turn."""
    home, plug = mem_home
    root = home / "memory"
    _seed_vectorless(root, "owner", 2)
    monkeypatch.setattr(plug, "_embed_fn", lambda: _fake_embed)
    monkeypatch.setattr(plug, "_live_engine", lambda: None)
    monkeypatch.setattr(plug._threading, "Thread", _SyncThread)
    monkeypatch.setattr(plug, "_maybe_auto_consolidate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    assert plug._memory_outlet("the reply", [], {}) == "the reply"
    assert vectorless_total(root) == 0, (
        "a consolidate-trigger failure must not skip the backfill sweep")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
