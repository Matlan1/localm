# SPDX-License-Identifier: AGPL-3.0-or-later
"""model_manager/registry.py's sync_models_dir() runs its whole reconcile loop
inside a single update_registry() call, so it is atomic and cross-process
lock-protected like every other write path in that file.

A bare load/mutate/save shape would load the registry once, reconcile
missing/restored/pruned entries with real per-entry filesystem stat calls, then
blind-save at the end, silently discarding a concurrent update_registry() write
(a `pull`/`rm`/`alias` in another thread or process) that landed during the scan.

This test drives a REAL in-process threading race, not a mock: _entry_path (the
per-entry accessor the reconcile loop calls) is wrapped with a delay, widening
the window a concurrent update_registry() writer can land in - the same
technique test_app_lifecycle_1_atomic_config.py's _install_slow_merge uses for
the config side."""

import json
import threading
import time

import pytest

import localm.config as cfg
import localm.model_manager as mm
import localm.model_manager.registry as registry_mod


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir()
    models_dir = h / "models"
    models_dir.mkdir()
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", models_dir)
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    # registry.py resolves _mm.MODELS_DIR (a live attribute lookup on this
    # package) for its directory scan, separately from config.py's own
    # MODELS_DIR - both must point at the same temp dir.
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    return h, models_dir


def _install_slow_entry_path(monkeypatch, delay=0.1):
    """Delay every _entry_path() call - the accessor sync_models_dir()'s
    reconcile loop calls per registry entry to get its stat-checkable path.
    In the OLD bare load_registry()/mutate/save_registry() shape this delay
    lands in the fully-unlocked window between the reconcile reload and the
    final save; in the FIXED shape it lands inside update_registry()'s lock -
    so only the fixed call site can survive a concurrent writer here."""
    real = registry_mod._entry_path

    def slow(entry):
        time.sleep(delay)
        return real(entry)

    monkeypatch.setattr(registry_mod, "_entry_path", slow)


def test_sync_models_dir_survives_a_concurrent_registry_writer(home, monkeypatch):
    """sync_models_dir()'s reconcile phase, racing a concurrent
    update_registry() write from another thread, must not lose either
    change - proving the reconcile loop now runs inside update_registry(),
    not a bare load_registry()/save_registry() pair."""
    home_dir, models_dir = home
    # One managed entry whose file is gone -> the reconcile loop flags it
    # ("missing") rather than nothing happening, exercising the write path.
    gone_path = models_dir / "gone.gguf"
    cfg.save_registry({"m": {"path": str(gone_path), "source": "local"}})
    _install_slow_entry_path(monkeypatch)
    barrier = threading.Barrier(2)

    def run_sync():
        barrier.wait()
        mm.sync_models_dir(prune=False)

    def concurrent_writer():
        barrier.wait()
        # sync_models_dir() calls _entry_path once before the reconcile
        # reload too (building its `known` set from the pre-scan registry),
        # so this must clear that single delay unit and still land inside
        # the ~4-delay-unit reconcile window that follows, not before it.
        time.sleep(0.25)
        cfg.update_registry(
            lambda reg: reg.__setitem__(
                "concurrent-writer-model",
                {"path": "Z:/concurrent.gguf", "source": "local"}))

    t1 = threading.Thread(target=run_sync)
    t2 = threading.Thread(target=concurrent_writer)
    t1.start(); t2.start()
    t1.join(timeout=15); t2.join(timeout=15)

    final = json.loads(cfg.REGISTRY_FILE.read_text(encoding="utf-8"))
    assert final["m"]["missing"] is True, "sync's own flagging must still land"
    assert final.get("concurrent-writer-model", {}).get("path") == "Z:/concurrent.gguf", (
        "sync_models_dir() must reconcile through update_registry() (atomic), "
        "not load_registry()/save_registry() - a concurrent registry write "
        "during its scan was lost")
