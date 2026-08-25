# SPDX-License-Identifier: AGPL-3.0-or-later
"""Follow-up to #584/#586: model_manager/registry.py's sync_models_dir() had the same bare load/mutate/save shape the update_config()/update_registry() atomic helpers exist to close, just with the registry instead of the config. sync_models_dir() loaded the registry once, reconciled missing/restored/pr..."""

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
    """Delay every _entry_path() call - the accessor sync_models_dir()'s reconcile loop calls per registry entry to get its stat-checkable path."""
    real = registry_mod._entry_path

    def slow(entry):
        time.sleep(delay)
        return real(entry)

    monkeypatch.setattr(registry_mod, "_entry_path", slow)


def test_sync_models_dir_survives_a_concurrent_registry_writer(home, monkeypatch):
    """sync_models_dir()'s reconcile phase, racing a concurrent update_registry() write from another thread, must not lose either change - proving the reconcile loop now runs inside update_registry(), not a bare load_registry()/save_registry() pair."""
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
