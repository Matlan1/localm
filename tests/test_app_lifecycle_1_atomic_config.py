# SPDX-License-Identifier: AGPL-3.0-or-later
"""cli/models.py's config_cmd() and inference/routes/config.py's
patch_config() must go through the atomic update_config(mutator) helper, which
holds config._io_lock across the whole read-modify-write.

A raw load_config()/mutate/save_config() sequence takes the lock only for each
of its own calls, so the window between them is unlocked and a concurrent
config write racing either call site can be silently lost.
"""

import threading
import time

import pytest

import localm.config as cfg


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir()
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


# --------------------------------------------------------------------------- #
#  A bare load/mutate/save pair loses a concurrent update_config() write in   #
#  the unlocked window between load and save.                                 #
# --------------------------------------------------------------------------- #

def _bare_load_mutate_save(key, value, *, delay_before_save):
    """Reproduces the OLD buggy pattern inline (not calling into cli/models.py
    or routes/config.py, which are now fixed) - this is the shape being
    proven hazardous, so the fix's rationale is verified against real
    load_config/save_config behavior, not just asserted."""
    c = cfg.load_config()
    time.sleep(delay_before_save)   # the unlocked window
    c[key] = value
    cfg.save_config(c)


def test_bare_load_save_pair_can_lose_a_concurrent_update(home):
    """Sanity check that the vulnerability this fix closes is REAL: a writer
    using the old bare pattern, racing an update_config() writer, can drop
    the concurrent update. If this test ever fails, the fixture/race no
    longer reproduces the hazard and the fix's own tests below need
    revisiting."""
    cfg.save_config({"n_ctx": 4096})

    barrier = threading.Barrier(2)

    def slow_bare_writer():
        barrier.wait()
        _bare_load_mutate_save("temperature", 0.9, delay_before_save=0.2)

    def fast_concurrent_writer():
        barrier.wait()
        time.sleep(0.05)  # land inside the bare writer's load->save window
        cfg.update_config(lambda c: c.__setitem__("main_gpu_index", 1))

    t1 = threading.Thread(target=slow_bare_writer)
    t2 = threading.Thread(target=fast_concurrent_writer)
    t1.start(); t2.start()
    t1.join(); t2.join()

    final = cfg.load_config()
    assert final["temperature"] == 0.9  # the bare writer's own change lands
    # The concurrent update_config() write, persisted DURING the bare writer's
    # window, is overwritten when the bare writer's stale in-memory copy is
    # saved afterward.
    assert final["main_gpu_index"] != 1, (
        "expected the bare load/mutate/save race to LOSE the concurrent "
        "write - if this assertion fails, the race no longer reproduces "
        "and the regression tests below should be re-examined")


# --------------------------------------------------------------------------- #
#  update_config() holds the lock across the whole read-modify-write, so two  #
#  concurrent update_config() writers never lose each other's change.         #
# --------------------------------------------------------------------------- #

def test_update_config_never_loses_a_concurrent_write(home):
    cfg.save_config({"n_ctx": 4096})
    barrier = threading.Barrier(2)

    def writer_a():
        barrier.wait()
        def _mutate(c):
            time.sleep(0.1)  # hold the lock across a window a bare pair would not
            c["temperature"] = 0.9
        cfg.update_config(_mutate)

    def writer_b():
        barrier.wait()
        time.sleep(0.02)  # would land inside writer_a's window if unlocked
        cfg.update_config(lambda c: c.__setitem__("main_gpu_index", 1))

    t1 = threading.Thread(target=writer_a)
    t2 = threading.Thread(target=writer_b)
    t1.start(); t2.start()
    t1.join(); t2.join()

    final = cfg.load_config()
    assert final["temperature"] == 0.9
    assert final["main_gpu_index"] == 1, (
        "update_config() must serialize writers so no concurrent change is lost")


# --------------------------------------------------------------------------- #
#  The actual call sites: config_cmd() (CLI) and patch_config() (HTTP route)  #
#  must go through update_config(), not a bare load/save pair.                #
# --------------------------------------------------------------------------- #

def _install_slow_merge(monkeypatch, delay=0.15):
    """Widen the read-modify-write critical section via _merge_stored_config,
    the one internal step whose LOCK STATUS actually differs between the fixed
    and the old, buggy call-site shape:

      - load_config() calls _merge_stored_config() AFTER its own `with _io_lock`
        block has already exited (config.py:694-697) - so in the OLD bare
        load_config()/save_config() pattern this delay lands OUTSIDE any lock,
        in the exact unlocked read-to-write gap the bug report describes.
      - update_config() calls _merge_stored_config() INSIDE its `with _io_lock`
        block (config.py) - so in the FIXED call-site shape this delay widens
        a window that IS held under the lock.

    _atomic_write_json is NOT a usable patch point here: it sits inside the
    lock in BOTH shapes (bare save_config() also takes _io_lock around its own
    call), so delaying it widens an already-locked window and cannot tell the
    two call-site shapes apart."""
    real = cfg._merge_stored_config

    def slow(cfgd, stored):
        time.sleep(delay)
        return real(cfgd, stored)

    monkeypatch.setattr(cfg, "_merge_stored_config", slow)


def test_cli_config_cmd_survives_a_concurrent_writer(home, monkeypatch):
    """localm config <key> <value>, racing a concurrent update_config() write
    from another thread, must not lose either change - proving config_cmd()
    itself now uses the atomic path."""
    from localm.cli.models import config_cmd
    cfg.save_config({"n_ctx": 4096})
    _install_slow_merge(monkeypatch)
    barrier = threading.Barrier(2)

    def run_cli_command():
        barrier.wait()
        config_cmd.callback(key="temperature", value="0.9")

    def concurrent_writer():
        barrier.wait()
        time.sleep(0.02)  # start just after - lands inside the widened window
        cfg.update_config(lambda c: c.__setitem__("main_gpu_index", 1))

    t1 = threading.Thread(target=run_cli_command)
    t2 = threading.Thread(target=concurrent_writer)
    t1.start(); t2.start()
    t1.join(timeout=10); t2.join(timeout=10)

    final = cfg.load_config()
    assert final["temperature"] == 0.9
    assert final["main_gpu_index"] == 1, (
        "config_cmd() must use update_config() (atomic), not load_config()/"
        "save_config() - a concurrent write during its mutation was lost")


def test_http_patch_config_survives_a_concurrent_writer(home, monkeypatch):
    """The /v1/config PATCH handler, racing a concurrent update_config() write,
    must not lose either change - proving patch_config() itself now uses the
    atomic path (the same property set_media_config(), 20 lines below it in
    the same file, already had)."""
    import asyncio
    from fastapi import FastAPI
    import localm.inference.routes.config as config_routes

    cfg.save_config({"n_ctx": 4096})
    app = FastAPI()

    class _Ctx:
        class mode:
            value = "standard"
    config_routes.register(app, _Ctx())
    patch_config = next(
        r for r in app.routes if getattr(r, "path", None) == "/v1/config"
        and "PATCH" in getattr(r, "methods", set())
    ).endpoint

    _install_slow_merge(monkeypatch)
    barrier = threading.Barrier(2)

    class _FakeRequest:
        pass

    def run_patch():
        barrier.wait()
        asyncio.run(patch_config({"temperature": 0.9}, _FakeRequest()))

    def concurrent_writer():
        barrier.wait()
        time.sleep(0.02)
        cfg.update_config(lambda c: c.__setitem__("main_gpu_index", 1))

    t1 = threading.Thread(target=run_patch)
    t2 = threading.Thread(target=concurrent_writer)
    t1.start(); t2.start()
    t1.join(timeout=10); t2.join(timeout=10)

    final = cfg.load_config()
    assert final["temperature"] == 0.9
    assert final["main_gpu_index"] == 1, (
        "patch_config() must use update_config() (atomic), not load_config()/"
        "save_config() - a concurrent write during its mutation was lost")


# --------------------------------------------------------------------------- #
#  Four more call sites with the same bare load_config()/mutate/save_config() #
#  shape. Same technique (_install_slow_merge): each must survive a           #
#  concurrent update_config() writer landing in its widened merge window.     #
# --------------------------------------------------------------------------- #

def test_save_plugin_config_survives_a_concurrent_writer(home, monkeypatch):
    """PluginHost.save_plugin_config() (the public plugin Host API,
    docs/plugins.md's "write a plugin's config atomically") must not lose a
    concurrent update_config() write during its own mutation."""
    from unittest.mock import MagicMock

    from localm.plugins.engine import PluginHost

    cfg.save_config({"n_ctx": 4096})
    _install_slow_merge(monkeypatch)
    spec = MagicMock()
    spec.name = "alpha"
    host = PluginHost(MagicMock(), MagicMock(), spec)
    barrier = threading.Barrier(2)

    def run_save():
        barrier.wait()
        host.save_plugin_config(cfg=[{"secret": "alpha-owns-this"}][0])

    def concurrent_writer():
        barrier.wait()
        time.sleep(0.02)
        cfg.update_config(lambda c: c.__setitem__("main_gpu_index", 1))

    t1 = threading.Thread(target=run_save)
    t2 = threading.Thread(target=concurrent_writer)
    t1.start(); t2.start()
    t1.join(timeout=10); t2.join(timeout=10)

    final = cfg.load_config()
    assert final["plugins"]["alpha"]["secret"] == "alpha-owns-this"
    assert final["main_gpu_index"] == 1, (
        "save_plugin_config() must use update_config() (atomic), not "
        "load_config()/save_config() - a concurrent write during its "
        "mutation was lost")


def test_set_auto_deps_survives_a_concurrent_writer(home, monkeypatch):
    """localm.cli.plugins._set_auto_deps() must not lose a concurrent
    update_config() write during its own mutation."""
    from localm.cli.plugins import _set_auto_deps

    cfg.save_config({"n_ctx": 4096})
    _install_slow_merge(monkeypatch)
    barrier = threading.Barrier(2)

    def run_set():
        barrier.wait()
        _set_auto_deps(False)

    def concurrent_writer():
        barrier.wait()
        time.sleep(0.02)
        cfg.update_config(lambda c: c.__setitem__("main_gpu_index", 1))

    t1 = threading.Thread(target=run_set)
    t2 = threading.Thread(target=concurrent_writer)
    t1.start(); t2.start()
    t1.join(timeout=10); t2.join(timeout=10)

    final = cfg.load_config()
    assert final["auto_install_plugin_deps"] is False
    assert final["main_gpu_index"] == 1, (
        "_set_auto_deps() must use update_config() (atomic), not "
        "load_config()/save_config() - a concurrent write during its "
        "mutation was lost")


def test_remember_func_shim_survives_a_concurrent_writer(home, monkeypatch):
    """localm.cli.media._remember_func_shim() must not lose a concurrent
    update_config() write during its own mutation."""
    from localm.cli.media import _remember_func_shim

    cfg.save_config({"n_ctx": 4096})
    _install_slow_merge(monkeypatch)
    barrier = threading.Barrier(2)

    def run_remember():
        barrier.wait()
        _remember_func_shim()

    def concurrent_writer():
        barrier.wait()
        time.sleep(0.02)
        cfg.update_config(lambda c: c.__setitem__("main_gpu_index", 1))

    t1 = threading.Thread(target=run_remember)
    t2 = threading.Thread(target=concurrent_writer)
    t1.start(); t2.start()
    t1.join(timeout=10); t2.join(timeout=10)

    final = cfg.load_config()
    assert final["comfy_func_shim"] is True
    assert final["main_gpu_index"] == 1, (
        "_remember_func_shim() must use update_config() (atomic), not "
        "load_config()/save_config() - a concurrent write during its "
        "mutation was lost")


def test_mark_managed_comfy_setup_offered_survives_a_concurrent_writer(home, monkeypatch):
    """localm.media.comfy_client.mark_managed_comfy_setup_offered() must not
    lose a concurrent update_config() write during its own mutation (lower
    stakes than the other three - its own docstring frames it as best-effort -
    but it now uses the same atomic helper for consistency)."""
    from localm.media.comfy_client import mark_managed_comfy_setup_offered

    cfg.save_config({"n_ctx": 4096})
    _install_slow_merge(monkeypatch)
    barrier = threading.Barrier(2)

    def run_mark():
        barrier.wait()
        mark_managed_comfy_setup_offered()

    def concurrent_writer():
        barrier.wait()
        time.sleep(0.02)
        cfg.update_config(lambda c: c.__setitem__("main_gpu_index", 1))

    t1 = threading.Thread(target=run_mark)
    t2 = threading.Thread(target=concurrent_writer)
    t1.start(); t2.start()
    t1.join(timeout=10); t2.join(timeout=10)

    final = cfg.load_config()
    assert final["comfy_managed_setup_offered"] is True
    assert final["main_gpu_index"] == 1, (
        "mark_managed_comfy_setup_offered() must use update_config() "
        "(atomic), not load_config()/save_config() - a concurrent write "
        "during its mutation was lost")
