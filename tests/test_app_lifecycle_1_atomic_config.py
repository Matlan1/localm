# SPDX-License-Identifier: AGPL-3.0-or-later
"""APP-LIFECYCLE-1 (correctness, race): cli/models.py's config_cmd() and
inference/routes/config.py's patch_config() used to do a raw
load_config()/mutate/save_config() sequence instead of the atomic
update_config(mutator) helper config.py already provides for exactly this
reason. load_config() and save_config() each take config._io_lock only for
their OWN call, so the window between them is unlocked: a concurrent config
write racing either call site (or racing each other) can be silently lost.

Regression: both call sites now go through update_config(), which holds the
lock across the whole read-modify-write, so a concurrent writer's change is
never silently dropped.
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
#  Demonstrate the hazard the fix closes: a bare load/mutate/save pair CAN    #
#  lose a concurrent update_config() write in the unlocked window between    #
#  load and save. This is the exact shape config_cmd()/patch_config() used   #
#  to have.                                                                   #
# --------------------------------------------------------------------------- #

def _bare_load_mutate_save(key, value, *, delay_before_save):
    """Reproduces the OLD buggy pattern inline (not calling into cli/models.py
    or routes/config.py, which are now fixed) - this is the shape being
    proven hazardous, so the fix's rationale is verified against real
    load_config/save_config behavior, not just asserted."""
    c = cfg.load_config()
    time.sleep(delay_before_save)   # the unlocked window (AGENTS.md rule: prove
    c[key] = value                  # the race, don't just assert the fix ran)
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
    # THE BUG: the concurrent update_config() write, which landed and was
    # persisted DURING the bare writer's window, gets silently overwritten
    # when the bare writer's stale in-memory copy is saved afterward.
    assert final["main_gpu_index"] != 1, (
        "expected the bare load/mutate/save race to LOSE the concurrent "
        "write - if this assertion fails, the race no longer reproduces "
        "and the regression tests below should be re-examined")


# --------------------------------------------------------------------------- #
#  The fix: update_config() holds the lock across the whole read-modify-     #
#  write, so two concurrent update_config() writers never lose each other's  #
#  change - this is what config_cmd()/patch_config() now use.                #
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

def _install_slow_write(monkeypatch, delay=0.15):
    """Widen update_config()'s critical section (still INSIDE its real
    _io_lock) so a genuinely concurrent second writer reliably overlaps it,
    without needing to touch either call site's source - this proves the
    ACTUAL call sites are race-safe, not a reimplementation of them."""
    real = cfg._atomic_write_json

    def slow(path, data):
        time.sleep(delay)
        return real(path, data)

    monkeypatch.setattr(cfg, "_atomic_write_json", slow)


def test_cli_config_cmd_survives_a_concurrent_writer(home, monkeypatch):
    """localm config <key> <value>, racing a concurrent update_config() write
    from another thread, must not lose either change - proving config_cmd()
    itself now uses the atomic path."""
    from localm.cli.models import config_cmd
    cfg.save_config({"n_ctx": 4096})
    _install_slow_write(monkeypatch)
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

    _install_slow_write(monkeypatch)
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
