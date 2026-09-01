# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm can stop and restart the ComfyUI IT launched.

Covers the retained-handle registry, the graceful stop (abort render + clear
queue + free VRAM), the never-kill-a-user's-ComfyUI rule, and the HTTP routes.
"""

from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
#  comfy_client core
# --------------------------------------------------------------------------- #

class _FakeProc:
    def __init__(self, pid=4321, alive=True):
        self.pid = pid
        self._alive = alive
        self.killed = False

    def poll(self):
        return None if self._alive else 0


def _patch_common(monkeypatch, cc, *, alive_after=False):
    """Stub the network side-effects so the tests never touch a real ComfyUI."""
    calls = {"interrupt": 0, "vram": 0, "killed": []}
    monkeypatch.setattr(cc, "interrupt_comfy", lambda url: calls.__setitem__("interrupt", calls["interrupt"] + 1) or True)
    monkeypatch.setattr(cc, "free_comfy_vram", lambda url=None: calls.__setitem__("vram", calls["vram"] + 1) or True)
    monkeypatch.setattr(cc, "_comfy_alive", lambda url, timeout=3.0: alive_after)
    monkeypatch.setattr(cc, "_kill_process_tree", lambda proc: calls["killed"].append(getattr(proc, "pid", None)))
    return calls


def test_stop_terminates_a_localm_launched_comfy(monkeypatch):
    from localm.media import comfy_client as cc
    calls = _patch_common(monkeypatch, cc)
    url = "http://127.0.0.1:8188"
    proc = _FakeProc(pid=999)
    cc._remember_spawned(url, proc)

    ok, msg = cc.stop_comfy(url)
    assert ok
    assert "launched" in msg.lower()
    assert calls["interrupt"] == 1 and calls["vram"] == 1
    assert calls["killed"] == [999]                     # our tree was killed
    assert cc.spawned_pid(url) is None                  # handle dropped


def test_stop_does_not_kill_a_user_launched_comfy(monkeypatch):
    from localm.media import comfy_client as cc
    calls = _patch_common(monkeypatch, cc, alive_after=True)   # alive, but not ours
    url = "http://127.0.0.1:8188"
    # no _remember_spawned -> localm did not launch it
    ok, msg = cc.stop_comfy(url)
    assert ok
    assert "did not launch" in msg.lower()
    assert calls["interrupt"] == 1                      # render still aborted
    assert calls["killed"] == []                        # process left alone


def test_spawned_pid_reflects_liveness(monkeypatch):
    from localm.media import comfy_client as cc
    url = "http://127.0.0.1:8188"
    cc._remember_spawned(url, _FakeProc(pid=555, alive=True))
    assert cc.spawned_pid(url) == 555
    cc._remember_spawned(url, _FakeProc(pid=556, alive=False))   # exited
    assert cc.spawned_pid(url) is None
    cc._take_spawned(url)


def test_restart_does_not_relaunch_a_still_running_foreign_comfy(monkeypatch):
    """The reported bug: restarting a ComfyUI localm did not launch must not
    report a green success as if a fresh instance came up - nothing did."""
    from localm.media import comfy_client as cc
    calls = _patch_common(monkeypatch, cc, alive_after=True)   # alive, but not ours
    url = "http://127.0.0.1:8188"
    # no _remember_spawned -> localm did not launch it
    launches = []
    monkeypatch.setattr(cc, "ensure_comfy",
                        lambda **kw: launches.append(kw) or (True, "should not be reached"))

    ok, msg = cc.restart_comfy(url)

    assert ok
    assert "did not launch" in msg.lower()
    assert "nothing was restarted" in msg.lower()
    assert calls["interrupt"] == 1                      # the render was still aborted
    assert calls["killed"] == []                         # the foreign process was left alone
    assert launches == []                                 # no relaunch was attempted


def test_restart_relaunches_when_localm_launched_it(monkeypatch):
    """Regression guard: the ordinary case - localm's own ComfyUI - must still
    stop and relaunch exactly as before."""
    from localm.media import comfy_client as cc
    calls = _patch_common(monkeypatch, cc, alive_after=False)
    url = "http://127.0.0.1:8188"
    cc._remember_spawned(url, _FakeProc(pid=777))
    launches = []
    monkeypatch.setattr(cc, "ensure_comfy",
                        lambda **kw: launches.append(kw) or (True, "ComfyUI is up."))

    ok, msg = cc.restart_comfy(url, wait_seconds=42)

    assert ok
    assert msg == "ComfyUI is up."
    assert calls["killed"] == [777]                      # our old process was terminated
    assert len(launches) == 1                             # ensure_comfy was actually asked to relaunch
    assert launches[0]["api_url"] == url
    assert launches[0]["wait_seconds"] == 42


def test_restart_still_attempts_a_launch_when_nothing_is_running_at_all(monkeypatch):
    """Not-ours-and-not-alive is a DIFFERENT case from the reported bug: there
    is no foreign process to leave running, so restart must still try to
    bring one up rather than silently doing nothing."""
    from localm.media import comfy_client as cc
    calls = _patch_common(monkeypatch, cc, alive_after=False)   # not ours, not alive either
    url = "http://127.0.0.1:8188"
    # no _remember_spawned -> localm did not launch it, and it is not alive
    launches = []
    monkeypatch.setattr(cc, "ensure_comfy",
                        lambda **kw: launches.append(kw) or (True, "ComfyUI is up."))

    ok, msg = cc.restart_comfy(url)

    assert ok
    assert msg == "ComfyUI is up."
    assert len(launches) == 1                              # a launch was attempted
    assert calls["killed"] == []


def test_restart_holds_the_launch_lock_across_the_ownership_check_and_stop(monkeypatch):
    """TOCTOU regression guard: without a lock spanning the ownership read and
    the stop, a concurrent restart_comfy() on the same url can pop the
    tracked proc between the two, making a genuine owner falsely conclude
    "not ours". Pin the mechanism that prevents it: the per-url launch lock
    must be held for that whole span."""
    from localm.media import comfy_client as cc
    calls = _patch_common(monkeypatch, cc, alive_after=False)
    url = "http://127.0.0.1:8188"
    cc._remember_spawned(url, _FakeProc(pid=321))
    lock = cc._launch_lock_for(url)
    seen_locked_during_stop = []

    def _spy_stop_comfy(api_url=None):
        seen_locked_during_stop.append(lock.locked())
        return True, "Stopped the ComfyUI that localm launched."

    monkeypatch.setattr(cc, "stop_comfy", _spy_stop_comfy)
    monkeypatch.setattr(cc, "ensure_comfy", lambda **kw: (True, "ComfyUI is up."))

    cc.restart_comfy(url)

    assert seen_locked_during_stop == [True]
    assert not lock.locked()                             # released once restart_comfy returns
    assert calls == {"interrupt": 0, "vram": 0, "killed": []}   # stop_comfy was the spy, not the real one


def test_restart_does_not_deadlock_under_concurrent_restarts(monkeypatch):
    """Sanity check for the lock added above: two real, overlapping
    restart_comfy() calls on the same url must both resolve rather than
    wait on each other forever."""
    import threading
    from localm.media import comfy_client as cc
    _patch_common(monkeypatch, cc, alive_after=False)
    url = "http://127.0.0.1:8188"
    cc._remember_spawned(url, _FakeProc(pid=321))
    monkeypatch.setattr(cc, "ensure_comfy", lambda **kw: (True, "ComfyUI is up."))
    results = {}

    def _run(name):
        results[name] = cc.restart_comfy(url)

    threads = [threading.Thread(target=_run, args=(n,), name=n) for n in ("A", "B")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not any(t.is_alive() for t in threads), "restart_comfy deadlocked under contention"
    assert len(results) == 2
    for name, (ok, msg) in results.items():
        assert ok, (name, msg)


# --------------------------------------------------------------------------- #
#  routes
# --------------------------------------------------------------------------- #

def _keyless_app(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.inference.http_server import create_app
    return create_app(None)


def test_status_reports_launched_by_localm(tmp_path, monkeypatch):
    # get_comfy_status resolves _comfy_alive via `from localm.image_gen.comfy
    # import _comfy_alive` (a fresh local import each call), and image_gen.comfy
    # imports it as its OWN module-level name (`from localm.media.comfy_client
    # import _comfy_alive`), a SEPARATE binding from comfy_client's own attribute.
    # Patching comfy_client._comfy_alive does not reach the route.
    from localm.image_gen import comfy as ic
    monkeypatch.setattr(ic, "_comfy_alive", lambda url, timeout=1.0: True)
    app = _keyless_app(tmp_path, monkeypatch)
    client = TestClient(app)
    tok = {"Authorization": f"Bearer {app.state.shell_token}"}
    r = client.get("/v1/comfy/status", headers=tok)
    assert r.status_code == 200
    body = r.json()
    assert body["alive"] is True, (
        f"expected the patched _comfy_alive's True to reach the response, got "
        f"{body!r} - the patch may not be intercepting the real call")
    assert "launched_by_localm" in body


def test_stop_route_calls_stop_comfy(tmp_path, monkeypatch):
    from localm.media import comfy_client as cc
    monkeypatch.setattr(cc, "stop_comfy", lambda api_url=None: (True, "stopped ok"))
    app = _keyless_app(tmp_path, monkeypatch)
    client = TestClient(app)
    tok = {"Authorization": f"Bearer {app.state.shell_token}"}
    r = client.post("/v1/comfy/stop", headers=tok)
    assert r.status_code == 200
    assert r.json() == {"ok": True, "message": "stopped ok"}


def test_stop_route_504s_when_stop_comfy_hangs_past_budget(tmp_path, monkeypatch):
    """A wedged stop_comfy() call (a taskkill that never returns, say) returns
    a 504 within the configured budget rather than hanging the HTTP request."""
    import time

    from localm.inference.routes import config as config_routes
    from localm.media import comfy_client as cc

    def _hangs(api_url=None):
        time.sleep(2.0)
        return True, "should never get here in time"

    monkeypatch.setattr(cc, "stop_comfy", _hangs)
    monkeypatch.setattr(config_routes, "_COMFY_STOP_TIMEOUT_S", 0.2)
    app = _keyless_app(tmp_path, monkeypatch)
    client = TestClient(app)
    tok = {"Authorization": f"Bearer {app.state.shell_token}"}

    start = time.monotonic()
    r = client.post("/v1/comfy/stop", headers=tok)
    elapsed = time.monotonic() - start

    assert r.status_code == 504, r.text
    assert "timed out" in r.json()["detail"].lower()
    assert elapsed < 1.5, (
        f"the route waited {elapsed:.2f}s despite a 0.2s budget - the "
        "timeout did not actually bound the request")
