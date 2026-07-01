# SPDX-License-Identifier: AGPL-3.0-or-later
"""NEW-STOPCOMFY - localm can stop/restart the ComfyUI IT launched.

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
    from localm.media import comfy_client as cc
    monkeypatch.setattr(cc, "_comfy_alive", lambda url, timeout=1.0: True)
    app = _keyless_app(tmp_path, monkeypatch)
    client = TestClient(app)
    tok = {"Authorization": f"Bearer {app.state.shell_token}"}
    r = client.get("/v1/comfy/status", headers=tok)
    assert r.status_code == 200
    body = r.json()
    assert "alive" in body and "launched_by_localm" in body


def test_stop_route_calls_stop_comfy(tmp_path, monkeypatch):
    from localm.media import comfy_client as cc
    monkeypatch.setattr(cc, "stop_comfy", lambda api_url=None: (True, "stopped ok"))
    app = _keyless_app(tmp_path, monkeypatch)
    client = TestClient(app)
    tok = {"Authorization": f"Bearer {app.state.shell_token}"}
    r = client.post("/v1/comfy/stop", headers=tok)
    assert r.status_code == 200
    assert r.json() == {"ok": True, "message": "stopped ok"}
