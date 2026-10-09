# SPDX-License-Identifier: AGPL-3.0-or-later
"""POST /v1/images/generations: the OpenAI-compatible image generation route.

The real route, the real background-job registry, size validation, the privacy
and gallery handling and the error mapping run; only the image backend (ComfyUI)
is replaced, at the image plugin's ``backend`` module.
"""

import base64
import io
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

URL = "/v1/images/generations"
GOLDEN = Path(__file__).parent / "fixtures" / "openai_sdk"


def _png(width=8, height=8) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 200)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALM_TMPDIR", str(tmp_path / "scratch"))
    (tmp_path / "scratch").mkdir()
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


class _Backend:
    """Stands in for the ComfyUI image backend and records every call."""

    def __init__(self):
        self.calls = []
        self.result = None
        self.fail = None

    def settings(self, cfg):
        return {"reload_after": True, "warning": "",
                "api_url": "http://127.0.0.1:8188"}

    def ensure_available(self, s, on_progress=None):
        return True, "ComfyUI is up."

    def generate(self, s, prompt, out_path, **kw):
        self.calls.append({"prompt": prompt, "out_path": out_path, **kw})
        if self.fail:
            return False, self.fail
        w, h = kw.get("width", 8), kw.get("height", 8)
        out_path.write_bytes(self.result or _png(w, h))
        return True, f"Image saved to {out_path.name} (seed 1)"


@pytest.fixture
def image_app(home, monkeypatch):
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.jobs import JobManager
    app = FastAPI()
    PluginManager(app, external_root=home / "noplugins").install("image")
    app.state.jobs = JobManager()
    app.state.self_url = "http://127.0.0.1:8642/v1"
    backend = _Backend()
    mod = sys.modules["_localm_plugin_image.backend"]
    for name in ("settings", "ensure_available", "generate"):
        monkeypatch.setattr(mod, name, getattr(backend, name))
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: False)
    monkeypatch.setattr("localm.vram.media_single_device_shortfall",
                        lambda s: None)
    app.state.test_backend = backend
    return app


@pytest.fixture
def backend(image_app):
    return image_app.state.test_backend


@pytest.fixture
def client(image_app):
    with TestClient(image_app) as c:
        yield c


@pytest.fixture
def log_mode(monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")


def _gallery(home):
    d = home / "gui_images"
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else []


def _scratch(home):
    return sorted(p.name for p in (home / "scratch").iterdir())


def _decode(item):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(item["b64_json"])))


class TestPrivacyMode:
    """The default session mode: nothing is kept on disk."""

    def test_default_response_is_b64_json_and_nothing_is_kept(
            self, client, backend, home):
        r = client.post(URL, json={"prompt": "a cat", "size": "64x128"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert isinstance(body["created"], int)
        assert len(body["data"]) == 1 and list(body["data"][0]) == ["b64_json"]
        assert _decode(body["data"][0]).size == (64, 128)
        assert _gallery(home) == []
        assert _scratch(home) == []

    def test_url_format_is_refused_up_front(self, client, backend, home):
        r = client.post(URL, json={"prompt": "a cat", "response_format": "url"})
        assert r.status_code == 400
        assert "privacy" in r.json()["detail"]
        assert "b64_json" in r.json()["detail"]
        assert backend.calls == []

    def test_private_directory_is_removed_when_generation_fails(
            self, client, backend, home):
        backend.fail = "ComfyUI rejected the workflow"
        r = client.post(URL, json={"prompt": "a cat"})
        assert r.status_code == 502
        assert "ComfyUI rejected the workflow" in r.json()["detail"]
        assert _scratch(home) == []
        assert _gallery(home) == []

    def test_no_sidecar_is_requested_and_the_image_is_not_stamped(
            self, client, backend, home):
        client.post(URL, json={"prompt": "a cat"})
        assert backend.calls[0]["write_sidecar"] is False
        assert backend.calls[0]["delete_outputs"] is True
        assert not (home / "gui_images").exists() or _gallery(home) == []

    def test_a_directory_that_cannot_be_removed_fails_the_request(
            self, client, backend, home, monkeypatch):
        import shutil
        monkeypatch.setattr(shutil, "rmtree",
                            lambda p: (_ for _ in ()).throw(PermissionError("locked")))
        r = client.post(URL, json={"prompt": "a cat"})
        assert r.status_code == 500
        assert "not returned" in r.json()["detail"]


class TestLoggedMode:
    """Outside privacy mode the image is kept in the gallery."""

    @pytest.fixture
    def key_header(self, image_app):
        from localm import auth
        return {"Authorization": f"Bearer {auth.create_key('i', ['image'])['key']}"}

    def test_with_a_key_the_default_is_a_gallery_url_that_serves_the_png(
            self, client, backend, home, log_mode, key_header):
        r = client.post(URL, json={"prompt": "a cat"}, headers=key_header)
        assert r.status_code == 200, r.text
        item = r.json()["data"][0]
        assert list(item) == ["url"]
        assert item["url"].startswith("http://testserver/api/imagine/file/")
        name = item["url"].rsplit("/", 1)[1]
        assert name in _gallery(home)
        got = client.get(f"/api/imagine/file/{name}", headers=key_header)
        assert got.status_code == 200 and got.content[:8] == b"\x89PNG\r\n\x1a\n"
        assert backend.calls[0]["write_sidecar"] is True

    def test_the_url_is_not_served_to_a_caller_without_the_key(
            self, client, backend, home, log_mode, key_header):
        r = client.post(URL, json={"prompt": "a cat"}, headers=key_header)
        name = r.json()["data"][0]["url"].rsplit("/", 1)[1]
        assert client.get(f"/api/imagine/file/{name}").status_code == 401

    def test_without_a_key_the_default_is_b64_json_and_url_is_refused(
            self, client, backend, home, log_mode):
        r = client.post(URL, json={"prompt": "a cat"})
        assert r.status_code == 200
        assert list(r.json()["data"][0]) == ["b64_json"]
        refused = client.post(URL, json={"prompt": "a cat", "response_format": "url"})
        assert refused.status_code == 400
        assert "API key" in refused.json()["detail"]
        assert len(backend.calls) == 1

    def test_b64_json_is_also_kept_in_the_gallery(
            self, client, backend, home, log_mode):
        r = client.post(URL, json={"prompt": "a cat", "response_format": "b64_json"})
        assert _decode(r.json()["data"][0]).size == (8, 8)
        assert len([n for n in _gallery(home) if n.endswith(".png")]) == 1

    def test_n_images_are_generated_one_after_another(
            self, client, backend, home, log_mode, key_header):
        r = client.post(URL, json={"prompt": "a cat", "n": 3}, headers=key_header)
        assert r.status_code == 200
        urls = [d["url"] for d in r.json()["data"]]
        assert len(set(urls)) == 3
        assert len(backend.calls) == 3

    @pytest.mark.parametrize("n", [0, 5, -1])
    def test_n_out_of_range_is_422(self, client, backend, log_mode, n):
        assert client.post(URL, json={"prompt": "a cat", "n": n}).status_code == 422
        assert backend.calls == []


class TestSize:
    @pytest.mark.parametrize("size", ["1024x1024", "512x768", "64x64",
                                      "2048x2048", "1792X1024", " 256x256 "])
    def test_valid_sizes_reach_the_backend(self, client, backend, size):
        r = client.post(URL, json={"prompt": "p", "size": size})
        assert r.status_code == 200, r.text
        w, h = (int(v) for v in size.strip().lower().split("x"))
        assert (backend.calls[0]["width"], backend.calls[0]["height"]) == (w, h)

    @pytest.mark.parametrize("size", [None, "auto", "AUTO", ""])
    def test_no_size_leaves_the_workflow_size(self, client, backend, size):
        body = {"prompt": "p"} if size is None else {"prompt": "p", "size": size}
        assert client.post(URL, json=body).status_code == 200
        assert "width" not in backend.calls[0] and "height" not in backend.calls[0]

    @pytest.mark.parametrize("size", [
        "1024", "axb", "1024x", "x1024", "0x0", "63x64", "64x63", "1023x1024",
        "2056x2048", "4096x4096", "-1024x1024", "1024x1024x3", "8x8"])
    def test_invalid_sizes_are_400(self, client, backend, size):
        r = client.post(URL, json={"prompt": "p", "size": size})
        assert r.status_code == 400, r.text
        assert "size" in r.json()["detail"]
        assert backend.calls == []


class TestRefusals:
    @pytest.mark.parametrize("body", [
        {"prompt": ""}, {"prompt": "   "}, {"prompt": "x" * 32001},
        {"prompt": "p", "response_format": "gif"},
        {"prompt": "p", "output_format": "jpeg"},
        {"prompt": "p", "output_format": "webp"}])
    def test_bad_request_is_400(self, client, backend, body):
        assert client.post(URL, json=body).status_code == 400
        assert backend.calls == []

    def test_missing_prompt_is_422(self, client, backend):
        assert client.post(URL, json={}).status_code == 422

    def test_compat_fields_are_accepted_and_ignored(self, client, backend):
        r = client.post(URL, json={
            "prompt": "p", "model": "dall-e-3", "quality": "hd", "style": "vivid",
            "user": "u1", "output_format": "PNG", "background": "auto"})
        assert r.status_code == 200, r.text

    def test_backend_failure_is_a_502_naming_the_reason(self, client, backend):
        backend.fail = "ComfyUI is not running and could not be started"
        r = client.post(URL, json={"prompt": "p"})
        assert r.status_code == 502
        assert "could not be started" in r.json()["detail"]

    def test_unavailable_backend_is_a_502_with_its_message(
            self, client, backend, image_app, monkeypatch):
        mod = sys.modules["_localm_plugin_image.backend"]
        monkeypatch.setattr(mod, "ensure_available",
                            lambda s, on_progress=None: (False, "ComfyUI missing"))
        r = client.post(URL, json={"prompt": "p"})
        assert r.status_code == 502 and "ComfyUI missing" in r.json()["detail"]

    def test_no_job_registry_is_503(self, client, image_app):
        image_app.state.jobs = None
        assert client.post(URL, json={"prompt": "p"}).status_code == 503

    def test_unknown_server_address_is_503(self, client, image_app):
        image_app.state.self_url = ""
        assert client.post(URL, json={"prompt": "p"}).status_code == 503


class TestBackendSizeSeam:
    def test_comfy_backend_forwards_size_only_when_set(self, monkeypatch):
        from localm.plugins.builtin.image import backend as mod
        seen = []
        monkeypatch.setattr(mod._comfy, "generate_image",
                            lambda *a, **k: seen.append(k) or (True, "ok"))
        s = {"api_url": "http://127.0.0.1:8188", "launch_cmd": "", "workdir": "",
             "output_dir": ""}
        from pathlib import Path
        mod._comfy_generate(s, "p", Path("x.png"), self_url="u", write_sidecar=False)
        mod._comfy_generate(s, "p", Path("x.png"), self_url="u",
                            write_sidecar=False, width=512, height=768)
        assert "width" not in seen[0] and "height" not in seen[0]
        assert (seen[1]["width"], seen[1]["height"]) == (512, 768)


class TestAuth:
    @pytest.fixture
    def keys(self, image_app):
        from localm import auth
        return {
            "image": auth.create_key("i", ["image"])["key"],
            "chat": auth.create_key("c", ["chat"])["key"],
            "voice": auth.create_key("v", ["voice"])["key"],
            "admin": auth.create_key("a", ["admin"], allow_privileged=True)["key"],
        }

    @staticmethod
    def _bearer(key):
        return {"Authorization": f"Bearer {key}"}

    def test_no_key_is_401_and_nothing_is_generated(self, client, backend, keys):
        r = client.post(URL, json={"prompt": "p"})
        assert r.status_code == 401
        assert backend.calls == []

    def test_garbage_key_is_401(self, client, backend, keys):
        r = client.post(URL, json={"prompt": "p"}, headers=self._bearer("nope"))
        assert r.status_code == 401

    @pytest.mark.parametrize("who", ["chat", "voice"])
    def test_a_key_without_the_image_scope_is_403(self, client, backend, keys, who):
        r = client.post(URL, json={"prompt": "p"}, headers=self._bearer(keys[who]))
        assert r.status_code == 403 and "image" in r.json()["detail"]
        assert backend.calls == []

    @pytest.mark.parametrize("who", ["image", "admin"])
    def test_image_and_admin_keys_are_served(self, client, backend, keys, who):
        r = client.post(URL, json={"prompt": "p"}, headers=self._bearer(keys[who]))
        assert r.status_code == 200, r.text
        assert len(backend.calls) == 1

    def test_open_mode_serves_without_a_key(self, client, backend):
        assert client.post(URL, json={"prompt": "p"}).status_code == 200


class TestJobIsTrackedLikeAnyOtherGeneration:
    def test_the_generation_runs_as_an_imagine_job(self, client, image_app, backend):
        client.post(URL, json={"prompt": "p"})
        kinds = [row["kind"] for row in image_app.state.jobs.snapshot()]
        assert kinds == ["imagine"]
        assert image_app.state.jobs.snapshot()[0]["status"] == "done"


class TestRealSdkWireFormat:
    """The JSON body the official openai SDK itself sent (captured with a mock
    transport; see tests/fixtures/openai_sdk), replayed against the route."""

    def test_the_sdk_request_is_accepted_and_honoured(self, client, backend):
        rec = json.loads((GOLDEN / "image_generation.json").read_text(encoding="utf-8"))
        assert "/v1" + rec["path"] == URL
        r = client.post(URL, content=base64.b64decode(rec["body_b64"]),
                        headers={"content-type": rec["content_type"]})
        assert r.status_code == 200, r.text
        assert len(r.json()["data"]) == 2
        assert [(c["width"], c["height"]) for c in backend.calls] == [(1024, 1024)] * 2
        assert all(c["prompt"] == "a lighthouse" for c in backend.calls)


def test_openapi_describes_the_json_request_body(image_app):
    schema = image_app.openapi()
    ref = schema["paths"][URL]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    props = schema["components"]["schemas"][ref.rsplit("/", 1)[1]]["properties"]
    assert {"prompt", "n", "size", "response_format"} <= set(props)


class _FakeJob:
    def __init__(self, events):
        self.cancelled = False
        self._events = events
        self._queue = None

    def subscribe(self):
        import asyncio
        self._queue = asyncio.Queue()
        for e in self._events:
            self._queue.put_nowait(e)
        return self._queue

    def unsubscribe(self, q):
        pass

    def cancel(self):
        self.cancelled = True


class _FakeRequest:
    """A request whose own is_disconnected() is always False, as behind the
    app's BaseHTTPMiddleware handlers; the working poll is in the ASGI scope."""

    def __init__(self, disconnected):
        async def poll():
            return disconnected

        self.scope = {"localm.disconnect_poll": poll}

    async def is_disconnected(self):
        return False


class TestAwaitJob:
    @pytest.fixture
    def plug(self, image_app, monkeypatch):
        mod = sys.modules["_localm_plugin_image"]
        monkeypatch.setattr(mod, "JOB_POLL_SECONDS", 0.02)
        monkeypatch.setattr(mod, "CANCEL_SETTLE_SECONDS", 0.3)
        return mod

    def test_returns_status_and_last_line_when_the_job_ends(self, plug):
        import asyncio
        job = _FakeJob([{"type": "line", "text": "first"},
                        {"type": "line", "text": "boom"},
                        {"type": "end", "status": "failed"}])
        got = asyncio.run(plug._await_job(job, _FakeRequest(False)))
        assert got == ("failed", "boom") and not job.cancelled

    def test_disconnect_cancels_the_job_and_waits_for_it_to_stop(self, plug):
        import asyncio
        job = _FakeJob([])
        events = []

        async def run():
            task = asyncio.ensure_future(plug._await_job(job, _FakeRequest(True)))
            await asyncio.sleep(0.1)
            events.append(job.cancelled)
            job._queue.put_nowait({"type": "end", "status": "cancelled"})
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=3)

        asyncio.run(run())
        assert events == [True]

    def test_a_job_that_never_stops_is_given_up_on_after_the_settle_time(self, plug):
        import asyncio
        import time
        job = _FakeJob([])
        t0 = time.monotonic()
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(asyncio.wait_for(
                plug._await_job(job, _FakeRequest(True)), timeout=3))
        assert job.cancelled and 0.25 <= time.monotonic() - t0 < 5


class TestKernelOriginGate:
    """On the real kernel app, in open mode, the OpenAI SDK (no Origin header,
    an arbitrary bearer) and a cross-origin local app both get past the
    origin/shell-token gate."""

    @pytest.fixture
    def kernel_client(self, home, monkeypatch):
        from localm.inference.http_server import create_app
        from localm.plugins.engine import PluginManager
        from localm.plugins.gui.jobs import JobManager
        app = create_app(None)
        PluginManager(app, external_root=home / "noplugins").install("image")
        app.state.jobs = JobManager()
        app.state.self_url = "http://127.0.0.1:8642/v1"
        backend = _Backend()
        mod = sys.modules["_localm_plugin_image.backend"]
        for name in ("settings", "ensure_available", "generate"):
            monkeypatch.setattr(mod, name, getattr(backend, name))
        monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: False)
        monkeypatch.setattr("localm.vram.media_single_device_shortfall",
                            lambda s: None)
        with TestClient(app) as c:
            yield c

    def test_sdk_style_request_without_origin_is_served(self, kernel_client):
        r = kernel_client.post(URL, json={"prompt": "p"},
                               headers={"Authorization": "Bearer sk-anything"})
        assert r.status_code == 200, r.text

    def test_cross_origin_local_app_is_served(self, kernel_client):
        r = kernel_client.post(URL, json={"prompt": "p"},
                               headers={"Origin": "http://localhost:9999"})
        assert r.status_code == 200, r.text

    def test_the_gui_route_is_still_refused_cross_origin(self, kernel_client):
        r = kernel_client.post("/api/imagine", json={"prompt": "p"},
                               headers={"Origin": "http://localhost:9999"})
        assert r.status_code == 403 and "cross-origin" in r.json()["detail"].lower()


class TestRemoveTreeWhenIdle:
    @pytest.fixture
    def plug(self, image_app, monkeypatch):
        mod = sys.modules["_localm_plugin_image"]
        monkeypatch.setattr(mod, "JOB_POLL_SECONDS", 0.02)
        return mod

    @staticmethod
    def _tree(tmp_path):
        d = tmp_path / "private"
        d.mkdir()
        (d / "a.png").write_bytes(b"x")
        return d

    def test_removes_at_once_when_every_job_has_stopped(self, plug, tmp_path):
        from types import SimpleNamespace
        d = self._tree(tmp_path)
        plug._remove_tree_when_idle(d, [SimpleNamespace(finished_at=1.0)])
        assert not d.exists()

    def test_waits_for_a_job_that_is_still_writing(self, plug, tmp_path):
        import time
        from types import SimpleNamespace
        d = self._tree(tmp_path)
        job = SimpleNamespace(finished_at=None)
        plug._remove_tree_when_idle(d, [job])
        time.sleep(0.2)
        assert d.exists(), "removed while a job could still write into it"
        job.finished_at = 2.0
        deadline = time.monotonic() + 5
        while d.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not d.exists()

    def test_no_jobs_removes_at_once(self, plug, tmp_path):
        d = self._tree(tmp_path)
        plug._remove_tree_when_idle(d, [])
        assert not d.exists()
