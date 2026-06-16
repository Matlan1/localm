"""
Output-containment regression tests (B4).

localm's hard rule: nothing a ComfyUI generation produces may remain visible
inside ComfyUI - the only copy is the one localm saved. These tests stand up a
real (loopback) HTTP stub that behaves like ComfyUI's relevant endpoints and
drive the actual generation/containment code over real sockets, so they fail if
containment regresses (the previous tests only asserted "a file was deleted",
which is exactly how B4 slipped through).

Covered:
  * contain_comfy_artifacts clears the /history entry, deletes ComfyUI's on-disk
    output copy, and deletes an uploaded img2img source.
  * when the ComfyUI output dir cannot be resolved it still clears history but
    returns a loud WARNING instead of leaking silently.
  * generate_image end-to-end: saves locally, deletes the ComfyUI copy, clears
    history (with dir) / warns (without dir).
  * generate_music end-to-end: music had NO cleanup before this fix; prove it
    now contains via the COMFY_OUTPUT_DIR env resolution path.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from localm.image_gen import comfy
from localm.music_gen import comfy as music_comfy


class _ComfyStub(HTTPServer):
    """Minimal stand-in for the ComfyUI HTTP API used by the generators."""

    def __init__(self, output_dir, input_dir):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.history: dict = {}        # prompt_id -> outputs dict
        self.history_deleted: list = []  # prompt_ids deleted via POST /history
        self.output_kind = "images"    # "images" | "audio" | gifs ...
        self.file_ext = ".png"
        self._counter = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a):  # keep the test output quiet
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _raw(self, code, data: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 (stdlib naming)
        s = self.server
        p = urlparse(self.path)
        if p.path == "/system_stats":
            return self._json(200, {"system": {}})
        if p.path.startswith("/history/"):
            pid = p.path.rsplit("/", 1)[-1]
            if pid in s.history:
                return self._json(200, {pid: {"outputs": s.history[pid]}})
            return self._json(200, {})
        if p.path == "/view":
            q = parse_qs(p.query)
            fn = q.get("filename", [""])[0]
            sub = q.get("subfolder", [""])[0]
            typ = q.get("type", ["output"])[0]
            root = s.output_dir if typ == "output" else s.input_dir
            f = root / sub / fn
            if f.is_file():
                return self._raw(200, f.read_bytes())
            return self._json(404, {})
        return self._json(404, {})

    def do_POST(self):  # noqa: N802
        s = self.server
        p = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        if p.path == "/prompt":
            s._counter += 1
            pid = f"pid-{s._counter}"
            fn = f"ComfyUI_{s._counter:05d}_{s.file_ext}"
            s.output_dir.mkdir(parents=True, exist_ok=True)
            (s.output_dir / fn).write_bytes(b"FAKEMEDIADATA")
            s.history[pid] = {"9": {s.output_kind: [
                {"filename": fn, "subfolder": "", "type": "output"}]}}
            return self._json(200, {"prompt_id": pid})
        if p.path == "/upload/image":
            s.input_dir.mkdir(parents=True, exist_ok=True)
            name = "uploaded_input.png"
            (s.input_dir / name).write_bytes(b"INPUTDATA")
            return self._json(200, {"name": name})
        if p.path == "/history":
            data = json.loads(body or b"{}")
            for pid in data.get("delete", []):
                s.history.pop(pid, None)
                s.history_deleted.append(pid)
            return self._json(200, {})
        if p.path == "/free":
            return self._json(200, {})
        return self._json(404, {})


@pytest.fixture
def stub(tmp_path):
    out = tmp_path / "comfy" / "output"
    inp = tmp_path / "comfy" / "input"
    out.mkdir(parents=True)
    inp.mkdir(parents=True)
    srv = _ComfyStub(out, inp)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


@pytest.fixture(autouse=True)
def _no_localm_url(monkeypatch):
    # the unload-before-generate step is a no-op without this; keep tests offline
    monkeypatch.delenv("LOCALM_URL", raising=False)
    monkeypatch.delenv("COMFY_OUTPUT_DIR", raising=False)


# --------------------------------------------------------------------------- #
#  contain_comfy_artifacts - the shared containment helper
# --------------------------------------------------------------------------- #

def test_contain_with_dir_clears_history_and_deletes_copies(stub):
    fn = "ComfyUI_99999_.png"
    (stub.output_dir / fn).write_bytes(b"X")
    (stub.input_dir / "src.png").write_bytes(b"Y")
    stub.history["pidX"] = {"9": {"images": [
        {"filename": fn, "subfolder": "", "type": "output"}]}}

    warn = comfy.contain_comfy_artifacts(
        stub.base_url, "pidX",
        {"filename": fn, "subfolder": "", "type": "output"},
        comfy_output_dir=str(stub.output_dir),
        uploaded_input="src.png",
    )

    assert warn == ""                                  # fully contained
    assert not (stub.output_dir / fn).exists()         # ComfyUI copy deleted
    assert not (stub.input_dir / "src.png").exists()   # uploaded input deleted
    assert "pidX" in stub.history_deleted              # /history entry cleared
    assert "pidX" not in stub.history


def test_contain_without_dir_warns_but_still_clears_history(stub, monkeypatch):
    # no explicit dir, no env, empty config -> dir cannot be resolved
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    fn = "leak.png"
    (stub.output_dir / fn).write_bytes(b"X")
    stub.history["pidY"] = {"9": {"images": [
        {"filename": fn, "subfolder": "", "type": "output"}]}}

    warn = comfy.contain_comfy_artifacts(
        stub.base_url, "pidY",
        {"filename": fn, "subfolder": "", "type": "output"},
        comfy_output_dir=None,
    )

    assert "WARNING" in warn                       # loud, not silent
    assert (stub.output_dir / fn).exists()         # could not delete the copy
    assert "pidY" in stub.history_deleted          # but history WAS cleared


def test_contain_skips_delete_for_temp_artifacts(stub):
    # type "temp" is auto-purged by ComfyUI; we must not error, and no warning
    warn = comfy.contain_comfy_artifacts(
        stub.base_url, "pidT",
        {"filename": "x.png", "subfolder": "", "type": "temp"},
        comfy_output_dir=None,
    )
    assert warn == ""


# --------------------------------------------------------------------------- #
#  generate_image - end to end over the stub
# --------------------------------------------------------------------------- #

def test_generate_image_contains_with_dir(stub, tmp_path, monkeypatch):
    # force the committed example workflow so the test is independent of any
    # personal flux_workflow.json present on the dev's machine
    monkeypatch.setattr(comfy, "_workflow_path",
                        lambda: comfy._WORKFLOW_EXAMPLE_PATH)
    out = tmp_path / "saved" / "img.png"

    ok, msg = comfy.generate_image(
        "a cat", out, api_url=stub.base_url,
        comfy_output_dir=str(stub.output_dir), write_sidecar=False,
    )

    assert ok, msg
    assert out.is_file()                                    # localm saved a copy
    assert list(stub.output_dir.glob("ComfyUI_*")) == []    # ComfyUI copy gone
    assert stub.history_deleted                             # history cleared
    assert "WARNING" not in msg


def test_generate_image_warns_without_dir(stub, tmp_path, monkeypatch):
    monkeypatch.setattr(comfy, "_workflow_path",
                        lambda: comfy._WORKFLOW_EXAMPLE_PATH)
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    out = tmp_path / "saved2" / "img.png"

    ok, msg = comfy.generate_image(
        "a dog", out, api_url=stub.base_url,
        comfy_output_dir=None, write_sidecar=False,
    )

    assert ok, msg
    assert out.is_file()
    assert "WARNING" in msg                                 # loud, not silent
    assert list(stub.output_dir.glob("ComfyUI_*"))          # copy remains
    assert stub.history_deleted                             # history still cleared


# --------------------------------------------------------------------------- #
#  generate_music - end to end (music had NO cleanup before this fix)
# --------------------------------------------------------------------------- #

def test_generate_music_contains_via_env_dir(stub, tmp_path, monkeypatch):
    monkeypatch.setattr(music_comfy, "_workflow_path",
                        lambda: music_comfy._WORKFLOW_PATH)
    # music's generate_music has no comfy_output_dir param; resolution falls
    # back to the COMFY_OUTPUT_DIR env var
    monkeypatch.setenv("COMFY_OUTPUT_DIR", str(stub.output_dir))
    stub.output_kind = "audio"
    stub.file_ext = ".flac"
    out = tmp_path / "saved" / "track.flac"

    ok, msg = music_comfy.generate_music(
        "lofi, chill", out, api_url=stub.base_url,
        duration_seconds=5.0, write_sidecar=False,
    )

    assert ok, msg
    assert out.is_file()                                   # localm saved a copy
    assert list(stub.output_dir.glob("ComfyUI_*")) == []   # ComfyUI copy gone
    assert stub.history_deleted                            # history cleared
    assert "WARNING" not in msg
