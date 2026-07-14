# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Output-containment regression tests.

Containment is OPT-IN. By DEFAULT localm LEAVES ComfyUI's own copies and /history
entry alone, because a user may run ComfyUI for its own gallery and want the
files. When the user opts in (delete_outputs=True, or privacy mode forces it),
nothing a generation produces may remain visible inside ComfyUI - the only copy
is the one localm saved. These tests stand up a real (loopback) HTTP stub that
behaves like ComfyUI's relevant endpoints and drive the actual
generation/containment code over real sockets.

Covered:
  * default (delete_outputs=False) is a NO-OP: ComfyUI keeps its copy + history.
  * contain_comfy_artifacts(delete_outputs=True) clears the /history entry,
    deletes ComfyUI's on-disk output copy, and deletes an uploaded img2img source.
  * when deletion is requested but the output dir cannot be resolved it still
    clears history and returns a loud WARNING instead of leaking silently.
  * generate_image / generate_music end-to-end with delete_outputs=True contain;
    by default they keep ComfyUI's copy.
"""

from __future__ import annotations

import json
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from localm.image_gen import comfy
from localm.music_gen import comfy as music_comfy


def _minimal_png() -> bytes:
    """A structurally valid 1x1 PNG for the stub's image outputs.

    ComfyUI writes real PNGs, so the stub must too: otherwise generate_image's
    _strip_png_metadata gets non-PNG bytes and (correctly) warns the strip could
    not run, which is not what these containment tests are exercising. Real PNG
    bytes let the actual strip path run clean, as it does in production."""
    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (len(data).to_bytes(4, "big") + ctype + data
                + zlib.crc32(ctype + data).to_bytes(4, "big"))
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")  # 1 filter byte + one RGB pixel
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


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
            # Images get real PNG bytes so the strip path runs as in production;
            # audio (music tests) is not PNG-stripped, so placeholder bytes are fine.
            media = _minimal_png() if s.output_kind == "images" else b"FAKEMEDIADATA"
            (s.output_dir / fn).write_bytes(media)
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
#  contain_comfy_artifacts - the shared containment helper                     #
# --------------------------------------------------------------------------- #

def test_contain_default_keeps_comfy_copy_and_history(stub):
    # DEFAULT (delete_outputs not set / False): a no-op. ComfyUI keeps its copy
    # AND its history entry, because a user may want them.
    fn = "ComfyUI_00001_.png"
    (stub.output_dir / fn).write_bytes(b"X")
    stub.history["pidK"] = {"9": {"images": [
        {"filename": fn, "subfolder": "", "type": "output"}]}}

    warn = comfy.contain_comfy_artifacts(
        stub.base_url, "pidK",
        {"filename": fn, "subfolder": "", "type": "output"},
        comfy_output_dir=str(stub.output_dir),
    )

    assert warn == ""                              # nothing to warn about
    assert (stub.output_dir / fn).exists()         # ComfyUI copy kept
    assert "pidK" not in stub.history_deleted      # history NOT cleared
    assert "pidK" in stub.history


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
        delete_outputs=True,
    )

    assert warn == ""                                  # fully contained
    assert not (stub.output_dir / fn).exists()         # ComfyUI copy deleted
    assert not (stub.input_dir / "src.png").exists()   # uploaded input deleted
    assert "pidX" in stub.history_deleted              # /history entry cleared
    assert "pidX" not in stub.history


def test_contain_without_dir_warns_but_still_clears_history(stub, monkeypatch):
    # delete requested, but no dir / env / config -> dir cannot be resolved
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    fn = "leak.png"
    (stub.output_dir / fn).write_bytes(b"X")
    stub.history["pidY"] = {"9": {"images": [
        {"filename": fn, "subfolder": "", "type": "output"}]}}

    warn = comfy.contain_comfy_artifacts(
        stub.base_url, "pidY",
        {"filename": fn, "subfolder": "", "type": "output"},
        comfy_output_dir=None,
        delete_outputs=True,
    )

    assert "WARNING" in warn                       # loud, not silent
    assert (stub.output_dir / fn).exists()         # could not delete the copy
    assert "pidY" in stub.history_deleted          # but history WAS cleared


def test_contain_skips_delete_for_temp_artifacts(stub):
    # type "temp" is auto-purged by ComfyUI; even when delete is requested we must
    # not error, and no warning.
    warn = comfy.contain_comfy_artifacts(
        stub.base_url, "pidT",
        {"filename": "x.png", "subfolder": "", "type": "temp"},
        comfy_output_dir=None,
        delete_outputs=True,
    )
    assert warn == ""


# --------------------------------------------------------------------------- #
#  generate_image - end to end over the stub                                  #
# --------------------------------------------------------------------------- #

def test_generate_image_default_keeps_comfy_copy(stub, tmp_path, monkeypatch):
    # By default a generation leaves ComfyUI's own copy + history in place.
    monkeypatch.setattr(comfy, "workflow_path",
                        lambda: comfy._WORKFLOW_EXAMPLE_PATH)
    out = tmp_path / "saved" / "img.png"

    ok, msg = comfy.generate_image(
        "a cat", out, api_url=stub.base_url,
        comfy_output_dir=str(stub.output_dir), write_sidecar=False,
    )

    assert ok, msg
    assert out.is_file()                                  # localm saved a copy
    assert list(stub.output_dir.glob("ComfyUI_*"))        # ComfyUI copy KEPT
    assert not stub.history_deleted                       # history NOT cleared
    assert "WARNING" not in msg


def test_generate_image_contains_with_dir(stub, tmp_path, monkeypatch):
    # force the committed example workflow so the test is independent of any
    # personal flux_workflow.json present on the dev's machine
    monkeypatch.setattr(comfy, "workflow_path",
                        lambda: comfy._WORKFLOW_EXAMPLE_PATH)
    out = tmp_path / "saved" / "img.png"

    ok, msg = comfy.generate_image(
        "a cat", out, api_url=stub.base_url,
        comfy_output_dir=str(stub.output_dir), delete_outputs=True,
        write_sidecar=False,
    )

    assert ok, msg
    assert out.is_file()                                    # localm saved a copy
    assert list(stub.output_dir.glob("ComfyUI_*")) == []    # ComfyUI copy gone
    assert stub.history_deleted                             # history cleared
    assert "WARNING" not in msg


def test_generate_image_warns_without_dir(stub, tmp_path, monkeypatch):
    monkeypatch.setattr(comfy, "workflow_path",
                        lambda: comfy._WORKFLOW_EXAMPLE_PATH)
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    out = tmp_path / "saved2" / "img.png"

    ok, msg = comfy.generate_image(
        "a dog", out, api_url=stub.base_url,
        comfy_output_dir=None, delete_outputs=True, write_sidecar=False,
    )

    assert ok, msg
    assert out.is_file()
    assert "WARNING" in msg                                 # loud, not silent
    assert list(stub.output_dir.glob("ComfyUI_*"))          # copy remains
    assert stub.history_deleted                             # history still cleared


# --------------------------------------------------------------------------- #
#  generate_music - end to end (delete_outputs threads through the env path)  #
# --------------------------------------------------------------------------- #

def test_generate_music_contains_via_env_dir(stub, tmp_path, monkeypatch):
    monkeypatch.setattr(music_comfy, "workflow_path",
                        lambda: music_comfy._WORKFLOW_PATH)
    # music's generate_music has no comfy_output_dir param; resolution falls
    # back to the COMFY_OUTPUT_DIR env var
    monkeypatch.setenv("COMFY_OUTPUT_DIR", str(stub.output_dir))
    stub.output_kind = "audio"
    stub.file_ext = ".flac"
    out = tmp_path / "saved" / "track.flac"

    ok, msg = music_comfy.generate_music(
        "lofi, chill", out, api_url=stub.base_url,
        duration_seconds=5.0, delete_outputs=True, write_sidecar=False,
    )

    assert ok, msg
    assert out.is_file()                                   # localm saved a copy
    assert list(stub.output_dir.glob("ComfyUI_*")) == []   # ComfyUI copy gone
    assert stub.history_deleted                            # history cleared
    assert "WARNING" not in msg
