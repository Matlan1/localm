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
  * generate_music / generate_video strip the workflow (prompt, lyrics) that
    ComfyUI's SaveAudio / SaveVideo embed as container metadata from the copy
    localm saves, and warn loudly when the fetched file is a container the
    strip does not handle.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

import pytest

from localm.image_gen import comfy
from localm.media import comfy_client
from localm.media.comfy_client import _comfy_output_root
from localm.music_gen import comfy as music_comfy
from localm.video_gen import comfy as video_comfy

# A non-routable RFC5737 (TEST-NET-1) address, so nothing here can dial a real
# host from this machine or CI.
_UNC = r"\\192.0.2.1\share"
_UNC_FWD = "//192.0.2.1/share"
_DEVICE = r"\\.\PhysicalDrive0"
_X_SHA256 = hashlib.sha256(b"X").hexdigest()


def _is_unc_or_device(s: str) -> bool:
    return s[:2] in ("\\\\", "//", "\\/", "/\\")


def _minimal_png() -> bytes:
    """A structurally valid 1x1 PNG for the stub's image outputs.

    ComfyUI writes real PNGs, so the stub does too: with non-PNG bytes
    generate_image's _strip_png_metadata warns that the strip could not run,
    which is not what these containment tests exercise."""
    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (len(data).to_bytes(4, "big") + ctype + data
                + zlib.crc32(ctype + data).to_bytes(4, "big"))
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")  # 1 filter byte + one RGB pixel
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


_MEDIA_FIXTURES = Path(__file__).parent / "fixtures" / "media"
_EMBEDDED_PROMPT_MARKER = b"SECRET-LYRICS-MARKER-7Q4M"


def _stub_media_bytes(output_kind: str) -> bytes:
    """What the stub's save node writes for an output kind: a real PNG, or the
    committed FLAC / MP4 fixture carrying an embedded workflow (with
    _EMBEDDED_PROMPT_MARKER in its lyrics) the way ComfyUI embeds it."""
    if output_kind == "images":
        return _minimal_png()
    if output_kind == "audio":
        return (_MEDIA_FIXTURES / "prompt.flac").read_bytes()
    if output_kind in ("videos", "gifs"):
        return (_MEDIA_FIXTURES / "prompt.mp4").read_bytes()
    return b"FAKEMEDIADATA"


class _ComfyStub(HTTPServer):
    """Minimal stand-in for the ComfyUI HTTP API used by the generators."""

    def __init__(self, output_dir, input_dir):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.history: dict = {}        # prompt_id -> outputs dict
        self.history_deleted: list = []  # prompt_ids deleted via POST /history
        self.fail_history_clear = False  # force POST /history to fail
        self.output_kind = "images"    # "images" | "audio" | gifs ...
        self.file_ext = ".png"
        self.media_bytes = None        # override what the save node writes
        self.uploaded_names: list = []  # filename each upload was sent under
        self.upload_reply_name = None  # a reply name other than the stored one
        self.store_other_bytes = False  # an upload stores bytes other than those sent
        self.prompts: list = []        # workflow of each POST /prompt
        self.submit_mode = None        # "http_error" | "no_id" | "drop"
        self.job_state = "finished"    # "finished" | "exec_error" | "no_output" | "running"
        self.view_fails = False        # force GET /view to fail
        self.object_info = None        # GET /object_info reply, None answers 404
        self.queue_running: list = []  # prompt ids GET /queue lists as running
        self.queue_unreadable = False  # force GET /queue to fail
        self.cancel_refused = False    # force the cancel and interrupt routes to fail
        self.cancel_calls: list = []   # prompt ids of each cancel request
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
        if p.path == "/object_info" and s.object_info is not None:
            return self._json(200, s.object_info)
        if p.path == "/queue":
            if s.queue_unreadable:
                return self._json(500, {"error": "boom"})
            return self._json(200, {"queue_running": [[0, pid] for pid in s.queue_running],
                                    "queue_pending": []})
        if p.path.startswith("/history/"):
            pid = p.path.rsplit("/", 1)[-1]
            if pid in s.history:
                entry = {"outputs": s.history[pid]}
                if s.job_state == "exec_error":
                    entry["status"] = {"status_str": "error", "messages": [
                        ["execution_error", {"node_type": "VAEDecode",
                                             "exception_message": "out of memory"}]]}
                return self._json(200, {pid: entry})
            return self._json(200, {})
        if p.path == "/view":
            if s.view_fails:
                return self._json(500, {"error": "boom"})
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
            s.prompts.append(json.loads(body)["prompt"])
            if s.submit_mode == "http_error":
                return self._json(400, {"error": {"type": "invalid_prompt",
                                                  "message": "Cannot execute because a node is missing",
                                                  "details": ""}, "node_errors": {}})
            if s.submit_mode == "no_id":
                return self._json(200, {})
            if s.submit_mode == "drop":
                self.close_connection = True
                return None
            s._counter += 1
            pid = f"pid-{s._counter}"
            if s.job_state == "running":
                s.queue_running.append(pid)
                return self._json(200, {"prompt_id": pid})
            if s.job_state in ("exec_error", "no_output"):
                s.history[pid] = {}
                return self._json(200, {"prompt_id": pid})
            fn = f"ComfyUI_{s._counter:05d}_{s.file_ext}"
            s.output_dir.mkdir(parents=True, exist_ok=True)
            # Real container bytes for every kind, so each generator's metadata
            # strip runs as in production: a PNG for images, and the PyAV-written
            # FLAC / MP4 fixtures (embedded prompt included) for audio / videos.
            (s.output_dir / fn).write_bytes(
                s.media_bytes if s.media_bytes is not None
                else _stub_media_bytes(s.output_kind))
            s.history[pid] = {"9": {s.output_kind: [
                {"filename": fn, "subfolder": "", "type": "output"}]}}
            return self._json(200, {"prompt_id": pid})
        if p.path == "/upload/image":
            s.input_dir.mkdir(parents=True, exist_ok=True)
            sent = re.search(rb'filename="([^"]+)"', body)
            name = sent.group(1).decode() if sent else "uploaded_input.png"
            s.uploaded_names.append(name)
            data = re.search(rb"\r\n\r\n(.*)\r\n--[^\r\n]+--\r\n$", body, re.S).group(1)
            if s.store_other_bytes:
                data = b"other bytes"
            # A clash takes ComfyUI's name: same bytes reuse the file, others get " (n)".
            stem, ext = os.path.splitext(name)
            n = 0
            while (s.input_dir / name).exists() and (s.input_dir / name).read_bytes() != data:
                n += 1
                name = f"{stem} ({n}){ext}"
            (s.input_dir / name).write_bytes(data)
            return self._json(200, {"name": s.upload_reply_name or name})
        if p.path.startswith("/api/jobs/") and p.path.endswith("/cancel"):
            pid = p.path.split("/")[3]
            s.cancel_calls.append(pid)
            if s.cancel_refused:
                return self._json(500, {"error": "boom"})
            return self._json(200, {"cancelled": pid in s.queue_running})
        if p.path == "/interrupt":
            return self._json(500 if s.cancel_refused else 200, {})
        if p.path == "/queue":
            return self._json(200, {})
        if p.path == "/history":
            if s.fail_history_clear:
                return self._json(500, {"error": "boom"})
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
    # AND its history entry.
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
    stub.history["pidX"] = {"9": {"images": [
        {"filename": fn, "subfolder": "", "type": "output"}]}}

    warn = comfy.contain_comfy_artifacts(
        stub.base_url, "pidX",
        {"filename": fn, "subfolder": "", "type": "output"},
        comfy_output_dir=str(stub.output_dir),
        output_sha256=_X_SHA256,
        delete_outputs=True,
    )

    assert warn == ""                                  # fully contained
    assert not (stub.output_dir / fn).exists()         # ComfyUI copy deleted
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


def test_contain_warns_when_history_clear_fails(stub):
    # NEW-COMFY-HISTORY-CLEAR-DISCARDED: clear_comfy_history()'s return value
    # must not be discarded - a failed clear is a containment step that failed
    # and must be surfaced, even though the rest of containment still proceeds.
    fn = "ComfyUI_history_fail.png"
    (stub.output_dir / fn).write_bytes(b"X")
    stub.history["pidHF"] = {"9": {"images": [
        {"filename": fn, "subfolder": "", "type": "output"}]}}
    stub.fail_history_clear = True

    warn = comfy.contain_comfy_artifacts(
        stub.base_url, "pidHF",
        {"filename": fn, "subfolder": "", "type": "output"},
        comfy_output_dir=str(stub.output_dir),
        output_sha256=_X_SHA256,
        delete_outputs=True,
    )

    assert "WARNING" in warn
    assert "history" in warn.lower()
    assert not (stub.output_dir / fn).exists()      # the rest of containment still ran
    assert "pidHF" not in stub.history_deleted      # history genuinely was not cleared


class TestContainmentRejectsOutOfBoundsNames:
    """Every name below is parsed straight out of ComfyUI's own HTTP JSON
    (/history outputs, the /upload/image reply), so a remote or compromised
    ComfyUI - which sanitize_comfy_url permits over plaintext http on a LAN or
    public api_url - controls it. Under plain pathlib an ABSOLUTE component
    REPLACES the base and a traversing subfolder walks out of it.

    Two properties per vector:
      1. the file outside the ComfyUI dirs SURVIVES (containment held), and
      2. a WARNING is returned, so a rejected name is never silently skipped.

    The tests above all use well-formed names.
    """

    @staticmethod
    def _victim(tmp_path):
        """A file OUTSIDE ComfyUI's output/ and input/ dirs. The stub's dirs are
        tmp_path/comfy/{output,input}, so tmp_path/victim.txt is two levels up
        from output/ - exactly what '../../victim.txt' reaches."""
        v = tmp_path / "victim.txt"
        v.write_text("do not delete me", encoding="utf-8")
        return v

    # EVERY vector resolves INSIDE tmp_path. The absolute vector is BUILT from
    # tmp_path, which is itself absolute AND drive-qualified on Windows, so it
    # exercises the same escape (an absolute component REPLACES the base under
    # pathlib) without naming a path outside the fixture.
    ABS = "<ABS_OUTSIDE>"

    @pytest.mark.parametrize("subfolder,filename", [
        ("", "../../victim.txt"),          # traversal in the filename
        ("../..", "victim.txt"),           # traversal in the subfolder
        ("", ABS),                         # absolute + drive-qualified
        ("", ""),                          # collapses to the output dir itself
    ])
    def test_output_name_escape_is_refused_and_warned(
            self, stub, tmp_path, subfolder, filename):
        victim = self._victim(tmp_path)
        if filename == self.ABS:
            filename = str(victim)

        warn = comfy.contain_comfy_artifacts(
            stub.base_url, "pidEsc",
            {"filename": filename, "subfolder": subfolder, "type": "output"},
            comfy_output_dir=str(stub.output_dir),
            delete_outputs=True,
        )

        assert victim.exists(), f"containment escaped: {subfolder}/{filename}"
        assert victim.read_text(encoding="utf-8") == "do not delete me"
        assert "WARNING" in warn, "a refused name must be surfaced, not skipped"
        # The output dir itself must survive the '' case.
        assert stub.output_dir.is_dir()

    @pytest.mark.parametrize("uploaded", [
        "../../victim.txt",
        ABS,                               # absolute + drive-qualified, in tmp
        "sub/../../../victim.txt",
    ])
    def test_uploaded_input_escape_is_refused_and_warned(
            self, stub, tmp_path, uploaded):
        victim = self._victim(tmp_path)
        if uploaded == self.ABS:
            uploaded = str(victim)

        warn = comfy_client.remove_uploaded_input(
            stub.base_url, uploaded, comfy_output_dir=str(stub.output_dir))

        assert victim.exists(), f"containment escaped via uploaded_input: {uploaded}"
        assert victim.read_text(encoding="utf-8") == "do not delete me"
        assert "was not removed" in warn, "a refused name must be surfaced, not skipped"

    def test_legitimate_nesting_still_deletes(self, stub):
        """`subfolder` is a real ComfyUI feature, so confinement permits depth
        and rejects only escape."""
        nested = stub.output_dir / "batch01"
        nested.mkdir()
        (nested / "ComfyUI_00007_.png").write_bytes(b"X")

        warn = comfy.contain_comfy_artifacts(
            stub.base_url, "pidNest",
            {"filename": "ComfyUI_00007_.png", "subfolder": "batch01",
             "type": "output"},
            comfy_output_dir=str(stub.output_dir),
            output_sha256=_X_SHA256,
            delete_outputs=True,
        )

        assert warn == "", f"legitimate nested output warned: {warn!r}"
        assert not (nested / "ComfyUI_00007_.png").exists()

    def test_output_alias_name_does_not_delete_a_different_real_file(
            self, stub, monkeypatch):
        """A compromised or malicious ComfyUI (sanitize_comfy_url permits a LAN
        or public api_url over plaintext http) could report a `filename` that
        is a short-name-alias for a DIFFERENT, real file already sitting in the
        same output directory - e.g. an earlier generation's output. That
        target stays strictly under output/, so the escape checks above cannot
        see it; only the resolved-name check catches it. The OS-level
        substitution is simulated, so no 8.3-enabled volume is needed."""
        victim = stub.output_dir / "LongModelNameThatIsVeryLong.png"
        victim.write_bytes(b"do not delete me")
        alias = "LONGMO~1.PNG"
        victim_resolved = victim.resolve()

        real_resolve = Path.resolve

        def fake_resolve(self, *a, **k):
            if self.name == alias:
                return victim_resolved
            return real_resolve(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", fake_resolve)

        warn = comfy.contain_comfy_artifacts(
            stub.base_url, "pidAlias",
            {"filename": alias, "subfolder": "", "type": "output"},
            comfy_output_dir=str(stub.output_dir),
            delete_outputs=True,
        )

        assert victim.exists(), "alias substitution deleted the wrong file"
        assert victim.read_bytes() == b"do not delete me"
        assert "WARNING" in warn, "a refused name must be surfaced, not skipped"

    def test_symlinked_subfolder_pointing_outside_is_refused(self, stub, tmp_path):
        """Lexical checks alone are not enough: a symlink INSIDE output/ that
        points out of it turns a perfectly well-formed name into an escape. This
        is why the helper asserts on the RESOLVED path, not just the string."""
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "victim.txt"
        target.write_text("do not delete me", encoding="utf-8")
        try:
            (stub.output_dir / "link").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError) as e:
            pytest.skip(f"symlink creation not permitted here: {e}")

        warn = comfy.contain_comfy_artifacts(
            stub.base_url, "pidLink",
            {"filename": "victim.txt", "subfolder": "link", "type": "output"},
            comfy_output_dir=str(stub.output_dir),
            delete_outputs=True,
        )

        assert target.exists(), "symlinked subfolder escaped containment"
        assert "WARNING" in warn


class TestComfyOutputRootUncGuard:
    """_comfy_output_root() is the read-time choke point every caller of
    contain_comfy_artifacts goes through, and comfy_output_dir is settable by a
    config:write-scoped (privileged, not ADMIN) caller, so the UNC/device guard
    lives there."""

    @pytest.mark.parametrize("bad", [_UNC, _UNC_FWD, _DEVICE])
    def test_unc_and_device_dir_returns_none(self, bad):
        assert _comfy_output_root(bad) is None

    def test_ordinary_dir_still_resolves(self, tmp_path):
        assert _comfy_output_root(str(tmp_path)) == Path(tmp_path)

    def test_env_var_source_is_also_guarded(self, monkeypatch):
        monkeypatch.setenv("COMFY_OUTPUT_DIR", _UNC)
        assert _comfy_output_root() is None

    def test_config_source_is_also_guarded(self, monkeypatch):
        monkeypatch.delenv("COMFY_OUTPUT_DIR", raising=False)
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"comfy_output_dir": _UNC})
        assert _comfy_output_root() is None


class TestContainmentRejectsUntrustedRoot:
    """confined_under() validates only the RELATIVE path handed to it, never
    whether the BASE it is confined under is itself safe. comfy_output_dir (the
    source of that base) is settable by a config:write-scoped caller
    (privileged, but not ADMIN). Every test above this class passes a trusted,
    test-fixed root (stub.output_dir), so a UNC-shaped comfy_output_dir reaching
    confined_under's own .resolve() call is only covered here."""

    @pytest.mark.parametrize("bad", [_UNC, _UNC_FWD, _DEVICE])
    def test_unc_and_device_output_dir_rejected_without_touching_the_filesystem(
            self, stub, monkeypatch, bad):
        fn = "ComfyUI_00042_.png"
        (stub.output_dir / fn).write_bytes(b"X")
        stub.history["pidRoot"] = {"9": {"images": [
            {"filename": fn, "subfolder": "", "type": "output"}]}}

        real_resolve = Path.resolve
        seen: list = []

        def spy(self, *a, **kw):
            s = str(self)
            seen.append(s)
            if _is_unc_or_device(s):
                raise AssertionError(
                    f"Path.resolve() reached the filesystem with a UNC/device "
                    f"string: {s!r} - this is the SMB dial (and the "
                    "net-NTLMv2 leak), which happens before any containment "
                    "check can refuse it")
            return real_resolve(self, *a, **kw)

        monkeypatch.setattr(Path, "resolve", spy)
        warn = comfy.contain_comfy_artifacts(
            stub.base_url, "pidRoot",
            {"filename": fn, "subfolder": "", "type": "output"},
            comfy_output_dir=bad,
            delete_outputs=True,
        )

        assert "WARNING" in warn, "an unresolvable root must be surfaced, not silent"
        assert (stub.output_dir / fn).exists(), (
            "the real ComfyUI output must survive - containment must fail "
            "safe, not touch an arbitrary path")
        assert not any(_is_unc_or_device(s) for s in seen), (
            "the UNC/device root reached Path.resolve() - the whole finding "
            "is that this syscall happens before any containment check")

    def test_ordinary_root_is_unaffected(self, stub):
        """Control: an ordinary (non-UNC) root still resolves and contains
        normally."""
        fn = "ComfyUI_00043_.png"
        (stub.output_dir / fn).write_bytes(b"X")
        stub.history["pidOrd"] = {"9": {"images": [
            {"filename": fn, "subfolder": "", "type": "output"}]}}

        warn = comfy.contain_comfy_artifacts(
            stub.base_url, "pidOrd",
            {"filename": fn, "subfolder": "", "type": "output"},
            comfy_output_dir=str(stub.output_dir),
            output_sha256=_X_SHA256,
            delete_outputs=True,
        )
        assert warn == ""
        assert not (stub.output_dir / fn).exists()


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


def test_generate_image_threads_instance_token_to_unload(stub, tmp_path, monkeypatch):
    """The instance_token the image route resolved from its own app state (for
    keyless-mode auth on the localm_url unload call) reaches _localm_unload,
    rather than being dropped between the plug.py route and comfy.py's call
    site."""
    monkeypatch.setattr(comfy, "workflow_path",
                        lambda: comfy._WORKFLOW_EXAMPLE_PATH)
    unload_spy = MagicMock(return_value=None)
    out = tmp_path / "saved" / "img.png"

    with patch.object(comfy, "_localm_unload", unload_spy):
        ok, msg = comfy.generate_image(
            "a cat", out, api_url=stub.base_url,
            comfy_output_dir=str(stub.output_dir), write_sidecar=False,
            localm_url="http://127.0.0.1:9/v1", instance_token="tok-abc",
        )

    assert ok, msg
    unload_spy.assert_called_once()
    assert unload_spy.call_args.args == ("http://127.0.0.1:9/v1", "tok-abc")


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


# --------------------------------------------------------------------------- #
#  The saved track / clip carries none of the prompt ComfyUI embedded in it    #
# --------------------------------------------------------------------------- #

def test_generate_music_strips_the_embedded_lyrics_from_the_saved_track(
        stub, tmp_path, monkeypatch):
    monkeypatch.setattr(music_comfy, "workflow_path",
                        lambda: music_comfy._WORKFLOW_PATH)
    stub.output_kind = "audio"
    stub.file_ext = ".flac"
    out = tmp_path / "saved" / "track.flac"

    ok, msg = music_comfy.generate_music(
        "lofi, chill", out, api_url=stub.base_url, lyrics="[verse] la la",
        duration_seconds=5.0, write_sidecar=False,
    )

    assert ok, msg
    comfy_copy = next(stub.output_dir.glob("ComfyUI_*"))
    assert _EMBEDDED_PROMPT_MARKER in comfy_copy.read_bytes()   # what was fetched
    saved = out.read_bytes()
    assert saved[:4] == b"fLaC"
    assert len(saved) == comfy_copy.stat().st_size
    assert _EMBEDDED_PROMPT_MARKER not in saved
    assert b"prompt" not in saved
    assert "WARNING" not in msg
    assert not out.with_suffix(".flac.json").exists()           # no sidecar either


def test_generate_video_strips_the_embedded_prompt_from_the_saved_clip(
        stub, tmp_path, monkeypatch):
    monkeypatch.setattr(video_comfy, "workflow_path",
                        lambda: video_comfy._WORKFLOW_PATH)
    stub.output_kind = "videos"
    stub.file_ext = ".mp4"
    out = tmp_path / "saved" / "clip.mp4"

    ok, msg = video_comfy.generate_video(
        "a fox in the snow", out, api_url=stub.base_url,
        seconds=1.0, write_sidecar=False,
    )

    assert ok, msg
    comfy_copy = next(stub.output_dir.glob("ComfyUI_*"))
    assert _EMBEDDED_PROMPT_MARKER in comfy_copy.read_bytes()   # what was fetched
    saved = out.read_bytes()
    assert saved[4:8] == b"ftyp"
    assert len(saved) == comfy_copy.stat().st_size
    assert _EMBEDDED_PROMPT_MARKER not in saved
    assert b"prompt" not in saved
    assert "WARNING" not in msg
    assert not out.with_suffix(".mp4.json").exists()            # no sidecar either


def test_generate_music_warns_when_the_track_is_not_a_container_it_can_strip(
        stub, tmp_path, monkeypatch):
    monkeypatch.setattr(music_comfy, "workflow_path",
                        lambda: music_comfy._WORKFLOW_PATH)
    stub.output_kind = "audio"
    stub.file_ext = ".flac"
    stub.media_bytes = b"FAKEMEDIADATA"
    out = tmp_path / "saved" / "track.flac"

    ok, msg = music_comfy.generate_music(
        "lofi, chill", out, api_url=stub.base_url,
        duration_seconds=5.0, write_sidecar=False,
    )

    assert ok, msg
    assert out.read_bytes() == b"FAKEMEDIADATA"                 # bytes untouched
    assert "WARNING: generated file is not FLAC or MP3" in msg   # loud, not silent


# --------------------------------------------------------------------------- #
#  The file deleted is the one this job produced, on the instance that ran it  #
# --------------------------------------------------------------------------- #

@pytest.fixture
def managed_stub(tmp_path, monkeypatch):
    """A stub standing in for localm's own ComfyUI, serving from
    <LOCALM_HOME>/comfyui/output, with the user's own ComfyUI folder beside it
    holding a same-named file."""
    from localm.media import managed_comfy as mc
    home = tmp_path / "home"
    out = home / "comfyui" / "output"
    inp = home / "comfyui" / "input"
    out.mkdir(parents=True)
    inp.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))
    srv = _ComfyStub(out, inp)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    monkeypatch.setattr(mc, "MANAGED_COMFY_API_URL", srv.base_url)
    monkeypatch.setattr(mc, "is_managed_comfy_installed", lambda: True)
    user_comfy = tmp_path / "users-comfyui"
    (user_comfy / "output").mkdir(parents=True)
    (user_comfy / "input").mkdir()
    srv.user_comfy = user_comfy
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def _users_file(srv, name: str) -> Path:
    victim = srv.user_comfy / "output" / name
    victim.write_bytes(b"the user's own picture")
    return victim


@pytest.mark.parametrize("configured_workdir", [True, False])
def test_managed_job_deletes_its_own_copy_and_never_the_users_file(
        managed_stub, tmp_path, monkeypatch, configured_workdir):
    monkeypatch.setattr(comfy, "workflow_path", lambda: comfy._WORKFLOW_EXAMPLE_PATH)
    cfg = {"comfy_target": "own"}
    if configured_workdir:
        cfg["comfy_workdir"] = str(managed_stub.user_comfy)
    monkeypatch.setattr("localm.config.load_config", lambda: cfg)
    victim = _users_file(managed_stub, "ComfyUI_00001_.png")

    ok, msg = comfy.generate_image(
        "a cat", tmp_path / "saved" / "img.png", api_url=managed_stub.base_url,
        delete_outputs=True, write_sidecar=False)

    assert ok, msg
    assert victim.read_bytes() == b"the user's own picture", "a user's file was deleted"
    assert list(managed_stub.output_dir.glob("ComfyUI_*")) == [], "the managed copy stayed"
    assert "WARNING" not in msg


@pytest.mark.parametrize("kind,ext,gen", [
    ("audio", ".flac", "music"),
    ("videos", ".mp4", "video"),
])
def test_managed_music_and_video_delete_their_own_copy_and_never_the_users_file(
        managed_stub, tmp_path, monkeypatch, kind, ext, gen):
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"comfy_target": "own",
                                 "comfy_workdir": str(managed_stub.user_comfy)})
    managed_stub.output_kind = kind
    managed_stub.file_ext = ext
    victim = _users_file(managed_stub, f"ComfyUI_00001_{ext}")
    out = tmp_path / "saved" / f"out{ext}"

    if gen == "music":
        monkeypatch.setattr(music_comfy, "workflow_path", lambda: music_comfy._WORKFLOW_PATH)
        ok, msg = music_comfy.generate_music(
            "lofi", out, api_url=managed_stub.base_url, duration_seconds=5.0,
            delete_outputs=True, write_sidecar=False)
    else:
        monkeypatch.setattr(video_comfy, "workflow_path", lambda: video_comfy._WORKFLOW_PATH)
        ok, msg = video_comfy.generate_video(
            "a fox", out, api_url=managed_stub.base_url, seconds=1.0,
            delete_outputs=True, write_sidecar=False)

    assert ok, msg
    assert victim.read_bytes() == b"the user's own picture", "a user's file was deleted"
    assert list(managed_stub.output_dir.glob("ComfyUI_*")) == [], "the managed copy stayed"
    assert "WARNING" not in msg


def test_managed_job_ignores_an_output_dir_naming_the_users_comfyui(managed_stub, tmp_path, monkeypatch):
    monkeypatch.setattr(comfy, "workflow_path", lambda: comfy._WORKFLOW_EXAMPLE_PATH)
    monkeypatch.setattr("localm.config.load_config", lambda: {"comfy_target": "own"})
    victim = _users_file(managed_stub, "ComfyUI_00001_.png")

    ok, msg = comfy.generate_image(
        "a cat", tmp_path / "saved" / "img.png", api_url=managed_stub.base_url,
        comfy_output_dir=str(victim.parent), delete_outputs=True, write_sidecar=False)

    assert ok, msg
    assert victim.read_bytes() == b"the user's own picture"
    assert list(managed_stub.output_dir.glob("ComfyUI_*")) == []
    assert "WARNING" not in msg


def test_job_on_another_instance_keeps_a_same_named_file_and_warns(stub, tmp_path, monkeypatch):
    """The output folder names a different ComfyUI than the one that ran the job:
    the file there has the job's name and other content."""
    monkeypatch.setattr(comfy, "workflow_path", lambda: comfy._WORKFLOW_EXAMPLE_PATH)
    elsewhere = tmp_path / "elsewhere" / "output"
    elsewhere.mkdir(parents=True)
    victim = elsewhere / "ComfyUI_00001_.png"
    victim.write_bytes(b"the user's own picture")

    ok, msg = comfy.generate_image(
        "a cat", tmp_path / "saved" / "img.png", api_url=stub.base_url,
        comfy_output_dir=str(elsewhere), delete_outputs=True, write_sidecar=False)

    assert ok, msg
    assert victim.read_bytes() == b"the user's own picture"
    assert "WARNING" in msg and "different content" in msg
    assert list(stub.output_dir.glob("ComfyUI_*")), "the job's real copy was not found, so it stays"


def test_deletion_that_finds_nothing_warns(stub, tmp_path):
    empty = tmp_path / "empty-output"
    empty.mkdir()
    warn = comfy.contain_comfy_artifacts(
        stub.base_url, "pidMissing",
        {"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"},
        comfy_output_dir=str(empty), output_sha256="0" * 64, delete_outputs=True)
    assert "WARNING" in warn and "not in the folder localm checked" in warn


def test_deletion_without_a_checksum_keeps_the_file_and_warns(stub):
    fn = "ComfyUI_00050_.png"
    (stub.output_dir / fn).write_bytes(b"X")
    warn = comfy.contain_comfy_artifacts(
        stub.base_url, "pidNoSum",
        {"filename": fn, "subfolder": "", "type": "output"},
        comfy_output_dir=str(stub.output_dir), delete_outputs=True)
    assert (stub.output_dir / fn).exists()
    assert "WARNING" in warn and "checksum" in warn


def test_failed_unlink_warns(stub, monkeypatch):
    fn = "ComfyUI_00051_.png"
    (stub.output_dir / fn).write_bytes(b"X")

    def refuse(self, *a, **k):
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "unlink", refuse)
    warn = comfy.contain_comfy_artifacts(
        stub.base_url, "pidLocked",
        {"filename": fn, "subfolder": "", "type": "output"},
        comfy_output_dir=str(stub.output_dir), output_sha256=_X_SHA256, delete_outputs=True)
    assert (stub.output_dir / fn).exists()
    assert "could not delete" in warn


def test_plugin_workdir_locates_the_output_folder(stub, monkeypatch):
    """stub.output_dir is <workdir>/output; only the job's own workdir names it."""
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    fn = "ComfyUI_00052_.png"
    (stub.output_dir / fn).write_bytes(b"X")
    warn = comfy.contain_comfy_artifacts(
        stub.base_url, "pidWd",
        {"filename": fn, "subfolder": "", "type": "output"},
        comfy_workdir=str(stub.output_dir.parent),
        output_sha256=_X_SHA256, delete_outputs=True)
    assert warn == ""
    assert not (stub.output_dir / fn).exists()


def test_input_copy_cleanup_cannot_reach_a_users_same_named_image(stub, tmp_path, monkeypatch):
    """The user's own input folder holds photo.png, and the image they generate
    from has that name too."""
    monkeypatch.setattr(comfy, "workflow_path", lambda: comfy._WORKFLOW_EXAMPLE_PATH)
    users_photo = stub.input_dir / "photo.png"
    users_photo.write_bytes(b"the user's own photo")
    source = tmp_path / "photo.png"
    source.write_bytes(_minimal_png())
    out = tmp_path / "saved" / "img.png"

    ok, msg = comfy.generate_image(
        "a cat", out, api_url=stub.base_url, input_image=source, denoise=0.5,
        comfy_output_dir=str(stub.output_dir), delete_outputs=True, write_sidecar=False)

    assert ok, msg
    assert users_photo.read_bytes() == b"the user's own photo", "a user's input image was deleted"
    assert stub.uploaded_names and stub.uploaded_names[0].startswith("localm-input-")
    assert [p.name for p in stub.input_dir.iterdir()] == ["photo.png"], "localm's upload stayed"
    assert "WARNING" not in msg


def test_input_copy_warns_when_the_input_folder_cannot_be_located(stub, monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    warn = comfy_client.remove_uploaded_input(
        stub.base_url, f"localm-input-{'a' * 32}.png")
    assert "input image remains" in warn


# --------------------------------------------------------------------------- #
#  The uploaded source image, on every way out of a job                        #
# --------------------------------------------------------------------------- #

def _upload_name(data: bytes, ext: str = ".png") -> str:
    return f"localm-input-{hashlib.sha256(data).hexdigest()[:32]}{ext}"


def _input_files(stub) -> list:
    return sorted(p.name for p in stub.input_dir.iterdir())


def _generate(gen, stub, tmp_path, monkeypatch, **kwargs):
    """Run the real generator for *gen* on an image to image job against the stub."""
    source = tmp_path / "photo.png"
    source.write_bytes(_minimal_png())
    workdir = str(stub.output_dir.parent)
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    kwargs.setdefault("delete_outputs", True)
    if gen == "image":
        monkeypatch.setattr(comfy, "workflow_path", lambda: comfy._WORKFLOW_EXAMPLE_PATH)
        return comfy.generate_image(
            "a cat", tmp_path / "saved" / "img.png", api_url=stub.base_url,
            input_image=source, denoise=0.5, workdir=workdir, write_sidecar=False, **kwargs)
    monkeypatch.setattr(video_comfy, "workflow_path", lambda: video_comfy._WORKFLOW_PATH)
    stub.output_kind = "videos"
    stub.file_ext = ".mp4"
    return video_comfy.generate_video(
        "a fox", tmp_path / "saved" / "clip.mp4", api_url=stub.base_url, seconds=1.0,
        input_image=source, workdir=workdir, write_sidecar=False, **kwargs)


def _cancelled(stub):
    stub.job_state = "running"
    return {"cancel_check": lambda: True}


def _timed_out(stub):
    stub.job_state = "running"
    return {"max_poll_seconds": 0}


def _set(**attrs):
    def apply(stub):
        for key, value in attrs.items():
            setattr(stub, key, value)
        return {}
    return apply


_MISSING_VAE = {"VAELoader": {"input": {"required": {"vae_name": [["other.safetensors"], {}]}}}}

_FAILED_EXITS = {
    "cancel": (_cancelled, "cancelled"),
    "timeout": (_timed_out, "timed out"),
    "exec_error": (_set(job_state="exec_error"), "out of memory"),
    "preflight": (_set(object_info=_MISSING_VAE), "not installed"),
    "submit_http_error": (_set(submit_mode="http_error"), "HTTP 400"),
    "submit_no_prompt_id": (_set(submit_mode="no_id"), "no prompt_id"),
    "submit_dropped": (_set(submit_mode="drop"), "Error queuing"),
    "no_output": (_set(job_state="no_output"), "no "),
    "failed_fetch": (_set(view_fails=True), "Failed to download"),
}
_EXIT_IDS = list(_FAILED_EXITS)
_GENS = ["image", "video"]


@pytest.mark.parametrize("gen", _GENS)
@pytest.mark.parametrize("exit_path", _EXIT_IDS)
def test_a_failed_job_removes_its_input_copy(stub, tmp_path, monkeypatch, gen, exit_path):
    prepare, says = _FAILED_EXITS[exit_path]
    kwargs = prepare(stub)
    users_file = stub.input_dir / "family.png"
    users_file.write_bytes(b"the user's own photo")

    ok, msg = _generate(gen, stub, tmp_path, monkeypatch, **kwargs)

    assert len(stub.uploaded_names) == 1, "the job never uploaded, so this proves nothing"
    assert not ok
    assert _input_files(stub) == ["family.png"], "the input copy stayed"
    assert users_file.read_bytes() == b"the user's own photo"
    assert says in msg
    assert "WARNING" not in msg


@pytest.mark.parametrize("gen", _GENS)
@pytest.mark.parametrize("exit_path", _EXIT_IDS)
def test_a_failed_removal_is_in_the_failed_jobs_message(stub, tmp_path, monkeypatch, gen, exit_path):
    prepare, says = _FAILED_EXITS[exit_path]
    kwargs = prepare(stub)
    stub.store_other_bytes = True

    ok, msg = _generate(gen, stub, tmp_path, monkeypatch, **kwargs)

    assert not ok
    assert len(_input_files(stub)) == 1
    assert says in msg
    assert "WARNING:" in msg and "input image" in msg and "different content" in msg


@pytest.mark.parametrize("gen", _GENS)
@pytest.mark.parametrize("exit_path", _EXIT_IDS)
def test_without_delete_outputs_a_failed_job_keeps_its_input_copy(
        stub, tmp_path, monkeypatch, gen, exit_path):
    prepare, _says = _FAILED_EXITS[exit_path]
    kwargs = prepare(stub)

    ok, msg = _generate(gen, stub, tmp_path, monkeypatch, delete_outputs=False, **kwargs)

    assert not ok
    assert _input_files(stub) == stub.uploaded_names
    assert "WARNING" not in msg


@pytest.mark.parametrize("gen", _GENS)
def test_a_url_error_at_submit_removes_the_input_copy(stub, tmp_path, monkeypatch, gen):
    module = comfy if gen == "image" else video_comfy
    monkeypatch.setattr(module, "comfy_submit_prompt",
                        lambda *_a, **_k: (comfy.SUBMIT_URL_ERROR, URLError("refused")))
    ok, msg = _generate(gen, stub, tmp_path, monkeypatch)
    assert stub.uploaded_names
    assert not ok and "refused" in msg
    assert _input_files(stub) == []


@pytest.mark.parametrize("gen", _GENS)
def test_a_successful_job_removes_its_input_copy_once(stub, tmp_path, monkeypatch, gen):
    ok, msg = _generate(gen, stub, tmp_path, monkeypatch)
    assert ok, msg
    assert stub.uploaded_names
    assert _input_files(stub) == []
    assert "WARNING" not in msg, "the copy was removed, then reported missing"


@pytest.mark.parametrize("gen", _GENS)
def test_a_successful_job_names_an_input_copy_it_could_not_remove(stub, tmp_path, monkeypatch, gen):
    stub.store_other_bytes = True
    ok, msg = _generate(gen, stub, tmp_path, monkeypatch)
    assert ok, msg
    assert len(_input_files(stub)) == 1
    assert "WARNING:" in msg and "different content" in msg


@pytest.mark.parametrize("gen", _GENS)
@pytest.mark.parametrize("prepare", [_cancelled, _timed_out], ids=["cancel", "timeout"])
class TestAnUnfinishedJob:
    def test_still_listed_is_cancelled_again_then_the_copy_goes(
            self, stub, tmp_path, monkeypatch, gen, prepare):
        ok, msg = _generate(gen, stub, tmp_path, monkeypatch, **prepare(stub))
        assert not ok
        assert stub.queue_running, "the job left the queue, so this proves nothing"
        assert len(stub.cancel_calls) == 2
        assert _input_files(stub) == []
        assert "WARNING" not in msg

    def test_a_refused_cancel_keeps_the_copy_and_says_so(
            self, stub, tmp_path, monkeypatch, gen, prepare):
        kwargs = prepare(stub)
        stub.cancel_refused = True
        ok, msg = _generate(gen, stub, tmp_path, monkeypatch, **kwargs)
        assert not ok
        assert _input_files(stub) == stub.uploaded_names, "a running job lost its input"
        assert "WARNING: ComfyUI did not stop this job" in msg

    def test_an_unreadable_queue_counts_as_stopped(
            self, stub, tmp_path, monkeypatch, gen, prepare):
        kwargs = prepare(stub)
        stub.queue_unreadable = True
        stub.cancel_refused = True
        ok, msg = _generate(gen, stub, tmp_path, monkeypatch, **kwargs)
        assert not ok
        assert _input_files(stub) == []
        assert "WARNING" not in msg


@pytest.mark.parametrize("gen", _GENS)
class TestAnExceptionInsideTheJob:
    @staticmethod
    def _boom():
        raise OSError("boom")

    def test_the_copy_goes_and_the_exception_stays(self, stub, tmp_path, monkeypatch, gen):
        stub.job_state = "running"
        with pytest.raises(OSError, match="boom") as caught:
            _generate(gen, stub, tmp_path, monkeypatch, cancel_check=self._boom)
        assert _input_files(stub) == []
        assert stub.cancel_calls, "a job left running was not stopped"
        assert type(caught.value) is OSError

    def test_a_failed_removal_is_in_the_exception(self, stub, tmp_path, monkeypatch, gen):
        stub.job_state = "running"
        stub.store_other_bytes = True
        with pytest.raises(RuntimeError) as caught:
            _generate(gen, stub, tmp_path, monkeypatch, cancel_check=self._boom)
        assert "boom" in str(caught.value) and "different content" in str(caught.value)
        assert isinstance(caught.value.__cause__, OSError)

    def test_keyboard_interrupt_is_not_turned_into_an_error(self, stub, tmp_path, monkeypatch, gen):
        def interrupted():
            raise KeyboardInterrupt
        stub.job_state = "running"
        stub.store_other_bytes = True
        with pytest.raises(KeyboardInterrupt):
            _generate(gen, stub, tmp_path, monkeypatch, cancel_check=interrupted)


def test_a_failed_unlink_of_the_input_copy_is_named(stub, tmp_path, monkeypatch):
    real_unlink = Path.unlink

    def refuse(self, *a, **k):
        if self.name.startswith("localm-input-"):
            raise PermissionError("locked")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", refuse)
    stub.job_state = "exec_error"
    ok, msg = _generate("image", stub, tmp_path, monkeypatch)
    assert not ok
    assert len(_input_files(stub)) == 1
    assert "input image remains" in msg and "locked" in msg


def test_a_removal_that_raises_cannot_lose_the_jobs_message(stub, tmp_path, monkeypatch):
    def broken(*_a, **_k):
        raise TypeError("hostile queue reply")

    monkeypatch.setattr(comfy_client, "_comfy_job_stopped", broken)
    ok, msg = _generate("image", stub, tmp_path, monkeypatch, **_timed_out(stub))
    assert not ok
    assert "timed out" in msg
    assert "WARNING:" in msg and "hostile queue reply" in msg


def test_the_uploaded_name_reaches_the_load_image_node(stub, tmp_path, monkeypatch):
    ok, msg = _generate("image", stub, tmp_path, monkeypatch)
    assert ok, msg
    nodes = [n for n in stub.prompts[0].values() if n["class_type"] == "LoadImage"]
    assert [n["inputs"]["image"] for n in nodes] == stub.uploaded_names


def test_a_refused_image_workflow_uploads_nothing(stub, tmp_path, monkeypatch):
    """Every failure the image builder can return comes before its upload."""
    monkeypatch.setattr(comfy, "workflow_path", lambda: comfy._WORKFLOW_EXAMPLE_PATH)
    source = tmp_path / "photo.png"
    source.write_bytes(_minimal_png())
    ok, msg = comfy.generate_image(
        "a cat", tmp_path / "img.png", api_url=stub.base_url, input_image=source,
        lora_name="../bad", delete_outputs=True, write_sidecar=False)
    assert not ok and "Invalid LoRA name" in msg
    assert stub.uploaded_names == []


def test_a_workflow_the_image_builder_cannot_shape_uploads_nothing(stub, tmp_path, monkeypatch):
    workflow = tmp_path / "custom.json"
    workflow.write_text(json.dumps({"6": {"class_type": "CLIPTextEncode",
                                          "inputs": {"text": ""}}}), encoding="utf-8")
    monkeypatch.setattr(comfy, "workflow_path", lambda: workflow)
    source = tmp_path / "photo.png"
    source.write_bytes(_minimal_png())
    with pytest.raises(KeyError):
        comfy.generate_image(
            "a cat", tmp_path / "img.png", api_url=stub.base_url, input_image=source,
            delete_outputs=True, write_sidecar=False)
    assert stub.uploaded_names == []


# --------------------------------------------------------------------------- #
#  What a ComfyUI reply can make localm delete                                 #
# --------------------------------------------------------------------------- #

def test_a_hostile_upload_reply_cannot_delete_a_users_file(stub, tmp_path, monkeypatch):
    victim = stub.input_dir / "family.png"
    victim.write_bytes(b"the user's own photo")
    stub.upload_reply_name = "family.png"

    ok, msg = _generate("image", stub, tmp_path, monkeypatch)

    assert ok, msg
    assert victim.read_bytes() == b"the user's own photo", "the reply named a file and it went"
    assert "WARNING:" in msg and "'family.png'" in msg and "left alone" in msg


def test_a_reply_with_localms_form_cannot_delete_other_content(stub):
    name = f"localm-input-{'a' * 32}.png"
    other = stub.input_dir / name
    other.write_bytes(b"the user's own photo")
    warn = comfy_client.remove_uploaded_input(
        stub.base_url, name, comfy_output_dir=str(stub.output_dir))
    assert other.read_bytes() == b"the user's own photo"
    assert "different content" in warn


@pytest.mark.parametrize("name", [
    "family.png",
    "localm-input-abc.png",
    f"localm-input-{'a' * 32}family.png",
    f"sub/localm-input-{'a' * 32}.png",
    12345,
])
def test_a_reply_outside_the_form_localm_sends_is_refused(stub, name):
    target = stub.input_dir / str(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"data")
    warn = comfy_client.remove_uploaded_input(
        stub.base_url, name, comfy_output_dir=str(stub.output_dir))
    assert target.exists()
    assert "not a name localm sent" in warn


def test_comfyuis_clash_rename_removes_only_our_copy(stub, tmp_path, monkeypatch):
    """A file already holds the name localm sends, with other content: ComfyUI
    stores the upload as "<stem> (1)<ext>" and answers with that name."""
    occupant = stub.input_dir / _upload_name(_minimal_png())
    occupant.write_bytes(b"something else")

    ok, msg = _generate("image", stub, tmp_path, monkeypatch)

    assert ok, msg
    assert _input_files(stub) == [occupant.name], "ours stayed, or the occupant went"
    assert occupant.read_bytes() == b"something else"
    assert "WARNING" not in msg


def test_a_directory_with_the_uploaded_name_is_reported(stub):
    name = _upload_name(b"x")
    (stub.input_dir / name).mkdir()
    warn = comfy_client.remove_uploaded_input(
        stub.base_url, name, comfy_output_dir=str(stub.output_dir))
    assert (stub.input_dir / name).is_dir()
    assert "may remain" in warn
