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
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from localm.image_gen import comfy
from localm.media.comfy_client import _comfy_output_root
from localm.music_gen import comfy as music_comfy

# A non-routable RFC5737 (TEST-NET-1) address, so nothing here can dial a real
# host from this machine or CI.
_UNC = r"\\192.0.2.1\share"
_UNC_FWD = "//192.0.2.1/share"
_DEVICE = r"\\.\PhysicalDrive0"


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

        warn = comfy.contain_comfy_artifacts(
            stub.base_url, "pidEsc2",
            {"filename": "ok.png", "subfolder": "", "type": "output"},
            comfy_output_dir=str(stub.output_dir),
            uploaded_input=uploaded,
            delete_outputs=True,
        )

        assert victim.exists(), f"containment escaped via uploaded_input: {uploaded}"
        assert victim.read_text(encoding="utf-8") == "do not delete me"
        assert "WARNING" in warn, "a refused name must be surfaced, not skipped"

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
    unload_spy.assert_called_once_with("http://127.0.0.1:9/v1", "tok-abc")


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
