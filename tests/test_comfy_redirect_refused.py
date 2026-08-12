# SPDX-License-Identifier: AGPL-3.0-or-later
"""CHK-COMFY-REDIRECT (LM-DA-045): sanitize_comfy_url (CHK-COMFY-APIURL) only
screens the CONFIGURED comfy_api_url, at resolution time. A redirect target is
chosen by the remote ComfyUI AFTER that check has already run, and urllib's
default opener follows up to 10 such redirects with no validation at all - so a
hostile or compromised ComfyUI (SECURITY.md: it "may be another machine, over
plain http") could answer any request with a 3xx straight past the guard, e.g.
to a cloud-metadata address. ComfyUI's HTTP API has no legitimate reason to
redirect, so the connection itself must refuse every hop outright.

These tests stand up a real (loopback) HTTP stub that answers every endpoint
the client calls with a 302 to a link-local/metadata-shaped address, and drive
the real comfy_client functions over real sockets. The redirect target is
never actually dialed by a correct fix - ComfyRedirectRefused fires inside
redirect_request, before urllib attempts the follow-up connection - so this
suite cannot reach out even if the fix under test regresses; a mistaken revert
would instead try (and fail/hang) to reach 169.254.169.254, which is exactly
the failure this guards against.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from localm.media import comfy_client as c

_METADATA_TARGET = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
_PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8   # looks_like_image only checks the head


class _RedirectStub(HTTPServer):
    """Every endpoint answers 302 to a link-local/metadata address."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _RedirectHandler)
        self.hit_paths: list = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _RedirectHandler(BaseHTTPRequestHandler):
    def log_message(self, *_a):  # keep test output quiet
        pass

    def _redirect(self):
        self.server.hit_paths.append(self.path)
        self.send_response(302)
        self.send_header("Location", _METADATA_TARGET)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802 (stdlib naming)
        self._redirect()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self._redirect()


@pytest.fixture
def stub():
    srv = _RedirectStub()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def test_object_info_refuses_redirect_and_reports_unfetchable(stub):
    # Documented best-effort: a refused redirect must read the same as "could
    # not be fetched", never raise past this function's boundary.
    assert c.comfy_object_info(stub.base_url, timeout=2) is None
    assert stub.hit_paths == ["/object_info"]   # the redirect target was NEVER dialed


def test_comfy_alive_refuses_redirect_and_reports_down(stub):
    assert c._comfy_alive(stub.base_url, timeout=2) is False
    assert stub.hit_paths == ["/system_stats"]


def test_free_comfy_vram_refuses_redirect(stub):
    assert c.free_comfy_vram(stub.base_url) is False
    assert stub.hit_paths == ["/free"]


def test_interrupt_comfy_refuses_redirect(stub):
    assert c.interrupt_comfy(stub.base_url) is False
    assert stub.hit_paths == ["/interrupt", "/queue"]


def test_clear_comfy_history_refuses_redirect(stub):
    assert c.clear_comfy_history(stub.base_url, "some-prompt-id") is False
    assert stub.hit_paths == ["/history"]


def test_submit_prompt_refuses_redirect(stub):
    kind, value = c.comfy_submit_prompt(stub.base_url, {"1": {}}, timeout=2)
    assert kind == c.SUBMIT_ERROR
    assert isinstance(value, c.ComfyRedirectRefused)
    assert stub.hit_paths == ["/prompt"]


def test_fetch_output_refuses_redirect(stub, tmp_path):
    with pytest.raises(c.ComfyRedirectRefused):
        c.comfy_fetch_output(
            stub.base_url, {"filename": "x.png"}, tmp_path / "out.png", timeout=2)
    assert stub.hit_paths == ["/view?filename=x.png&subfolder=&type=output"]
    assert not (tmp_path / "out.png").exists()


def test_upload_image_refuses_redirect(stub, tmp_path):
    img = tmp_path / "in.png"
    img.write_bytes(_PNG_HEADER)
    with pytest.raises(c.ComfyRedirectRefused):
        c._upload_image(img, stub.base_url)
    assert stub.hit_paths == ["/upload/image"]


def test_poll_until_done_refuses_redirect_and_times_out(stub):
    # comfy_poll_until_done retries within its loop on ANY exception (ComfyUI
    # may just be busy) rather than propagating, so the observable contract
    # here is POLL_TIMEOUT with the refusal as the last error - never a
    # follow of the redirect, and never a false POLL_FINISHED.
    status, value = c.comfy_poll_until_done(
        stub.base_url, "pid-1", max_poll_seconds=0.3, history_timeout=1,
        sleep_seconds=0.05)
    assert status == c.POLL_TIMEOUT
    assert isinstance(value, c.ComfyRedirectRefused)
    assert set(stub.hit_paths) == {"/history/pid-1"}
