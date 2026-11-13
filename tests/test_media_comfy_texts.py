# SPDX-License-Identifier: AGPL-3.0-or-later
"""What localm tells a person about their ComfyUI: a ComfyUI address that
redirects is named as such (not "not reachable"), a launch timeout reads in the
unit it was set in and names the Settings field, the address guard's warning
names no config key, ``ComfyUI to use: user`` alone does not make ``auto``
choose ComfyUI, and the update refusal for a tree without a source record does
not blame a copy fallback.

The redirecting ComfyUI is a real HTTP server on loopback."""

from __future__ import annotations

import http.server
import threading
from unittest.mock import patch

import pytest

from localm.media import backend_choice, comfy_client


@pytest.fixture(autouse=True)
def _isolated_comfy_handles(monkeypatch):
    """A fresh ComfyUI handle registry per test, so a stub launch this file
    leaves "running" is never seen as a running ComfyUI by another test."""
    monkeypatch.setattr(comfy_client, "_spawned_procs", {})
    monkeypatch.setattr(comfy_client, "_device_errors", {})


class _Redirects(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(302)
        self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def redirecting_comfy():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Redirects)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_a_redirecting_address_is_named_as_a_redirect(redirecting_comfy, tmp_path):
    from localm.config import home_dir
    home_dir().mkdir(parents=True, exist_ok=True)
    comfy_client.mark_comfy_dead(redirecting_comfy)
    with patch("localm.config.load_config",
               return_value={"comfy_workdir": None, "comfy_launch_cmd": None}):
        ok, msg = comfy_client.ensure_comfy(redirecting_comfy)
    assert ok is False
    assert "not reachable" not in msg
    assert "redirect" in msg and "localm refuses" in msg
    assert redirecting_comfy in msg


def test_the_redirect_is_remembered_only_until_the_address_answers(redirecting_comfy):
    assert comfy_client._comfy_alive(redirecting_comfy) is False
    assert "redirect" in comfy_client._redirect_detail(redirecting_comfy)
    with patch.object(comfy_client, "_comfy_urlopen", side_effect=OSError("refused")):
        assert comfy_client._comfy_alive(redirecting_comfy) is False
    assert comfy_client._redirect_detail(redirecting_comfy) is None


@pytest.mark.parametrize("seconds,text", [(30, "30 seconds"), (119, "119 seconds"),
                                          (120, "2 minutes"), (600, "10 minutes")])
def test_the_launch_wait_reads_in_seconds_or_minutes(seconds, text):
    assert comfy_client.launch_wait_text(seconds) == text


def test_the_launch_timeout_names_the_settings_field_and_an_empty_log(monkeypatch, tmp_path):
    import time as time_mod
    from unittest.mock import MagicMock

    from localm.config import home_dir
    from tests._popen_scope import ignore_background_threads
    home_dir().mkdir(parents=True, exist_ok=True)
    launcher = tmp_path / ("comfyui.bat" if __import__("sys").platform == "win32"
                           else "comfyui.sh")
    launcher.write_text("echo\n", encoding="utf-8")
    cfg = {"comfy_launch_cmd": None, "comfy_workdir": str(tmp_path),
           "comfy_launch_timeout": 30}

    def fake_popen(argv, cwd=None, stdout=None, **kw):
        proc = MagicMock()
        proc.poll.return_value = None
        return proc
    ticks = {"n": 0}

    def fake_monotonic():
        ticks["n"] += 1
        return 0.0 if ticks["n"] == 1 else 10_000.0
    monkeypatch.setattr(time_mod, "monotonic", fake_monotonic)
    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    with patch("localm.config.load_config", return_value=cfg), \
         patch.object(comfy_client, "_comfy_alive", side_effect=lambda *a, **k: False), \
         patch("subprocess.Popen", ignore_background_threads(fake_popen)):
        ok, msg = comfy_client.ensure_comfy("http://127.0.0.1:8188")
    assert ok is False
    assert "within 30 seconds" in msg and "0 minutes" not in msg
    assert "'ComfyUI launch timeout (s)' in Settings > Media (or run" in msg
    assert "launcher window" not in msg
    assert "wrote nothing to its log" in msg


def test_the_func_shim_advice_names_the_settings_field():
    msg = comfy_client.comfy_exec_error_message(
        "'function' object has no attribute '__func__'", "http://127.0.0.1:8188")
    assert "'Fix ComfyUI ACE-Step __func__ crash (in-memory)' in Settings > Media (or run" in msg


@pytest.mark.parametrize("url", ["http://169.254.169.254:8188", "http://[::1"])
def test_the_address_guard_warning_names_no_config_key(url):
    sanitized, warning = comfy_client.sanitize_comfy_url_checked(url)
    assert sanitized == "http://127.0.0.1:8188"
    assert warning.startswith("The ComfyUI address ")
    assert "comfy_api_url" not in warning


def _no_comfy_seen(monkeypatch):
    monkeypatch.setattr(comfy_client, "is_comfy_confirmed", lambda url: False)
    monkeypatch.delenv("FLUX_API_URL", raising=False)


def test_comfy_target_user_alone_keeps_auto_on_native(monkeypatch):
    _no_comfy_seen(monkeypatch)
    cfg = {"comfy_target": "user"}
    backend, note = backend_choice.resolve("auto", cfg, {}, "http://127.0.0.1:8188", "Image")
    assert backend == "native"
    assert "no ComfyUI is set up" in note


@pytest.mark.parametrize("cfg,blk", [
    ({"comfy_target": "user", "comfy_api_url": "http://127.0.0.1:8190"}, {}),
    ({"comfy_target": "user"}, {"workdir": "D:/path/to/ComfyUI"}),
    ({"comfy_launch_cmd": "run.bat"}, {}),
])
def test_a_configured_comfyui_still_makes_auto_choose_it(monkeypatch, cfg, blk):
    _no_comfy_seen(monkeypatch)
    backend, _note = backend_choice.resolve("auto", cfg, blk, "http://127.0.0.1:8188", "Image")
    assert backend == "comfy"


def test_comfy_target_user_with_a_comfyui_that_answered_chooses_it(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    monkeypatch.setattr(comfy_client, "is_comfy_confirmed", lambda url: True)
    backend, _note = backend_choice.resolve("auto", {"comfy_target": "user"}, {},
                                            "http://127.0.0.1:8188", "Image")
    assert backend == "comfy"


def test_the_pin_update_refusal_does_not_blame_a_copy_fallback(monkeypatch):
    from localm.media import managed_comfy_update as mcu
    monkeypatch.setattr(mcu.mc, "is_managed_comfy_installed", lambda: True)
    monkeypatch.setattr(mcu, "_rev_parse_head", lambda root: None)
    result = mcu.update_managed_comfy()
    assert result.ok is False
    assert "no recorded source commit" in result.message
    assert "fallback" not in result.message
