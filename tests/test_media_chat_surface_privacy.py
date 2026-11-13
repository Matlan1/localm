# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chat-surface privacy setting (``chat_mode = privacy`` under ``mode = log``)
reaches media generated from the GUI chat and the bug report of a failed
``localm run``.

The GUI chat's media commands send ``X-Client-Id: chat``; the request-privacy
middleware runs such a request inside ``audit.request_privacy``, and every
media job carries that into its worker thread."""

import io
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def chat_private(monkeypatch):
    from localm.config import update_config
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    monkeypatch.delenv("LOCALM_KEEP_DIAGNOSTICS", raising=False)
    update_config(lambda c: c.update(mode="log", chat_mode="privacy",
                                     keep_diagnostics=False))


def test_media_session_is_private_follows_request_privacy(chat_private):
    from localm.audit import request_privacy
    from localm.plugins.media_config import media_session_is_private
    assert media_session_is_private() is False
    with request_privacy(True):
        assert media_session_is_private() is True


def test_media_session_is_private_in_server_privacy_mode(monkeypatch):
    from localm.plugins.media_config import media_session_is_private
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    assert media_session_is_private() is True


def test_diagnostics_are_off_inside_a_private_request(chat_private):
    from localm.audit import diagnostics_allowed, request_privacy
    assert diagnostics_allowed() is True
    with request_privacy(True):
        assert diagnostics_allowed() is False


def test_keep_diagnostics_still_wins_inside_a_private_request(chat_private, monkeypatch):
    from localm.audit import diagnostics_allowed, request_privacy
    monkeypatch.setenv("LOCALM_KEEP_DIAGNOSTICS", "1")
    with request_privacy(True):
        assert diagnostics_allowed() is True


def _png_writer(calls):
    def fake_generate_image(prompt, out, **kwargs):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "PNG")
        Path(out).write_bytes(buf.getvalue())
        calls.append(kwargs)
        return True, f"Image saved to {out}"
    return fake_generate_image


@pytest.fixture
def server(monkeypatch, tmp_path):
    from localm.inference.http_server import create_app
    from localm.plugins.engine import PluginManager
    monkeypatch.setattr("localm.plugins.lookup.is_available", lambda name: True)
    monkeypatch.setattr("localm.media.backend_choice.comfy_is_set_up", lambda *a, **k: True)
    monkeypatch.setattr("localm.image_gen.comfy.ensure_comfy", lambda *a, **k: (True, "up"))
    monkeypatch.setattr("localm.image_gen.comfy.free_comfy_vram", lambda *a, **k: True)
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: False)
    monkeypatch.setattr("localm.vram.media_single_device_shortfall", lambda s: None)
    engine = MagicMock()
    engine.display_name = "test-model"
    app = create_app(engine)
    PluginManager(app, external_root=tmp_path / "noplugins").install("image")
    app.state.self_url = "http://127.0.0.1:8642/v1"
    with TestClient(app) as client:
        client.headers["Authorization"] = f"Bearer {app.state.shell_token}"
        yield app, client


def _imagine(app, client, headers):
    calls: list = []
    with patch("localm.image_gen.comfy.generate_image", _png_writer(calls)):
        r = client.post("/api/imagine", json={"prompt": "a private cat"}, headers=headers)
        assert r.status_code == 200, r.text
        job = app.state.jobs.get(r.json()["job_id"])
        deadline = time.monotonic() + 30
        while job.status not in ("done", "failed", "cancelled") and time.monotonic() < deadline:
            time.sleep(0.05)
    assert job.status == "done", job.status
    assert len(calls) == 1
    return calls[0]


def test_gui_chat_imagine_follows_the_chat_privacy_setting(chat_private, server):
    app, client = server
    kwargs = _imagine(app, client, {"X-Client-Id": "chat"})
    assert kwargs["write_sidecar"] is False
    assert kwargs["delete_outputs"] is True


def test_imagine_without_the_chat_header_follows_the_server_mode(chat_private, server):
    app, client = server
    kwargs = _imagine(app, client, {})
    assert kwargs["write_sidecar"] is True


def test_every_media_job_carries_the_request_privacy():
    """Each ``jobs.start_fn`` in a media plugin route runs its job body through
    ``carry_request_privacy``, so the job thread sees the request's mode."""
    import ast
    root = Path(__file__).resolve().parent.parent / "localm" / "plugins" / "builtin"
    found = []
    for kind in ("image", "video", "music"):
        path = root / kind / "plug.py"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "start_fn"):
                body = node.args[1]
                carried = (isinstance(body, ast.Call) and isinstance(body.func, ast.Name)
                           and body.func.id == "carry_request_privacy")
                found.append((f"{kind}/plug.py:{node.lineno}", carried))
    assert len(found) >= 5
    assert [where for where, carried in found if not carried] == []


def _report_failure(command):
    from localm.cli._core import _GracefulGroup
    ctx = MagicMock()
    ctx.invoked_subcommand = command
    with patch("localm.bugreport.stdin_is_interactive", return_value=False):
        _GracefulGroup._report_failure(ctx, RuntimeError("boom"))


def _saved_reports():
    from localm.config import home_dir
    return list((home_dir() / "bug-reports").glob("bug-*.md"))


def test_a_failed_run_saves_no_report_when_chat_is_private(chat_private):
    _report_failure("run")
    assert _saved_reports() == []


def test_another_failed_command_still_saves_its_report(chat_private):
    _report_failure("pull")
    assert len(_saved_reports()) == 1
