# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat and coder document attachments are extracted by the always-on chat
plugin (``POST /api/chat/extract``), so they work without the Knowledge (rag)
plugin installed."""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.gui.web import attach_gui


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.fixture
def chat_only_app(tmp_path, monkeypatch):
    """A GUI app with the chat plugin and NOT the rag plugin."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")

    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("chat")

    async def switch_model(name):
        pass
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app, home


def _extract(client, name, blob, **body):
    payload = {"filename": name, "content_b64": _b64(blob)}
    payload.update(body)
    return client.post("/api/chat/extract", json=payload)


def test_text_attachment_extracts_without_the_rag_plugin(chat_only_app):
    app, _ = chat_only_app
    paths = {getattr(r, "path", None) for r in app.router.routes}
    assert "/api/rag/extract" not in paths
    with TestClient(app) as client:
        r = _extract(client, "notes.txt", b"plain text attachment")
    assert r.status_code == 200
    assert r.json() == {"filename": "notes.txt", "text": "plain text attachment",
                        "chars": 21, "truncated": False}


def test_attachment_is_the_whole_file_unless_an_excerpt_is_asked(chat_only_app):
    app, _ = chat_only_app
    blob = b"x" * 5000
    with TestClient(app) as client:
        whole = _extract(client, "big.txt", blob).json()
        part = _extract(client, "big.txt", blob, max_chars=1000).json()
    assert (whole["chars"], len(whole["text"]), whole["truncated"]) == (5000, 5000, False)
    assert (part["chars"], len(part["text"]), part["truncated"]) == (5000, 1000, True)


def test_image_attachment_is_refused_with_the_chat_pointer(chat_only_app):
    app, _ = chat_only_app
    with TestClient(app) as client:
        r = _extract(client, "photo.png", b"\x89PNG\r\n\x1a\n")
    assert r.status_code == 422
    assert r.json()["detail"] == ("No extractable text in photo.png. Images are not "
                                  "converted to text here; attach them in Chat with "
                                  "a vision model.")


@pytest.mark.parametrize("name,content,status", [
    ("x.txt", "!!not-b64!!", 400),
    ("x.exe", _b64(b"\x00\x01"), 422),
])
def test_bad_attachments_are_refused(chat_only_app, name, content, status):
    app, _ = chat_only_app
    with TestClient(app) as client:
        r = client.post("/api/chat/extract",
                        json={"filename": name, "content_b64": content})
    assert r.status_code == status


def test_oversized_attachment_is_refused_before_decoding(chat_only_app, monkeypatch):
    from localm.rag import attachment
    app, _ = chat_only_app
    decoded = []
    monkeypatch.setattr(attachment, "MAX_B64_CHARS", 8)
    monkeypatch.setattr(attachment.base64, "b64decode",
                        lambda *a, **k: decoded.append(a) or b"")
    with TestClient(app) as client:
        r = _extract(client, "big.txt", b"0123456789")
    assert r.status_code == 413
    assert decoded == []


def test_attachment_extraction_writes_nothing_to_disk(chat_only_app):
    app, home = chat_only_app
    before = {str(p) for p in home.rglob("*")} if home.exists() else set()
    with TestClient(app) as client:
        assert _extract(client, "secret.txt", b"secret content").status_code == 200
    after = {str(p) for p in home.rglob("*")} if home.exists() else set()
    assert after == before


def test_chat_extract_offloads_extraction_off_the_event_loop(monkeypatch):
    from localm.plugins.builtin.chat.plug import AttachmentExtractRequest, chat_extract
    from localm.rag import attachment

    def _slow_extract(data, filename):
        time.sleep(0.3)
        return "extracted text"

    monkeypatch.setattr(attachment._extract, "extract_bytes", _slow_extract)
    ticks: list[float] = []

    async def ticker(t0: float, stop_at: float):
        while time.monotonic() - t0 < stop_at:
            ticks.append(time.monotonic() - t0)
            await asyncio.sleep(0.02)

    async def scenario():
        req = AttachmentExtractRequest(filename="x.txt", content_b64=_b64(b"hello"))
        t0 = time.monotonic()
        t_extract = asyncio.create_task(chat_extract(req))
        t_tick = asyncio.create_task(ticker(t0, 0.4))
        result = await t_extract
        await t_tick
        return result

    result = asyncio.run(scenario())
    assert result["text"] == "extracted text"
    assert len([t for t in ticks if t < 0.25]) >= 3, ticks
