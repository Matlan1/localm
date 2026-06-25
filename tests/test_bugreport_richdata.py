# SPDX-License-Identifier: AGPL-3.0-or-later
"""A bug report must carry ACTUALLY useful data - app state (loaded model,
backend, session mode, plugins), an allowlisted config subset, dependency
versions, the in-memory recent-activity log, and (for the GUI) browser context -
while NEVER leaking the API key, config secrets, or chat content.

This pins the "reports still contain no useful data" fix: the rich sections are
present and correct, and the privacy boundary holds (DEBUG-level model output and
non-allowlisted config keys stay out).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localm import bugreport, debuglog


# --------------------- in-memory recent-activity buffer ------------------- #

@pytest.fixture
def fresh_ring():
    """Install a clean ring buffer on the localm logger for one test, then
    restore the logger's prior handlers/level/global so tests don't bleed."""
    logger = debuglog.logger
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_ring = debuglog._ring_handler
    for h in list(logger.handlers):
        if isinstance(h, debuglog._RingBufferHandler):
            logger.removeHandler(h)
    debuglog._ring_handler = None
    assert debuglog.install_ring_buffer() is True
    yield logger
    logger.handlers[:] = saved_handlers
    logger.setLevel(saved_level)
    debuglog._ring_handler = saved_ring


def test_ring_buffer_captures_info_but_never_debug(fresh_ring):
    logger = fresh_ring
    logger.info("model load: gemma-3 on vulkan")
    logger.warning("VRAM low, falling back to CPU clip")
    # The raw, pre-scrub model output is logged at DEBUG (llama.py) - it must
    # NEVER enter the buffer, even though the buffer is always on.
    logger.debug("raw model output:\nthis is private chat content")
    joined = "\n".join(debuglog.recent_activity())
    assert "model load: gemma-3 on vulkan" in joined
    assert "VRAM low" in joined
    assert "private chat content" not in joined


def test_install_ring_buffer_is_idempotent(fresh_ring):
    # The fixture already installed it; a second call is a no-op.
    assert debuglog.install_ring_buffer() is False


def test_recent_activity_empty_when_uninstalled(monkeypatch):
    monkeypatch.setattr(debuglog, "_ring_handler", None)
    assert debuglog.recent_activity() == []


# ----------------------------- app state ---------------------------------- #

def test_build_report_has_app_state_and_dependency_sections():
    text = bugreport.build_report("x", context={"operation": "chat"})
    assert "## App state" in text
    assert "## Dependencies" in text
    # fastapi is always installed (the server stack); its version is reported.
    assert "fastapi:" in text


def test_build_report_includes_loaded_model(monkeypatch):
    import localm.inference.http_server as hs

    class _Eng:
        display_name = "gemma-3-4b-it"
        loaded = True
        effective_ctx_max = 8192
        _backend = object()

    monkeypatch.setattr(hs, "_engine", _Eng(), raising=False)
    text = bugreport.build_report("x")
    assert "gemma-3-4b-it" in text
    assert "loaded" in text.lower()
    assert "ctx<=8192" in text


def test_build_report_no_engine_is_graceful(monkeypatch):
    import localm.inference.http_server as hs
    monkeypatch.setattr(hs, "_engine", None, raising=False)
    text = bugreport.build_report("x")
    assert "no engine loaded in this process" in text


def test_build_report_includes_recent_activity(fresh_ring):
    fresh_ring.info("backend selected: vulkan")
    text = bugreport.build_report("x")
    assert "## Recent activity (in-memory log)" in text
    assert "backend selected: vulkan" in text


# --------------------- config subset: allowlist + scrub ------------------- #

def test_config_subset_allowlisted_and_scrubbed(monkeypatch):
    home = str(Path.home())
    fake = {
        "net_search_url": "https://user:pass@searx.example/search",
        "binary_dir": home + "/llama/bin",
        "port": 8642,
        "mode": "privacy",
        # Non-allowlisted, secret-looking keys MUST NOT be echoed.
        "api_key": "SHOULD-NOT-LEAK",
        "hf_token": "hf_NOPE",
    }
    monkeypatch.setattr("localm.config.load_config", lambda: fake)
    sub = bugreport._safe_config_subset()
    assert sub["port"] == 8642 and sub["mode"] == "privacy"
    # URL credentials redacted.
    assert "user:pass" not in sub["net_search_url"]
    assert "<redacted>" in sub["net_search_url"]
    # Home path scrubbed (no username segment), structure kept.
    assert home not in sub["binary_dir"] and "llama" in sub["binary_dir"]
    # Secrets never present.
    assert "api_key" not in sub and "hf_token" not in sub


def test_build_report_does_not_leak_non_allowlisted_secret(monkeypatch):
    monkeypatch.setattr(
        "localm.config.load_config",
        lambda: {"api_key": "TOP-SECRET-XYZ", "port": 8642})
    text = bugreport.build_report("x")
    assert "TOP-SECRET-XYZ" not in text


# ----------------------- client / browser sanitizer ----------------------- #

def test_sanitize_client_context_caps_and_filters():
    from localm.inference.http_server import _sanitize_client_context
    out = _sanitize_client_context({
        "userAgent": "u" * 2000,
        "page": "#settings",
        "console": (["err", 7, {"bad": "obj"}, "err2"] * 20),
        "unknownField": "drop me",
    })
    assert len(out["userAgent"]) <= 500
    assert out["page"] == "#settings"
    assert "unknownField" not in out
    assert len(out["console"]) <= 40
    assert all(isinstance(x, str) for x in out["console"])  # dicts dropped, ints coerced
    assert _sanitize_client_context("not-a-dict") == {}
    assert _sanitize_client_context(None) == {}


def test_client_context_rendered_in_report():
    text = bugreport.build_report("x", context={"client": {
        "userAgent": "Mozilla/5.0 TestBrowser",
        "page": "#chat",
        "console": ["TypeError: foo is not a function"],
    }})
    assert "## Browser / client" in text
    assert "TestBrowser" in text
    assert "TypeError: foo is not a function" in text
