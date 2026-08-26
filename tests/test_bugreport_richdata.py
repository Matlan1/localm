# SPDX-License-Identifier: AGPL-3.0-or-later
"""A bug report must carry ACTUALLY useful data - app state (loaded model,
backend, session mode, plugins), an allowlisted config subset, dependency
versions, the in-memory recent-activity log, and (for the GUI) browser context -
while NEVER leaking the API key, config secrets, or chat content.

These pin both halves: the rich sections are present and correct, and the
privacy boundary holds (DEBUG-level model output and non-allowlisted config keys
stay out).
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
    # The raw, pre-scrub model output logged at DEBUG never enters the buffer,
    # even though the buffer is always on.
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


def test_config_subset_redacts_query_string_credential_by_name(monkeypatch):
    """comfy_api_url / net_search_url / coder_reviewer are user-supplied URLs
    that routinely carry a credential as a query parameter, not only via
    user:pass@. The config subset chain does not call _scrub_secrets (it has its
    own narrower chain), so this is a genuinely separate path from the
    _scrub_secrets tests and must be verified independently."""
    fake = {
        "net_search_url": "https://searx.example.com/search?api_key=CANARY1",
        "comfy_api_url": "http://qauser:CANARY2@127.0.0.1:8188",
        "coder_reviewer": "https://review.example.com/v1?token=CANARY3",
    }
    monkeypatch.setattr("localm.config.load_config", lambda: fake)
    sub = bugreport._safe_config_subset()
    assert "CANARY1" not in sub["net_search_url"]
    assert "api_key=<redacted>" in sub["net_search_url"]
    assert "CANARY2" not in sub["comfy_api_url"]
    assert "CANARY3" not in sub["coder_reviewer"]
    assert "token=<redacted>" in sub["coder_reviewer"]


def test_build_report_does_not_leak_non_allowlisted_secret(monkeypatch):
    monkeypatch.setattr(
        "localm.config.load_config",
        lambda: {"api_key": "TOP-SECRET-XYZ", "port": 8642})
    text = bugreport.build_report("x")
    assert "TOP-SECRET-XYZ" not in text


def test_build_report_end_to_end_no_canary_survives_with_credentialed_config_urls(
        monkeypatch):
    """The report-level regression test for the QA finding: a unit test on the
    scrub regex alone cannot prove the regex is actually REACHED from a real
    report. Render a full report with all three affected config keys set to
    credentialed URLs and assert no canary substring survives ANYWHERE in the
    rendered output - not just in the isolated config-subset dict."""
    fake = {
        "net_search_url": "https://searx.example.com/search?api_key=QACANARYQUERYKEY77d1e5c2",
        "comfy_api_url": "http://qauser:QACANARYURLCRED31bf90aa@127.0.0.1:8188",
        "coder_reviewer": "https://review.example.com/v1?token=QACANARYREVIEWTOK5a9c33e1",
        "port": 8642,
    }
    monkeypatch.setattr("localm.config.load_config", lambda: fake)
    text = bugreport.build_report("testing config url redaction")
    assert "## Configuration (safe subset)" in text
    assert "comfy_api_url" in text and "net_search_url" in text
    assert "QACANARYQUERYKEY77d1e5c2" not in text
    assert "QACANARYURLCRED31bf90aa" not in text
    assert "QACANARYREVIEWTOK5a9c33e1" not in text


def test_corrupt_config_flagged_unreadable_not_silently_defaulted(tmp_path, monkeypatch):
    """A corrupt config.json must not render identically to a genuinely absent
    one - see bugreport._config_unreadable / config.load_config_checked.

    Uses REAL files on disk rather than monkeypatching load_config, unlike the
    tests above: a lambda can never be "unreadable", so that fixture shape
    cannot express this case at all.
    """
    import localm.config as cfg
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")

    # arm A: a real, valid config on disk.
    cfg.CONFIG_FILE.write_text('{"n_ctx": 31337}', encoding="utf-8")
    report_present = bugreport.build_report("x")
    # The injection took: CONFIG_FILE points at the file just written.
    assert "31337" in report_present

    # arm B: overwrite the SAME path with corrupt JSON.
    cfg.CONFIG_FILE.write_text("{ not json ", encoding="utf-8")
    assert cfg.CONFIG_FILE.is_file() and cfg.CONFIG_FILE.stat().st_size > 0
    report_corrupt = bugreport.build_report("x")

    # arm C: no config.json at all.
    cfg.CONFIG_FILE.unlink()
    report_absent = bugreport.build_report("x")

    # The discriminator: only the corrupt arm may say the file could not be read.
    marker = "config.json exists but could not be read"
    assert marker in report_corrupt
    assert marker not in report_absent

    # The two Configuration sections are not byte-identical: corrupt is
    # distinguishable from never configured.
    def _config_section(text):
        start = text.index("## Configuration (safe subset)")
        end = text.find("\n\n## ", start)
        return text[start:] if end == -1 else text[start:end]

    assert _config_section(report_corrupt) != _config_section(report_absent)


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
