# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat CONTENT must never reach the debug log in privacy mode, even when the
debug log is on (via --debug or the keep_diagnostics toggle). debug_content_enabled()
is the gate the GGUF backend's "raw model output" line and the jobs web-tool arg
line both use; operational lines stay on the looser debug_enabled()."""

from localm import debuglog


def _debug(monkeypatch, on):
    if on:
        monkeypatch.setenv("LOCALM_DEBUG", "1")
    else:
        monkeypatch.delenv("LOCALM_DEBUG", raising=False)


def _mode(monkeypatch, **cfg):
    monkeypatch.delenv("LOCALM_MODE", raising=False)   # config decides
    monkeypatch.setattr("localm.config.load_config", lambda: cfg)


def test_no_content_when_debug_off(monkeypatch):
    _debug(monkeypatch, False)
    _mode(monkeypatch, mode="full")
    assert debuglog.debug_content_enabled() is False   # log off -> no content


def test_no_content_in_privacy_even_with_debug_on(monkeypatch):
    _debug(monkeypatch, True)
    _mode(monkeypatch, mode="privacy")
    assert debuglog.debug_content_enabled() is False


def test_content_allowed_in_log_mode(monkeypatch):
    _debug(monkeypatch, True)
    _mode(monkeypatch, mode="log", chat_mode=None)
    assert debuglog.debug_content_enabled() is True


def test_content_allowed_in_full_mode(monkeypatch):
    _debug(monkeypatch, True)
    _mode(monkeypatch, mode="full", chat_mode=None)
    assert debuglog.debug_content_enabled() is True


def test_suppressed_when_a_per_surface_mode_is_privacy(monkeypatch):
    # Global log but the chat surface is privacy -> content suppressed (the backend
    # is surface-agnostic, so ANY relevant surface being privacy wins).
    _debug(monkeypatch, True)
    _mode(monkeypatch, mode="log", chat_mode="privacy")
    assert debuglog.debug_content_enabled() is False


def test_suppressed_when_the_coder_surface_is_privacy(monkeypatch):
    # Global log, chat/server unset (so they fall back to the global log), coder
    # alone set to privacy: the coder-only override suppresses the raw model
    # output line the shared llama.cpp backend writes for a coder session.
    _debug(monkeypatch, True)
    _mode(monkeypatch, mode="log", coder_mode="privacy")
    assert debuglog.debug_content_enabled() is False


def test_content_allowed_when_no_surface_including_coder_is_privacy(monkeypatch):
    # An explicit non-privacy coder_mode, with every other surface also
    # non-privacy, still allows content.
    _debug(monkeypatch, True)
    _mode(monkeypatch, mode="log", coder_mode="full")
    assert debuglog.debug_content_enabled() is True


def test_toml_pinned_coder_privacy_is_seen_by_debug_content_gate(tmp_path, monkeypatch):
    # Regression: debug_content_enabled() calls effective_mode("coder") with NO
    # cwd, so a project's own .localcoder/config.toml privacy pin (visible only
    # when cwd is passed) was invisible to it - a coder session running under a
    # toml-pinned privacy override still leaked its raw model output to the
    # debug log whenever the GLOBAL coder_mode was not itself privacy. A coder
    # session now publishes its own cwd-resolved mode into a small registry
    # this gate also checks.
    import localm.audit as _audit
    monkeypatch.setattr(_audit, "_active_coder_privacy_count", 0)   # isolate

    _debug(monkeypatch, True)
    _mode(monkeypatch, mode="log")   # global default: NOT privacy
    proj = tmp_path / "proj"
    (proj / ".localcoder").mkdir(parents=True)
    (proj / ".localcoder" / "config.toml").write_text('mode = "privacy"\n', encoding="utf-8")

    resolved = _audit.effective_mode("coder", cwd=proj)
    assert resolved == _audit.SessionMode.PRIVACY   # sanity: the toml pin itself resolves

    from unittest.mock import MagicMock, patch
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    with patch("localm.plugins.coder.agent.make_audit_log") as mock_factory, \
         patch("localm.plugins.coder.agent.load_memory", return_value=""), \
         patch("localm.plugins.coder.agent.ProjectMap") as mock_pm:
        mock_pm.build.return_value.file_count.return_value = 0
        mock_factory.return_value = _audit.NullAuditLog()
        agent = Agent(backend=backend, cwd=proj, mode=resolved)

    try:
        # The gate must see privacy for this session, even though the global
        # coder_mode ("log") alone would have allowed content.
        assert debuglog.debug_content_enabled() is False
    finally:
        agent.close()

    # Closing the session releases the registry entry.
    assert debuglog.debug_content_enabled() is True


def test_child_agent_does_not_double_register(tmp_path, monkeypatch):
    # A child (sub-agent) inherits the parent's mode rather than resolving its
    # own, so only the top-level agent should touch the registry - otherwise a
    # parent that outlives a closed child would under-count on the child's
    # close() (or a child that outlives the parent would over-count).
    import localm.audit as _audit
    monkeypatch.setattr(_audit, "_active_coder_privacy_count", 0)

    _debug(monkeypatch, True)
    _mode(monkeypatch, mode="log")

    from unittest.mock import MagicMock, patch
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    with patch("localm.plugins.coder.agent.make_audit_log") as mock_factory, \
         patch("localm.plugins.coder.agent.load_memory", return_value=""), \
         patch("localm.plugins.coder.agent.ProjectMap") as mock_pm:
        mock_pm.build.return_value.file_count.return_value = 0
        mock_factory.return_value = _audit.NullAuditLog()
        parent = Agent(backend=backend, cwd=tmp_path, mode=_audit.SessionMode.PRIVACY)
        child = Agent(backend=backend, cwd=tmp_path, mode=_audit.SessionMode.PRIVACY,
                      parent=parent)

    assert _audit._active_coder_privacy_count == 1
    child.close()
    assert _audit._active_coder_privacy_count == 1   # unaffected: the child never counted
    parent.close()
    assert _audit._active_coder_privacy_count == 0


def test_fails_safe_to_no_content_on_error(monkeypatch):
    _debug(monkeypatch, True)
    monkeypatch.delenv("LOCALM_MODE", raising=False)

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("localm.config.load_config", _boom)
    assert debuglog.debug_content_enabled() is False   # fail-safe: no content
