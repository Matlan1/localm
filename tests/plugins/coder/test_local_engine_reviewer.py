# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU-resident reviewer: a different small model loaded in-process on CPU, for a
heterogeneous review that is still fully local + private (allowed in privacy mode).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from localm.audit import SessionMode
from localm.plugins.coder.reviewer import reviewer_for_agent


# --------------------------------------------------------------------------- #
#  LocalEngineBackend adapter                                                  #
# --------------------------------------------------------------------------- #

def _patched_engine(monkeypatch, stream_tokens):
    """Patch the inference Engine so no real model loads; return the fake engine."""
    fake_engine = MagicMock()
    fake_engine.chat_stream.return_value = iter(stream_tokens)
    monkeypatch.setattr("localm.inference.engine.Engine",
                        lambda *a, **k: fake_engine)
    monkeypatch.setattr("localm.inference.engine.model_display_name",
                        lambda p: "reviewer-mini")
    return fake_engine


def test_local_backend_chat_joins_stream(monkeypatch):
    from localm.plugins.coder.backends.local_engine import LocalEngineBackend
    eng = _patched_engine(monkeypatch, ["Looks ", "good", "."])
    b = LocalEngineBackend("/models/mini.gguf")
    assert b.chat([{"role": "user", "content": "review"}]) == "Looks good."
    assert b.model_id == "reviewer-mini"
    eng.load.assert_called_once()                  # lazy-loaded once


def test_local_backend_filters_unsupported_gen_kwargs(monkeypatch):
    from localm.plugins.coder.backends.local_engine import LocalEngineBackend
    eng = _patched_engine(monkeypatch, ["ok"])
    b = LocalEngineBackend("/models/mini.gguf")
    b.chat([{"role": "user", "content": "x"}],
           max_tokens=600, temperature=0.2, bogus="drop", none_val=None)
    _, kwargs = eng.chat_stream.call_args
    assert kwargs == {"max_tokens": 600, "temperature": 0.2}   # bogus/None dropped


def test_local_backend_loads_only_once(monkeypatch):
    from localm.plugins.coder.backends.local_engine import LocalEngineBackend
    eng = _patched_engine(monkeypatch, ["a"])
    b = LocalEngineBackend("/models/mini.gguf")
    b.chat([{"role": "user", "content": "1"}])
    eng.chat_stream.return_value = iter(["b"])
    b.chat([{"role": "user", "content": "2"}])
    eng.load.assert_called_once()                  # not reloaded on the 2nd call


def test_local_backend_supports_grammar_mirrors_the_engine(monkeypatch):
    """LocalEngineBackend must not compute its own guess at grammar support
    from the backend's TYPE. It defers entirely to Engine.supports_grammar
    (engine.py), the install-derived answer from the actual backend:
    GgufBackend always True, HFBackend True only when xgrammar is importable.
    Both booleans are exercised, so a hardcoded-False regression shows up as a
    mismatch on the True case."""
    from localm.plugins.coder.backends.local_engine import LocalEngineBackend

    for expected in (True, False):
        fake_engine = _patched_engine(monkeypatch, [])
        fake_engine.supports_grammar = expected
        b = LocalEngineBackend("/models/mini.gguf")
        assert b.supports_grammar is expected


# --------------------------------------------------------------------------- #
#  Factory: the "local" reviewer kind                                          #
# --------------------------------------------------------------------------- #

def _cfg(monkeypatch, **over):
    base = {"coder_review": True, "coder_reviewer": "local",
            "coder_reviewer_model": "reviewer-mini"}
    base.update(over)
    monkeypatch.setattr("localm.config.load_config", lambda: base)


def _agent_backend():
    b = MagicMock()
    b.model_id = "main-model"
    return b


def test_local_reviewer_built_heterogeneous(monkeypatch):
    _cfg(monkeypatch)
    # **kw absorbs the allow_direct_path argument reviewer.py passes.
    monkeypatch.setattr("localm.model_manager.get_model_path",
                        lambda n, **kw: __import__("pathlib").Path("/models/mini.gguf"))
    fake = MagicMock()
    with patch("localm.plugins.coder.backends.local_engine.LocalEngineBackend",
               return_value=fake) as mk:
        rv = reviewer_for_agent(_agent_backend(), SessionMode.FULL, False)
    assert rv.heterogeneous is True and rv.backend is fake
    mk.assert_called_once()


def test_local_reviewer_allowed_in_privacy_mode(monkeypatch):
    _cfg(monkeypatch)
    # **kw absorbs the allow_direct_path argument reviewer.py passes.
    monkeypatch.setattr("localm.model_manager.get_model_path",
                        lambda n, **kw: __import__("pathlib").Path("/models/mini.gguf"))
    fake = MagicMock()
    with patch("localm.plugins.coder.backends.local_engine.LocalEngineBackend",
               return_value=fake):
        rv = reviewer_for_agent(_agent_backend(), SessionMode.PRIVACY, False)
    assert rv.heterogeneous is True and rv.backend is fake


def test_local_reviewer_missing_model_falls_back(monkeypatch):
    _cfg(monkeypatch)
    monkeypatch.setattr("localm.model_manager.get_model_path", lambda n: None)
    backend = _agent_backend()
    with patch("localm.plugins.coder.display.print_warning") as warn:
        rv = reviewer_for_agent(backend, SessionMode.FULL, False)
    assert rv.backend is backend and rv.heterogeneous is False   # fell back to same-model
    assert warn.called


def test_local_reviewer_no_model_name_falls_back(monkeypatch):
    _cfg(monkeypatch, coder_reviewer_model="")
    backend = _agent_backend()
    with patch("localm.plugins.coder.display.print_warning") as warn:
        rv = reviewer_for_agent(backend, SessionMode.FULL, False)
    assert rv.backend is backend and rv.heterogeneous is False
    assert warn.called


def test_local_reviewer_still_gated_off_for_restricted(monkeypatch):
    _cfg(monkeypatch)
    assert reviewer_for_agent(_agent_backend(), SessionMode.FULL, True) is None
