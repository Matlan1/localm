# SPDX-License-Identifier: AGPL-3.0-or-later
"""The loaded model's resolved context ceiling is reported by
Engine.context_capacity() and consumed by its callers, so compaction budgets
against the real window rather than the static config or the initial
4096-token one.
"""

from __future__ import annotations

import types

import pytest

from localm.inference.engine import Engine


def _engine_with_backend(backend) -> Engine:
    e = Engine.__new__(Engine)
    e._backend = backend
    return e


def test_capacity_prefers_effective_ctx_max():
    b = types.SimpleNamespace(effective_ctx_max=65536, n_ctx_max=16384, n_ctx=4096)
    assert _engine_with_backend(b).context_capacity() == 65536


def test_capacity_falls_back_to_configured_ceiling():
    b = types.SimpleNamespace(effective_ctx_max=None, n_ctx_max=16384, n_ctx=4096)
    assert _engine_with_backend(b).context_capacity() == 16384


def test_capacity_falls_back_to_base_window():
    b = types.SimpleNamespace(effective_ctx_max=None, n_ctx_max=0, n_ctx=8192)
    assert _engine_with_backend(b).context_capacity() == 8192


def test_capacity_none_when_nothing_resolvable():
    b = types.SimpleNamespace()          # bare backend, no attrs, no _llm
    assert _engine_with_backend(b).context_capacity() is None


def test_capacity_uses_low_level_llm_when_present():
    b = types.SimpleNamespace(
        effective_ctx_max=None, n_ctx_max=0, n_ctx=0,
        _llm=types.SimpleNamespace(n_ctx=lambda: 32768))
    assert _engine_with_backend(b).context_capacity() == 32768


# ---------------------------------------------------- coder fill measurement #

def test_coder_ctx_window_prefers_backend_capacity():
    from localm.plugins.coder.agent.context import _ContextMixin

    class _Agent(_ContextMixin):
        def __init__(self, backend):
            self.backend = backend

    # A backend reporting a large resolved window: the coder budgets against it,
    # not the 4096 default.
    backend = types.SimpleNamespace(context_capacity=lambda: 65536)
    assert _Agent(backend)._ctx_window_tokens() == 65536


def test_coder_ctx_window_falls_back_without_capacity(monkeypatch):
    from localm.plugins.coder.agent.context import _ContextMixin

    class _Agent(_ContextMixin):
        def __init__(self, backend):
            self.backend = backend

    monkeypatch.setattr("localm.config.load_config", lambda: {"n_ctx": 4096})
    # Backend with no context_capacity (e.g. a third-party OpenAI server).
    backend = types.SimpleNamespace()
    assert _Agent(backend)._ctx_window_tokens() == 4096


# ---------------------------------------------------- coder HTTP backend fetch #

def test_http_backend_context_capacity_reads_effective_ctx_max(monkeypatch):
    from localm.plugins.coder.backends import http as http_mod

    calls = {"n": 0}

    class _Resp:
        ok = True

        def json(self):
            return {"effective_ctx_max": 32768, "n_ctx": 4096}

    def fake_get(url, **kw):
        calls["n"] += 1
        assert url.endswith("/config")
        return _Resp()

    monkeypatch.setattr(http_mod.requests, "get", fake_get)
    be = http_mod.HTTPBackend("http://127.0.0.1:8642/v1", model="m")
    assert be.context_capacity() == 32768
    # Cached: a second call does not re-fetch.
    assert be.context_capacity() == 32768
    assert calls["n"] == 1


def test_http_backend_context_capacity_none_on_error(monkeypatch):
    from localm.plugins.coder.backends import http as http_mod

    def boom(url, **kw):
        raise ConnectionError("no server")

    monkeypatch.setattr(http_mod.requests, "get", boom)
    be = http_mod.HTTPBackend("http://127.0.0.1:8642/v1", model="m")
    assert be.context_capacity() is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
