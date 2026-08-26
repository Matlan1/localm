# SPDX-License-Identifier: AGPL-3.0-or-later
"""HttpEngine.context_capacity must not permanently latch a None from an early
failure.

Setting _ctx_capacity_cached = True BEFORE the fetch caches whatever comes back,
so a transient or early failure (the model is not loaded yet, a non-200, a
network blip) latches None forever and every later call returns None even once
/v1/config would answer. The cache is populated after the first SUCCESSFUL
fetch, not after the first attempt.
"""

from unittest.mock import MagicMock

from localm.inference.http_engine import HttpEngine


def _resp(status, payload):
    m = MagicMock()
    m.status_code = status
    m.json = lambda: payload
    return m


def test_context_capacity_retries_after_early_failure(monkeypatch):
    eng = HttpEngine("http://x/v1", token="t", model="m")

    calls = {"n": 0}

    def flaky_get(url, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            # First call: model not loaded yet -> a non-200 (must NOT latch None).
            return _resp(503, {"detail": "no model"})
        return _resp(200, {"effective_ctx_max": 8192})

    monkeypatch.setattr("requests.get", flaky_get)

    assert eng.context_capacity() is None          # early failure, not cached
    assert eng.context_capacity() == 8192          # retried and resolved
    # ...and now it IS cached (no third network call).
    assert eng.context_capacity() == 8192
    assert calls["n"] == 2


def test_context_capacity_caches_success(monkeypatch):
    """Guard: a successful resolution is still cached (one fetch only)."""
    eng = HttpEngine("http://x/v1", token="t", model="m")
    calls = {"n": 0}

    def ok_get(url, headers=None, timeout=None):
        calls["n"] += 1
        return _resp(200, {"effective_ctx_max": 4096})

    monkeypatch.setattr("requests.get", ok_get)
    assert eng.context_capacity() == 4096
    assert eng.context_capacity() == 4096
    assert calls["n"] == 1
