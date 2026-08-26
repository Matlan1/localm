# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for read_env, tool-result compression, and HTTP 429 retry."""

from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder.tools import tool_read_env


class TestReadEnv:
    def test_env_file_secrets_redacted(self, tmp_path):
        (tmp_path / ".env").write_text(
            "API_KEY=supersecret\n"
            "export DB_PASSWORD=hunter2\n"
            "DEBUG=true\n"
            "# comment\n",
            encoding="utf-8",
        )
        result = tool_read_env(tmp_path, path=".env")
        assert result.ok
        assert "supersecret" not in result.output
        assert "hunter2" not in result.output
        assert "<redacted, 11 chars>" in result.output   # supersecret
        assert "DEBUG=true" in result.output

    def test_missing_explicit_file_errors(self, tmp_path):
        result = tool_read_env(tmp_path, path="nope.env")
        assert not result.ok

    def test_process_env_included_without_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_TEST_TOKEN", "abc123")
        monkeypatch.setenv("LOCALM_TEST_PLAIN", "visible")
        result = tool_read_env(tmp_path)
        assert result.ok
        assert "abc123" not in result.output       # TOKEN → redacted
        assert "visible" in result.output


class TestGgufEmbedUnsupported:
    def test_clear_not_implemented_when_binding_lacks_embeddings(self):
        # The real model lives in an isolated worker process (llamacpp/_runner.py),
        # not as a self._llm attribute here - embed() always raises (the ctypes
        # binding never implements create_embedding), gated only on "loaded".
        from localm.inference.backends.gguf import GgufBackend
        backend = GgufBackend.__new__(GgufBackend)
        backend._loaded = True
        try:
            backend.embed(["text"])
            assert False, "expected NotImplementedError"
        except NotImplementedError as e:
            assert "GGUF" in str(e)


class TestResultCompression:
    def _agent(self, fill):
        from localm.plugins.coder.agent import Agent
        agent = Agent.__new__(Agent)
        agent._fill_ratio = lambda: fill
        return agent

    def test_untouched_below_threshold(self):
        agent = self._agent(0.3)
        blocks = ["x" * 20_000]
        assert agent._compress_results(blocks) == blocks

    def test_compressed_above_threshold(self):
        agent = self._agent(0.7)
        big = "a" * 20_000
        out = agent._compress_results([big])[0]
        assert len(out) < len(big)
        assert "elided" in out
        assert out.startswith("a" * 100)
        assert out.endswith("a" * 100)

    def test_small_blocks_never_touched(self):
        agent = self._agent(0.9)
        blocks = ["short result", "y" * 5000]
        assert agent._compress_results(blocks) == blocks


class TestRetryOn429:
    def test_retries_then_succeeds(self):
        from localm.plugins.coder.backends.http import _post_with_retry

        limited = MagicMock(status_code=429, headers={"Retry-After": "0"})
        ok = MagicMock(status_code=200, headers={})
        with patch("localm.plugins.coder.backends.http.requests.post",
                   side_effect=[limited, ok]) as post, \
             patch("localm.plugins.coder.backends.http.time.sleep") as sleep:
            resp = _post_with_retry("http://x/v1/chat/completions",
                                    headers={}, json_body={}, timeout=5)
        assert resp is ok
        assert post.call_count == 2
        sleep.assert_called_once()

    def test_gives_up_after_max_retries(self):
        from localm.plugins.coder.backends.http import (
            _MAX_RETRIES, _post_with_retry)

        limited = MagicMock(status_code=429, headers={})
        with patch("localm.plugins.coder.backends.http.requests.post",
                   return_value=limited) as post, \
             patch("localm.plugins.coder.backends.http.time.sleep"):
            resp = _post_with_retry("http://x", headers={}, json_body={}, timeout=5)
        assert resp.status_code == 429
        assert post.call_count == _MAX_RETRIES + 1

    def test_no_retry_on_success(self):
        from localm.plugins.coder.backends.http import _post_with_retry
        ok = MagicMock(status_code=200, headers={})
        with patch("localm.plugins.coder.backends.http.requests.post",
                   return_value=ok) as post:
            _post_with_retry("http://x", headers={}, json_body={}, timeout=5)
        assert post.call_count == 1

    def test_respects_retry_after_header(self):
        from localm.plugins.coder.backends.http import _retry_delay
        resp = MagicMock(headers={"Retry-After": "7"})
        assert _retry_delay(resp, 0) == 7.0


class TestRetryOn503:
    """A LOCAL server's 503 is never retried. Every 503 reachable from this
    backend's own call path (/v1/chat/completions, /v1/completions, both
    resolving their engine via switch_engine(..., preempt=False)) is either a
    deterministic fault (grammar-check/embedding worker crash) or has already
    exhausted its own resolution window server-side (wait_for_vram_release's 5s
    wait) - see the comment above _RETRY_STATUSES in http.py. A non-local
    (cloud/other OpenAI-compatible) endpoint keeps the full budget."""

    def test_local_503_is_never_retried(self):
        from localm.plugins.coder.backends.http import _post_with_retry

        faulted = MagicMock(status_code=503, headers={})
        with patch("localm.plugins.coder.backends.http.requests.post",
                   return_value=faulted) as post, \
             patch("localm.plugins.coder.backends.http.time.sleep") as sleep:
            resp = _post_with_retry("http://x", headers={}, json_body={},
                                    timeout=5, retry_503=False)
        assert resp.status_code == 503
        # Exactly ONE request, zero retries - not 429's _MAX_RETRIES + 1 = 5.
        # A deterministic worker-fault 503 is not retried like a transient
        # cloud rate limit.
        assert post.call_count == 1
        sleep.assert_not_called()

    def test_non_local_503_keeps_the_full_retry_budget(self):
        """retry_503 defaults to True (unset), so a cloud or other
        OpenAI-compatible endpoint's 503 keeps the full retry budget."""
        from localm.plugins.coder.backends.http import (
            _MAX_RETRIES, _post_with_retry)

        errored = MagicMock(status_code=503, headers={})
        with patch("localm.plugins.coder.backends.http.requests.post",
                   return_value=errored) as post, \
             patch("localm.plugins.coder.backends.http.time.sleep"):
            resp = _post_with_retry("http://x", headers={}, json_body={}, timeout=5)
        assert resp.status_code == 503
        assert post.call_count == _MAX_RETRIES + 1

    def test_non_local_503_succeeding_on_retry_is_unaffected(self):
        from localm.plugins.coder.backends.http import _post_with_retry

        faulted = MagicMock(status_code=503, headers={"Retry-After": "0"})
        ok = MagicMock(status_code=200, headers={})
        with patch("localm.plugins.coder.backends.http.requests.post",
                   side_effect=[faulted, ok]) as post, \
             patch("localm.plugins.coder.backends.http.time.sleep"):
            resp = _post_with_retry("http://x", headers={}, json_body={}, timeout=5)
        assert resp is ok
        assert post.call_count == 2

    def test_other_5xx_statuses_keep_the_full_retry_budget_even_with_retry_503_false(self):
        """retry_503=False gates 503 specifically: 429/500/502/529 keep the full
        budget, even on a localm_server=True backend."""
        from localm.plugins.coder.backends.http import (
            _MAX_RETRIES, _post_with_retry)

        for status in (429, 500, 502, 529):
            errored = MagicMock(status_code=status, headers={})
            with patch("localm.plugins.coder.backends.http.requests.post",
                       return_value=errored) as post, \
                 patch("localm.plugins.coder.backends.http.time.sleep"):
                resp = _post_with_retry("http://x", headers={}, json_body={},
                                        timeout=5, retry_503=False)
            assert resp.status_code == status
            assert post.call_count == _MAX_RETRIES + 1, status


class TestHTTPBackendLocalServer503Wiring:
    """HTTPBackend passes retry_503 through, keyed on localm_server; a unit test
    on _post_with_retry alone would not reach the real caller's wiring."""

    def test_localm_server_backend_fails_fast_on_persistent_503(self):
        from localm.plugins.coder.backends.http import (
            CoderServerError, HTTPBackend)

        faulted = MagicMock(status_code=503, headers={},
                            json=MagicMock(return_value={"detail": "worker faulted"}))
        backend = HTTPBackend("http://127.0.0.1:8642/v1", "test-model",
                              localm_server=True, verify=False)
        with patch("localm.plugins.coder.backends.http.requests.post",
                   return_value=faulted) as post, \
             patch("localm.plugins.coder.backends.http.time.sleep") as sleep:
            with pytest.raises(CoderServerError):
                backend.chat([{"role": "user", "content": "hi"}])
        assert post.call_count == 1
        sleep.assert_not_called()

    def test_non_localm_backend_retries_persistent_503(self):
        from localm.plugins.coder.backends.http import (
            _MAX_RETRIES, CoderServerError, HTTPBackend)

        faulted = MagicMock(status_code=503, headers={},
                            json=MagicMock(return_value={"detail": "overloaded"}))
        backend = HTTPBackend("https://api.example.com/v1", "test-model",
                              verify=False)   # localm_server defaults False
        with patch("localm.plugins.coder.backends.http.requests.post",
                   return_value=faulted) as post, \
             patch("localm.plugins.coder.backends.http.time.sleep"):
            with pytest.raises(CoderServerError):
                backend.chat([{"role": "user", "content": "hi"}])
        assert post.call_count == _MAX_RETRIES + 1
