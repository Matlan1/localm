# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for read_env, tool-result compression, and HTTP 429 retry."""

from unittest.mock import MagicMock, patch

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
        # The real model now lives in an isolated worker process (see
        # llamacpp/_runner.py), not as a self._llm attribute here - embed()
        # always raises regardless (the ctypes binding never implements
        # create_embedding), gated only on "loaded", not on _llm.
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
