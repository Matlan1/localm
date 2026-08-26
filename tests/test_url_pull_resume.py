# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for resumable direct-URL model downloads (_pull_url): the Range/206
resume logic and the server-ignores-Range fallback, which must NOT append to a
stale .part file.
"""

from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm


def _resp(status, body: bytes, content_length=None):
    """A fake requests streaming response."""
    r = MagicMock()
    r.status_code = status
    r.raise_for_status = MagicMock()
    cl = len(body) if content_length is None else content_length
    r.headers = {"content-length": str(cl)}

    def _iter(chunk_size):
        for i in range(0, len(body), chunk_size):
            yield body[i:i + chunk_size]
    r.iter_content = _iter
    return r


@pytest.fixture()
def url_env(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])
    reg_spy = MagicMock()
    monkeypatch.setattr(mm, "_register", reg_spy)
    monkeypatch.setattr(mm, "_register_with_dedup", MagicMock())
    # check_url resolves the host; pin it to a public IP so the SSRF guard passes
    # hermetically (no real DNS) and the pinned-transport seam is what the tests
    # double below.
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    return models, reg_spy


def _wire_http(monkeypatch, head_total: int, response):
    """Double netpolicy.pinned_request (the pinned transport seam the pull path
    uses); return a dict that captures the GET headers."""
    captured = {}

    def fake_pinned_request(method, url, **kwargs):
        if method == "HEAD":
            h = MagicMock()
            h.status_code = 200                 # not a redirect (SSRF resolver reads this)
            h.headers = {"content-length": str(head_total)}
            return h
        captured["headers"] = dict(kwargs.get("headers") or {})
        return response

    monkeypatch.setattr("localm.netpolicy.pinned_request", fake_pinned_request)
    return captured


class TestUrlPull:
    def test_fresh_download_writes_and_registers(self, url_env, monkeypatch):
        models, reg_spy = url_env
        cap = _wire_http(monkeypatch, 10, _resp(200, b"0123456789"))
        mm._pull_url("http://example.com/model.gguf", "mymodel")
        dest = models / "model.gguf"
        assert dest.read_bytes() == b"0123456789"
        assert not (models / "model.gguf.part").exists()   # renamed on success
        assert "Range" not in cap["headers"]               # nothing to resume
        reg_spy.assert_called_once()

    def test_gui_mode_streams_json_progress(self, url_env, monkeypatch, capsys):
        # In GUI mode (LOCALM_PROGRESS_JSON=1) a direct-URL pull streams the same
        # PROGRESS_SENTINEL JSON lines the GUI parses, not just a Rich bar it
        # cannot render.
        import json
        _, _ = url_env
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        _wire_http(monkeypatch, 10, _resp(200, b"0123456789"))
        mm._pull_url("http://example.com/model.gguf", "mymodel")
        out = capsys.readouterr().out
        lines = [l for l in out.splitlines() if mm.PROGRESS_SENTINEL in l]
        assert lines, "GUI mode must stream progress sentinels for a direct-URL pull"
        payloads = [json.loads(l.split(mm.PROGRESS_SENTINEL, 1)[1]) for l in lines]
        assert any(p.get("total") == 10 for p in payloads)   # known size reported
        assert payloads[-1]["downloaded"] == 10              # finishes at 100%

    def test_resume_appends_from_part_file(self, url_env, monkeypatch):
        models, _ = url_env
        (models / "model.gguf.part").write_bytes(b"01234")   # 5 bytes already have
        cap = _wire_http(monkeypatch, 10,
                         _resp(206, b"56789", content_length=5))
        mm._pull_url("http://example.com/model.gguf", "mymodel")
        dest = models / "model.gguf"
        assert dest.read_bytes() == b"0123456789"            # suffix appended
        assert cap["headers"].get("Range") == "bytes=5-"

    def test_server_ignoring_range_restarts_clean(self, url_env, monkeypatch):
        models, _ = url_env
        (models / "model.gguf.part").write_bytes(b"STALE")   # partial/garbage
        # 200 (full file) despite our Range request -> must overwrite, not append
        cap = _wire_http(monkeypatch, 11,
                         _resp(200, b"FULLCONTENT", content_length=11))
        mm._pull_url("http://example.com/model.gguf", "mymodel")
        dest = models / "model.gguf"
        assert dest.read_bytes() == b"FULLCONTENT"           # NOT b"STALEFULL..."
        assert cap["headers"].get("Range") == "bytes=5-"     # we did request resume

    def test_already_downloaded_skips_network(self, url_env, monkeypatch):
        models, _ = url_env
        (models / "model.gguf").write_bytes(b"already here")
        pin_spy = MagicMock()
        monkeypatch.setattr("localm.netpolicy.pinned_request", pin_spy)
        mm._pull_url("http://example.com/model.gguf", "mymodel")
        pin_spy.assert_not_called()


class TestUrlPullResult:
    """The bool return drives the CLI exit code and the GUI job status."""

    def test_success_returns_true(self, url_env, monkeypatch):
        _wire_http(monkeypatch, 10, _resp(200, b"0123456789"))
        assert mm._pull_url("http://example.com/model.gguf", "m") is True

    def test_already_downloaded_returns_true(self, url_env):
        models, _ = url_env
        (models / "model.gguf").write_bytes(b"already here")
        assert mm._pull_url("http://example.com/model.gguf", "m") is True

    def test_empty_stem_url_returns_false(self, url_env, monkeypatch):
        """A URL whose path has no file name is rejected before any network I/O."""
        pin_spy = MagicMock()
        monkeypatch.setattr("localm.netpolicy.pinned_request", pin_spy)
        assert mm._pull_url("https://huggingface.co/asdasd/", "m") is False
        pin_spy.assert_not_called()

    def test_http_error_returns_false(self, url_env, monkeypatch, capsys):
        """A 404/bad URL yields a clear message and False, not a traceback."""
        import requests

        def boom_pinned(method, url, **kwargs):
            if method == "HEAD":
                return MagicMock(status_code=200, headers={"content-length": "0"})
            resp = MagicMock()
            resp.status_code = 404
            err = requests.HTTPError("404 Not Found")
            err.response = resp
            raise err

        monkeypatch.setattr("localm.netpolicy.pinned_request", boom_pinned)
        assert mm._pull_url("http://example.com/model.gguf", "m") is False
        assert "download failed" in capsys.readouterr().out.lower()

    def test_size_head_policy_refusal_fails_closed(self, url_env, monkeypatch, capsys):
        """A NetworkPolicyError at the size-HEAD stage must SURFACE and fail
        closed, not be collapsed into total=0: a policy refusal is a different
        thing from a benign connect error."""
        from localm.netpolicy import NetworkPolicyError
        calls = {"head": 0}

        def fake_pinned(method, url, **kw):
            if method == "HEAD":
                calls["head"] += 1
                if calls["head"] == 1:                 # redirect-resolver hop: ok
                    return MagicMock(status_code=200,
                                     headers={"content-length": "10"})
                raise NetworkPolicyError("rebind blocked at the size HEAD")
            raise AssertionError("GET must not run after a refused size HEAD")

        monkeypatch.setattr("localm.netpolicy.pinned_request", fake_pinned)
        assert mm._pull_url("http://example.com/model.gguf", "m") is False
        assert "network policy" in capsys.readouterr().out.lower()


class TestPullModelDispatch:
    def test_unknown_spec_returns_false(self, monkeypatch, capsys):
        monkeypatch.setattr(mm, "resolve_spec", lambda s: s)
        assert mm.pull_model("garbage-no-slash") is False
        assert "unknown spec" in capsys.readouterr().out.lower()
