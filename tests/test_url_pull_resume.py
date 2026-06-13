"""Tests for resumable direct-URL model downloads (_pull_url).

Only the fresh path was exercised before; the Range/206 resume logic and the
server-ignores-Range fallback (which must NOT append to a stale .part file) had
no coverage despite being a correctness- and data-integrity-sensitive path.
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
    return models, reg_spy


def _wire_http(monkeypatch, head_total: int, response):
    """Patch requests.head/get; return a dict that captures the GET headers."""
    captured = {}

    def fake_head(url, allow_redirects=None, timeout=None):
        h = MagicMock()
        h.headers = {"content-length": str(head_total)}
        return h

    def fake_get(url, headers=None, stream=None, timeout=None):
        captured["headers"] = dict(headers or {})
        return response

    monkeypatch.setattr("requests.head", fake_head)
    monkeypatch.setattr("requests.get", fake_get)
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
        get_spy = MagicMock()
        monkeypatch.setattr("requests.get", get_spy)
        monkeypatch.setattr("requests.head", MagicMock())
        mm._pull_url("http://example.com/model.gguf", "mymodel")
        get_spy.assert_not_called()
