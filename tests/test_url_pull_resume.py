# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for resumable direct-URL model downloads (_pull_url): the Range/206
resume logic and the server-ignores-Range fallback, which must NOT append to a
stale .part file, and the record beside a .part that binds it to the URL it
came from.
"""

import hashlib
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


URL_A = "http://host1.example/model.gguf"
URL_B = "http://host2.example/model.gguf"
_A = b"A" * 10
_B = b"B" * 10


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


class _RangeServer:
    """A fake ``pinned_request`` serving *body*: the size HEAD reports its
    length (nothing when *head_length* is False), a ``Range: bytes=N-`` GET gets
    the tail from byte N as a 206 (a 416 past the end), any other GET the whole
    body. With *drop_after*, a GET's body stops with a connection error after
    that many bytes. ``gets`` records every GET's headers."""

    def __init__(self, body: bytes, head_length: bool = True, drop_after=None):
        self.body = body
        self.head_length = head_length
        self.drop_after = drop_after
        self.gets: list = []

    def __call__(self, method, url, **kw):
        if method == "HEAD":
            h = MagicMock()
            h.status_code = 200
            h.headers = ({"content-length": str(len(self.body))}
                         if self.head_length else {})
            return h
        headers = dict(kw.get("headers") or {})
        self.gets.append(headers)
        rng = headers.get("Range")
        if rng is None:
            r = _resp(200, self.body)
        else:
            start = int(rng.removeprefix("bytes=").removesuffix("-"))
            if start >= len(self.body):
                return _resp(416, b"")
            r = _resp(206, self.body[start:])
        if self.drop_after is not None:
            r.iter_content = _dropped_after(r, self.drop_after)
        return r


def _dropped_after(resp, keep: int):
    """An ``iter_content`` that yields *resp*'s first *keep* body bytes, then
    fails as a dropped connection does."""
    import requests
    whole = b"".join(resp.iter_content(1 << 20))

    def _iter(chunk_size):
        yield whole[:keep]
        raise requests.ConnectionError("connection reset by peer")
    return _iter


def _interrupt_pull(monkeypatch, models, url, head_total, delivered, **kw):
    """Run a pull of *url* whose transfer drops after *delivered* bytes, and
    return the partial it leaves behind. *head_total* is the size the size HEAD
    reports."""
    import requests

    def _dropped(chunk_size):
        yield delivered
        raise requests.ConnectionError("connection reset by peer")

    resp = _resp(200, b"", content_length=head_total)
    resp.iter_content = _dropped
    _wire_http(monkeypatch, head_total, resp)
    try:
        mm._pull_url(url, "mymodel", **kw)
    except requests.ConnectionError:
        pass
    part = models / "model.gguf.part"
    assert part.read_bytes() == delivered, "the interrupted pull left no partial"
    return part


class TestUrlPull:
    def test_fresh_download_writes_and_registers(self, url_env, monkeypatch):
        models, reg_spy = url_env
        cap = _wire_http(monkeypatch, 10, _resp(200, b"0123456789"))
        mm._pull_url("http://example.com/model.gguf", "mymodel")
        dest = models / "model.gguf"
        assert dest.read_bytes() == b"0123456789"
        assert sorted(p.name for p in models.iterdir()) == ["model.gguf"]
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
        _interrupt_pull(monkeypatch, models, "http://example.com/model.gguf",
                        10, b"01234")                        # 5 bytes already have
        cap = _wire_http(monkeypatch, 10,
                         _resp(206, b"56789", content_length=5))
        mm._pull_url("http://example.com/model.gguf", "mymodel")
        dest = models / "model.gguf"
        assert dest.read_bytes() == b"0123456789"            # suffix appended
        assert cap["headers"].get("Range") == "bytes=5-"

    def test_server_ignoring_range_restarts_clean(self, url_env, monkeypatch):
        models, _ = url_env
        _interrupt_pull(monkeypatch, models, "http://example.com/model.gguf",
                        11, b"STALE")                        # partial/garbage
        # 200 (full file) despite our Range request -> must overwrite, not append
        cap = _wire_http(monkeypatch, 11,
                         _resp(200, b"FULLCONTENT", content_length=11))
        mm._pull_url("http://example.com/model.gguf", "mymodel")
        dest = models / "model.gguf"
        assert dest.read_bytes() == b"FULLCONTENT"           # NOT b"STALEFULL..."
        assert cap["headers"].get("Range") == "bytes=5-"     # we did request resume

    def test_range_not_satisfiable_resets_and_retries(self, url_env, monkeypatch):
        models, _ = url_env
        # Every byte landed but the rename never ran: the next Range starts at
        # the end of the file.
        _interrupt_pull(monkeypatch, models, "http://example.com/model.gguf",
                        10, b"0123456789")
        calls = []
        on_disk_at_retry = []

        def fake_pinned_request(method, url, **kwargs):
            if method == "HEAD":
                h = MagicMock()
                h.status_code = 200
                h.headers = {"content-length": "10"}
                return h
            calls.append(dict(kwargs.get("headers") or {}))
            if len(calls) == 1:
                return _resp(416, b"")
            on_disk_at_retry.extend(p.name for p in models.iterdir())
            return _resp(200, b"0123456789")

        monkeypatch.setattr("localm.netpolicy.pinned_request", fake_pinned_request)
        assert mm._pull_url("http://example.com/model.gguf", "mymodel") is True
        assert len(calls) == 2
        assert calls[0].get("Range") == "bytes=10-"
        assert "Range" not in calls[1]
        assert on_disk_at_retry == [], "the retry ran beside the rejected partial"
        dest = models / "model.gguf"
        assert dest.read_bytes() == b"0123456789"
        assert sorted(p.name for p in models.iterdir()) == ["model.gguf"]

    def test_already_downloaded_skips_network(self, url_env, monkeypatch):
        models, _ = url_env
        (models / "model.gguf").write_bytes(b"already here")
        pin_spy = MagicMock()
        monkeypatch.setattr("localm.netpolicy.pinned_request", pin_spy)
        mm._pull_url("http://example.com/model.gguf", "mymodel")
        pin_spy.assert_not_called()


class TestUrlPartIdentity:
    """A direct-URL ``.part`` has a record beside it of which URL, digest and
    size it holds, and a later pull appends to that partial only when the
    record matches the file it is pulling."""

    def test_an_interrupted_pull_keeps_the_partial_and_its_record(
            self, url_env, monkeypatch):
        models, _ = url_env

        _interrupt_pull(monkeypatch, models, URL_A, 10, b"AAAAA")

        assert sorted(p.name for p in models.iterdir()) == [
            "model.gguf.part", "model.gguf.part.json"]

    def test_the_record_never_holds_the_url(self, url_env, monkeypatch):
        models, _ = url_env
        url = "http://host1.example/model.gguf?token=SECRET-TOKEN"

        _interrupt_pull(monkeypatch, models, url, 10, b"AAAAA")

        record = (models / "model.gguf.part.json").read_text(encoding="utf-8")
        assert "SECRET-TOKEN" not in record
        assert "host1.example" not in record

    def test_the_same_url_resumes_from_its_partial(self, url_env, monkeypatch):
        models, _ = url_env
        _interrupt_pull(monkeypatch, models, URL_A, 10, b"AAAAA")
        server = _RangeServer(_A)
        monkeypatch.setattr("localm.netpolicy.pinned_request", server)

        ok = mm._pull_url(URL_A, "mymodel")

        dest = models / "model.gguf"
        assert dest.is_file() and dest.read_bytes() == _A
        assert sorted(p.name for p in models.iterdir()) == ["model.gguf"]
        assert [g.get("Range") for g in server.gets] == ["bytes=5-"]
        assert ok is True

    def test_a_partial_of_another_url_is_not_resumed(self, url_env, monkeypatch):
        models, _ = url_env
        _interrupt_pull(monkeypatch, models, URL_A, 10, b"AAAAA")
        server = _RangeServer(_B)
        monkeypatch.setattr("localm.netpolicy.pinned_request", server)

        exc = None
        try:
            ok = mm._pull_url(URL_B, "mymodel")
        except Exception as e:
            ok, exc = None, e

        dest = models / "model.gguf"
        assert dest.is_file() and dest.read_bytes() == _B, (
            "the new file was not downloaded whole: "
            f"{dest.read_bytes() if dest.is_file() else None!r}")
        assert [g.get("Range") for g in server.gets] == [None]
        assert exc is None, f"the pull raised {exc!r}"
        assert ok is True

    @pytest.mark.parametrize("record", [None, b"{not json", b"[]"],
                             ids=["no-record", "unreadable-record", "not-an-object"])
    def test_a_partial_without_a_usable_record_is_not_resumed(
            self, record, url_env, monkeypatch):
        models, _ = url_env
        (models / "model.gguf.part").write_bytes(b"STALE")   # left by an older localm
        if record is not None:
            (models / "model.gguf.part.json").write_bytes(record)
        server = _RangeServer(_A)
        monkeypatch.setattr("localm.netpolicy.pinned_request", server)

        ok = mm._pull_url(URL_A, "mymodel")

        dest = models / "model.gguf"
        assert dest.is_file() and dest.read_bytes() == _A
        assert [g.get("Range") for g in server.gets] == [None]
        assert sorted(p.name for p in models.iterdir()) == ["model.gguf"]
        assert ok is True

    def test_a_record_with_no_partial_beside_it_is_ignored(
            self, url_env, monkeypatch):
        models, _ = url_env
        _interrupt_pull(monkeypatch, models, URL_A, 10, b"AAAAA").unlink()
        server = _RangeServer(_A)
        monkeypatch.setattr("localm.netpolicy.pinned_request", server)

        ok = mm._pull_url(URL_A, "mymodel")

        dest = models / "model.gguf"
        assert dest.is_file() and dest.read_bytes() == _A
        assert sorted(p.name for p in models.iterdir()) == ["model.gguf"]
        assert [g.get("Range") for g in server.gets] == [None]
        assert ok is True

    @pytest.mark.parametrize("first, second, body", [
        pytest.param({"expected_sha256": _sha(_A)}, {"expected_sha256": _sha(_B)},
                     _B, id="other-digest"),
        pytest.param({}, {"expected_sha256": _sha(_B)}, _B, id="digest-added"),
        pytest.param({}, {}, _B + b"+", id="other-size"),
    ])
    def test_a_partial_of_another_digest_or_size_is_not_resumed(
            self, first, second, body, url_env, monkeypatch):
        models, _ = url_env
        _interrupt_pull(monkeypatch, models, URL_A, 10, b"AAAAA", **first)
        server = _RangeServer(body)
        monkeypatch.setattr("localm.netpolicy.pinned_request", server)

        ok = mm._pull_url(URL_A, "mymodel", **second)

        dest = models / "model.gguf"
        assert dest.is_file() and dest.read_bytes() == body
        assert [g.get("Range") for g in server.gets] == [None]
        assert ok is True

    def test_a_digest_in_another_case_is_the_same_digest(self, url_env, monkeypatch):
        models, _ = url_env
        _interrupt_pull(monkeypatch, models, URL_A, 10, b"AAAAA",
                        expected_sha256=_sha(_A).upper())
        server = _RangeServer(_A)
        monkeypatch.setattr("localm.netpolicy.pinned_request", server)

        ok = mm._pull_url(URL_A, "mymodel", expected_sha256=_sha(_A))

        dest = models / "model.gguf"
        assert dest.is_file() and dest.read_bytes() == _A
        assert [g.get("Range") for g in server.gets] == ["bytes=5-"]
        assert ok is True

    def test_a_pull_that_learns_no_size_still_resumes_its_own_partial(
            self, url_env, monkeypatch):
        models, _ = url_env
        _interrupt_pull(monkeypatch, models, URL_A, 0, b"AAAAA")
        server = _RangeServer(_A, head_length=False)
        monkeypatch.setattr("localm.netpolicy.pinned_request", server)

        ok = mm._pull_url(URL_A, "mymodel")

        dest = models / "model.gguf"
        assert dest.is_file() and dest.read_bytes() == _A
        assert [g.get("Range") for g in server.gets] == ["bytes=5-"]
        assert ok is True

    def test_a_pull_interrupted_again_after_resuming_resumes_again(
            self, url_env, monkeypatch):
        import requests
        models, _ = url_env
        _interrupt_pull(monkeypatch, models, URL_A, 10, b"AAAAA")
        second = _RangeServer(_A, drop_after=3)
        monkeypatch.setattr("localm.netpolicy.pinned_request", second)
        try:
            mm._pull_url(URL_A, "mymodel")
        except requests.ConnectionError:
            pass
        assert (models / "model.gguf.part").read_bytes() == b"A" * 8
        third = _RangeServer(_A)
        monkeypatch.setattr("localm.netpolicy.pinned_request", third)

        ok = mm._pull_url(URL_A, "mymodel")

        dest = models / "model.gguf"
        assert dest.is_file() and dest.read_bytes() == _A
        assert [g.get("Range") for g in second.gets] == ["bytes=5-"]
        assert [g.get("Range") for g in third.gets] == ["bytes=8-"]
        assert ok is True

    def test_a_retry_that_fails_before_any_byte_keeps_the_partial_and_its_record(
            self, url_env, monkeypatch):
        import requests
        models, _ = url_env
        _interrupt_pull(monkeypatch, models, URL_A, 10, b"AAAAA")

        def _refused(method, url, **kwargs):
            if method == "HEAD":
                return MagicMock(status_code=200, headers={"content-length": "10"})
            resp = MagicMock()
            resp.status_code = 503
            err = requests.HTTPError("503 Service Unavailable")
            err.response = resp
            raise err

        monkeypatch.setattr("localm.netpolicy.pinned_request", _refused)
        failed = mm._pull_url(URL_A, "mymodel")
        kept = sorted(p.name for p in models.iterdir())
        kept_bytes = (models / "model.gguf.part").read_bytes()
        server = _RangeServer(_A)
        monkeypatch.setattr("localm.netpolicy.pinned_request", server)

        ok = mm._pull_url(URL_A, "mymodel")

        assert kept == ["model.gguf.part", "model.gguf.part.json"]
        assert kept_bytes == b"AAAAA"
        assert failed is False
        dest = models / "model.gguf"
        assert dest.is_file() and dest.read_bytes() == _A
        assert [g.get("Range") for g in server.gets] == ["bytes=5-"]
        assert ok is True

    def test_a_redirect_target_that_changes_each_pull_still_resumes(
            self, url_env, monkeypatch):
        models, _ = url_env
        hops = iter(range(10))
        monkeypatch.setattr("localm.model_manager.pull._ssrf_resolve_final_url",
                            lambda url: f"{url}?sig={next(hops)}")
        _interrupt_pull(monkeypatch, models, URL_A, 10, b"AAAAA")
        server = _RangeServer(_A)
        monkeypatch.setattr("localm.netpolicy.pinned_request", server)

        ok = mm._pull_url(URL_A, "mymodel")

        dest = models / "model.gguf"
        assert dest.is_file() and dest.read_bytes() == _A
        assert [g.get("Range") for g in server.gets] == ["bytes=5-"]
        assert ok is True

    def test_a_restart_over_another_urls_partial_reports_progress_from_zero(
            self, url_env, monkeypatch, capsys):
        import json
        models, _ = url_env
        _interrupt_pull(monkeypatch, models, URL_A, 10, b"AAAAA")
        capsys.readouterr()
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        monkeypatch.setattr("localm.netpolicy.pinned_request", _RangeServer(_B))

        ok = mm._pull_url(URL_B, "mymodel")

        out = capsys.readouterr().out
        events = [json.loads(line.split(mm.PROGRESS_SENTINEL, 1)[1])
                  for line in out.splitlines() if mm.PROGRESS_SENTINEL in line]
        downloads = [e for e in events if e.get("phase") == "download"]
        assert downloads, f"no download progress was reported: {out!r}"
        assert downloads[0]["downloaded"] == 0, downloads
        assert ok is True

    def test_a_checksum_mismatch_leaves_no_file_and_no_record(
            self, url_env, monkeypatch):
        models, _ = url_env
        monkeypatch.setattr("localm.netpolicy.pinned_request", _RangeServer(_A))

        ok = mm._pull_url(URL_A, "mymodel", expected_sha256=_sha(_B))

        assert list(models.iterdir()) == []
        assert ok is False


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
