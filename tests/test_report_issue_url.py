# SPDX-License-Identifier: AGPL-3.0-or-later
"""The standalone reporter must only ever POST to an http/https endpoint, and
must show the user WHERE the report is going before they confirm.

scripts/report_issue.py builds its outbound Request from the URL read_proxy()
returns. That URL has only two producers - the LOCALM_BUGREPORT_URL environment
variable and a regex over the install's own localm/config.py - so this is NOT
remotely reachable and these tests do not claim it is; an actor who can set
either already controls the process or the code. It is defence in depth on a sink
that hands whatever it is given to urlopen(), plus a visible destination so a
redirected reporter cannot be silent.

The override's documented purpose - pointing the reporter at your own proxy or a
test double - is preserved: http and https both stay accepted.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load the standalone script by path (scripts/ is not an importable package).
_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_issue.py"
_spec = importlib.util.spec_from_file_location("report_issue_url_under_test", _MOD_PATH)
ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ri)

REJECTED = ["file:///etc/passwd", "ftp://x/", "file://C:/Windows/win.ini",
            "gopher://x/1", "data:text/plain,hi", "javascript:alert(1)",
            "not-a-url", "", "https://", "https:///nohost"]
ACCEPTED = ["https://example.invalid/x", "http://127.0.0.1:8080/x",
            "https://proxy.example.workers.dev", "HTTPS://Example.Invalid/x"]


def _no_tty(monkeypatch):
    monkeypatch.setattr(
        ri.sys, "stdin", type("S", (), {"isatty": staticmethod(lambda: False)})())


def _explode_urlopen(monkeypatch):
    """Make the REAL sink fail the test if anything reaches it."""
    def _boom(*a, **k):
        raise AssertionError("urlopen() was reached for a rejected endpoint")
    monkeypatch.setattr(ri.urllib.request, "urlopen", _boom)


# ---------------------------- endpoint_is_allowed -------------------------- #

@pytest.mark.parametrize("url", REJECTED)
def test_endpoint_is_allowed_rejects_non_http(url):
    assert ri.endpoint_is_allowed(url) is False


@pytest.mark.parametrize("url", ACCEPTED)
def test_endpoint_is_allowed_accepts_http_and_https(url):
    assert ri.endpoint_is_allowed(url) is True


def test_endpoint_is_allowed_handles_a_malformed_authority():
    # urlsplit raises ValueError on a bad IPv6 literal; that must be a rejection,
    # not a crash inside the reporter's own failure path.
    assert ri.endpoint_is_allowed("http://[oops/") is False


# -------------------------------- read_proxy ------------------------------- #

@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x/"])
def test_read_proxy_drops_a_non_http_override(url, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOCALM_BUGREPORT_URL", url)
    monkeypatch.setenv("LOCALM_BUGREPORT_TOKEN", "tok")
    got_url, got_token = ri.read_proxy(tmp_path / "no-config.py")
    assert got_url is None
    assert got_token == "tok"          # only the endpoint is dropped
    # The rejection is reported, not silent.
    out = capsys.readouterr().out
    assert "Ignoring bug-report endpoint" in out
    assert "http:// and https://" in out


def test_read_proxy_does_not_silently_fall_back_to_the_shipped_url(tmp_path, monkeypatch):
    """A bad override must NOT be quietly replaced by the config default - that
    would override an explicit choice in silence. It returns nothing instead."""
    cfg = tmp_path / "config.py"
    cfg.write_text('D = {\n    "bugreport_upload_url": "https://shipped.example",\n'
                   '    "bugreport_upload_token": "t",\n}\n', encoding="utf-8")
    monkeypatch.setenv("LOCALM_BUGREPORT_URL", "file:///etc/passwd")
    url, _ = ri.read_proxy(cfg)
    assert url is None


@pytest.mark.parametrize("url", ["https://example.invalid/x", "http://127.0.0.1:8080/x"])
def test_read_proxy_keeps_an_http_override(url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_BUGREPORT_URL", url)
    monkeypatch.delenv("LOCALM_BUGREPORT_TOKEN", raising=False)
    got_url, _ = ri.read_proxy(tmp_path / "no-config.py")
    assert got_url == url


def test_read_proxy_still_reads_the_real_shipped_config(monkeypatch):
    monkeypatch.delenv("LOCALM_BUGREPORT_URL", raising=False)
    url, _ = ri.read_proxy()
    assert url and ri.endpoint_is_allowed(url)


# -------------------------------- post_report ------------------------------ #

@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x/", None, ""])
def test_post_report_refuses_before_any_opener_call(url, monkeypatch):
    _explode_urlopen(monkeypatch)

    def _must_not_open(u, data, hdrs, to):
        raise AssertionError(f"opener was called for a rejected endpoint {u!r}")

    with pytest.raises(RuntimeError, match="only http:// and https://"):
        ri.post_report(url, "tok", "t", "b", opener=_must_not_open)


@pytest.mark.parametrize("url", ["https://example.invalid/x", "http://127.0.0.1:8080/x"])
def test_post_report_still_sends_over_http_and_https(url):
    seen = {}

    def opener(u, data, hdrs, to):
        seen["url"] = u
        return 201, '{"url": "https://github.com/o/r/issues/1"}'

    res = ri.post_report(url, "tok", "t", "b", opener=opener)
    assert seen["url"] == url
    assert res["url"].endswith("/issues/1")


def test_post_report_rejection_message_does_not_leak_url_credentials(monkeypatch):
    # The opener and the real sink are BOTH stubbed to fail the test, so nothing
    # here can make a live outbound connection.
    _explode_urlopen(monkeypatch)

    def _must_not_open(u, data, hdrs, to):
        raise AssertionError(f"opener was called for a rejected endpoint {u!r}")

    with pytest.raises(RuntimeError) as e:
        ri.post_report("ftp://admin:SECRETPASS@host/x", None, "t", "b",
                       opener=_must_not_open)
    assert "SECRETPASS" not in str(e.value)
    # Rejected for the SCHEME, not incidentally by a network failure.
    assert "only http:// and https://" in str(e.value)


# ------------------------- end to end through main() ----------------------- #

@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x/"])
def test_main_never_sends_to_a_rejected_endpoint(url, tmp_path, monkeypatch, capsys):
    """The real path: a poisoned override reaches neither post_report nor
    urlopen, the run reports failure, and the report is saved locally."""
    _no_tty(monkeypatch)
    _explode_urlopen(monkeypatch)
    monkeypatch.setattr(ri, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("LOCALM_BUGREPORT_URL", url)

    rc = ri.main(["--yes", "--summary", "boom"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "Sent to the maintainer" not in out
    assert "Ignoring bug-report endpoint" in out
    saved = list((tmp_path / "home" / "bug-reports").glob("bug-*.md"))
    assert saved, "a report that could not be sent must still be saved"


def test_main_previews_the_destination_host_before_sending(tmp_path, monkeypatch, capsys):
    _no_tty(monkeypatch)
    monkeypatch.setattr(ri, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("LOCALM_BUGREPORT_URL", "https://redirected.example.invalid/report")
    monkeypatch.setattr(ri, "post_report", lambda *a, **k: {"url": "https://x/1"})

    rc = ri.main(["--yes", "--summary", "x"])

    assert rc == 0
    out = capsys.readouterr().out
    # The destination is visible in the block the user reviews, so a redirected
    # reporter cannot be silent.
    assert "Destination: redirected.example.invalid" in out
    assert out.index("Destination:") < out.index("Sent to the maintainer")


def test_main_says_when_no_destination_is_configured(tmp_path, monkeypatch, capsys):
    _no_tty(monkeypatch)
    monkeypatch.setattr(ri, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ri, "read_proxy", lambda: (None, None))

    rc = ri.main(["--yes", "--summary", "x"])

    assert rc == 1
    assert "Destination: none configured" in capsys.readouterr().out
