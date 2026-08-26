# SPDX-License-Identifier: AGPL-3.0-or-later
"""The "Send to maintainer" upload channel: the app POSTs a user-reviewed report
to the configured proxy (a Cloudflare Worker holding the GitHub token), which
files it as a GitHub issue. No token ships in the app.

Pins: config gating, the POST shape + optional shared secret, and the honesty
rule - a failed upload raises (never a false success), and the saved file stays.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm import bugreport
from localm.inference.http_server import create_app
from tests._fake_https import patch_https_transport


# ------------------------------- config gating --------------------------- #

def test_upload_gate_off_when_url_absent(monkeypatch):
    # Opt-out path: a config with no bugreport_upload_url has no hosted upload
    # channel. A report still saves to a file.
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    assert bugreport.upload_config() == (None, None)
    assert bugreport.upload_available() is False


def test_hosted_channel_on_by_default(monkeypatch):
    # A fresh install (pure DEFAULT_CONFIG, no user config.json) has the hosted
    # channel live out of the box. All three surfaces read the one shipped default;
    # update/issues fall back to bugreport_upload_url/token.
    import copy

    from localm import config, issue_tracker, updater
    default = copy.deepcopy(config.DEFAULT_CONFIG)
    assert default["bugreport_upload_url"].startswith("https://")   # a real URL, not None
    monkeypatch.setattr("localm.config.load_config",
                        lambda: copy.deepcopy(config.DEFAULT_CONFIG))
    assert bugreport.upload_available() is True
    assert updater.available() is True
    assert issue_tracker.available() is True


def test_upload_config_reads_url_and_token(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "https://proxy.example/",
        "bugreport_upload_token": "  s3cret  ",
    })
    url, token = bugreport.upload_config()
    assert url == "https://proxy.example/"
    assert token == "s3cret"
    assert bugreport.upload_available() is True


# ------------------------------- upload_report ---------------------------- #

def test_upload_report_posts_and_returns_issue_url():
    seen = {}

    def opener(url, data, headers, timeout):
        import json
        seen["url"] = url
        seen["headers"] = headers
        seen["payload"] = json.loads(data.decode("utf-8"))
        return 201, '{"ok": true, "url": "https://github.com/x/localm/issues/9"}'

    res = bugreport.upload_report(
        "image gen froze", "## report\nbody",
        url="https://proxy.example/file", token="tok", opener=opener)
    assert res["url"].endswith("/issues/9")
    assert seen["url"] == "https://proxy.example/file"
    assert seen["payload"]["title"] == "image gen froze"
    assert seen["payload"]["body"] == "## report\nbody"
    # The shared secret rides a header, never the GitHub token (which is not ours).
    assert seen["headers"]["X-Localm-Token"] == "tok"
    assert not any("github" in k.lower() for k in seen["headers"])


def test_upload_report_strips_edit_disclaimer_from_body():
    """build_report()'s "you can edit anything above before sending"
    disclaimer (and the maintainer's email it names) is TRUE for a human
    reading the saved file or a downloaded copy. It stops being true, and
    re-publishes the email, once the SAME text is what actually got uploaded
    into a PUBLIC GitHub issue, so it is stripped at the upload choke point
    every caller (report_failure, inference/routes/admin.py) flows through."""
    from localm import bugreport as br

    real_report = br.build_report("image gen froze", context={"operation": "run"})
    assert br.MAINTAINER_EMAIL in real_report          # it really is there
    assert "You can edit anything above" in real_report

    seen = {}

    def opener(url, data, headers, timeout):
        import json
        seen["payload"] = json.loads(data.decode("utf-8"))
        return 201, '{"url": "https://github.com/x/localm/issues/2"}'

    bugreport.upload_report(
        "image gen froze", real_report,
        url="https://proxy.example/file", token="tok", opener=opener)

    sent_body = seen["payload"]["body"]
    assert bugreport.MAINTAINER_EMAIL not in sent_body
    assert "You can edit anything above" not in sent_body
    # The real diagnostic content is untouched - only the trailing footer is gone.
    assert "image gen froze" in sent_body
    assert "## App state" in sent_body


def test_upload_report_body_without_footer_is_unaffected():
    """A body that never carried the disclaimer (e.g. a hand-typed test body,
    or a user who deleted the footer themselves) uploads byte-for-byte."""
    seen = {}

    def opener(url, data, headers, timeout):
        import json
        seen["payload"] = json.loads(data.decode("utf-8"))
        return 201, '{"url": "https://github.com/x/localm/issues/3"}'

    bugreport.upload_report(
        "t", "## report\nno footer in this body at all",
        url="https://proxy.example/file", token="tok", opener=opener)
    assert seen["payload"]["body"] == "## report\nno footer in this body at all"


def test_upload_report_scrubs_home_path_in_title():
    """The title becomes a PUBLIC GitHub issue title. Scrubbing at the upload
    choke point means a home path (username) in ANY caller's title is redacted in
    what is actually SENT on the wire, whichever caller passed it (report_failure
    passes the raw summary, the GUI route the raw first line)."""
    seen = {}

    def opener(url, data, headers, timeout):
        import json
        seen["payload"] = json.loads(data.decode("utf-8"))
        return 201, '{"url": "https://github.com/x/localm/issues/1"}'

    bugreport.upload_report(
        r"froze at Z:\Users\bob\localm\gui", "## body\nno secrets here",
        url="https://proxy.example/file", token="tok", opener=opener)
    assert "bob" not in seen["payload"]["title"]
    assert "<redacted>" in seen["payload"]["title"]


def test_upload_report_no_endpoint_raises(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    with pytest.raises(bugreport.LocalmError):
        bugreport.upload_report("t", "b")


def test_upload_report_non_2xx_raises():
    def opener(url, data, headers, timeout):
        return 502, '{"error": "GitHub rejected the issue"}'

    with pytest.raises(bugreport.LocalmError) as ei:
        bugreport.upload_report("t", "b", url="https://proxy.example", opener=opener)
    assert "502" in (ei.value.reason or "")


def test_upload_report_429_raises_rate_limited():
    # A 429 raises the distinct RateLimitedError carrying retry_after (parsed from the
    # body), so callers can count down and retry instead of failing outright.
    def opener(url, data, headers, timeout):
        return 429, '{"error":"rate limited","retry_after":30}'

    with pytest.raises(bugreport.RateLimitedError) as ei:
        bugreport.upload_report("t", "b", url="https://proxy.example", token="x", opener=opener)
    assert ei.value.retry_after == 30
    assert isinstance(ei.value, bugreport.LocalmError)   # still a LocalmError subclass


def test_upload_report_omits_token_header_when_none(monkeypatch):
    # With no configured token and token=None, no X-Localm-Token header is sent.
    # The empty config is simulated explicitly, since a token ships in DEFAULT_CONFIG.
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    seen = {}

    def opener(url, data, headers, timeout):
        seen["headers"] = headers
        return 201, "{}"

    bugreport.upload_report("t", "b", url="https://proxy.example", token=None, opener=opener)
    assert "X-Localm-Token" not in seen["headers"]


def test_upload_report_fills_url_from_config_when_only_token_passed(monkeypatch):
    """An explicit token must not suppress loading the url from config (each of
    url/token defaults from config independently)."""
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "https://cfg.example/file",
        "bugreport_upload_token": "cfg-tok"})
    seen = {}

    def opener(url, data, headers, timeout):
        seen["url"] = url
        seen["headers"] = dict(headers)
        return 201, "{}"

    bugreport.upload_report("t", "b", token="explicit-tok", opener=opener)
    assert seen["url"] == "https://cfg.example/file"          # url loaded from config
    assert seen["headers"]["X-Localm-Token"] == "explicit-tok"  # explicit token wins


# ------------------------------- CLI menu --------------------------------- #

def test_cli_menu_upload_branch_calls_upload(tmp_path, monkeypatch):
    """When an upload endpoint is configured, the CLI report menu offers "[1] Send
    now", and picking it uploads the (edited) report rather than opening a browser."""
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    monkeypatch.setattr(bugreport, "upload_config",
                        lambda: ("https://proxy.example", "tok"))
    sent = {}

    def fake_upload(title, body, *, url=None, token=None, **kw):
        sent.update(title=title, url=url, token=token, body=body)
        return {"url": "https://github.com/x/localm/issues/3"}

    monkeypatch.setattr(bugreport, "upload_report", fake_upload)
    opened = []
    bugreport.report_failure(
        summary="bug", interactive=True,
        open_browser=lambda u: opened.append(u), prompt=lambda _t: "1")
    assert sent.get("url") == "https://proxy.example"
    assert sent.get("token") == "tok"
    assert opened == []   # upload does not open a browser


def test_cli_menu_no_upload_option_when_unconfigured(tmp_path, monkeypatch, capsys):
    """Without an endpoint, the menu has no "Send now" option and "1" maps to nothing
    (email is [2], not [1]) - so picking "1" opens no browser."""
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    monkeypatch.setattr(bugreport, "upload_config", lambda: (None, None))
    opened = []
    bugreport.report_failure(
        summary="bug", interactive=True,
        open_browser=lambda u: opened.append(u), prompt=lambda _t: "1")
    out = capsys.readouterr().out
    assert "Send to the maintainer now" not in out
    assert opened == []


def test_cli_menu_channels_stable_when_upload_configured(tmp_path, monkeypatch, capsys):
    """With upload configured, the upload option is [1] but email stays [2] and
    the manual/self channel stays [3] - the always-present channels are not
    renumbered."""
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    monkeypatch.setattr(bugreport, "upload_config",
                        lambda: ("https://proxy.example", "tok"))
    opened = []
    bugreport.report_failure(
        summary="bug", interactive=True,
        open_browser=lambda u: opened.append(u), prompt=lambda _t: "2")
    assert opened and opened[0].startswith(f"mailto:{bugreport.MAINTAINER_EMAIL}")
    opened.clear()
    bugreport.report_failure(
        summary="bug", interactive=True,
        open_browser=lambda u: opened.append(u), prompt=lambda _t: "3")
    assert opened == []   # [3] is the manual/self channel, not a browser open
    assert bugreport.MAINTAINER_EMAIL in capsys.readouterr().out


def test_cli_menu_upload_failure_shows_hint_and_retries(tmp_path, monkeypatch, capsys):
    """A failed send tells the user WHERE it failed (the diagnosed hint) and offers
    to retry creating the issue; answering yes re-attempts and can succeed."""
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    monkeypatch.setattr(bugreport, "upload_config",
                        lambda: ("https://proxy.example", "tok"))
    calls = {"n": 0}

    def flaky(title, body, *, url=None, token=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise bugreport.LocalmError(
                "could not reach the bug-report server", reason="getaddrinfo failed",
                stage="offline_or_dns", hint="You may be offline.")
        return {"url": "https://github.com/x/localm/issues/9"}

    monkeypatch.setattr(bugreport, "upload_report", flaky)
    answers = iter(["1", "y"])   # pick [1] Send, then retry = yes
    bugreport.report_failure(summary="bug", interactive=True,
                             prompt=lambda _t: next(answers, ""))
    out = capsys.readouterr().out
    assert "You may be offline." in out          # the diagnosed hint is shown
    assert calls["n"] == 2                        # retried after the failure
    assert "Sent to the maintainer." in out       # the retry succeeded


def test_cli_menu_upload_failure_decline_retry_keeps_file(tmp_path, monkeypatch, capsys):
    """Declining the retry does not re-attempt; the saved report + email fallback
    are pointed at (a failed send is never a false success)."""
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    monkeypatch.setattr(bugreport, "upload_config",
                        lambda: ("https://proxy.example", "tok"))
    calls = {"n": 0}

    def always_fail(title, body, *, url=None, token=None, **kw):
        calls["n"] += 1
        raise bugreport.LocalmError("could not reach the bug-report server",
                                    reason="refused", stage="unreachable",
                                    hint="The server may be down.")

    monkeypatch.setattr(bugreport, "upload_report", always_fail)
    answers = iter(["1", "n"])   # pick [1] Send, then decline retry
    bugreport.report_failure(summary="bug", interactive=True,
                             prompt=lambda _t: next(answers, ""))
    out = capsys.readouterr().out
    assert "The server may be down." in out
    assert calls["n"] == 1                        # NOT retried
    assert bugreport.MAINTAINER_EMAIL in out       # email fallback pointed at


# ------------------------------- endpoint --------------------------------- #

def _engine():
    e = MagicMock()
    e.display_name = "test-model"
    e.loaded = True
    return e


def _post(app, body):
    with TestClient(app) as c:
        return c.post("/api/bug-report", json=body,
                      headers={"Authorization": f"Bearer {app.state.shell_token}"})


def test_endpoint_uploads_on_request(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(bugreport, "upload_report",
                        lambda title, body: {"url": "https://github.com/x/localm/issues/12"})
    r = _post(create_app(_engine()),
              {"description": "the thing broke", "upload": True})
    assert r.status_code == 200
    data = r.json()
    assert data["saved"] is True
    assert data["uploaded"] is True
    assert data["issue_url"].endswith("/issues/12")
    assert Path(data["path"]).is_file()   # still saved to disk


def test_endpoint_upload_scrubs_home_path_end_to_end(tmp_path, monkeypatch):
    """Drive the REAL /api/bug-report upload route end to end - description ->
    save_user_report -> build_report -> upload_report -> the network POST - and
    assert the actual bytes on the wire carry NO username in the title OR the
    body. Only the socket is faked, so nothing between the user's field and the
    wire is mocked away."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    monkeypatch.setattr(bugreport, "upload_config",
                        lambda: ("https://proxy.example/file", "tok"))

    import json
    sent = {}

    class _Resp:
        status = 201

        def read(self):
            return b'{"url": "https://github.com/x/localm/issues/7"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None, context=None):
        sent["data"] = req.data
        return _Resp()

    patch_https_transport(monkeypatch, fake_urlopen)

    # The home path is on the FIRST line so it also flows into the derived title.
    desc = (r"Upload broke when I ran Z:\Users\bob\localm\gui\index.html"
            "\nClicked send five times, nothing happened.")
    r = _post(create_app(_engine()), {"description": desc, "upload": True})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["uploaded"] is True
    payload = json.loads(sent["data"].decode("utf-8"))
    # The GUI route builds the title from the raw first line; the upload choke point
    # scrubs it, so the username never reaches the public issue title OR the body.
    assert "bob" not in payload["title"]
    assert "bob" not in payload["body"]
    assert "<redacted>" in payload["title"]


def test_endpoint_upload_failure_is_surfaced_not_hidden(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)

    def boom(title, body):
        raise bugreport.LocalmError("could not reach the bug-report server",
                                    reason="timed out")

    monkeypatch.setattr(bugreport, "upload_report", boom)
    r = _post(create_app(_engine()),
              {"description": "broke again", "upload": True})
    # The save still succeeds (200), but the upload failure is reported honestly.
    assert r.status_code == 200
    data = r.json()
    assert data["saved"] is True
    assert data["uploaded"] is False
    assert "could not reach" in data["upload_error"]
    assert "issue_url" not in data
    assert Path(data["path"]).is_file()


def test_endpoint_does_not_upload_without_flag(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    called = []
    monkeypatch.setattr(bugreport, "upload_report",
                        lambda *a, **k: called.append(1) or {"url": "x"})
    r = _post(create_app(_engine()), {"description": "just save it"})
    assert r.status_code == 200
    assert "uploaded" not in r.json()
    assert called == []   # never uploads unless explicitly asked
