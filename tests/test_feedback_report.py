# SPDX-License-Identifier: AGPL-3.0-or-later
"""Feedback and suggestions (Settings > Feedback and Suggestion, ``localm
feedback``): only the person's text, the localm version and the type are saved
and sent, scrubbed like a bug report, through the same review and send path. A
bug report is sent without a type."""

import json
import re
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

import localm.config as cfg
from localm import bugreport
from localm._version import read_version
from localm.bugreport.persistence import _is_saved_report_name
from localm.inference.http_server import create_app


def _engine():
    e = MagicMock()
    e.display_name = "secret-model-name"
    e.loaded = True
    return e


@pytest.fixture
def open_mode(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)


def _post(c, app, body, path="/api/feedback"):
    return c.post(path, json=body,
                  headers={"Authorization": f"Bearer {app.state.shell_token}"})


def _expected(title, text):
    return (f"# {title}\n\n{text}\n\nlocalm {read_version()}\n\n---\n"
            f"Sent to the localm maintainer ({bugreport.MAINTAINER_EMAIL}). "
            "You can edit anything above before sending.\n")


def _saved():
    d = cfg.HOME_DIR / "bug-reports"
    return sorted(d.glob("*.md")) if d.is_dir() else []


def _fake_server(monkeypatch):
    """Route uploads to a list of the JSON bodies that would have been sent."""
    sent = []

    def opener(url, data, headers, timeout):
        sent.append(json.loads(data))
        return 201, json.dumps({"url": "https://example.invalid/issues/9"})

    real = bugreport.upload_report

    def upload(title, body, **kw):
        kw.update(url="https://example.invalid/r", token="", opener=opener)
        return real(title, body, **kw)

    monkeypatch.setattr(bugreport, "upload_config", lambda: ("https://example.invalid/r", ""))
    monkeypatch.setattr(bugreport, "upload_report", upload)
    return sent


# --- the text -------------------------------------------------------------

@pytest.mark.parametrize("text, title, body", [
    ("  An idea.\nMore of it.  ", "An idea.", "An idea.\nMore of it."),
    ("\ufeff\n\nAn idea", "An idea", "An idea"),
    ("# a heading\ntext", "# a heading", "# a heading\ntext"),
    ("---\nafter", "---", "---\nafter"),
])
def test_the_report_holds_only_the_text_and_the_version(text, title, body):
    assert bugreport.build_feedback(text) == _expected(title, body)


def test_the_text_and_the_title_are_scrubbed():
    text = bugreport.build_feedback(
        "key sk-" + "Arch08OpenAiKey0123456789" + " in C:\\Users\\alice\\notes.txt mail bob@example.com")
    assert "sk-Arch08" not in text
    assert "alice" not in text
    assert "bob@example.com" not in text


def test_a_long_first_line_is_cut_in_the_title_only():
    line = "word " * 100
    report = bugreport.build_feedback(line)
    title = bugreport.report_title_from_markdown(report)
    assert len(title) <= 120 and title == title.strip()
    assert line.strip() in report


# --- the saved file -------------------------------------------------------

@pytest.mark.parametrize("name, kind", [
    ("bug-20261124-120000.md", "bug"), ("feedback-20261124-120000.md", "feedback"),
    ("suggestion-20261124-120000-2.md", "suggestion"),
    ("feedback-.md", None), ("feedback.md", None), ("feedback", None), ("bug.md", None),
    ("other-1.md", None), ("../feedback-1.md", None), ("feedback-a/b.md", None),
    ("feedback-a\\b.md", None), ("feedback-1.txt", None), ("feedback-..md", None),
    ("Feedback-1.md", None), ("", None), (None, None), (5, None),
])
def test_saved_report_names(name, kind):
    assert _is_saved_report_name(name) is (kind is not None)
    if kind is None:
        with pytest.raises(ValueError, match="not a saved bug-report file name"):
            bugreport.saved_report_kind(name)
    else:
        assert bugreport.saved_report_kind(name) == kind


def test_another_kind_is_refused():
    with pytest.raises(ValueError, match="not a feedback kind"):
        bugreport.save_feedback("x", "bug")
    assert _saved() == []


# --- the GUI route --------------------------------------------------------

@pytest.mark.parametrize("kind", ["feedback", "suggestion"])
def test_the_route_saves_the_text_and_sends_nothing(open_mode, monkeypatch, kind):
    sent = _fake_server(monkeypatch)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = _post(c, app, {"kind": kind, "text": "An idea", "include_log": True,
                           "client": {"userAgent": "UAMARK"}})
    assert r.status_code == 200, r.text
    data = r.json()
    assert re.fullmatch(kind + r"-[\w-]+\.md", data["filename"])
    assert data["report_markdown"] == _expected("An idea", "An idea")
    assert [p.name for p in _saved()] == [data["filename"]]
    assert sent == []


@pytest.mark.parametrize("body", [
    {"text": "x"}, {"kind": "bug", "text": "x"}, {"kind": ["feedback"], "text": "x"},
    {"kind": None, "text": "x"}, {"kind": "feedback"}, {"kind": "feedback", "text": "  "},
    {"kind": "feedback", "text": "\ufeff"}, {"kind": "feedback", "text": 5},
])
def test_the_route_refuses_and_saves_nothing(open_mode, body):
    app = create_app(_engine())
    with TestClient(app) as c:
        r = _post(c, app, body)
    assert r.status_code == 400, r.text
    assert _saved() == []


def test_the_route_needs_the_shell_token_in_open_mode(open_mode):
    with TestClient(create_app(_engine())) as c:
        r = c.post("/api/feedback", json={"kind": "feedback", "text": "x"})
    assert r.status_code == 403
    assert _saved() == []


def test_a_caller_other_than_the_operator_gets_the_text_path_scrubbed(open_mode, monkeypatch):
    from localm.inference import http_server
    monkeypatch.setattr(http_server, "path_formatter",
                        lambda request: lambda value: str(value).replace("SECRETPATH", "<path>"))
    app = create_app(_engine())
    with TestClient(app) as c:
        r = _post(c, app, {"kind": "feedback", "text": "Why SECRETPATH?\nSECRETPATH again"})
    assert r.status_code == 200, r.text
    (path,) = _saved()
    assert path.read_text(encoding="utf-8") == _expected("Why <path>?", "Why <path>?\n<path> again")


_SAVES = [
    ("/api/feedback", {"kind": "feedback", "text": "x"}, "feedback"),
    ("/api/feedback", {"kind": "suggestion", "text": "x"}, "suggestion"),
    ("/api/bug-report", {"description": "x"}, "bug report"),
]


@pytest.mark.parametrize("path, body, what", _SAVES)
def test_a_failed_save_names_what_was_not_saved(open_mode, monkeypatch, path, body, what):
    monkeypatch.setattr(bugreport, "save_report", lambda *a, **k: None)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = _post(c, app, body, path=path)
    assert r.status_code == 500
    assert r.json()["detail"] == f"Could not save the {what} to disk."


@pytest.mark.parametrize("path, body, what", _SAVES)
def test_a_slow_save_names_what_timed_out(open_mode, monkeypatch, path, body, what):
    from localm.inference.routes import admin

    async def too_slow(*a, **k):
        raise admin.ThreadCallTimeout("save")

    monkeypatch.setattr(admin, "run_in_threadpool_bounded", too_slow)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = _post(c, app, body, path=path)
    assert r.status_code == 504
    assert r.json()["detail"].startswith(f"Saving the {what} timed out: ")


# --- sending --------------------------------------------------------------

@pytest.mark.parametrize("kind", ["feedback", "suggestion"])
def test_the_reviewed_text_is_sent_with_its_type(open_mode, monkeypatch, kind):
    sent = _fake_server(monkeypatch)
    app = create_app(_engine())
    with TestClient(app) as c:
        saved = _post(c, app, {"kind": kind, "text": "An idea"}).json()
        r = _post(c, app, {"filename": saved["filename"], "sha256": saved["report_sha256"]},
                  path="/api/bug-report/send")
    assert r.json()["uploaded"] is True, r.text
    assert sent == [{"title": "An idea", "type": kind,
                     "body": _expected("An idea", "An idea").split("\n\n---\n")[0]}]


def test_a_bug_report_is_sent_without_a_type(open_mode, monkeypatch):
    sent = _fake_server(monkeypatch)
    app = create_app(_engine())
    with TestClient(app) as c:
        saved = _post(c, app, {"description": "it broke"}, path="/api/bug-report").json()
        r = _post(c, app, {"filename": saved["filename"], "sha256": saved["report_sha256"]},
                  path="/api/bug-report/send")
    assert r.json()["uploaded"] is True, r.text
    assert [sorted(s) for s in sent] == [["body", "title"]]
    assert saved["filename"].startswith("bug-")


@pytest.mark.parametrize("kind, subject", [
    ("bug", "localm bug: x"), ("feedback", "localm feedback: x"), ("suggestion", "localm suggestion: x")])
def test_the_email_subject_names_the_kind(kind, subject):
    url = bugreport.mailto_url("x", "body", kind=kind)
    assert parse_qs(urlsplit(url).query)["subject"] == [subject]


# --- the terminal ---------------------------------------------------------

@pytest.mark.parametrize("flags, kind", [([], "feedback"), (["--suggestion"], "suggestion")])
def test_the_command_saves_and_sends_nothing_by_itself(cli_runner, monkeypatch, flags, kind):
    from localm.cli import main
    sent = _fake_server(monkeypatch)
    result = cli_runner.invoke(main, ["feedback", "-m", "An idea", *flags])
    assert result.exit_code == 0, result.output
    (path,) = _saved()
    assert path.name.startswith(kind + "-")
    assert path.read_text(encoding="utf-8") == _expected("An idea", "An idea")
    assert sent == []
    assert "localm bug-report" not in result.output


@pytest.mark.parametrize("args", [["feedback"], ["feedback", "-m", "   "], ["feedback", "-m", "\ufeff"]])
def test_the_command_refuses_empty_text(cli_runner, args):
    from localm.cli import main
    result = cli_runner.invoke(main, args)
    assert result.exit_code == 1, result.output
    assert _saved() == []


@pytest.mark.parametrize("flags, kind", [([], "feedback"), (["--suggestion"], "suggestion")])
def test_the_command_sends_the_type_on_request(cli_runner, monkeypatch, flags, kind):
    from localm.cli import main
    sent = _fake_server(monkeypatch)
    result = cli_runner.invoke(main, ["feedback", "-m", "An idea", "--send", *flags])
    assert result.exit_code == 0, result.output
    assert [(s["title"], s["type"]) for s in sent] == [("An idea", kind)]


def test_the_terminal_menu_sends_the_type(monkeypatch):
    sent = _fake_server(monkeypatch)
    path = bugreport.save_feedback("An idea", "suggestion")
    bugreport.offer_to_send("An idea", path, "", prompt=lambda text: "1", kind="suggestion")
    assert [s["type"] for s in sent] == ["suggestion"]


def test_the_terminal_menu_emails_under_the_kind():
    opened = []
    path = bugreport.save_feedback("An idea", "suggestion")
    bugreport.offer_to_send("An idea", path, "", prompt=lambda text: "2", kind="suggestion",
                            open_browser=opened.append)
    assert parse_qs(urlsplit(opened[0]).query)["subject"] == ["localm suggestion: An idea"]


def test_a_refused_send_names_the_command_that_was_used(cli_runner, monkeypatch):
    from localm.cli import main

    def refused(*a, **k):
        raise bugreport.LocalmError("off", reason="off", stage="network_off", hint="Network access is off.")

    monkeypatch.setattr(bugreport, "upload_config", lambda: ("https://example.invalid/r", ""))
    monkeypatch.setattr(bugreport, "upload_report", refused)
    result = cli_runner.invoke(main, ["feedback", "--suggestion", "-m", "An idea", "--send"])
    assert result.exit_code == 1, result.output
    assert "localm feedback --suggestion --send-anyway" in result.output
    assert "localm bug-report" not in result.output
