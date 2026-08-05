# SPDX-License-Identifier: AGPL-3.0-or-later
"""Item #25: a genuinely successful pull was reported as FAILED, because the
final status print - a bare console.print of a green checkmark - crashed AFTER
the download, checksum verification and registry write were already done. Two
independent, unrelated exceptions have hit this exact line in practice (a
ModuleNotFoundError from rich's cell-width lookup, a UnicodeEncodeError from a
legacy Windows console write path) - see
dev-notes/ROOTCAUSE-pull-success-reported-as-failed-2026-08-05.md.

The GUI runs `localm pull` as a subprocess (localm/plugins/gui/jobs.py) and
reduces the whole operation to `status = "done" if returncode == 0 else
"failed"`. So the return value of pull_model()/_pull_url()/_pull_hf_snapshot()
alone is one link short of the real bug: what actually decides the user-visible
outcome is the CLI PROCESS'S EXIT CODE, which is why the last test below drives
the real `pull` click command through CliRunner rather than stopping at the
Python-level return value.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from localm import model_manager as mm
from localm.model_manager import pull as pull_mod


# --------------------------------------------------------- _report_success itself

class TestReportSuccessNeverRaises:
    def test_falls_back_to_plain_text_and_logs_a_warning_on_render_failure(
            self, monkeypatch, caplog):
        calls = []

        def _fake_print(msg):
            calls.append(msg)
            if "✓" in msg:      # the checkmark glyph, exactly what crashes
                raise ModuleNotFoundError(
                    "No module named 'rich._unicode_data.unicode17-0-0'")

        monkeypatch.setattr(pull_mod.console, "print", _fake_print)
        with caplog.at_level(logging.WARNING):
            pull_mod._report_success("[green]✓[/green] ok", "[green]OK[/green] ok")

        assert calls == ["[green]✓[/green] ok", "[green]OK[/green] ok"], (
            "must try the rich message first, then fall back to the plain one - "
            f"got {calls}")
        assert any("could not render" in r.message for r in caplog.records), (
            "a render failure must be logged, not silently swallowed (rule 5)")

    def test_does_not_raise_even_if_the_plain_fallback_also_fails(
            self, monkeypatch, caplog):
        """The degrade path itself must not become a new way to crash the pull."""
        def _always_raise(msg):
            raise OSError("console is gone")

        monkeypatch.setattr(pull_mod.console, "print", _always_raise)
        with caplog.at_level(logging.WARNING):
            pull_mod._report_success("[green]✓[/green] ok", "[green]OK[/green] ok")
        assert sum("could not render" in r.message
                   or "fallback also failed" in r.message
                   for r in caplog.records) >= 2, (
            "both the render failure and the fallback failure must be logged")

    def test_the_happy_path_still_prints_the_rich_message_untouched(self, monkeypatch):
        calls = []
        monkeypatch.setattr(pull_mod.console, "print", calls.append)
        pull_mod._report_success("[green]✓[/green] ok", "[green]OK[/green] ok")
        assert calls == ["[green]✓[/green] ok"], (
            "the fix must not change output on the working case - only the "
            "plain fallback is new, and it must not fire when nothing is broken")


# --------------------------------------------------- the real _pull_url path

def _resp(status, body: bytes, content_length=None):
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
    monkeypatch.setattr(mm, "_register", MagicMock())
    monkeypatch.setattr(mm, "_register_with_dedup", MagicMock())
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    return models


def _wire_http(monkeypatch, head_total: int, response):
    def fake_pinned_request(method, url, **kwargs):
        if method == "HEAD":
            h = MagicMock()
            h.status_code = 200
            h.headers = {"content-length": str(head_total)}
            return h
        return response
    monkeypatch.setattr("localm.netpolicy.pinned_request", fake_pinned_request)


def _raise_on_checkmark(msg):
    # console.print() also carries non-string renderables (rich.control.Control
    # instances from Progress/Live's cursor-management calls) - only the final
    # status line is ever a plain string containing the glyph, so anything else
    # must pass through untouched or this mock crashes on input the real bug
    # never touches.
    if isinstance(msg, str) and "✓" in msg:
        raise ModuleNotFoundError(
            "No module named 'rich._unicode_data.unicode17-0-0'")


class TestPullUrlSurvivesACrashingSuccessPrint:
    def test_pull_url_still_returns_true_when_the_checkmark_print_raises(
            self, url_env, monkeypatch):
        body = b"0123456789"
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        _wire_http(monkeypatch, len(body), _resp(200, body))
        monkeypatch.setattr(pull_mod.console, "print", _raise_on_checkmark)

        result = mm._pull_url("http://example.com/model.gguf", "mymodel")

        assert result is True, (
            "the download, checksum path and registry write all completed - "
            "a crash in the trailing status print must not turn this into a "
            "reported failure")


# ------------------------------------- the actual CLI process's exit code

class TestCliPullExitCode:
    """The GUI job runner (localm/plugins/gui/jobs.py) decides success/failure
    from the SUBPROCESS EXIT CODE alone, not from any Python-level return
    value it never sees. Driving the real `pull` click command is what
    actually proves the user-visible bug is fixed - a passing return-value
    test one layer down could still leave `sys.exit(1)` reachable if the
    click command itself, or something between it and _pull_url, re-raised."""

    def test_pull_command_exits_zero_when_the_success_print_raises(
            self, url_env, monkeypatch):
        from localm.cli.models import pull

        body = b"0123456789"
        # Matches the real incident exactly: the GUI spawns `localm pull` with
        # LOCALM_PROGRESS_JSON=1 (jobs.py:296/618-619).
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        _wire_http(monkeypatch, len(body), _resp(200, body))
        monkeypatch.setattr(pull_mod.console, "print", _raise_on_checkmark)

        result = CliRunner().invoke(pull, ["http://example.com/model.gguf",
                                           "-n", "mymodel"])

        assert result.exit_code == 0, (
            f"a provably successful pull must not exit non-zero just because "
            f"its trailing status print crashed - exit_code={result.exit_code}, "
            f"output={result.output!r}"
        )
