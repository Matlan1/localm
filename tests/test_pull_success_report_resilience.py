# SPDX-License-Identifier: AGPL-3.0-or-later
"""A genuinely successful pull must not be reported as FAILED when the final
status print - a bare console.print of a green checkmark - crashes AFTER the
download, checksum verification and registry write are already done. Two
independent, unrelated exceptions hit that exact line in practice: a
ModuleNotFoundError from rich's cell-width lookup, and a UnicodeEncodeError from
a legacy Windows console write path.

The GUI runs `localm pull` as a subprocess (localm/plugins/gui/jobs.py) and
reduces the whole operation to `status = "done" if returncode == 0 else
"failed"`. So the return value of pull_model()/_pull_url()/_pull_hf_snapshot()
alone is one link short: what decides the user-visible outcome is the CLI
PROCESS'S EXIT CODE, so the last test below drives the real `pull` click command
through CliRunner rather than stopping at the Python-level return value.

The classes near the bottom of this file cover the SAME shape at three more call
sites: the mid-function "SHA256 verified" checkmark in pull_model()'s local-path
branch, in _pull_gguf_file(), and in _pull_url() - each printed BEFORE the
function's own trailing _report_success()-guarded message, but still AFTER the
real work (hashing, and in two of the three cases the registry write) is done.
Unlike the trailing checkmark, these sit in the MIDDLE of their function, so the
property is not just "does the return value survive" but "does execution reach
the registration code that follows the print at all".
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
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
            if "✓" in msg:      # the checkmark glyph
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
    # must pass through untouched.
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
        # The GUI spawns `localm pull` with LOCALM_PROGRESS_JSON=1.
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


# ------------------------------------------------------- the mid-function
# ------------------------------------------------ "SHA256 verified" checkmarks
#
# Three raw console.print(f"[green]checkmark[/green] SHA256 verified...") calls
# share the identical shape but sit BEFORE the rest of their function's work,
# so these tests assert that execution reaches the registration code that
# follows the print.

class TestLocalPathSurvivesACrashingVerifiedPrint:
    """pull_model()'s local-path branch (pull.py, is_local_path with
    --sha256): the checkmark sits between the real digest check and the
    add_local() call that actually registers the file."""

    def test_pull_model_local_path_still_registers_when_the_verified_print_raises(
            self, tmp_path, monkeypatch):
        body = b"a real local model file's bytes"
        f = tmp_path / "private-finetune.gguf"
        f.write_bytes(body)
        digest = hashlib.sha256(body).hexdigest()

        add_local_calls = []
        monkeypatch.setattr(
            pull_mod._mm, "add_local",
            lambda path_str, **kw: add_local_calls.append((path_str, kw)) or True)
        monkeypatch.setattr(pull_mod.console, "print", _raise_on_checkmark)

        result = pull_mod.pull_model(f.as_posix(), expected_sha256=digest)

        assert result is True, (
            "the SHA256 was already verified against the real bytes before "
            "this print ran - a crash rendering the checkmark must not turn "
            f"a verified local file into a reported failure, got {result!r}")
        assert add_local_calls, (
            "add_local() must still run after the crashing verified-checkmark "
            "print - it sits AFTER the print in the source, so a caller that "
            "does not swallow the crash never reaches it at all")


class TestPullGgufFileSurvivesACrashingVerifiedPrint:
    """_pull_gguf_file()'s --sha256 branch: the checkmark sits between the real
    digest check on the freshly-downloaded bytes and the metadata probe +
    _register() call that follows it."""

    def test_pull_gguf_file_still_registers_when_the_verified_print_raises(
            self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
        monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
        monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
        # None (not a real digest): the pre-download reconciliation at pull.py's
        # "want and expected and want != expected" only fires when HF metadata
        # is known: keep it unknown so the caller's --sha256 is the one actually
        # verified against the downloaded bytes below.
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo_id, fn: None)

        register_calls = []
        monkeypatch.setattr(
            mm, "_register",
            lambda *a, **k: register_calls.append((a, k)))

        body = b"a freshly downloaded gguf file's bytes"
        digest = hashlib.sha256(body).hexdigest()

        def _fake_download(repo_id, filename, local_dir, **kw):
            p = Path(local_dir) / filename
            p.write_bytes(body)
            return str(p)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
        monkeypatch.setattr(pull_mod.console, "print", _raise_on_checkmark)

        result = mm._pull_gguf_file("o/r:new.gguf", None, expected_sha256=digest)

        assert result is True, (
            "the download and SHA256 verification both completed - a crash "
            f"rendering the checkmark must not report this as failed, got {result!r}")
        assert register_calls, (
            "_register() must still run after the crashing verified-checkmark "
            "print - it sits AFTER the print in the source")


class TestPullUrlSurvivesACrashingVerifiedPrint:
    """_pull_url()'s --sha256 branch: the checkmark sits between the real
    digest check on the downloaded bytes and the duplicate-check + _register()
    call that follows it (distinct from TestPullUrlSurvivesACrashingSuccessPrint
    above, which covers the function's TRAILING checkmark and never passes
    expected_sha256, so it never reaches this earlier print at all)."""

    def test_pull_url_still_registers_when_the_verified_print_raises(
            self, url_env, monkeypatch):
        body = b"0123456789"
        digest = hashlib.sha256(body).hexdigest()
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        _wire_http(monkeypatch, len(body), _resp(200, body))
        monkeypatch.setattr(pull_mod.console, "print", _raise_on_checkmark)

        result = mm._pull_url("http://example.com/model.gguf", "mymodel",
                              expected_sha256=digest)

        assert result is True, (
            "the download and SHA256 verification both completed - a crash "
            f"rendering the checkmark must not report this as failed, got {result!r}")
        assert mm._register.call_count == 1, (
            "_register() must still run after the crashing verified-checkmark "
            "print - it sits AFTER the print in the source")
