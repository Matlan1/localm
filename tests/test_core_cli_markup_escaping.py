# SPDX-License-Identifier: AGPL-3.0-or-later
"""Nine ``console.print(f"..."`` sites in localm/cli/_core.py interpolate a
config value, an exception, or a caller-supplied description directly into a
Rich markup f-string without escaping - the same bug
tests/test_rag_cli_markup_escaping.py documents for rag.py:

    Console().print('report[draft].txt')      -> prints "report.txt"
    Console().print('notes[bold red].md')     -> prints "notes.md"

The bracketed span is either dropped outright or consumed as a (bogus) style
directive, in both cases silently. ``_core.py`` is shared infrastructure
(the root CLI group, the graceful crash handler, the TLS/bind-host resolvers,
and the server-discovery reporting helpers every other CLI file calls
through), so a corrupted message here can surface from almost any command.

Two of the nine sites (`report_server_failure`'s *what*, `no_server_message`'s
*what*) are defense-in-depth only: every CURRENT caller passes a hardcoded
literal, so there is no real trigger today - tested directly against the
function, the same convention test_cli_comfy_lifecycle_and_cancel.py already
uses for `report_server_failure` itself
(test_the_two_404_meanings_do_not_print_the_same_sentence calls it directly
with capsys, not through a CLI command). The remaining sites have a genuine
real-world trigger (a hand-edited config.json's bind_host/tls_cert/tls_key, an
arbitrary exception message, an HTTP error body, a real filesystem directory
name) and are driven through the real function/command with that content
forced via monkeypatch, matching test_rag_cli_markup_escaping.py's own
TestLockMessageEscaping precedent (forcing a real exception rather than
fabricating one that cannot occur naturally).
"""

from __future__ import annotations

import click
import pytest
import requests
from click.testing import CliRunner

import localm.cli._core as core
import localm.cli.comfy as comfy_cli
from localm.cli._core import (
    _config_tls_pair, _resolve_bind_host, _setup_tls_or_exit, no_server_message,
    report_server_failure,
)

# One value Rich DROPS outright, one it consumes as a (bogus) style tag - the
# two distinct failure shapes test_rag_cli_markup_escaping.py's docstring
# describes.
BRACKET_DROP_TEXT = "report[draft].txt"
BRACKET_STYLE_TEXT = "notes[bold red].md"


@pytest.fixture(autouse=True)
def wide_console(monkeypatch):
    """rich.console.Console().size reads COLUMNS from os.environ live (not
    cached at construction - verified against the installed rich source), so
    this also widens _core.py's module-level `console` singleton even though
    it was already constructed at import time. Without it, an 80-col default
    can hard-wrap one of these longer messages mid-word, exactly as
    test_rag_cli_markup_escaping.py's own `env` fixture explains."""
    monkeypatch.setenv("COLUMNS", "300")


# --------------------------------------------------------------------------
#  A minimal, self-contained fake HTTP layer for the one real-CLI test below.
#  Own copy rather than importing test_cli_comfy_lifecycle_and_cancel.py's
#  richer _Server/_Resp/_install (module-private test helpers) - same
#  reasoning test_rag_cli_markup_escaping.py's own `env` fixture gives for
#  not importing a sibling file's fixture.
# --------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status, body):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body

    def json(self):
        return self._body

    @property
    def text(self):
        return str(self._body)


def _fake_request_returning(status, body):
    def _req(method, url, **kw):
        return _FakeResp(status, body)
    return _req


# --------------------------------------------------------------------------
#  _resolve_bind_host: a hand-edited config.json bypasses write-time
#  validation (see tests/test_bind_host_config.py's identical scenario,
#  which this borrows the fixture shape from).
# --------------------------------------------------------------------------

@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return cfg


class TestResolveBindHostMarkupEscaping:
    def test_bracket_style_bind_host_warning_survives_verbatim(self, cfg_home, capsys):
        import json
        cfg_home.CONFIG_FILE.write_text(
            json.dumps({"bind_host": BRACKET_STYLE_TEXT}), encoding="utf-8")
        host, from_config = _resolve_bind_host(None)
        assert host == "127.0.0.1" and from_config is False   # unchanged behavior
        out = capsys.readouterr().out
        assert BRACKET_STYLE_TEXT in out, (
            f"an invalid bind_host warning must echo the exact config value "
            f"verbatim, not have it mangled by Rich markup parsing: {out!r}")


# --------------------------------------------------------------------------
#  _config_tls_pair: a config-sourced tls_cert/tls_key path that does not
#  exist on disk - a real, reachable "file not found" report.
# --------------------------------------------------------------------------

class TestConfigTlsPairMarkupEscaping:
    def test_bracket_drop_cert_path_survives_verbatim(self, capsys):
        bad_path = f"/no/such/{BRACKET_DROP_TEXT}"
        result = _config_tls_pair({"tls_cert": bad_path, "tls_key": bad_path})
        assert result is None
        out = capsys.readouterr().out
        assert bad_path in out, (
            f"an unusable tls_cert path must be shown verbatim in the "
            f"fallback warning: {out!r}")


# --------------------------------------------------------------------------
#  _setup_tls_or_exit: TLS resolution can raise for many reasons (a broken
#  crypto stack, cert-generation failure); the handler must not corrupt
#  whatever text that exception carries.
# --------------------------------------------------------------------------

class TestSetupTlsOrExitMarkupEscaping:
    def test_bracket_style_exception_survives_verbatim(self, monkeypatch, capsys):
        def _boom(host, *, no_tls, tls_cert, tls_key):
            raise RuntimeError(f"cert generation failed: {BRACKET_STYLE_TEXT}")
        monkeypatch.setattr(core, "_resolve_tls", _boom)
        with pytest.raises(SystemExit) as ei:
            _setup_tls_or_exit("0.0.0.0", no_tls=False, tls_cert=None, tls_key=None)
        assert ei.value.code == 2
        out = capsys.readouterr().out
        assert BRACKET_STYLE_TEXT in out, (
            f"a TLS setup failure must echo the real exception text "
            f"verbatim: {out!r}")


# --------------------------------------------------------------------------
#  _GracefulGroup._report_failure's own fallback print (bugreport.report_
#  failure itself blowing up) - same scenario
#  tests/test_cli_graceful_handler.py::test_reporting_failure_does_not_recurse
#  exercises, extended with a bracketed exception message.
# --------------------------------------------------------------------------

class TestGracefulGroupFallbackMarkupEscaping:
    def test_bracket_drop_exception_survives_verbatim(self, tmp_path, monkeypatch):
        from localm import bugreport
        monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)

        def boom_reporter(**k):
            raise RuntimeError("reporter itself broke")
        monkeypatch.setattr(bugreport, "report_failure", boom_reporter)

        @click.group(cls=core._GracefulGroup)
        def g():
            pass

        @g.command()
        def boom():
            raise RuntimeError(f"real failure: {BRACKET_DROP_TEXT}")

        res = CliRunner().invoke(g, ["boom"])
        assert res.exit_code == 1
        assert "localm failed" in res.output
        assert BRACKET_DROP_TEXT in res.output, (
            f"the fallback failure line must echo the real exception text "
            f"verbatim, not have it mangled: {res.output!r}")


# --------------------------------------------------------------------------
#  report_server_failure - direct calls, matching
#  test_cli_comfy_lifecycle_and_cancel.py::
#  test_the_two_404_meanings_do_not_print_the_same_sentence's own precedent
#  of calling this function directly with capsys.
# --------------------------------------------------------------------------

class TestReportServerFailureMarkupEscaping:
    def test_unreachable_payload_and_what_survive_verbatim(self, capsys):
        """Covers both `what` (defense-in-depth: no real caller sends
        anything but a literal) and `payload` (defense-in-depth for the
        "unreachable" branch: real payload is always type(e).__name__, a
        Python identifier, but this shared helper cannot enforce that for a
        future caller) in one call, one of each bracket shape."""
        what = f"reach {BRACKET_STYLE_TEXT}"
        report_server_failure("unreachable", BRACKET_DROP_TEXT, what)
        out = capsys.readouterr().out
        assert BRACKET_DROP_TEXT in out, f"payload must survive verbatim: {out!r}"
        assert BRACKET_STYLE_TEXT in out, f"what must survive verbatim: {out!r}"

    def test_missing_what_survives_verbatim(self, capsys):
        report_server_failure("missing", None, BRACKET_STYLE_TEXT)
        out = capsys.readouterr().out
        assert BRACKET_STYLE_TEXT in out, f"what must survive verbatim: {out!r}"

    def test_unsupported_what_survives_verbatim(self, capsys):
        report_server_failure("unsupported", None, BRACKET_DROP_TEXT)
        out = capsys.readouterr().out
        assert BRACKET_DROP_TEXT in out, f"what must survive verbatim: {out!r}"

    def test_unauthorized_what_survives_verbatim(self, capsys):
        report_server_failure("unauthorized", None, BRACKET_STYLE_TEXT)
        out = capsys.readouterr().out
        assert BRACKET_STYLE_TEXT in out, f"what must survive verbatim: {out!r}"

    def test_http_detail_survives_verbatim_direct(self, capsys):
        """`detail` is genuinely untrusted (server-response text). `code` is
        r.status_code from `requests` - always a real int in every path that
        reaches here (verified against server_call's every "http" return),
        so it is left unescaped and asserted to render plainly here."""
        report_server_failure("http", (500, BRACKET_DROP_TEXT), "do the thing")
        out = capsys.readouterr().out
        assert BRACKET_DROP_TEXT in out, f"detail must survive verbatim: {out!r}"
        assert "500" in out

    def test_http_detail_survives_verbatim_through_real_comfy_start(self, monkeypatch):
        """The genuinely reachable case: a real `localm comfy start` whose
        discovered server answers the status probe with a non-2xx response
        carrying a 'detail' body - server-response text, not typed by the
        local operator at all. Drives the actual CLI command
        (localm/cli/comfy.py:219-226), matching
        test_cli_comfy_lifecycle_and_cancel.py's own _install()/_Server()
        convention (a fake requests.request plus a patched running_server),
        just with a self-contained fake response (see _FakeResp above)."""
        monkeypatch.setattr(
            requests, "request",
            _fake_request_returning(500, {"detail": BRACKET_STYLE_TEXT}))
        monkeypatch.setattr(comfy_cli, "running_server",
                            lambda **kw: ("http://127.0.0.1:9999", {}))
        result = CliRunner().invoke(comfy_cli.comfy_start, [])
        assert result.exit_code == 1, result.output
        assert BRACKET_STYLE_TEXT in result.output, (
            f"a server error body's 'detail' text must survive verbatim "
            f"through the real 'localm comfy start' command: {result.output!r}")


# --------------------------------------------------------------------------
#  no_server_message
# --------------------------------------------------------------------------

class TestNoServerMessageMarkupEscaping:
    def test_what_survives_verbatim_direct(self, capsys):
        """Defense-in-depth: every current caller passes a hardcoded literal
        (comfy.py's "starting/stopping/restarting ComfyUI", models.py's
        "cancelling an operation"), so there is no real trigger today -
        called directly, matching report_server_failure's own precedent."""
        no_server_message(BRACKET_DROP_TEXT)
        out = capsys.readouterr().out
        assert BRACKET_DROP_TEXT in out, f"what must survive verbatim: {out!r}"

    def test_directory_survives_verbatim_through_real_comfy_start(self, monkeypatch):
        """The genuinely reachable case: `instances.resolve_root_dir()`
        returns a real filesystem path (walked up from cwd), which can
        legitimately contain brackets - a directory named "Project [WIP]" is
        an ordinary real-world name. Drives the real `localm comfy start`
        with NO discovered server (the natural, un-mocked path - no
        registry entry exists in the throwaway LOCALM_HOME the autouse
        fixture already provides), with resolve_root_dir's return value
        forced via monkeypatch, matching TestLockMessageEscaping's own
        approach of forcing a value that cannot be produced any other way
        in a test."""
        import localm.instances as instances_mod
        monkeypatch.setattr(instances_mod, "resolve_root_dir",
                            lambda *a, **k: BRACKET_STYLE_TEXT)
        monkeypatch.setattr(comfy_cli, "running_server", lambda **kw: None)
        result = CliRunner().invoke(comfy_cli.comfy_start, [])
        assert result.exit_code == 1, result.output
        assert BRACKET_STYLE_TEXT in result.output, (
            f"the 'Directory:' line must show the real resolved path "
            f"verbatim through the real 'localm comfy start' command: "
            f"{result.output!r}")
